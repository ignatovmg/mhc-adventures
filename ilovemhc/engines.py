import pandas as pd
import numpy as np
import os

import torch
from torch.utils.data import Dataset, DataLoader

from wrappers import *
import utils
import grids
import dataset

import ignite
from ignite.engine.engine import Engine, State, Events
from ignite.engine import Events, create_supervised_trainer
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.handlers import ModelCheckpoint, Timer

from torch import nn
import torch.nn.functional as F
import torch.optim

from sklearn import metrics
import scipy.stats as stats

def _my_create_evaluator(model, 
                         metrics={}, 
                         add_index=False, 
                         device=None, 
                         non_blocking=False, 
                         prepare_batch=ignite.engine._prepare_batch):
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch[:2], device=device, non_blocking=non_blocking)
            y_pred = model(x)
            
            if add_index:
                index = batch[2]
                return {'prediction': y_pred, 'target': y, 'index': index}
            else:
                return {'prediction': y_pred, 'target': y}

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine

def prepended_print(prepend, output):
    lines = str(output).split('\n')
    result = '\n'.join([prepend + x for x in lines])
    print(result)

def regression_stats(df, ascending=False):
    '''
    df (pandas.DataFrame) - Must contain columns "prediction", "target", "target_orig", "tag"
    '''
    
    output = {}
    for name, group in df.groupby('tag'):
        to_fill = {}
        
        group = group.sort_values('prediction', ascending=ascending)
        to_fill['best_found_value'] = group['target_orig'].iloc[0]                                                      
        group['order'] = range(group.shape[0])
        
        group = group.sort_values('target', ascending=ascending)
        order = group['order'].values
        to_fill['best_existing_value'] = group['target_orig'].iloc[0]
        
        rank = list(order).index(0)
        to_fill['rank'] = rank
        to_fill['percentile'] = float(rank) / len(order)
        to_fill['corr'] = np.corrcoef(group['prediction'], group['target'])[0, 1]
        to_fill['ktau'] = stats.kendalltau(order, np.arange(len(order))).correlation
        
        output[name] = to_fill
        
    return output

def regression_trainer_with_tagwise_statistics(model, optim, loss, test_loader, test_table, device, model_dir='models', model_prefix='model'):
    # set up trainer
    trainer = create_supervised_trainer(model, optim, loss, device=device)
    
    # keep all the models
    handler = ModelCheckpoint(model_dir, 
                              model_prefix, 
                              save_interval=1, 
                              create_dir=True, 
                              require_empty=False, 
                              atomic=True, n_saved=100)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {'model': model})

    # set up evaluator
    running_loss = RunningAverage(Loss(loss, output_transform=lambda x: [x['prediction'], x['target']]))
    loss_metric = Loss(loss, output_transform=lambda x: [x['prediction'], x['target']])
    metrics = {'Loss': loss_metric, 'Running MSE': running_loss}
    evaluator = _my_create_evaluator(model, metrics=metrics, device=device, add_index=True)

    # set up timers
    timer_it = Timer(average=False)
    timer_it.attach(trainer, start=Events.ITERATION_STARTED, pause=Events.ITERATION_COMPLETED, resume=Events.ITERATION_STARTED)
    timer_it.attach(evaluator, start=Events.ITERATION_STARTED, pause=Events.ITERATION_COMPLETED, resume=Events.ITERATION_STARTED)
    
    timer_epoch = Timer(average=False)

    # here we collect the output of the model for the entire validation set
    acc_val_output = []
    def trans_output_to_store(output):
        newdict = {}
        newdict['index'] = output['index'].cpu().numpy()
        newdict['prediction'] = output['prediction'].cpu().numpy()[:, 0]
        newdict['target'] = output['target'].cpu().numpy()
        return pd.DataFrame(newdict)
    
    current_epoch = 0
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_every_iteration(trainer):
        print("EPOCH[{}] TRAIN Iteration[{}] Time[{:.2f}s] LOSS {:.2f}".format(trainer.state.epoch, 
                                                                                  trainer.state.iteration, 
                                                                                  trainer.state.output,
                                                                                  timer_it.value()))
    @trainer.on(Events.EPOCH_STARTED)
    def log_epoch_started(trainer):
        global current_epoch
        current_epoch = trainer.state.epoch
        
        timer_epoch.resume()
        print("\nEPOCH[{}] TRAIN STARTED").format(trainer.state.epoch)    

    @trainer.on(Events.EPOCH_COMPLETED)   
    def log_epoch_completion(trainer):
        global acc_val_output
        
        timer_epoch.pause()
        print("EPOCH[{}] TRAIN COMPLETED Time[{:.2f}s]").format(trainer.state.epoch, timer_epoch.value())
        
        timer_epoch.reset()

        # Run validation
        evaluator.run(test_loader)
        
        # Calculate statistics
        acc_val_output = pd.concat(acc_val_output)
        val_index = acc_val_output['index'].values
        acc_val_output.index = val_index
        acc_val_output['tag'] = test_table['tag'].iloc[val_index]
        acc_val_output['target_orig'] = test_table['target'].iloc[val_index]
        
        print acc_val_output
        stats_table = pd.DataFrame(regression_stats(acc_val_output))
        stats_table = stats_table.transpose()
        
        # Save statistics
        stats_file = os.path.join(model_dir, model_prefix) + '_' + str(trainer.state.epoch) + '.csv'
        stats_table.to_csv(stats_file, float_format='%.3f')
        
        prepend = "EPOCH[{}] VALID ".format(trainer.state.epoch)
        print(prepend + '========    Statistics   =======')
        prepended_print(prepend, stats_table)
        print(prepend + '======== Mean statistics =======')
        prepended_print(prepend, stats_table.mean())

    @evaluator.on(Events.EPOCH_STARTED)
    def log_epoch_started_eval(evaluator):
        global acc_val_output, current_epoch
        acc_val_output = []
        
        timer_epoch.resume()
        print("EPOCH[{}] VALID STARTED").format(current_epoch)  
        
    @evaluator.on(Events.EPOCH_COMPLETED)   
    def log_epoch_completion_eval(evaluator):
        global current_epoch
        timer_epoch.pause()
        print("EPOCH[{}] VALID COMPLETED Time[{:.2f}s] Loss {:.2f}".format(current_epoch, 
                                                                          timer_epoch.value(), 
                                                                          evaluator.state.metrics['Loss']))
        timer_epoch.reset()
    
    @evaluator.on(Events.ITERATION_COMPLETED)
    def log_every_iteration_eval(evaluator):
        global acc_val_output, current_epoch
        
        niter = evaluator.state.iteration
        mse = evaluator.state.metrics['Running MSE']
        print("EPOCH[{}] VALID Iteration[{}] Time[{:.2f}s] RUNNING_MSE {:.2f}".format(current_epoch, 
                                                                                      niter, 
                                                                                      timer_it.value(), 
                                                                                      mse))
        timer_it.reset()
        acc_val_output.append(trans_output_to_store(evaluator.state.output))
    
    return trainer
    
    