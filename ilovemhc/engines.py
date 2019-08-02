import pandas as pd
import numpy as np
import os
import logging

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score

from wrappers import *
import utils
import grids
import dataset

import ignite
from ignite.engine.engine import Engine, State, Events
from ignite.engine import Events, create_supervised_trainer
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.handlers import ModelCheckpoint, Timer

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim

from sklearn import metrics
import scipy.stats as stats

from path import Path

def cuda_is_avail():
    logging.info("Checking CUDA availability ...")
    avail = torch.cuda.is_available()
    logging.info("CUDA is available yay" if avail else "CUDA is not available :-(")
    return avail


def get_device(device_name):
    avail = cuda_is_avail()
    
    if avail:
        ngpu = torch.cuda.device_count()
        logging.info('There are %i GPUs in total' % ngpu)
        device = torch.device(device_name)
    else:
        ngpu = None
        device = torch.device("cpu")
        
    logging.info('Using ' + str(device))
    
    return avail, device
    

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
            
            #if add_index:
            index = batch[2]
            return {'prediction': y_pred, 'target': y, 'idx': index}
            #else:
            #    return {'prediction': y_pred, 'target': y}

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def _my_create_supervised_trainer(model, optimizer, loss_fn,
                              device=None, 
                              non_blocking=False,
                              prepare_batch=ignite.engine._prepare_batch):
    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            x, y = prepare_batch(batch[:2], device=device, non_blocking=non_blocking)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
        #logging.info(y_pred.shape)
        return {'loss': loss.item(), 'prediction': y_pred, 'target': y}

    return Engine(_update)


def prepended_print(prepend, output):
    lines = str(output).split('\n')
    result = '\n'.join([prepend + x for x in lines])
    logging.info('\n' + result)


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
        to_fill['min_value'] = group['target_orig'].min()
        to_fill['med_value'] = group['target_orig'].median()
        to_fill['max_value'] = group['target_orig'].max()
        
        rank = list(order).index(0)
        to_fill['rank'] = rank
        to_fill['percentile'] = float(rank) / len(order)

        try:
            rocauc = roc_auc_score(group['target'], group['prediction'])
        except Exception as e:
            logging.exception(e)
            rocauc = np.nan
        to_fill['AUC'] = rocauc

        to_fill['corr'] = np.corrcoef(group['prediction'], group['target'])[0, 1]
        to_fill['ktau'] = stats.kendalltau(order, np.arange(len(order))).correlation
        to_fill['size'] = group.shape[0]
        
        output[name] = to_fill
        
    return output


def make_evaluator(model, loss_fn, device, model_dir='models', model_prefix='model', every_niter=100):
    def _inference_model(batch):
        with torch.no_grad():
            x = batch[0].to(device)
            #y = batch[1].to(device)
            y = batch[1].type(torch.LongTensor).to(device)
            index = batch[2]
            y_pred = model(x)

            #y_pred = y_pred[:, 0]
            #loss = loss_fn(y_pred, y)

            matrix = torch.cat([1 - y_pred, y_pred], 1)
            loss = loss_fn(matrix, y)
            y_pred = y_pred[:, 0]

            return {'loss': loss.item(), 'prediction': y_pred, 'target': y, 'idx': index}
        
    def run(test_loader, test_table, epoch=1):
        logging.info('VALID EPOCH[%i] STARTED' % epoch)
        
        all_output = []
        def _trans_output_to_store(output):
            newdict = {}
            newdict['idx'] = output['idx'].cpu().numpy()
            newdict['prediction'] = output['prediction'].cpu().numpy() #[:, 0]
            newdict['target'] = output['target'].cpu().numpy()
            return pd.DataFrame(newdict)
        
        model.eval()
        
        for niter, batch in enumerate(test_loader, 1):
            batch_size = batch[0].shape[0]
            output = _inference_model(batch)
            logging.info('VALID EPOCH[%i] Iteration[%i] Loss = %.5f' % (epoch, niter, output['loss']))
            
            if niter % every_niter == 0:
                logging.info(output)
            
            loc_table = _trans_output_to_store(output)
            all_output.append(loc_table)
            
        # Calculate statistics
        all_output = pd.concat(all_output)
        indices = all_output['idx'].values
        all_output.index = indices
        all_output['tag'] = test_table['tag'].iloc[indices]
        all_output['target_orig'] = test_table['target'].iloc[indices]
        
        # Save output
        save_dir = Path(model_dir)
        save_dir.mkdir_p()
        output_file = save_dir.joinpath(model_prefix + '-output-%i.csv' % epoch)
        logging.info('Saving model output to %s' % output_file)
        all_output.to_csv(output_file, float_format='%.3f', index=True)
        
        # Save statistics
        stats_table = pd.DataFrame(regression_stats(all_output))
        stats_table = stats_table.transpose()
        stats_file = save_dir.joinpath(model_prefix + '-stats-%i.csv' % epoch)
        logging.info('Saving model statistics to %s' % stats_file)
        stats_table.to_csv(stats_file, float_format='%.3f')
        
        logging.info('========    Statistics   =======')
        logging.info(stats_table)
        logging.info('======== Mean statistics =======')
        logging.info(stats_table.mean())
        
        logging.info('VALID EPOCH[%i] COMPLETED' % epoch)
        
    return run


def make_trainer(model, 
                 optimizer, 
                 loss_fn, 
                 device, 
                 evaluator, 
                 test_loader, 
                 test_table, 
                 model_dir='models', 
                 model_prefix='model', 
                 every_niter=100):
    
    def _update_model(batch):
        x = batch[0].to(device)
        #y = batch[1].type(torch.FloatTensor).to(device)
        y = batch[1].type(torch.LongTensor).to(device)
        optimizer.zero_grad()
        
        with torch.set_grad_enabled(True):
            y_pred = model(x) # [:, 0]
            matrix = torch.cat([1 - y_pred, y_pred], 1)
            y_pred = y_pred[:, 0]

            #loss = loss_fn(y_pred, y)
            loss = loss_fn(matrix, y)
            loss.backward()
            optimizer.step()

        return {'loss': loss.item(), 'prediction': y_pred, 'target': y}

    def run(train_loader, nepoch):
        
        for epoch in range(1, nepoch + 1):
            epoch_loss = []
            block_loss = []
            block_nitems = 0
            epoch_nitems = 0
            
            model.train()
            
            logging.info('TRAIN EPOCH[%i] STARTED' % epoch)
            for niter, batch in enumerate(train_loader, 1):
                output = _update_model(batch)
                
                batch_size = batch[0].size(0)
                epoch_loss.append(output['loss'] * batch_size)
                block_loss.append(output['loss'] * batch_size)
                block_nitems += batch_size
                epoch_nitems += batch_size
                
                logging.info('TRAIN EPOCH[%i] Iteration[%i] Loss = %.5f' % (epoch, niter, output['loss']))
                if niter % every_niter == 0:
                    
                    logging.info('TRAIN EPOCH[%i] Iteration[%i-%i] BlockLoss = %.5f' % (epoch, max(1, niter-every_niter), niter, 
                                                                                   (sum(block_loss) / block_nitems)))
                    logging.info(output)
                    block_loss = []
                    block_nitems = 0
                    
            logging.info('TRAIN EPOCH[%i] COMPLETED' % epoch)
            logging.info('TRAIN EPOCH[%i] AverageLoss = %.5f' % (epoch, sum(epoch_loss) / epoch_nitems))
            epoch_loss = []
            epoch_nitems = 0
            
            save_dir = Path(model_dir)
            save_dir.mkdir_p()
            save_file = save_dir.joinpath(model_prefix + '-model-%i.pth' % epoch)
            logging.info('TRAIN EPOCH[%i] Saving model to %s' % (epoch, save_file))
            
            with open(str(save_file), 'w') as f:
                torch.save(model, f)

            evaluator(test_loader, test_table, epoch)
            
    return run


# ============== deprecated ==============
def regression_evaluator_with_tagwise_statistics(model, loss, test_table, device, stats_prefix):
    assert(test_table.loc[:, ['tag','target','path']].dropna().shape[0] == test_table.shape[0])
    
    # set up evaluator
    running_loss = RunningAverage(Loss(loss, output_transform=lambda x: [x['prediction'], x['target']]))
    loss_metric = Loss(loss, output_transform=lambda x: [x['prediction'], x['target']])
    metrics = {'Loss': loss_metric, 'Running MSE': running_loss}
    evaluator = _my_create_evaluator(model, metrics=metrics, device=device, add_index=True)
    
    timer_epoch = Timer(average=False)
    timer_epoch.attach(evaluator, start=Events.EPOCH_STARTED, pause=Events.EPOCH_COMPLETED)
    
    timer_it = Timer(average=False)
    timer_it.attach(evaluator, start=Events.ITERATION_STARTED, pause=Events.ITERATION_COMPLETED, resume=Events.ITERATION_STARTED)
    
    if not hasattr(evaluator, 'trainer_epoch'):
        evaluator.trainer_epoch = 1
    
    # here we collect the output of the model for the entire validation set
    acc_val_output = []
    def trans_output_to_store(output):
        newdict = {}
        newdict['idx'] = output['idx'].cpu().numpy()
        newdict['prediction'] = output['prediction'].cpu().numpy()[:, 0]
        newdict['target'] = output['target'].cpu().numpy()
        return pd.DataFrame(newdict)
    
    @evaluator.on(Events.EPOCH_STARTED)
    def log_epoch_started_eval(evaluator):
        global acc_val_output
        acc_val_output = []
        
        logging.info("EPOCH[{}] VALID STARTED".format(evaluator.trainer_epoch))
        
    @evaluator.on(Events.EPOCH_COMPLETED)   
    def log_epoch_completion_eval(evaluator):
        global acc_val_output
        logging.info("EPOCH[{}] VALID COMPLETED Time[{:.2f}s] Loss {:.2f}".format(evaluator.trainer_epoch, 
                                                                          timer_epoch.value(), 
                                                                          evaluator.state.metrics['Loss']))
        
        # Calculate statistics
        acc_val_output = pd.concat(acc_val_output)
        val_index = acc_val_output['idx'].values
        acc_val_output.index = val_index
        acc_val_output['tag'] = test_table['tag'].iloc[val_index]
        acc_val_output['target_orig'] = test_table['target'].iloc[val_index]
        
        output_file = stats_prefix + '_output_' + str(evaluator.trainer_epoch) + '.csv'
        acc_val_output.to_csv(output_file, float_format='%.3f', index=False)
        
        stats_table = pd.DataFrame(regression_stats(acc_val_output))
        stats_table = stats_table.transpose()
        
        # Save statistics
        stats_file = stats_prefix + '_stats_' + str(evaluator.trainer_epoch) + '.csv'
        stats_table.to_csv(stats_file, float_format='%.3f')
        
        prepend = "EPOCH[{}] VALID ".format(evaluator.trainer_epoch)
        logging.info('\n' + prepend + '========    Statistics   =======')
        prepended_print(prepend, stats_table)
        logging.info('\n' + prepend + '======== Mean statistics =======')
        prepended_print(prepend, stats_table.mean())
    
    @evaluator.on(Events.ITERATION_COMPLETED)
    def log_every_iteration_eval(evaluator):
        global acc_val_output
        
        niter = evaluator.state.iteration
        mse = evaluator.state.metrics['Running MSE']
        logging.info("EPOCH[{}] VALID Iteration[{}] Time[{:.2f}s] RUNNING_MSE {:.2f}".format(evaluator.trainer_epoch, 
                                                                                      niter, 
                                                                                      timer_it.value(), 
                                                                                      mse))
        #timer_it.reset()
        acc_val_output.append(trans_output_to_store(evaluator.state.output))
        
    return evaluator

def regression_trainer_with_tagwise_statistics(model, optim, loss, test_loader, test_table, device, model_dir='models', model_prefix='model', every_n_iter=100):
    # set up trainer
    logging.info('Creating trainer..')
    trainer = _my_create_supervised_trainer(model, optim, loss, device=device)
    
    # keep all the models
    handler = ModelCheckpoint(model_dir, 
                              model_prefix, 
                              save_interval=1, 
                              create_dir=True, 
                              require_empty=False, 
                              atomic=True, n_saved=100)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {'model': model})

    stats_prefix = os.path.join(model_dir, model_prefix)
    logging.debug('Creating evaluator..')
    evaluator = regression_evaluator_with_tagwise_statistics(model, loss, test_table, device, stats_prefix)

    # set up timers
    timer_it = Timer(average=False)
    timer_it.attach(trainer, start=Events.ITERATION_STARTED, pause=Events.ITERATION_COMPLETED, resume=Events.ITERATION_STARTED)
    timer_epoch = Timer(average=False)
    
    current_epoch = 0
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_every_iteration(trainer):
        output = trainer.state.output
        logging.info("EPOCH[{}] TRAIN Iteration[{}] Time[{:.2f}s] LOSS {:f}".format(trainer.state.epoch, 
                                                                              trainer.state.iteration,
                                                                              timer_it.value(),
                                                                              output['loss']))
        if trainer.state.iteration % every_n_iter == 1:
            prepend = "EPOCH[{}] TRAIN Iteration[{}] ".format(trainer.state.epoch, trainer.state.iteration)
            pred = output['prediction'][:5]
            targ = output['target'][:5]
            outstr = ''.join(['%f %f\n' % x for x in zip(pred, targ)])
            output_example = 'prediction, target\n'
            output_example = output_example + outstr
            prepended_print(prepend, output_example)
            
        
    @trainer.on(Events.EPOCH_STARTED)
    def log_epoch_started(trainer):
        timer_epoch.resume()
        logging.info("\nEPOCH[{}] TRAIN STARTED".format(trainer.state.epoch))

    @trainer.on(Events.EPOCH_COMPLETED)   
    def log_epoch_completion(trainer):
        global acc_val_output
        
        timer_epoch.pause()
        logging.info("EPOCH[{}] TRAIN COMPLETED Time[{:.2f}s]".format(trainer.state.epoch, timer_epoch.value()))
        timer_epoch.reset()

        # Run validation
        evaluator.trainer_epoch = trainer.state.epoch
        evaluator.run(test_loader)
        
    return trainer
