import itertools
from pathlib import Path
import os
import json
import pandas as pd

subdirs = ["test", "train"]
subdir_list = "list"
sample_dir_rmsd = "rmsd"
standart_columns = [
    'nmin_aa',
    'nmin_bb',
    'min_aa',
    'min_bb',
    'ros_aa',
    'ros_bb',
]


def process_sub_dir(dataset_dir, subdir=None):
    if subdir is not None:
        sub_dir_path = dataset_dir / subdir
    else:
        sub_dir_path = dataset_dir
    sub_dir_content = os.listdir(sub_dir_path)

    if_list = next((s for s in sub_dir_content if subdir_list in s), None)

    get_sub_dir_set = lambda dir_path, dir_content, dataframe: {
        p: [
            (lambda path:
             next((path / s for s in os.listdir(path) if sample_dir_rmsd in s), None)
             )(dir_path / p),
            (lambda dframe, p:
             dframe.loc[dframe['dir'] == p, 'length'].iloc[0]
             if dframe is not None
             else None
             )(dataframe, p)
        ]
        for p in dir_content if os.path.isdir(dir_path / p)
    }

    if if_list is not None:
        sub_dir_list = sub_dir_path / if_list
        df = pd.read_csv(sub_dir_list, ",")
        sub_dir_set = get_sub_dir_set(sub_dir_path, sub_dir_content, df)
    else:
        sub_dir_set = get_sub_dir_set(sub_dir_path, sub_dir_content, None)
    return sub_dir_set


def process_dataset_dir(path: str) -> dict:
    dataset_dir = Path(path)
    dataset_dir_content = os.listdir(path)
    subsets = dict()
    for subdir in subdirs:
        if subdir in dataset_dir_content:
            subsets[subdir] = process_sub_dir(dataset_dir, subdir)
    if not subsets:
        subsets["dataset"] = process_sub_dir(dataset_dir)
    return subsets


def repr_jsonify_datasets(datasets: dict) -> dict:
    return {str(k):
                {str(s[0]):
                     {"path": str(s[1]), "length": str(s[2])}
                 for s in v}
            for k, v in datasets.items()}


def repr_jsonify_processed(datasets: dict, processed: dict, columns: list) -> dict:
    return {k:
        {s[0]:
            dict(
                zip(
                    ['rmsd_csv_path', 'length', *columns],
                    (lambda lst, k:
                     [str(ss)
                      if type(ss) is not str
                      else str(datasets[k][ss][0])
                      for ss in lst])
                    (s, k)
                )
            )
            for s in v}
        for k, v in processed.items()}


def calc_threshold_rmsd(dataset: dict, threshold: float, columns: list) -> list:
    samples = []
    for k, v in dataset.items():
        df = pd.read_csv(v[0], ",")
        df.rename(columns={'Unnamed: 0': 'conformation'}, inplace=True)

        samples.append([k, v[1], *list(itertools.chain(
            *[
                (lambda dataframe, column: [
                    len(dataframe[(dataframe[column] <= threshold)]),
                    len(dataframe.index)
                ])(df, column) for column in [column for column in df if column in columns]
            ]
        )
        )])

    return samples


def get_dataframe_stats(subset_name: str, subset: list, columns: list):
    columns_ext = ['complex', 'length', *columns]
    dataframe = pd.DataFrame(subset, columns=columns_ext)
    dataframe_1 = dataframe[['complex', 'length']]
    nine_dim = dataframe[dataframe['length'] <= 9]
    nine_dim_1 = nine_dim[['complex', 'length']]
    # dataframe.to_csv(str(subset_name) + "_complexes.csv", index_label='id', sep=",", encoding='utf-8')
    calc_percentages = lambda df, cols: pd.concat(
        list(itertools.chain([df[i] / df[j] for i, j in zip(columns[0::2], columns[1::2])])), axis=1,
        keys=columns[0::2])
    calculated_df = calc_percentages(dataframe, columns)

    calculated_nine_dim = calc_percentages(nine_dim, columns)
    summ_df = dict(calculated_df.sum() / len(calculated_df))
    summ_nine_dim = dict(calculated_nine_dim.sum() / len(calculated_nine_dim))

    return dict(zip(
        ["orig_average", "nine_dim_average", "orig_df", "nine_dim_df"],
        [summ_df, summ_nine_dim, pd.concat([dataframe_1, calculated_df], axis=1),
         pd.concat([nine_dim_1, calculated_nine_dim], axis=1)]
    ))


class Analytics:
    def __init__(self, path, rmsd_threshold, ratio_threshold):
        self.columns = self.set_columns()
        self.datasets = process_dataset_dir(path)
        self.set_rmsd_threshold(rmsd_threshold)
        self.set_ratio_threshold(ratio_threshold)
        self.processed_datasets = self.process_datasets()

    def set_columns(self):
        return list(itertools.chain(*[[item, item + "_c"] for item in standart_columns]))

    def set_rmsd_threshold(self, rmsd_threshold):
        self.rmsd_treshold = rmsd_threshold

    def set_ratio_threshold(self, ratio_threshold):
        self.ratio_treshold = ratio_threshold

    def process_datasets(self):
        return {k: calc_threshold_rmsd(v, self.rmsd_treshold, self.columns) for k, v in self.datasets.items()}

    def process_stats(self):
        res = {}
        for k, v in self.processed_datasets.items():
            res[k] = get_dataframe_stats(k, v, self.columns)
        return res

    def __repr__(self):
        return json.dumps(
            repr_jsonify_processed(self.datasets, self.processed_datasets, self.columns),
            indent=2
        )
