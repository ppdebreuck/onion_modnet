from modnet.preprocessing import MODData
import numpy as np
import pandas as pd
import copy
from modnet.hyper_opt import FitGenetic
from sklearn.model_selection import KFold
from modnet.preprocessing import MODData


def shuffle_MD(data, random_state=10):
    data = copy.deepcopy(data)
    ids = data.df_targets.sample(frac=1, random_state=random_state).index
    data.df_featurized = data.df_featurized.loc[ids]
    data.df_targets = data.df_targets.loc[ids]
    data.df_structure = data.df_structure.loc[ids]

    return data

def MDKsplit(data, n_splits=5, exp_split=True, random_state=10):

    if exp_split:  # hardcoded split
        f1, f2 = pd.read_pickle("data/fold-ids.pkl")
        fold_ids = [(f1, f2), (f2, f1)]
        folds = []
        for train_idx, val_idx in fold_ids:
            data_train = MODData(
                data.df_structure.loc[train_idx]["structure"].values,
                data.df_targets.loc[train_idx].values,
                target_names=data.df_targets.columns,
                structure_ids=train_idx,
            )
            data_train.df_featurized = data.df_featurized.loc[train_idx]
            # data_train.optimal_features = data.optimal_features

            data_val = MODData(
                data.df_structure.loc[val_idx]["structure"].values,
                data.df_targets.loc[val_idx].values,
                target_names=data.df_targets.columns,
                structure_ids=val_idx,
            )
            data_val.df_featurized = data.df_featurized.loc[val_idx]
            # data_val.optimal_features = data.optimal_features

            folds.append((data_train, data_val))
    else:
        data = shuffle_MD(data, random_state=random_state)
        ids = np.array(data.structure_ids)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        folds = []
        for train_idx, val_idx in kf.split(ids):
            data_train = MODData(
                data.df_structure.iloc[train_idx]["structure"].values,
                data.df_targets.iloc[train_idx].values,
                target_names=data.df_targets.columns,
                structure_ids=ids[train_idx],
            )
            data_train.df_featurized = data.df_featurized.iloc[train_idx]
            # data_train.optimal_features = data.optimal_features

            data_val = MODData(
                data.df_structure.iloc[val_idx]["structure"].values,
                data.df_targets.iloc[val_idx].values,
                target_names=data.df_targets.columns,
                structure_ids=ids[val_idx],
            )
            data_val.df_featurized = data.df_featurized.iloc[val_idx]
            # data_val.optimal_features = data.optimal_features

            folds.append((data_train, data_val))

    return folds


def MD_append(md, lmd):
    md = copy.deepcopy(md)
    for m in lmd:
        md.df_structure = md.df_structure.append(m.df_structure)
        md.df_targets = md.df_targets.append(m.df_targets)
        md.df_featurized = md.df_featurized.append(m.df_featurized)

    md = shuffle_MD(md)
    return md

def MD_append_and_set(md, lmd, model_denoiser):
    md = copy.deepcopy(md)
    for m in lmd:
        orig_len = len(md.df_structure)
        md.df_structure = md.df_structure.append(m.df_structure)
        md.df_featurized = md.df_featurized.append(m.df_featurized)
        md.df_targets = md.df_targets.append(m.df_targets)
        pred = model_denoiser.predict(md)
        for idx, p in enumerate(pred['gap']):
            if idx>orig_len and abs(p - md.df_targets['gap'][idx]) > 0.3:
                md.df_targets['gap'][idx] = p
    md = shuffle_MD(md)
    return md

def get_params(data, n_jobs=4):
    ga = FitGenetic(data)
    ga.run(n_jobs=n_jobs, fast=False)
    hp = ga.results
    hp["targets"] = [[["gap"]]]
    hp["weights"] = {"gap": 1}
    hp["epochs"] = 800
    hp["num_neurons"] = [
        [int(hp["n_neurons_first_layer"])],
        [int(hp["n_neurons_first_layer"] * hp["fraction1"])],
        [int(hp["n_neurons_first_layer"] * hp["fraction1"] * hp["fraction2"])],
        [
            int(
                hp["n_neurons_first_layer"]
                * hp["fraction1"]
                * hp["fraction2"]
                * hp["fraction3"]
            )
        ],
    ]
    return hp
