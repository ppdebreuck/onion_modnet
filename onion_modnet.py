from modnet.preprocessing import MODData
from modnet.models import MODNetModel
import numpy as np
import os
from tensorflow.keras.callbacks import EarlyStopping
from modnet.preprocessing import MODData
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

k = 2
random_state = 202010
n_jobs = 64


def train_phase(model, train, test, ph_int, f_int, **fit_params):
    es = EarlyStopping(
        monitor="loss",
        min_delta=0.001,
        patience=30,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
    )
    callbacks = [es]

    model.fit(train, callbacks=callbacks, **fit_params)

    model.save("out/MODNet_onion_{}_ph{}".format(f_int, ph_int))

    pred = model.predict(test)
    true = test.df_targets
    error = pred - true
    mae = np.abs(error.values).mean()

    with open("results/out_onion.txt", "a") as fp:
        fp.write("mae ph{} - f{}: {:.3f}\n".format(ph_int, f_int + 1, mae))
    return mae


def main():
    md_exp = MODData.load("data/exp_gap_all")
    md_exp.df_targets.columns = ["gap"]
    md_pbe = MODData.load("data/pbe_gap.zip")
    md_pbe.df_targets.columns = ["gap"]
    md_hse = MODData.load("data/hse_gap.zip")
    md_hse.df_targets.columns = ["gap"]
    md_gllb = MODData.load("data/gllb_gap.zip")
    md_gllb.df_targets.columns = ["gap"]
    md_scan = MODData.load("data/scan_md.zip")
    md_scan.df_targets.columns = ["gap"]

    # only use common features
    common_feats = set(md_exp.df_featurized.columns)
    for d in [md_pbe, md_hse, md_gllb, md_scan]:
        common_feats = common_feats.intersection(set(d.df_featurized.columns))
    for d in [md_exp, md_pbe, md_hse, md_gllb, md_scan]:
        d.df_featurized = d.df_featurized[list(common_feats)]

    folds = MDKsplit(md_exp, n_splits=k, random_state=random_state)
    maes_ph1 = np.ones(k)
    maes_ph2 = np.ones(k)
    maes_ph3 = np.ones(k)
    maes_ph4 = np.ones(k)
    maes_ph5 = np.ones(k)
    for i, f in enumerate(folds):
        train = f[0]
        test = f[1]
        fpath = "train_folds/train_{}_{}".format(random_state, i + 1)
        if os.path.exists(fpath):
            train = MODData.load(fpath)
            train.df_targets.columns = ["gap"]
        else:
            train.feature_selection(n=-1, n_jobs=n_jobs)
            train.save(fpath)

        # assure no overlap
        assert (
            len(set(train.df_targets.index).intersection(set(test.df_targets.index)))
            == 0
        )

        # find hyper_params
        hp = get_params(train, n_jobs)

        # fit params
        fit_params = {
            "loss": hp["loss"],
            "lr": hp["lr"],
            "epoch": hp["epochs"],
            "batch_size": hp["batch_size"],
            "xscale": hp["xscale"],
            "verbose": 0,
        }

        model = MODNetModel(
            targets=hp["targets"],
            weights=hp["weights"],
            n_feat=hp["n_feat"],
            num_neurons=hp["num_neurons"],
            act=hp["act"],
            num_classes={"gap": 0},
        )

        # phase 1
        md = MD_append(train, [md_pbe, md_hse, md_gllb, md_scan])
        mae = train_phase(model, md, test, 1, i, **fit_params)
        maes_ph1[i] = mae

        # phase 2
        md = MD_append(train, [md_pbe, md_hse, md_scan])
        mae = train_phase(model, md, test, 2, i, **fit_params)
        maes_ph2[i] = mae

        # phase 3
        md = MD_append(train, [md_hse, md_scan])
        mae = train_phase(model, md, test, 3, i, **fit_params)
        maes_ph3[i] = mae

        # phase 4
        md = MD_append(train, [md_hse])
        mae = train_phase(model, md, test, 4, i, **fit_params)
        maes_ph4[i] = mae

        # phase 5
        mae = train_phase(model, train, test, 5, i, **fit_params)
        maes_ph5[i] = mae

    with open("results/out_onion.txt", "a") as fp:
        fp.write("2-fold Summary\n")
        fp.write("mae ph1 : {:.3f}\n".format(maes_ph1.mean()))
        fp.write("mae ph2 : {:.3f}\n".format(maes_ph2.mean()))
        fp.write("mae ph3 : {:.3f}\n".format(maes_ph3.mean()))
        fp.write("mae ph4 : {:.3f}\n".format(maes_ph4.mean()))
        fp.write("mae ph5 : {:.3f}\n".format(maes_ph5.mean()))


if __name__ == "__main__":
    main()
