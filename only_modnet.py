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
n_jobs = 32


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

    # STEP 1: determine hyperparams based on E only !!
    params = []
    optimal_feats = []
    folds_exp = MDKsplit(md_exp, n_splits=k, random_state=random_state)
    for i, f in enumerate(folds_exp):
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
        print(f"Determining hyper params fold {i+1}...")
        params.append(get_params(train, n_jobs=n_jobs))
        optimal_feats.append(train.optimal_features)

    # STEP 2: Train models based on E params
    for data, name in zip(
        [md_exp, md_gllb, md_pbe, md_scan, md_hse],
        ["exp", "gllb", "pbe", "scan", "hse"],
    ):
        maes = np.ones(k)
        print(f"Start with {name}...")

        data.optimal_features = optimal_feats[i]

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

        model = MODNetModel(
            targets=params[i]["targets"],
            weights=params[i]["weights"],
            n_feat=params[i]["n_feat"],
            num_neurons=params[i]["num_neurons"],
            act=params[i]["act"],
            num_classes={"gap": 0},
        )

        model.fit(
            data,
            loss=params[i]["loss"],
            lr=params[i]["lr"],
            epochs=params[i]["epochs"],
            batch_size=params[i]["batch_size"],
            xscale=params[i]["xscale"],
            callbacks=callbacks,
            verbose=0,
        )

        model.save("out/MODNet_alone_{}_{}".format(i + 1, name))

        all_preds = []
        for i, test in enumerate(folds_exp):
            preds = model.predict(test)
            true = test.df_targets
            true.columns = ["true_gap"]
            error = preds - true
            mae = np.abs(error.values).mean()

            all_preds.append(preds.join(true))

            with open("results/out_alone.txt", "a") as fp:
                fp.write("mae {} - f{}: {:.3f}\n".format(name, i + 1, mae))
            maes[i] = mae

        with open("results/out_alone.txt", "a") as fp:
            fp.write("mae {} {}-fold: {:.3f}\n".format(name, k, maes.mean()))

        (pd.concat(all_preds)).to_csv("results/{}_only.csv".format(name))


if __name__ == "__main__":
    main()
