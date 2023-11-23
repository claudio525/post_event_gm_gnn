import os
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import typer


import sim_ranking as sr
import spatial_hazard as sh
import ml_tools as mlt


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

print(f"Using device: {device}")


def main(
    hyperparams_ffp: Path,
    max_dist: float = 75,
    debug: bool = False,
    val_percent: float = 0.33,
):
    ### Config
    # SITE_FEATURE_KEYS = ["vs30", "z1.0", "z2.5", "tsite"]
    SITE_FEATURE_KEYS = ["vs30"]

    db_ffp_orig = "$wdata/sim_ranking/db/gm_db_neil.sqlite"
    db_ffp = Path(os.path.expandvars(db_ffp_orig))
    db_avail_sites = sr.db.DB(db_ffp).get_avail_sites()

    corr_dir_orig = "$wdata/sim_ranking/sim_correlations/neil"
    corr_dir = Path(os.path.expandvars(corr_dir_orig))

    hp_config = sr.ml.corr.HyperParamsConfig.from_yaml(hyperparams_ffp)
    run_params = sr.ml.corr.RunParamsConfig(max_dist, debug, device)

    ### Data prep
    ims = sr.constants.PSA_KEYS
    corr_dfs = {
        cur_im: pd.read_csv(corr_dir / f"{cur_im}.csv", index_col=0) for cur_im in ims
    }
    assert np.all(
        [np.all(cur_df.index == corr_dfs[ims[0]].index) for cur_df in corr_dfs.values()]
    )

    sites = np.intersect1d(corr_dfs[ims[0]].site_1.unique().astype(str), db_avail_sites)

    db = sr.db.DB(db_ffp)
    site_df = db.get_site_df().loc[sites]

    dist_matrix = sh.im_dist.calculate_distance_matrix(
        site_df.index.values.astype(str), site_df
    )

    print(f"Pre-processing site features")
    site_features_df, site_feature_stats = sr.ml.features.preprocess_site_features(
        site_df, SITE_FEATURE_KEYS
    )

    print(f"Computing site-pairs")
    cur_corr_df = corr_dfs[ims[0]].loc[:, ["site_1", "site_2", "corr"]]
    site_comb = cur_corr_df.loc[
        cur_corr_df.site_1 != cur_corr_df.site_2, ["site_1", "site_2"]
    ].values.astype(str)

    print(f"Creating feature dataframe")
    X = pd.DataFrame(
        columns=mlt.array_utils.numpy_str_join("_", site_comb[:, 0], site_comb[:, 1]),
        data=[site_comb[:, 0], site_comb[:, 1]],
        index=["site_1", "site_2"],
    ).T

    # Combine
    X = X.merge(
        site_features_df.loc[:, SITE_FEATURE_KEYS].rename(
            columns={
                cur_feature: f"site_1_{cur_feature}"
                for cur_feature in SITE_FEATURE_KEYS
            }
        ),
        how="left",
        left_on="site_1",
        right_index=True,
    )
    X = X.merge(
        site_features_df.loc[:, SITE_FEATURE_KEYS].rename(
            columns={
                cur_feature: f"site_2_{cur_feature}"
                for cur_feature in SITE_FEATURE_KEYS
            }
        ),
        how="left",
        left_on="site_2",
        right_index=True,
    )

    # Add distance
    dist_row_ind, dist_col_ind = dist_matrix.index.get_indexer_for(
        site_comb[:, 0]
    ), dist_matrix.columns.get_indexer_for(site_comb[:, 1])
    X["dist"] = pre_dist_matrix.values[dist_row_ind, dist_col_ind]

    # Apply distance filter
    mask = dist_matrix.values[dist_row_ind, dist_col_ind] < max_dist
    X = X.loc[mask]

    # Compute labels
    y = pd.DataFrame(
        data=[corr_dfs[cur_im].loc[X.index, "corr"] for cur_im in ims],
        index=ims,
        columns=X.index,
    ).T

    metadata = {
        "hyperparams": hp_config.to_dict(),
        "data": {
            "db_ffp_orig": db_ffp_orig,
        },
        "ims": ims,
        "max_dist": run_params.max_dist,
    }

    ### Split into train and test
    # Have to re-define as some may have been filtered out
    sites = X.site_1.unique().astype(str)
    val_sites = np.random.choice(sites, int(0.33 * sites.size), replace=False)
    train_sites = np.setdiff1d(sites, val_sites)

    train_dataset, val_dataset = sr.ml.corr.get_datasets(
        X, y, train_sites, val_sites, ims
    )

    model = sr.ml.models.MLPModel(
        len(ims), hp_config.fc_units, (len(SITE_FEATURE_KEYS) * 2) + 1
    )
    model.to(device)

    best_model_state, best_model_epoch, metrics = sr.ml.corr.train(
        model, train_dataset, val_dataset, hp_config, run_params
    )
    model.load_state_dict(best_model_state)

    print(
        f"Best model epoch: {best_model_epoch + 1}, "
        f"Validation:\n"
        f"\tLoss: {metrics['val_loss_hist'][best_model_epoch]:.4f}\n"
    )

    # Save model
    (out_dir := run_params.results_dir / mlt.utils.create_run_id()).mkdir()
    torch.save(model, out_dir / "model.pt")

    # Get dataset predictions
    train_results = sr.ml.corr.get_dataset_predictions(model, train_dataset, run_params)
    val_results = sr.ml.corr.get_dataset_predictions(model, val_dataset, run_params)

    # Save results
    train_results.to_csv(out_dir / "train_results.csv")
    val_results.to_csv(out_dir / "val_results.csv")

    # Save metrics
    pd.to_pickle(metrics, out_dir / "metrics.pkl")

    mlt.utils.write_to_yaml(metadata, out_dir / "metadata.yaml")


if __name__ == "__main__":
    typer.run(main)
