"""Module for running custom cross-validation for GNN models."""

import copy
from pathlib import Path

import torch.multiprocessing as mp
import pandas as pd
import numpy as np

import ml_tools as mlt
from labelled_data_array import LabelledDataArray

from . import gnn_gm
from . import data as ml_data
from . import features
from .. import constants
from .. import data
from .. import utils
from .. import analysis


def run_cv(
    run_config: Path | gnn_gm.RunConfig,
    n_event_folds: int,
    n_site_folds: int,
    n_epochs: int = None,
    id_suffix: str = "",
    n_procs: int = mp.cpu_count(),
    device: str = None,
):
    """
    Performs a custom cross-validation run,
    with the specified number of event and site folds.

    Parameters
    ----------
    run_config: Path | RunConfig
        Either the RunConfig, or Path to a
        yaml-based run config file.
    n_event_folds: int
        Number of event folds.
    n_site_folds: int
        Number of site folds.
    n_epochs: int, optional
        Number of epochs to run the model for.
        Note: Only used if run_config is a Path.
    id_suffix: str, optional
        Suffix to append to the output directory.
    n_procs: int, optional
        Number of processes to use.
        Default is the number of CPUs.
    device: str, optional
        Device to run the model on.
        Note: Only used if run_config is a Path.
    """
    # Create the config
    if isinstance(run_config, Path):
        run_config = gnn_gm.RunConfig.from_config_kwargs(
            run_config, n_epochs=n_epochs, device=device
        )

    ### Data loading
    obs_data = data.load_obs_nzgmdb(run_config.obs_data_ffp)
    if len(run_config.ignore_events) > 0:
        obs_data = obs_data.drop_events(run_config.ignore_events)
    if run_config.ignore_sites is not None and len(run_config.ignore_sites) > 0:
        obs_data = obs_data.drop_sites(run_config.ignore_sites)

    # Load empirical GMM data & compute empirical residuals
    if run_config.use_emp_gm_model:
        emp_gm_params, emp_res_df = analysis.load_emp_gm_params_res(
            run_config.emp_gm_params_ffp, obs_data
        )

    events, all_sites = obs_data.events, obs_data.sites
    event_sites = obs_data.event_sites
    print(f"Number of events: {len(events)}")

    # Get the set of valid site-interests per event
    print("Getting valid sites of interest")
    int_sites, valid_event_int_sites, _ = ml_data.get_valid_site_ints_Lee2024(
        event_sites, obs_data.record_df.drop(columns=obs_data.ims)
    )
    events = np.intersect1d(events, np.asarray(list(valid_event_int_sites.keys())))

    print("Computing distance matrix")
    dist_matrix = utils.calculate_distance_matrix(all_sites, obs_data.site_df)

    print("Getting scalar features")
    scalar_features = features.get_scalar_features(
        event_sites,
        obs_data.event_df,
        obs_data.site_df,
        obs_data.record_df,
        run_config,
        constants.SCALAR_FEATURE_KEYS,
        dist_matrix,
    )

    # Cross-validation setup
    np.random.seed(run_config.seed)
    np.random.shuffle(events)
    np.random.shuffle(int_sites)
    event_folds = np.array_split(events, n_event_folds)
    site_folds = np.array_split(int_sites, n_site_folds)

    fold_combs = [(i, j) for i in range(n_event_folds) for j in range(n_site_folds)]

    id_suffix = f"_{id_suffix}" if len(id_suffix) > 0 else ""
    out_dir = run_config.results_dir / f"{mlt.utils.create_run_id(False)}_cv{id_suffix}"
    assert not out_dir.exists(), "Output directory already exists!"

    # Run CV
    if n_procs == 1:
        out_dirs = []
        for cv_iter, (train_folds_ind, val_fold_ind) in enumerate(
            get_cv_iterator(fold_combs)
        ):
            out_dirs.append(
                _run_mp_helper(
                    event_folds,
                    site_folds,
                    val_fold_ind,
                    train_folds_ind,
                    all_sites,
                    event_sites,
                    valid_event_int_sites,
                    dist_matrix,
                    obs_data,
                    emp_gm_params if run_config.use_emp_gm_model else None,
                    emp_res_df if run_config.use_emp_gm_model else None,
                    scalar_features,
                    copy.deepcopy(run_config),
                    out_dir,
                    cv_iter,
                    True,
                    graph_data_n_procs=mp.cpu_count(),
                    # graph_data_n_procs=1,
                )
            )
    else:
        with mp.Pool(processes=n_procs) as pool:
            out_dirs = pool.starmap(
                _run_mp_helper,
                [
                    (
                        event_folds,
                        site_folds,
                        val_fold_ind,
                        train_folds_ind,
                        all_sites,
                        event_sites,
                        valid_event_int_sites,
                        dist_matrix,
                        obs_data,
                        emp_gm_params if run_config.use_emp_gm_model else None,
                        emp_res_df if run_config.use_emp_gm_model else None,
                        scalar_features,
                        copy.deepcopy(run_config),
                        out_dir,
                        cv_iter,
                        (cv_iter % n_procs) == 0,  # Only print for the first process
                    )
                    for cv_iter, (train_folds_ind, val_fold_ind) in enumerate(
                        get_cv_iterator(fold_combs)
                    )
                ],
            )

    # Post-processing
    run_config.to_yaml(out_dir / "run_config.yaml")

    val_results, metrics = [], {}
    # val_attn_coeffs = []
    for cur_out_dir in out_dirs:
        cur_val_result = pd.read_parquet(cur_out_dir / "val_results.parquet")
        cur_val_result["cv_iter"] = cur_out_dir.stem
        val_results.append(cur_val_result)

        metrics[cur_out_dir.stem] = pd.read_parquet(cur_out_dir / "metrics.parquet")

    val_results = pd.concat(val_results, axis=0)
    val_results.to_parquet(out_dir / "val_results.parquet")

    # Sanity check
    assert np.all(
        [
            np.all(cur_df.columns == metrics["cv_0"].columns)
            for cur_df in metrics.values()
        ]
    )

    # Create 3D metrics array
    metrics_lda = LabelledDataArray(
        np.stack([cur_df.values for cur_df in metrics.values()], axis=1),
        (
            np.arange(run_config.n_epochs),
            list(metrics.keys()),
            list(metrics["cv_0"].columns),
        ),
        ("epoch", "cv_iter", "metric"),
    )
    pd.to_pickle(metrics_lda, out_dir / "metrics.pickle")

    # Compute aggregate metrics
    agg_metrics = {
        # Loss
        "mean_min_train_loss": float(
            metrics_lda.sel[:, :, "loss_hist_train"].min(axis=0).mean().item()
        ),
        "std_min_train_loss": float(
            metrics_lda.sel[:, :, "loss_hist_train"].min(axis=0).std().item()
        ),
        "mean_best_train_loss_epoch": float(
            metrics_lda.sel[:, :, "loss_hist_train"].values.argmin(axis=0).mean().item()
        ),
        "std_best_train_loss_epoch": float(
            metrics_lda.sel[:, :, "loss_hist_train"].values.argmin(axis=0).std().item()
        ),
        "mean_min_val_loss": float(
            metrics_lda.sel[:, :, "loss_hist_val"].min(axis=0).mean().item()
        ),
        "std_min_val_loss": float(
            metrics_lda.sel[:, :, "loss_hist_val"].min(axis=0).std().item()
        ),
        "mean_best_val_loss_epoch": float(
            metrics_lda.sel[:, :, "loss_hist_val"].values.argmin(axis=0).mean().item()
        ),
        "std_best_val_loss_epoch": float(
            metrics_lda.sel[:, :, "loss_hist_val"].values.argmin(axis=0).std().item()
        ),
        # Weighted Loss
        "mean_min_train_w_loss": float(
            metrics_lda.sel[:, :, "w_loss_hist_train"].min(axis=0).mean().item()
        ),
        "std_min_train_w_loss": float(
            metrics_lda.sel[:, :, "w_loss_hist_train"].min(axis=0).std().item()
        ),
        "mean_best_train_w_loss_epoch": float(
            metrics_lda.sel[:, :, "w_loss_hist_train"]
            .values.argmin(axis=0)
            .mean()
            .item()
        ),
        "std_best_train_w_loss_epoch": float(
            metrics_lda.sel[:, :, "w_loss_hist_train"]
            .values.argmin(axis=0)
            .std()
            .item()
        ),
        "mean_min_val_w_loss": float(
            metrics_lda.sel[:, :, "w_loss_hist_val"].min(axis=0).mean().item()
        ),
        "std_min_val_w_loss": float(
            metrics_lda.sel[:, :, "w_loss_hist_val"].min(axis=0).std().item()
        ),
        "mean_best_val_w_loss_epoch": float(
            metrics_lda.sel[:, :, "w_loss_hist_val"].values.argmin(axis=0).mean().item()
        ),
        "std_best_val_w_loss_epoch": float(
            metrics_lda.sel[:, :, "w_loss_hist_val"].values.argmin(axis=0).std().item()
        ),
        # MSE
        "mean_min_train_mse": float(
            metrics_lda.sel[:, :, "mse_hist_train"].min(axis=0).mean().item()
        ),
        "std_min_train_mse": float(
            metrics_lda.sel[:, :, "mse_hist_train"].min(axis=0).std().item()
        ),
        "mean_min_val_mse": float(
            metrics_lda.sel[:, :, "mse_hist_val"].min(axis=0).mean().item()
        ),
        "std_min_val_mse": float(
            metrics_lda.sel[:, :, "mse_hist_val"].min(axis=0).std().item()
        ),
    }
    mlt.utils.write_to_yaml(agg_metrics, out_dir / "agg_metrics.yaml")

    # Generate reports
    # cv_agg_notebook = (
    #     Path(__file__).parent.parent
    #     / "scripts/ml_models/report_notebooks/cv_agg_results.ipynb"
    # )
    # mlt.quarto.render_quarto(
    #     "mamba activate sim-ranking-pip",
    #     cv_agg_notebook,
    #     out_dir / "cv_agg_results.html",
    #     results_dir=out_dir,
    #     wdata=run_config.wdata,
    # )

    # ind_notebook = (
    #     Path(__file__).parent.parent
    #     / "scripts/ml_models/report_notebooks/ind_scenarios.ipynb"
    # )
    # mlt.quarto.render_quarto(
    #     "mamba activate sim-ranking-pip",
    #     ind_notebook,
    #     out_dir / "ind_scenarios.html",
    #     results_dir=out_dir,
    #     wdata=run_config.wdata,
    # )

    return out_dir, agg_metrics


def _run_mp_helper(
    event_folds: list[np.ndarray[str]],
    site_folds: list[np.ndarray[str]],
    val_fold_ind: tuple[int, int],
    train_folds_ind: list[tuple[int, int]],
    all_sites: np.ndarray[str],
    event_sites: dict[str, np.ndarray[str]],
    valid_event_int_sites: dict[str, np.ndarray[str]],
    dist_matrix: pd.DataFrame,
    obs_data: data.ObservedData,
    emp_gm_params: pd.DataFrame,
    emp_res_df: pd.DataFrame,
    scalar_features: ml_data.ScalarFeatures,
    run_config: gnn_gm.RunConfig,
    out_dir: Path,
    cv_iter: int,
    verbose: bool,
    graph_data_n_procs: int = 1,
):
    cur_val_events = event_folds[val_fold_ind[0]]
    cur_val_int_sites = site_folds[val_fold_ind[1]]

    cur_train_events = np.unique(
        np.concatenate([event_folds[i] for i, _ in train_folds_ind])
    )
    cur_train_int_sites = np.unique(
        np.concatenate([site_folds[i] for _, i in train_folds_ind])
    )
    assert (
        np.isin(cur_val_events, cur_train_events).sum() == 0
    ), "Event folds are not mutually exclusive!"
    assert (
        np.isin(cur_val_int_sites, cur_train_int_sites).sum() == 0
    ), "Site folds are not mutually exclusive!"

    obs_sites = np.setdiff1d(all_sites, cur_val_int_sites)

    cur_out_dir = out_dir / f"cv_{cv_iter}"

    gnn_gm.run_model_training(
        cur_out_dir,
        event_sites,
        valid_event_int_sites,
        cur_train_events,
        cur_val_events,
        cur_train_int_sites,
        cur_val_int_sites,
        obs_sites,
        dist_matrix,
        obs_data,
        scalar_features,
        run_config,
        emp_gm_params=emp_gm_params,
        emp_res_df=emp_res_df,
        graph_data_n_procs=graph_data_n_procs,
        verbose=verbose,
    )

    return cur_out_dir


def get_cv_iterator(fold_combs: list[tuple[int, int]]):
    for val_fold_ind in fold_combs:
        train_folds_ind = [
            cur_fold
            for cur_fold in fold_combs
            if (cur_fold[0] != val_fold_ind[0]) and (cur_fold[1] != val_fold_ind[1])
        ]
        yield train_folds_ind, val_fold_ind
