import copy
from pathlib import Path

import torch
import torch.multiprocessing as mp
import pandas as pd
import numpy as np
import typer

import ml_tools as mlt
import sim_ranking as sr

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

print(f"Using device: {device.upper()}")

app = typer.Typer()


@app.command("run-holdout")
def run_holdout(
    run_config_ffp: Path,
    holdout_config_ffp: Path,
    n_epochs: int = None,
    id_suffix: str = "",
):
    ### Create the configs
    run_config = sr.ml.gnn_gm.RunConfig.from_config_kwargs(
        run_config_ffp, n_epochs=n_epochs, ims=sr.constants.PSA_KEYS, device=device
    )
    holdout_config = sr.ml.gnn_gm.HoldoutConfig.from_yaml(holdout_config_ffp)

    ### Data loading
    obs_data = sr.data.load_obs_nzgmdb(run_config.obs_data_ffp)
    events, all_sites = obs_data.events, obs_data.sites
    event_sites = obs_data.event_sites
    print(f"Number of events: {len(events)}")

    ### Data setup
    # Get the set of valid site-interests per event
    print(f"Getting valid sites of interest")
    valid_int_sites, valid_event_int_sites, _ = sr.ml.data.get_valid_site_ints(
        event_sites, obs_data.record_df.drop(columns=obs_data.ims)
    )
    events = np.intersect1d(events, np.asarray(list(valid_event_int_sites.keys())))
    print(f"Number of valid events: {len(events)}/{len(obs_data.events)}")

    if holdout_config.test_events is not None:
        events = np.setdiff1d(events, holdout_config.test_events)
        print(f"Number of events after removing test events: {len(events)}")

    # Set the random seed
    if run_config.seed is not None:
        print(f"Using numpy random seed: {run_config.seed}")
        np.random.seed(run_config.seed)

    # Split into training and validation
    val_events = np.random.choice(events, holdout_config.n_val_events, replace=False)
    if holdout_config.val_events is not None:
        val_events = np.union1d(val_events, holdout_config.val_events)
    train_events = np.setdiff1d(events, val_events)

    print(f"----------------- Events Summary -----------------")
    print(f"Number of available events: {len(events)}")
    print(f"Number of training events: {train_events.size}")
    print(f"Number of validation events: {val_events.size}")

    if holdout_config.val_sites_ffp is not None:
        val_int_sites = np.load(holdout_config.val_sites_ffp)
    else:
        val_int_sites = np.random.choice(
            valid_int_sites, holdout_config.n_val_sites, replace=False
        )
    train_int_sites = np.setdiff1d(valid_int_sites, val_int_sites)
    obs_sites = np.setdiff1d(all_sites, val_int_sites)

    print(f"----------------- Sites Summary -----------------")
    print(f"Number of available sites: {len(all_sites)}")
    print(f"Number of valid sites of interests: {valid_int_sites.size}")
    print(f"Number of training sites of interests: {train_int_sites.size}")
    print(f"Number of validation sites of interests: {val_int_sites.size}")
    print(f"Number of observation sites: {obs_sites.size}")
    print(f"------------------------------------------------")

    print(f"Computing distance matrix")
    dist_matrix = sr.utils.calculate_distance_matrix(all_sites, obs_data.site_df)

    print(f"Getting scalar features")
    scalar_features = sr.ml.features.get_scalar_features(
        event_sites, obs_data, run_config, sr.constants.SCALAR_FEATURE_KEYS, dist_matrix
    )

    id_suffix = f"_{id_suffix}" if len(id_suffix) > 0 else ""
    cur_out_dir = (
        run_config.results_dir / f"{mlt.utils.create_run_id(False)}{id_suffix}"
    )

    sr.ml.gnn_gm.run_model_training(
        cur_out_dir,
        event_sites,
        valid_event_int_sites,
        train_events,
        val_events,
        train_int_sites,
        val_int_sites,
        obs_sites,
        dist_matrix,
        obs_data,
        scalar_features,
        run_config,
        graph_data_n_procs=1,
    )


@app.command("run-cv")
def run_cv(
    run_config_ffp: Path,
    n_event_folds: int,
    n_site_folds: int,
    n_epochs: int = None,
    id_suffix: str = "",
    n_procs: int = mp.cpu_count(),
):
    # Create the configs
    run_config = sr.ml.gnn_gm.RunConfig.from_config_kwargs(
        run_config_ffp, n_epochs=n_epochs, ims=sr.constants.PSA_KEYS, device=device
    )

    ### Data loading
    obs_data = sr.data.load_obs_nzgmdb(run_config.obs_data_ffp)
    if len(run_config.ignore_events) > 0:
        obs_data = obs_data.drop_events(run_config.ignore_events)

    events, all_sites = obs_data.events, obs_data.sites
    event_sites = obs_data.event_sites
    print(f"Number of events: {len(events)}")

    # Get the set of valid site-interests per event
    print(f"Getting valid sites of interest")
    int_sites, valid_event_int_sites, _ = sr.ml.data.get_valid_site_ints(
        event_sites, obs_data.record_df.drop(columns=obs_data.ims)
    )
    events = np.intersect1d(events, np.asarray(list(valid_event_int_sites.keys())))

    print(f"Computing distance matrix")
    dist_matrix = sr.utils.calculate_distance_matrix(all_sites, obs_data.site_df)

    print(f"Getting scalar features")
    scalar_features = sr.ml.features.get_scalar_features(
        event_sites, obs_data, run_config, sr.constants.SCALAR_FEATURE_KEYS, dist_matrix
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
                    scalar_features,
                    copy.deepcopy(run_config),
                    out_dir,
                    cv_iter,
                    True,
                    graph_data_n_procs=mp.cpu_count(),
                )
            )
    else:
        mp.set_start_method("spawn")
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
                        scalar_features,
                        run_config,
                        out_dir,
                        cv_iter,
                        (cv_iter % n_procs) == 0,
                    )
                    for cv_iter, (train_folds_ind, val_fold_ind) in enumerate(
                        get_cv_iterator(fold_combs)
                    )
                ],
            )

    # Post-processing
    run_config.to_yaml(out_dir / "run_config.yaml")

    val_results, metrics = [], {}
    val_attn_coeffs = []
    for cur_out_dir in out_dirs:
        cur_val_result = pd.read_parquet(cur_out_dir / "val_results.parquet")
        cur_val_result["cv_iter"] = cur_out_dir.stem
        val_results.append(cur_val_result)

        cur_val_attn_coeffs = pd.read_parquet(cur_out_dir / "val_attn_coeffs.parquet")
        cur_val_attn_coeffs["cv_iter"] = cur_out_dir.stem
        val_attn_coeffs.append(cur_val_attn_coeffs)

        metrics[cur_out_dir.stem] = pd.read_pickle(cur_out_dir / "metrics.pickle")

    val_results = pd.concat(val_results, axis=0)
    val_results.to_parquet(out_dir / "val_results.parquet")

    val_attn_coeffs = pd.concat(val_attn_coeffs, axis=0)
    val_attn_coeffs.to_parquet(out_dir / "val_attn_coeffs.parquet")

    pd.to_pickle(metrics, out_dir / "metrics.pickle")

    # Generate report
    cv_agg_notebook = Path(__file__).parent / "report_notebooks/cv_agg_results.ipynb"
    mlt.quarto.render_quarto(
        "mamba activate sim-ranking-pip",
        cv_agg_notebook,
        out_dir / "cv_agg_results.html",
        results_dir=out_dir,
        wdata=run_config.wdata,
    )

    # ind_notebook = Path(__file__).parent / "report_notebooks/ind_scenarios.ipynb"
    # mlt.quarto.render_quarto(
    #     "mamba activate sim-ranking-pip",
    #     ind_notebook,
    #     out_dir / "ind_scenarios.html",
    #     gnn_results_dir=out_dir,
    #     wdata=run_config.wdata,
    # )


def _run_mp_helper(
    event_folds: list[np.ndarray[str]],
    site_folds: list[np.ndarray[str]],
    val_fold_ind: tuple[int, int],
    train_folds_ind: list[tuple[int, int]],
    all_sites: np.ndarray[str],
    event_sites: dict[str, np.ndarray[str]],
    valid_event_int_sites: dict[str, np.ndarray[str]],
    dist_matrix: pd.DataFrame,
    obs_data: sr.ObservedData,
    scalar_features: sr.ml.data.ScalarFeatures,
    run_config: sr.ml.gnn_gm.RunConfig,
    out_dir: Path,
    cv_iter: int,
    verbose: bool,
    graph_data_n_procs: int = 1,
):
    cur_val_events = event_folds[val_fold_ind[0]]
    cur_val_int_sites = site_folds[val_fold_ind[1]]

    cur_train_events = np.concatenate([event_folds[i] for i, _ in train_folds_ind])
    cur_train_int_sites = np.concatenate([site_folds[i] for _, i in train_folds_ind])

    obs_sites = np.setdiff1d(all_sites, cur_val_int_sites)

    cur_out_dir = out_dir / f"cv_{cv_iter}"

    sr.ml.gnn_gm.run_model_training(
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
        graph_data_n_procs=graph_data_n_procs,
        verbose=verbose,
    )

    return cur_out_dir


def get_cv_iterator(fold_combs: list[tuple[int, int]]):
    for val_fold_ind in fold_combs:
        train_folds_ind = [
            cur_fold for cur_fold in fold_combs if (cur_fold[0] != val_fold_ind[0]) and (cur_fold[1] != val_fold_ind[1])
        ]
        yield train_folds_ind, val_fold_ind


if __name__ == "__main__":
    app()
