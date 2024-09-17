import time
import os
from pathlib import Path

import torch
import pandas as pd
import numpy as np
import torch_geometric.data as gdata
import torch_geometric.transforms as T
import torch_geometric.loader as gloader
import tqdm
import typer

import ml_tools as mlt
import sim_ranking as sr
import spatial_hazard as sh

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

print(f"Using device: {device.upper()}")


def run_gnn(config_ffp: Path, n_epochs: int = None):
    ### Create the run config
    run_config = sr.ml.gnn_gm.RunConfig.from_config_kwargs(
        config_ffp, n_epochs=n_epochs, ims=sr.constants.PSA_KEYS, device=device
    )

    ### Data loading
    obs_data = sr.ObservedData.from_nzgmdb_flat(run_config.obs_data_ffp)
    obs_data.drop_nan()
    events, all_sites = obs_data.events, obs_data.sites
    print(f"Number of events: {len(events)}")

    start_time = time.time()
    event_sites = obs_data.event_sites
    print(f"Took {time.time() - start_time} to get event sites")

    ### Data setup
    # Get the set of valid site-interests per event
    print(f"Getting valid sites of interest")
    valid_int_sites, valid_event_int_sites, _ = sr.ml.data.get_valid_site_ints(
        event_sites, obs_data.record_df.drop(columns=obs_data.IM_COLUMNS)
    )
    events = np.intersect1d(events, np.asarray(list(valid_event_int_sites.keys())))
    print(f"Number of valid events: {len(events)}/{len(obs_data.events)}")
    # valid_sc_ids = np.concatenate([mlt.array_utils.numpy_str_join("_", cur_event, cur_sites)  for cur_event, cur_sites in valid_event_int_sites.items()])

    if run_config.test_events is not None:
        events = np.setdiff1d(events, run_config.test_events)
        print(f"Number of events after removing test events: {len(events)}")

    # Set the random seed
    if run_config.seed is not None:
        print(f"Using numpy random seed: {run_config.seed}")
        np.random.seed(run_config.seed)

    # Split into training and validation
    val_events = np.random.choice(events, run_config.n_val_events, replace=False)
    if run_config.val_events is not None:
        val_events = np.union1d(val_events, run_config.val_events)
    train_events = np.setdiff1d(events, val_events)

    print(f"----------------- Events Summary -----------------")
    print(f"Number of available events: {len(events)}")
    print(f"Number of training events: {train_events.size}")
    print(f"Number of validation events: {val_events.size}")

    if run_config.val_sites_ffp is not None:
        val_int_sites = np.load(run_config.val_sites_ffp)
    else:
        val_int_sites = np.random.choice(
            valid_int_sites, run_config.n_val_sites, replace=False
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
    dist_matrix = sh.im_dist.calculate_distance_matrix(all_sites, obs_data.site_df)

    scalar_features = sr.ml.features.get_scalar_features(
        event_sites,
        obs_data,
        run_config,
        sr.constants.SCALAR_FEATURE_KEYS,
        dist_matrix
    )

    # Compute mean and standard deviation for each period
    # for normalisation (only training events)
    # ims_mean = np.mean(np.log(record_df.loc[:, run_config.ims]), axis=0)
    # ims_std = np.std(np.log(record_df.loc[:, run_config.ims]), axis=0)

    print(f"Creating site combinations")
    train_site_combs, train_event_sites = sr.ml.data.compute_site_combinations(
        event_sites,
        valid_event_int_sites,
        train_events,
        dist_matrix,
        obs_sites,
        train_int_sites,
        max_dist=run_config.max_dist,
    )
    val_site_combs, val_event_sites = sr.ml.data.compute_site_combinations(
        event_sites,
        valid_event_int_sites,
        val_events,
        dist_matrix,
        obs_sites,
        val_int_sites,
        max_dist=run_config.max_dist,
    )

    edge_feature_keys = ["dist", "angular_dist"]
    site_int_site_feature_keys = [
        "vs30_site_int",
        "z1p0_site_int",
        "z2p5_site_int",
        "tsite_site_int",
    ]
    site_int_feature_keys = site_int_site_feature_keys + [
        "rrup_site_int",
        "mag",
    ]
    site_obs_site_feature_keys = [
        "vs30_site_obs",
        "z1p0_site_obs",
        "z2p5_site_obs",
        "tsite_site_obs",
    ]
    site_obs_scalar_feature_keys = site_obs_site_feature_keys + [
        "rrup_site_obs",
    ]

    site_int_n_node_features = len(site_int_feature_keys)
    site_obs_n_node_features = len(site_obs_scalar_feature_keys) + len(run_config.ims)

    print(f"Getting graph data")
    train_graph_data, site_obs_scalar_feature_ind = sr.ml.gnn_gm.get_graph_data(
        obs_data,
        train_event_sites,
        train_site_combs,
        scalar_features,
        sr.constants.GRAPH_FEATURE_KEYS,
        run_config.ims,
        n_procs=8,
    )

    val_graph_data, _ = sr.ml.gnn_gm.get_graph_data(
        obs_data,
        val_event_sites,
        val_site_combs,
        scalar_features,
        sr.constants.GRAPH_FEATURE_KEYS,
        run_config.ims,
        n_procs=8,
    )

    train_loader = gloader.DataLoader(
        train_graph_data, batch_size=run_config.batch_size, shuffle=True
    )
    val_loader = gloader.DataLoader(
        val_graph_data, batch_size=run_config.batch_size, shuffle=True
    )

    gnn_model = sr.ml.gnn_modules.BasicAttentionGNN(
        site_obs_n_node_features,
        len(site_obs_scalar_feature_keys),
        site_int_n_node_features,
        len(edge_feature_keys),
        run_config,
        torch.from_numpy(site_obs_scalar_feature_ind),
    )
    gnn_model.to(device)

    print(f"----------------- Training -----------------")
    print(f"Number of training graphs: {len(train_graph_data)}")
    print(f"Number of validation graphs: {len(val_graph_data)}")

    metrics, best_model_state, best_model_epoch = sr.ml.gnn_gm.train(
        run_config, gnn_model, train_loader, val_loader
    )
    print(
        f"Best model epoch: {best_model_epoch + 1}, "
        f"Validation: \tLoss: {metrics['loss_hist_val'][best_model_epoch]:.4f}\n"
    )

    # Load the best model
    gnn_model.load_state_dict(best_model_state)

    id_suffix = ""
    (
        cur_out_dir := run_config.results_dir
        / f"{mlt.utils.create_run_id(False)}{id_suffix}"
    ).mkdir()

    # Save the training sites and validation sites
    np.save(cur_out_dir / "val_int_sites.npy", val_int_sites)
    np.save(cur_out_dir / "train_int_sites.npy", train_int_sites)
    np.save(cur_out_dir / "obs_sites.npy", obs_sites)

    # Save the run config
    run_config.to_yaml(cur_out_dir / "run_config.yaml")

    # Save loss history
    pd.to_pickle(metrics, cur_out_dir / "metrics.pickle")

    # Save the model
    torch.save(gnn_model, cur_out_dir / "model.pt")

    # Save the results
    train_results_df = sr.ml.gnn_gm.get_predictions(
        run_config, gnn_model, train_graph_data
    )
    train_results_df.to_parquet(cur_out_dir / "train_results.parquet")

    val_results_df = sr.ml.gnn_gm.get_predictions(run_config, gnn_model, val_graph_data)
    val_results_df.to_parquet(cur_out_dir / "val_results.parquet")

    # Write the metadata
    metadata = {
        "best_model_epoch": int(best_model_epoch),
        "best_model_loss": float(metrics["loss_hist_val"][best_model_epoch]),
        "n_train_scenarios": len(train_graph_data),
        "n_val_scenarios": len(val_graph_data),
    }
    mlt.utils.write_to_yaml(metadata, cur_out_dir / "metadata.yaml")


if __name__ == "__main__":
    typer.run(run_gnn)
