"""
Trains a model that given
- Response spectrum from observed GM at the observation site
- Response spectrum from simulation realisation at the observation site
- Response spectrum from simulation realisation at the site of interest
- Site properties of observation site
- Site properties of site of interest
predicts the similarity score of the simulation realisation and the
(unknown) observed GM at the site of interest
"""
import os
from pathlib import Path

import torch
from torch import nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchinfo import summary
from torchview import draw_graph

import sim_ranking as sr
import spatial_hazard as sh
import ml_tools as mlt


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

print(f"Using device: {device}")


def custom_loss(output, target, distance):
    w = (-1 / 100) * distance + 1
    return torch.mean(w * (target - output) ** 2)


def get_dataset_predictions(
    train_dataset: sr.ml.data.MetaDataDataset,
    train_meta_dataset: sr.ml.data.ResponseSpectrumDataset,
    model: nn.Module,
    device: str,
):
    pred_train_meta_dataloader = DataLoader(
        train_meta_dataset, shuffle=False, batch_size=4096, num_workers=0
    )
    pred_train_dataloader = DataLoader(
        train_dataset, shuffle=False, batch_size=4096, num_workers=0
    )

    model.eval()
    results = {
        "index": [],
        "event": [],
        "rel": [],
        "site_int": [],
        "site_obs": [],
        "sim_score": [],
        "predicted_sim_score": [],
        "distance": [],
    }
    for i, (
        (
            rs_int_sim,
            rs_obs_sim,
            rs_obs_obs,
            site_features,
            sim_score,
            distance,
        ),
        (event, rel, site_int, site_obs),
    ) in enumerate(zip(pred_train_dataloader, pred_train_meta_dataloader)):
        # Forward pass
        pred = _get_prediction(
            rs_int_sim, rs_obs_sim, rs_obs_obs, site_features, model, device
        )

        results["index"] = np.concatenate(
            (
                results["index"],
                np.char.add(
                    np.char.add(
                        np.char.add(
                            np.char.add(np.asarray(rel), "_"), np.asarray(site_int)
                        ),
                        "_",
                    ),
                    np.asarray(site_obs),
                ),
            )
        )
        results["event"] = np.concatenate((results["event"], np.asarray(event)))
        results["rel"] = np.concatenate((results["rel"], np.asarray(rel)))
        results["site_int"] = np.concatenate(
            (results["site_int"], np.asarray(site_int))
        )
        results["site_obs"] = np.concatenate(
            (results["site_obs"], np.asarray(site_obs))
        )
        results["distance"] = np.concatenate((results["distance"], distance.numpy()))
        results["sim_score"] = np.concatenate((results["sim_score"], sim_score.numpy()))
        results["predicted_sim_score"] = np.concatenate(
            (results["predicted_sim_score"], pred.cpu().detach().numpy())
        )

    results_df = pd.DataFrame(results).set_index("index", drop=True)

    return results_df


def _get_prediction(
    rs_int_sim, rs_obs_sim, rs_obs_obs, site_features, model: nn.Module, device: str
):
    # Put data onto the device
    rs_int_sim = rs_int_sim.to(device, dtype=torch.float32)[:, None, :]
    rs_obs_sim = rs_obs_sim.to(device, dtype=torch.float32)[:, None, :]
    rs_obs_obs = rs_obs_obs.to(device, dtype=torch.float32)[:, None, :]
    site_features = site_features.to(device, dtype=torch.float32)

    return model(rs_int_sim, rs_obs_sim, rs_obs_obs, site_features).ravel()


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    n_epochs: int,
    device: str,
    optimizer: torch.optim.Optimizer,
):
    loss_hist_train, loss_hist_val = torch.zeros(n_epochs), torch.zeros(n_epochs)

    best_val_loss = np.inf
    best_model_state = None
    best_model_epoch = None

    for epoch in range(n_epochs):
        print(f"Processing epoch {epoch+1}/{n_epochs}")

        ### Training
        model.train()
        for i, (
            rs_int_sim,
            rs_obs_sim,
            rs_obs_obs,
            site_features,
            sim_score,
            distance,
        ) in enumerate(train_dataloader):
            # Forward pass
            pred = _get_prediction(
                rs_int_sim, rs_obs_sim, rs_obs_obs, site_features, model, device
            )

            # Compute the loss
            distance = distance.to(device, dtype=torch.float32)
            sim_score = sim_score.to(device, dtype=torch.float32)
            loss = custom_loss(pred, sim_score, distance)
            # loss = loss_fn(pred, sim_score)

            # Update weights
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            loss_hist_train[epoch] += loss.item() * sim_score.shape[0]

        loss_hist_train[epoch] /= len(train_dataloader.dataset)

        ### Validation
        model.eval()
        with torch.no_grad():
            for i, (
                rs_int_sim,
                rs_obs_sim,
                rs_obs_obs,
                site_features,
                sim_score,
                distance,
            ) in enumerate(val_dataloader):
                # Forward pass
                pred = _get_prediction(
                    rs_int_sim, rs_obs_sim, rs_obs_obs, site_features, model, device
                )

                # Loss
                distance = distance.to(device, dtype=torch.float32)
                sim_score = sim_score.to(device, dtype=torch.float32)
                loss = custom_loss(pred, sim_score, distance)
                # loss = loss_fn(pred, sim_score)
                loss_hist_val[epoch] += loss.item() * sim_score.shape[0]

            loss_hist_val[epoch] /= len(val_dataloader.dataset)

        # Keep track of the best model
        if loss_hist_val[i] < best_val_loss:
            best_model_state = model.state_dict()
            best_val_loss = loss_hist_val[i]
            best_model_epoch = i

        print(
            f"Epoch {epoch+1} Loss:\n Train {loss_hist_train[epoch]:.4f} Val {loss_hist_val[epoch]:.4f}"
        )
    return loss_hist_train, loss_hist_val, best_model_state, best_model_epoch


if __name__ == "__main__":
    ### CONFIG ###
    N_RELS_USED = 10
    N_VAL_SITES = 10

    N_EPOCHS = 5
    BATCH_SIZE = 1024
    N_SCALAR_FEATURES = 7
    RS_N_CHANNELS = [1, 16, 32]
    RS_KERNEL_SIZES = [5, 3]
    FC_UNITS = [64, 32, 16]

    SITE_FEATURES = ["vs30", "Z_1.0", "Z_2.5"]
    ### END CONFIG ###

    # Fixing the random seed
    np.random.seed(42)

    sim_imdb_ffp_orig = "$wdata/sim_ranking/sim_im_data/simulations.imdb"
    # sim_imdb_ffp_orig = "$wdata/sim_ranking/sim_im_data/lee_val_dataset/simulations.imdb"
    sim_imdb_ffp = Path(os.path.expandvars(sim_imdb_ffp_orig))

    sim_im_dir_orig = "$wdata/sim_ranking/sim_im_data/lee_val_dataset/raw_im_data"
    sim_im_dir = Path(os.path.expandvars(sim_im_dir_orig))

    # sim_imdb_lee_ffp_orig = (
    #     "$wdata/sim_ranking/sim_im_data/lee_val_dataset/simulations.imdb"
    # )
    # sim_imdb_lee_ffp = Path(os.path.expandvars(sim_imdb_lee_ffp_orig))

    obs_ffp_orig = (
        "$wdata/gm_datasets/nz_gmdb/v3.0/Tables/ground_motion_im_table_rotd50_flat.csv"
    )
    obs_ffp = Path(os.path.expandvars(obs_ffp_orig))

    sites_dir_orig = "$wdata/gm_hazard/sites/23p1"
    sites_dir = Path(os.path.expandvars("$wdata/gm_hazard/sites/23p1"))

    results_dir = Path(os.path.expandvars("$wdata/sim_ranking/results/ml"))

    # Load the station data
    station_df = sr.data.load_ll_file(
        sites_dir / "non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.ll"
    )
    vs30_df = sr.data.load_vs30_file(
        sites_dir / "non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.vs30"
    )
    z_df = pd.read_csv(
        sites_dir / "non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.z",
        index_col=0,
    ).drop(columns=["sigma"])

    assert np.all(station_df.index == vs30_df.index) and np.all(
        station_df.index == z_df.index
    )
    station_df = pd.concat([station_df, vs30_df, z_df], axis=1)
    station_df = station_df.rename(columns={"Z_1.0(km)": "Z_1.0", "Z_2.5(km)": "Z_2.5"})

    # Load the observed data
    obs_df = sr.data.load_obs_data(obs_ffp)

    # Load the available events
    events = sr.data.load_avail_sim_events(sim_imdb_ffp)
    events = np.intersect1d(events, obs_df.evid.values.astype(str))
    print(f"Number of events: {len(events)}")

    # Load the IM data for each event
    print(f"Loading IM data")
    obs_im_data, sim_im_data, rels, event_sites = sr.ml.data.get_sim_obs_data_dicts(
        obs_df, sim_imdb_ffp, events, n_rels=N_RELS_USED, n_procs=8
    )

    # Get all relevant sites across all events
    all_sites = np.unique(np.concatenate(list(event_sites.values())))

    # Compute the distance matrix
    print(f"Computing distance matrix")
    dist_matrix = sh.im_dist.calculate_distance_matrix(all_sites, station_df)

    # Select one of the events for validation
    # val_events = np.random.choice(events, 1)
    val_events = np.asarray(["2016p118944"])
    train_events = events[np.isin(events, val_events, invert=True)]

    # Select a subset of the stations (of the validation events) as the validation sites
    val_sites = np.random.choice(
        np.concatenate([event_sites[cur_val_event] for cur_val_event in val_events]),
        N_VAL_SITES,
        replace=False,
    )
    train_sites = all_sites[np.isin(all_sites, val_sites, invert=True)]

    # Get the training and validation dataset site combinations
    print(f"Creating site combinations")
    train_site_combs, train_event_sites = sr.ml.data.compute_site_combinations(
        event_sites, train_events, dist_matrix, sites_to_use=train_sites
    )
    val_site_combs, val_event_sites = sr.ml.data.compute_site_combinations(
        event_sites, val_events, dist_matrix, sites_to_use=val_sites
    )

    # Get the periods and corresponding pSA keys
    periods, pSA_keys = sr.utils.get_periods(
        obs_im_data[events[0]].columns.values.astype(str)
    )

    # Run pre-processing for the site features
    print(f"Pre-processing site features")
    station_df_norm, site_feature_stats = sr.ml.data.preprocess_site_features(
        station_df, SITE_FEATURES
    )

    # Create the training and validation dataset
    print(f"Creating datasets and dataloaders")
    train_dataset = sr.ml.data.ResponseSpectrumDataset(
        train_event_sites,
        train_site_combs,
        train_events,
        obs_im_data,
        sim_im_data,
        rels,
        station_df_norm,
        periods,
        pSA_keys,
        dist_matrix,
        SITE_FEATURES,
    )
    val_dataset = sr.ml.data.ResponseSpectrumDataset(
        val_event_sites,
        val_site_combs,
        val_events,
        obs_im_data,
        sim_im_data,
        rels,
        station_df_norm,
        periods,
        pSA_keys,
        dist_matrix,
        SITE_FEATURES,
    )

    # Create the dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )

    # Create the model
    n_rs_layers = len(RS_KERNEL_SIZES)
    padding = [
        mlt.dl_utils.compute_same_conv_padding(31, RS_KERNEL_SIZES[0])
    ] * n_rs_layers
    out_size = mlt.dl_utils.get_conv_out_sizes(
        31, RS_KERNEL_SIZES, [1] * n_rs_layers, padding, [2] * n_rs_layers
    )

    model = sr.ml.models.ResponseSpectrumSimModel(
        RS_KERNEL_SIZES,
        RS_N_CHANNELS,
        padding,
        FC_UNITS,
        periods.size,
        N_SCALAR_FEATURES,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    summary(
        model,
        input_size=[
            (BATCH_SIZE, 1, 31),
            (BATCH_SIZE, 1, 31),
            (BATCH_SIZE, 1, 31),
            (BATCH_SIZE, N_SCALAR_FEATURES),
        ],
    )

    print(f"Running training")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of training batches: {len(train_dataloader)}")
    print(f"Number of validation batches: {len(val_dataloader)}")
    loss_hist_train, loss_hist_val, best_model_state, best_epoch = train(
        model, train_dataloader, val_dataloader, N_EPOCHS, device, optimizer
    )

    # Load the best model
    model.load_state_dict(best_model_state)

    # Get predictions for the training and validation datasets
    print(f"Getting predictions")
    train_meta_dataset = sr.ml.data.MetaDataDataset(train_dataset)
    train_results_df = get_dataset_predictions(
        train_dataset, train_meta_dataset, model, device
    )
    val_meta_dataset = sr.ml.data.MetaDataDataset(val_dataset)
    val_results_df = get_dataset_predictions(
        val_dataset, val_meta_dataset, model, device
    )

    # Save the results
    print(f"Savings results")
    run_id = mlt.utils.create_run_id(False)
    (results_dir := results_dir / run_id).mkdir(exist_ok=False)

    train_results_df.to_csv(results_dir / "train_results.csv", index=True)
    val_results_df.to_csv(results_dir / "val_results.csv", index=True)

    # Write the metadata
    meta_data = {
        **site_feature_stats.to_dict(),
        "n_rels_used": N_RELS_USED,
        "n_val_sites": N_VAL_SITES,
        "val_events": val_events.astype(str).tolist(),
        "train_events": train_events.astype(str).tolist(),
        "site_features": SITE_FEATURES,
        "model": {
            "n_channels": RS_N_CHANNELS,
            "kernel_sizes": RS_KERNEL_SIZES,
            "fc_units": FC_UNITS,
        },
        "training": {
            "batch_size": BATCH_SIZE,
            "n_epochs": N_EPOCHS,
            "best_epoch": best_epoch,
        },
        "data": {
            "sim_imdb_ffp": str(sim_imdb_ffp_orig),
            "obs_ffp": str(obs_ffp_orig),
            "sites_dir": str(sites_dir_orig),
        },
    }
    mlt.utils.write_to_yaml(meta_data, results_dir / "meta.yaml")

    # Save the model
    torch.save(model, results_dir / "model.pt")

    # Save the loss history
    np.save(str(results_dir / "loss_hist_train.npy"), loss_hist_train.numpy())
    np.save(str(results_dir / "loss_hist_val.npy"), loss_hist_val.numpy())

    # Create a model visualisation
    model_graph = draw_graph(
        model,
        input_size=[
            (BATCH_SIZE, 1, 31),
            (BATCH_SIZE, 1, 31),
            (BATCH_SIZE, 1, 31),
            (BATCH_SIZE, N_SCALAR_FEATURES),
        ],
        expand_nested=True,
        filename="model_vis",
        save_graph=True,
        directory=str(results_dir),
    )
