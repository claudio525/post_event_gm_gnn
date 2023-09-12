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
import time
import os
from pathlib import Path
from typing import Dict, Sequence

import torch
from torch import nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchinfo import summary

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


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    n_epochs: int,
    device: str,
    optimizer: torch.optim.Optimizer,
):
    loss_hist_train, loss_hist_val = torch.zeros(n_epochs), torch.zeros(n_epochs)
    loss_fn = nn.MSELoss()

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
            # if i % 100 == 0:
            #     print(f"\tProcessing batch {i+1}/{len(train_dataloader)}")

            # Put data onto the device
            rs_int_sim = rs_int_sim.to(device, dtype=torch.float32)[:, None, :]
            rs_obs_sim = rs_obs_sim.to(device, dtype=torch.float32)[:, None, :]
            rs_obs_obs = rs_obs_obs.to(device, dtype=torch.float32)[:, None, :]
            site_features = site_features.to(device, dtype=torch.float32)
            sim_score = sim_score.to(device, dtype=torch.float32)
            distance = distance.to(device, dtype=torch.float32)

            # Forward pass
            pred = model(rs_int_sim, rs_obs_sim, rs_obs_obs, site_features).ravel()

            # Compute the loss and update weights
            # loss = custom_loss(pred, sim_score, distance)
            loss = loss_fn(pred, sim_score)
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
                # Put data onto the device
                rs_int_sim = rs_int_sim.to(device, dtype=torch.float32)[:, None, :]
                rs_obs_sim = rs_obs_sim.to(device, dtype=torch.float32)[:, None, :]
                rs_obs_obs = rs_obs_obs.to(device, dtype=torch.float32)[:, None, :]
                site_features = site_features.to(device, dtype=torch.float32)
                sim_score = sim_score.to(device, dtype=torch.float32)
                distance = distance.to(device, dtype=torch.float32)

                # Compute prediction and save loss
                pred = model(rs_int_sim, rs_obs_sim, rs_obs_obs, site_features).ravel()
                # loss = loss_fn(pred, sim_score, distance)
                loss = loss_fn(pred, sim_score)
                loss_hist_val[epoch] += loss.item() * sim_score.shape[0]

            loss_hist_val[epoch] /= len(val_dataloader.dataset)

        print(
            f"Epoch {epoch+1} Loss:\n Train {loss_hist_train[epoch]:.4f} Val {loss_hist_val[epoch]:.4f}"
        )


if __name__ == "__main__":
    ### CONFIG ###
    N_RELS_USED = 10
    N_VAL_SITES = 10

    BATCH_SIZE = 1024
    N_SCALAR_FEATURES = 7
    RS_N_CHANNELS = [1, 16, 32]
    RS_KERNEL_SIZES = [5, 3]
    FC_UNITS = [64, 32, 16]
    ### END CONFIG ###

    np.random.seed(12)

    sim_imdb_ffp = Path(
        os.path.expandvars("$wdata/sim_ranking/sim_im_data/simulations.imdb")
    )
    obs_ffp = Path(
        os.path.expandvars(
            "$wdata/gm_datasets/nz_gmdb/v3.0/Tables/ground_motion_im_table_rotd50_flat.csv"
        )
    )
    sites_dir = Path(os.path.expandvars("$wdata/gm_hazard/sites/23p1"))

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

    # Load the available events
    events = sr.data.load_avail_sim_events(sim_imdb_ffp)

    # Load the IM data for each event
    print(f"Loading IM data")
    obs_im_data, sim_im_data = {}, {}
    rels, event_sites = {}, {}
    for i, cur_event in enumerate(events):
        # Load the observed IM data
        cur_obs_data = sr.data.load_obs_rupture_data(obs_ffp, cur_event)

        # Load the simulation IM data
        cur_sim_data = sr.data.load_sim_data(
            sim_imdb_ffp, cur_obs_data.index.values, event=cur_event
        )

        sim_sites = np.asarray(list(cur_sim_data.keys()))
        obs_sites = np.asarray(list(cur_obs_data.index.values.astype(str)))
        cur_sites = obs_sites[np.isin(obs_sites, sim_sites)]

        sim_im_data[cur_event] = cur_sim_data
        obs_im_data[cur_event] = cur_obs_data.loc[cur_sites]

        rels[cur_event] = np.random.choice(
            cur_sim_data[cur_sites[0]].index.values.astype(str),
            size=N_RELS_USED,
            replace=False,
        )
        event_sites[cur_event] = cur_sites

    # Get all relevant sites across all events
    all_sites = np.unique(np.concatenate(list(event_sites.values())))

    # Compute the distance matrix
    print(f"Computing distance matrix")
    dist_matrix = sh.im_dist.calculate_distance_matrix(all_sites, station_df)

    # Select one of the events for validation
    val_events = np.random.choice(events, 1)
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
    train_site_combs, train_event_sites = sr.ml_data.compute_site_combinations(
        event_sites, train_events, dist_matrix, sites_to_use=train_sites
    )
    val_site_combs, val_event_sites = sr.ml_data.compute_site_combinations(
        event_sites, val_events, dist_matrix, sites_to_use=val_sites
    )

    # Get the periods and corresponding pSA keys
    periods, pSA_keys = sr.utils.get_periods(
        obs_im_data[events[0]].columns.values.astype(str)
    )

    # Create the training and validation dataset
    print(f"Creating datasets and dataloaders")
    train_dataset = sr.ml_data.ResponseSpectrumDataset(
        train_event_sites,
        train_site_combs,
        train_events,
        obs_im_data,
        sim_im_data,
        rels,
        station_df,
        periods,
        pSA_keys,
        dist_matrix,
        device,
    )
    val_dataset = sr.ml_data.ResponseSpectrumDataset(
        val_event_sites,
        val_site_combs,
        val_events,
        obs_im_data,
        sim_im_data,
        rels,
        station_df,
        periods,
        pSA_keys,
        dist_matrix,
        device,
    )

    # Create the dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )

    # for ix in range(10):
    #     start_time = time.time()
    #     for i, (
    #             rs_int_sim,
    #             rs_obs_sim,
    #             rs_obs_obs,
    #             site_features,
    #             sim_score,
    #             distance,
    #     ) in enumerate(train_dataloader):
    #         pass
    #         # print(f"{i}/{len(train_dataloader)}")
    #     print(f"Took {time.time() - start_time}")
    #
    # print(f"wtf")
    # exit()

    # Create the model
    n_rs_layers = len(RS_KERNEL_SIZES)
    padding = [
        mlt.dl_utils.compute_same_conv_padding(31, RS_KERNEL_SIZES[0])
    ] * n_rs_layers
    out_size = mlt.dl_utils.get_conv_out_sizes(
        31, RS_KERNEL_SIZES, [1] * n_rs_layers, padding, [2] * n_rs_layers
    )

    model = sr.models.ResponseSpectrumSimModel(
        RS_KERNEL_SIZES, RS_N_CHANNELS, padding, FC_UNITS, periods.size, N_SCALAR_FEATURES
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    summary(
        model,
        input_size=[(BATCH_SIZE, 1, 31), (BATCH_SIZE, 1, 31), (BATCH_SIZE, 1, 31), (BATCH_SIZE, N_SCALAR_FEATURES)],
    )

    print(f"Running training")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of training batches: {len(train_dataloader)}")
    print(f"Number of validation batches: {len(val_dataloader)}")
    train(model, train_dataloader, val_dataloader, 10, device, optimizer)
