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
from typing import Dict, Sequence, List

import torch
from torch import nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary

import sim_ranking as sr
import spatial_hazard as sh
import ml_tools as mlt


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

print(f"Using device: {device}")


def similiarity_score(rs_obs: np.ndarray, rs_sim: np.ndarray) -> float:
    # Compute the residual
    res = np.log(rs_obs) - np.log(rs_sim)
    res_area = np.trapz(np.abs(res))

    return res_area * (-1 / 30) + 1 if res_area < 30 else 0


def compute_site_combinations(
    sites: Dict[str, np.ndarray],
    events: Sequence[str],
    dist_matrix: pd.DataFrame,
    sites_to_use: np.ndarray = None,
):
    """
    Compute the site combinations for each event

    Note: The resulting indices are into the index/columns
    of the distance matrix,
    i.e. all stations, NOT the event specific sites
    """
    site_combs, used_sites = {}, {}
    for cur_event in events:
        cur_sites = sites[cur_event]
        cur_sites = (
            cur_sites
            if sites_to_use is None
            else cur_sites[np.isin(cur_sites, sites_to_use)]
        )

        # Filter for the current event sites
        # and site-combinations less than 100km apart
        cur_dist_matrix = dist_matrix.loc[cur_sites, cur_sites]
        cur_dist_mask = (cur_dist_matrix.values < 100) & (cur_dist_matrix.values > 0)
        cur_row_ind, cur_col_ind = np.nonzero(cur_dist_mask)

        # Get the site combinations
        # First is the site of interest, second is the observation site
        cur_site_combs = np.stack((cur_row_ind, cur_col_ind), axis=1)

        site_combs[cur_event] = cur_site_combs
        used_sites[cur_event] = cur_sites

        # # Only interested in site-combinations within 100km
        # cur_dist_mask = (dist_matrix < 100) & (dist_matrix > 0)
        # cur_row_ind, cur_col_inds = np.nonzero(cur_dist_mask.values)
        #
        # # Get the site combinations
        # # First is the site of interest, second is the observation site
        # cur_site_combs = np.stack((cur_row_ind, cur_col_inds), axis=1)
        #
        # # Filter for the current event sites only
        # cur_site_ind = np.flatnonzero(np.isin(cur_dist_mask.index.values.astype(str), cur_sites))
        # cur_sites_mask = np.isin(cur_site_combs[:, 0], cur_site_ind) & np.isin(
        #     cur_site_combs[:, 1], cur_site_ind
        # )
        # site_combs[cur_event] = cur_site_combs[cur_sites_mask]

    return site_combs, used_sites


class ResponseSpectrumDataset(Dataset):
    def __init__(
        self,
        sites: Dict[str, np.ndarray],
        site_combs: Dict[str, np.ndarray],
        events: Sequence[str],
        obs_im_data: Dict[str, pd.DataFrame],
        sim_im_data: Dict[str, pd.DataFrame],
        rels: Dict[str, np.ndarray],
        station_df: pd.DataFrame,
        periods: np.ndarray,
        pSA_keys: np.ndarray,
        dist_matrix: pd.DataFrame,
        device: str,
        max_dist: float = 100,
        sim_score_fn=similiarity_score,
    ):
        self.device = device
        self.sim_score_fn = sim_score_fn
        self.sites = sites
        self.site_combs = site_combs
        self.events = events
        self.rels = rels
        self.dist_matrix = dist_matrix

        self.all_sites = dist_matrix.index.values.astype(str)

        # Some sanity checking
        for cur_event in sites.keys():
            assert sites[cur_event].size == site_combs[cur_event].max() + 1

        # Number of realisations have to be the same for all events
        # TODO: Could in theory be different set of realisation
        # per model epoch
        assert np.all(
            [rels[self.events[0]].size == rels[cur_event].size for cur_event in events]
        )
        self.n_rels_used = rels[self.events[0]].size

        # Compute the number of samples per event
        self.n_samples_event = np.asarray(
            [
                self.site_combs[cur_event].shape[0] * self.n_rels_used
                for cur_event in self.events
            ]
        )
        self._cum_n_samples = np.cumsum(self.n_samples_event)

        self.obs_im_data = obs_im_data

        self.pSA_keys = pSA_keys
        self.periods = periods

        # Normalise the station data
        self.station_df = station_df.copy()
        self.station_df["vs30"] = (
            self.station_df["vs30"] - self.station_df["vs30"].mean()
        ) / self.station_df["vs30"].std()
        self.station_df["Z_1.0"] = (
            self.station_df["Z_1.0"] - self.station_df["Z_1.0"].mean()
        ) / self.station_df["Z_1.0"].std()
        self.station_df["Z_2.5"] = (
            self.station_df["Z_2.5"] - self.station_df["Z_2.5"].mean()
        ) / self.station_df["Z_2.5"].std()

        # Scale the (used) site-to-site distances
        # such that they are between -1 and 1
        # as per the maximum allowed site-to-site
        # distance when computing the site combinations
        self.scaled_dist_matrix = dist_matrix.copy()
        self.scaled_dist_matrix = ((self.scaled_dist_matrix / max_dist) * 2) - 1

        # Organize the sim response spectra such that it is
        # in the format [n_rels, n_periods, n_sites]
        # per event
        self.sim_im_data = {}
        for cur_event in self.events:
            cur_sites = self.sites[cur_event]

            cur_sim_im_data = []
            for cur_site in cur_sites:
                cur_sim_im_data.append(
                    sim_im_data[cur_event][cur_site].loc[self.rels[cur_event], pSA_keys]
                )

            self.sim_im_data[cur_event] = np.stack(cur_sim_im_data, axis=2)

    @property
    def n_samples(self):
        return self._cum_n_samples[-1]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        # Break the index down
        event_ix = np.argmin(idx // self._cum_n_samples)
        site_ix = (idx % self._cum_n_samples[max(event_ix - 1, 0)]) // self.n_rels_used
        rel_ix = idx % self.n_rels_used

        # Get the site of interest and observation site
        event = self.events[event_ix]
        site_int_ix = self.site_combs[event][site_ix, 0]
        site_obs_ix = self.site_combs[event][site_ix, 1]

        site_int = self.sites[event][site_int_ix]
        site_obs = self.sites[event][site_obs_ix]

        # site_int = self.all_sites[site_int_ix]
        # site_obs = self.all_sites[site_obs_ix]

        # Features
        site_int_sim = self.sim_im_data[event][rel_ix, :, site_int_ix]
        site_obs_sim = self.sim_im_data[event][rel_ix, :, site_obs_ix]
        site_obs_obs = (
            self.obs_im_data[event].loc[site_obs, self.pSA_keys].values.astype(float)
        )

        site_features = np.concatenate(
            (
                self.station_df.loc[site_int].loc[["vs30", "Z_1.0", "Z_2.5"]].values,
                self.station_df.loc[site_int].loc[["vs30", "Z_1.0", "Z_2.5"]].values,
                [self.scaled_dist_matrix.loc[site_int, site_obs]],
            )
        )

        # Labels
        site_int_obs = (
            self.obs_im_data[event].loc[site_int, self.pSA_keys].values.astype(float)
        )
        sim_score = self.sim_score_fn(site_int_obs, site_int_sim)

        return (
            site_int_sim,
            site_obs_sim,
            site_obs_obs,
            site_features,
            sim_score,
            self.dist_matrix.loc[site_int, site_obs],
        )


class ResponseSpectrumSimModel(nn.Module):
    def __init__(
        self,
        rs_kernel_sizes: List[int],
        rs_n_channels: List[int],
        rs_padding: List[int],
        fc_units: List[int],
        rs_input_length: int,
        n_scalar_inputs: int,
    ):
        super().__init__()

        self.n_rs_layers = len(rs_kernel_sizes)
        self.n_fc_layers = len(fc_units)

        ### Add the convolutional layers
        self.rs_layers = nn.Sequential()
        for i in range(self.n_rs_layers):
            self.rs_layers.append(
                nn.Conv1d(
                    rs_n_channels[i],
                    rs_n_channels[i + 1],
                    rs_kernel_sizes[i],
                    padding=rs_padding[i],
                )
            )
            self.rs_layers.append(nn.ELU())
            self.rs_layers.append(nn.MaxPool1d(2))
        self.rs_layers.append(nn.Flatten())

        ### Add the fully connected layers
        # Get the conv out size
        conv_out_size = self.rs_layers(torch.zeros(1, 1, rs_input_length)).shape[-1]
        fc_input_size = (conv_out_size * 3) + n_scalar_inputs
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_units)):
            if i == 0:
                self.fc_layers.append(nn.Linear(fc_input_size, fc_units[i]))
            else:
                self.fc_layers.append(nn.Linear(fc_units[i - 1], fc_units[i]))

            self.fc_layers.append(nn.ELU())

        self.fc_layers.append(nn.Linear(self.fc_layers[-2].out_features, 1))
        self.fc_layers.append(nn.Sigmoid())

    def forward(self, rs_int_sim, rs_obs_sim, rs_obs_obs, scalar_features):
        rs_int_sim_out = self.rs_layers(rs_int_sim)
        rs_obs_sim_out = self.rs_layers(rs_obs_sim)
        rs_obs_obs_out = self.rs_layers(rs_obs_obs)

        rs_conv_out = torch.cat(
            (rs_int_sim_out, rs_obs_sim_out, rs_obs_obs_out, scalar_features), 1
        )

        return self.fc_layers(rs_conv_out)


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

    BATCH_SIZE = 32
    N_SCALAR_FEATURES = 7
    RS_N_CHANNELS = [1, 16, 32]
    RS_KERNEL_SIZES = [5, 3]
    FC_UNITS = [64, 32, 16]
    ### END CONFIG ###

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
    train_site_combs, train_event_sites = compute_site_combinations(
        event_sites, train_events, dist_matrix, sites_to_use=train_sites
    )
    val_site_combs, val_event_sites = compute_site_combinations(
        event_sites, val_events, dist_matrix, sites_to_use=val_sites
    )

    # Get the periods and corresponding pSA keys
    periods, pSA_keys = sr.utils.get_periods(
        obs_im_data[events[0]].columns.values.astype(str)
    )

    # Create the training and validation dataset
    print(f"Creating datasets and dataloaders")
    train_dataset = ResponseSpectrumDataset(
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
    val_dataset = ResponseSpectrumDataset(
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
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )

    # Create the model
    n_rs_layers = len(RS_KERNEL_SIZES)
    padding = [
        mlt.dl_utils.compute_same_conv_padding(31, RS_KERNEL_SIZES[0])
    ] * n_rs_layers
    out_size = mlt.dl_utils.get_conv_out_sizes(
        31, RS_KERNEL_SIZES, [1] * n_rs_layers, padding, [2] * n_rs_layers
    )

    model = ResponseSpectrumSimModel(
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
