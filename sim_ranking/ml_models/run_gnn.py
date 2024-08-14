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

import sim_ranking as sr
import spatial_hazard as sh


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

print(f"Using device: {device.upper()}")

def run_gnn(
    rel_db_ffp: Path = typer.Argument(
        ...,
        help="Relative path to the database file with "
        "respect to $wdata env variable",
    ),
    max_dist: float = typer.Option(
        50,
        help="Maximum allowed distance of an observation "
        "site from the site of interest",
    ),
    seed: int = typer.Option(None),
    n_val_events: int = typer.Option(100, help="Number of validation events"),
    n_val_sites: int = typer.Option(100, help="Number of validation sites"),
):
    # Config
    n_epochs = 50
    batch_size = 64

    ims = sr.constants.PSA_KEYS
    cur_im = "pSA_1.0"

    ### Data loading
    db_ffp = Path(os.path.expandvars("$wdata")) / rel_db_ffp
    db = sr.db.DB(db_ffp)

    events = db.get_avail_events(data_source=None)
    print(f"Number of events: {len(events)}")

    # Get all relevant sites across all events
    all_sites = db.get_avail_sites()

    ### Data setup
    # Get the sites per event
    event_sites = db.get_event_sites()

    # Get the set of valid site-interests per event
    print(f"Getting valid sites of interest")
    valid_int_sites, valid_event_int_sites = sr.ml.data.get_valid_site_ints(
        event_sites, db.get_record_df(), db.get_site_df()
    )

    # Split into training and validation
    if seed is not None:
        print(f"Using numpy random seed: {seed}")
        np.random.seed(seed)
    val_events = np.random.choice(events, n_val_events, replace=False)
    train_events = np.setdiff1d(events, val_events)

    print(f"----------------- Events Summary -----------------")
    print(f"Number of available events: {len(events)}")
    print(f"Number of training events: {train_events.size}")
    print(f"Number of validation events: {val_events.size}")

    val_int_sites = np.random.choice(valid_int_sites, n_val_sites, replace=False)
    train_int_sites = np.setdiff1d(valid_int_sites, val_int_sites)
    obs_sites = np.setdiff1d(all_sites, val_int_sites)

    print(f"----------------- Sites Summary -----------------")
    print(f"Number of available sites: {len(all_sites)}")
    print(f"Number of valid sites of interests: {valid_int_sites.size}")
    print(f"Number of training sites of interests: {train_int_sites.size}")
    print(f"Number of validation sites of interests: {val_int_sites.size}")
    print(f"Number of observation sites: {obs_sites.size}")
    print(f"------------------------------------------------")

    # Get the scalar feature keys
    scalar_feature_keys = sr.constants.SCALAR_FEATURE_SET_LOOKUP[
        sr.constants.ScalarFeatureSetKey.all
    ]
    event_feature_keys = scalar_feature_keys["event"]
    site_feature_keys = scalar_feature_keys["site"]

    event_df = db.get_event_df()
    record_df = db.get_record_df()

    print(f"Computing distance matrix")
    station_df = db.get_site_df()
    all_sites = db.get_avail_sites()
    dist_matrix = sh.im_dist.calculate_distance_matrix(all_sites, station_df)

    ### Scalar Features
    # Run pre-processing for the site features
    # TODO: This should be updated such that the normalisation
    # only happens on training sites, not all sites
    print(f"Pre-processing site & event features")
    site_features_df, site_feature_stats = sr.ml.features.preprocess_site_features(
        station_df, site_feature_keys
    )

    event_features_stats = pd.DataFrame(
        index=["mean", "std"], columns=event_feature_keys
    )
    event_features_stats.loc["mean"] = event_df.loc[events, event_feature_keys].mean()
    event_features_stats.loc["std"] = event_df.loc[events, event_feature_keys].std()
    event_features_df = event_df.loc[events, event_feature_keys]
    event_features_df[event_feature_keys] = (
        event_df.loc[events, event_feature_keys]
        - event_features_stats.loc["mean", event_feature_keys]
    ) / event_features_stats.loc["std", event_feature_keys]

    # Compute the site-to-site features
    print(f"Computing scalar features")
    (
        site_to_site_features,
        event_site_features,
        event_site_to_site_features,
    ) = sr.ml.features.compute_scalar_features(
        events,
        event_sites,
        event_df,
        station_df,
        record_df,
        dist_matrix,
        max_dist,
    )
    scalar_features = sr.ml.data.ScalarFeatures(
        event_features_df,
        event_feature_keys,
        site_features_df,
        site_feature_keys,
        site_to_site_features,
        scalar_feature_keys["site_to_site"],
        event_site_features,
        scalar_feature_keys["event_site"],
        event_site_to_site_features,
        scalar_feature_keys["event_site_to_site"],
    )

    # Compute mean and standard deviation for each period
    # for normalisation (only training events)
    obs_data = db.get_obs_df()
    ims_mean = np.mean(np.log(obs_data.loc[:, ims]), axis=0)
    ims_std = np.std(np.log(obs_data.loc[:, ims]), axis=0)

    print(f"Creating site combinations")
    train_site_combs, train_event_sites = sr.ml.data.compute_site_combinations(
        event_sites,
        valid_event_int_sites,
        train_events,
        dist_matrix,
        obs_sites,
        train_int_sites,
        max_dist=max_dist,
    )
    val_site_combs, val_event_sites = sr.ml.data.compute_site_combinations(
        event_sites,
        valid_event_int_sites,
        val_events,
        dist_matrix,
        obs_sites,
        val_int_sites,
        max_dist=max_dist,
    )

    edge_feature_keys = ["dist", "angular_dist"]
    site_int_scalar_feature_keys = [
        "vs30_site_int",
        "z1.0_site_int",
        "z2.5_site_int",
        "tsite_site_int",
        "r_rup_site_int",
        "mag",
    ]
    site_obs_scalar_feature_keys = [
        "vs30_site_obs",
        "z1.0_site_obs",
        "z2.5_site_obs",
        "tsite_site_obs",
        "r_rup_site_obs",
        "mag",
    ]

    site_int_n_node_features = len(site_int_scalar_feature_keys)
    site_obs_n_node_features = len(site_obs_scalar_feature_keys) + 1

    print(f"Getting graph data")
    train_graph_data = sr.ml.gnn_gm.get_graph_data(
        db,
        train_event_sites,
        train_site_combs,
        scalar_features,
        site_int_scalar_feature_keys,
        site_obs_scalar_feature_keys,
        edge_feature_keys,
        ims_mean,
        ims_std,
        [cur_im],
    )

    val_graph_data = sr.ml.gnn_gm.get_graph_data(
        db,
        val_event_sites,
        val_site_combs,
        scalar_features,
        site_int_scalar_feature_keys,
        site_obs_scalar_feature_keys,
        edge_feature_keys,
        ims_mean,
        ims_std,
        [cur_im],
    )

    train_loader = gloader.DataLoader(train_graph_data, batch_size=batch_size, shuffle=True)
    val_loader = gloader.DataLoader(val_graph_data, batch_size=batch_size, shuffle=True)


    gnn_model = sr.ml.gnn_modules.CustomGNN(
        site_obs_n_node_features, site_int_n_node_features, len(edge_feature_keys), 32
    )
    gnn_model.to(device)

    # cur_data = train_graph_data[12]
    # gnn_model.forward(cur_data)

    # cur_batch = next(iter(train_loader))
    # gnn_model.forward(cur_batch)

    print(f"----------------- Training -----------------")
    print(f"Number of training graphs: {len(train_graph_data)}")
    print(f"Number of validation graphs: {len(val_graph_data)}")


    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)
    for cur_epoch in range(n_epochs):
        print(f"Epoch: {cur_epoch}")

        epoch_loss, n_graphs = 0, 0
        gnn_model.train()
        for cur_batch in tqdm.tqdm(train_loader):
            cur_batch = cur_batch.to(device)

            optimizer.zero_grad()
            out = gnn_model(cur_batch)
            loss = torch.nn.functional.mse_loss(out, cur_batch.y[:, None])
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_graphs += cur_batch.num_graphs

        train_avg_loss = epoch_loss / n_graphs

        gnn_model.eval()
        epoch_loss, n_graphs = 0, 0
        for cur_batch in val_loader:
            cur_batch = cur_batch.to(device)

            out = gnn_model(cur_batch)
            loss = torch.nn.functional.mse_loss(out, cur_batch.y[:, None])

            epoch_loss += loss.item()
            n_graphs += cur_batch.num_graphs

        val_avg_loss = epoch_loss / n_graphs

        print(f"Epoch {cur_epoch}/{n_epochs}")
        print(
            f"\tTraining"
            f"\t\tLoss: {train_avg_loss:.4f}"
        )
        print(
            f"\tValidation"
            f"\t\tLoss: {val_avg_loss:.4f}"
        )

    print(f"WTF")


if __name__ == "__main__":
    typer.run(run_gnn)
