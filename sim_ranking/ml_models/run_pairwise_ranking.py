import os
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import typer
from scipy.cluster import hierarchy
from scipy.spatial import distance
from plotly import graph_objects as go

import sim_ranking as sr
import spatial_hazard as sh

from sim_ranking.ml import pairwise as pr

app = typer.Typer()

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

print(f"Using device: {device}")


@app.command("train-model")
def train_model(
    hyperparams_ffp: Path,
    max_dist: float = 75,
    debug: bool = False,
    max_n_rels: int = 25,
):
    """Trains a single model"""
    run_config = pr.RunParamsConfig(max_dist, max_n_rels, debug, device)
    hp_config = pr.HyperParamsConfig.from_yaml(hyperparams_ffp)

    ### Data loading
    # db_ffp_orig = "$wdata/sim_ranking/db/gm_db.sqlite"
    db_ffp_orig = "$wdata/sim_ranking/db/gm_db_neil.sqlite"
    db_ffp = Path(os.path.expandvars(db_ffp_orig))

    db = sr.db.DB(db_ffp)

    # events = db.get_avail_events(data_source="specific")
    events = db.get_avail_events(data_source="neil")
    print(f"Number of events: {len(events)}")

    # Get all relevant sites across all events
    # all_sites = np.unique(np.concatenate(list(event_sites.values())))
    all_sites = db.get_avail_sites()

    ### Data setup

    # Get the sites per event
    event_sites = db.get_event_sites()

    # TODO: Fix this!!
    np.random.seed(30)
    # train_sites = all_sites
    val_int_sites = np.random.choice(all_sites, 100, replace=False)
    train_sites = np.setdiff1d(all_sites, val_int_sites)
    # val_sites = all_sites

    # val_events = np.asarray(["3468575"])
    # val_events = np.asarray(["3468575", "2016p118944", "3525264", "3528839"])
    val_events = np.random.choice(events, 100, replace=False)
    train_events = np.setdiff1d(events, val_events)

    # Data prep
    train_dataset, val_dataset, scalar_features, data_metadata = pr.data_prep(
        event_sites,
        train_events,
        val_events,
        train_sites,
        val_int_sites,
        events,
        run_config,
        db,
    )

    # Create the model
    ranking_model = pr.create_model(hp_config, scalar_features)

    # Train
    metrics, best_model_state, best_model_epoch = pr.train(
        ranking_model, train_dataset, val_dataset, device, hp_config, run_config
    )
    ranking_model.load_state_dict(best_model_state)

    print(
        f"Best model epoch: {best_model_epoch + 1}, "
        f"Validation:\n"
        f"\tLoss: {metrics['loss_hist_val'][best_model_epoch]:.4f}\n"
        f"\tAccuracy: {metrics['acc_hist_val'][best_model_epoch]:.4f}\n"
        f"\tBCELoss: {metrics['bce_loss_hist_val'][best_model_epoch]:.4f}\n"
    )

    data_metadata["db"] = db_ffp_orig

    # Post-processing
    pr.post_processing(
        ranking_model,
        train_dataset,
        val_dataset,
        hp_config,
        run_config,
        metrics,
        best_model_epoch,
        scalar_features,
        data_metadata,
    )


@app.command("run-kfold")
def run_kfold(
    hyperparams_ffp: Path,
    site_folds_ffp: Path,
    max_dist: float = 75,
    debug: bool = False,
    max_n_rels: int = 25,
):
    """
    Uses k-fold to get a better estimate of the model performance
    """
    run_config = pr.RunParamsConfig(max_dist, max_n_rels, debug, device)
    hp_config = pr.HyperParamsConfig.from_yaml(hyperparams_ffp)

    ### Data loading
    # db_ffp_orig = "$wdata/sim_ranking/db/gm_db.sqlite"
    db_ffp_orig = "$wdata/sim_ranking/db/gm_db_neil.sqlite"
    db_ffp = Path(os.path.expandvars(db_ffp_orig))

    db = sr.db.DB(db_ffp)

    # Get all relevant sites across all events
    all_sites = db.get_avail_sites()

    # Get the fold clusters
    folds_labels = pd.read_csv(site_folds_ffp, index_col=0).squeeze()

    # Sanity check
    assert np.all(all_sites == folds_labels.index.values)

    # Some data loading
    events = db.get_avail_events(data_source="neil")
    print(f"Number of events: {len(events)}")

    obs_df = db.get_obs_df()
    obs_df = obs_df.loc[np.isin(obs_df.event_id, events)]

    # Get the sites per event
    event_sites = db.get_event_sites()

    for cur_fold_id in np.unique(folds_labels.values):
        # Get the sites for this fold
        cur_val_sites = folds_labels.index.values[
            folds_labels.values == cur_fold_id
        ].astype(str)
        cur_train_sites = np.setdiff1d(all_sites, cur_val_sites)

        cur_obs_df = obs_df.loc[np.isin(obs_df.site_id, cur_val_sites)]
        cur_val_events = (
            cur_obs_df[["event_id", "site_id"]]
            .groupby("event_id")
            .count()
            .sort_values("site_id")
            .tail(25)
            .index.values
        )
        cur_train_events = np.setdiff1d(events, cur_val_events)

        # Data prep
        train_dataset, val_dataset, scalar_features = pr.data_prep(
            event_sites,
            cur_train_events,
            cur_val_events,
            cur_train_sites,
            cur_val_sites,
            events,
            run_config,
            db,
        )

        # Create the model
        ranking_model = pr.create_model(hp_config, scalar_features)

        # Train
        pr.train(
            ranking_model, train_dataset, val_dataset, device, hp_config, run_config
        )


@app.command("gen-site-fold-clusters")
def gen_site_fold_clusters(db_ffp: Path, output_dir: Path, n_clusters: int = 4):
    def compute_hierarchical_linkage_matrix(dist_mat, method="complete"):
        if method == "complete":
            return hierarchy.complete(dist_mat)
        if method == "single":
            return hierarchy.single(dist_mat)
        if method == "average":
            return hierarchy.average(dist_mat)
        if method == "ward":
            return hierarchy.ward(dist_mat)

    # Load data
    db = sr.db.DB(db_ffp)
    sites = db.get_avail_sites()
    station_df = db.get_site_df()

    # Compute distance matrix
    dist_matrix = sh.im_dist.calculate_distance_matrix(sites, station_df)

    # Run hierarchical clustering
    condensed_dist_matrix = distance.squareform(dist_matrix.values)
    Z = compute_hierarchical_linkage_matrix(condensed_dist_matrix, method="ward")

    cluster_labels = hierarchy.fcluster(Z, n_clusters, criterion="maxclust")

    # Save the cluster labels
    pd.Series(index=sites, data=cluster_labels, name="cluster").to_csv(
        output_dir / "site_clusters.csv"
    )

    # Create figure
    fig = go.Figure()

    colors = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "black",
        "yellow",
        "pink",
        "brown",
        "grey",
    ]
    for ix, cur_cluster in enumerate(np.unique(cluster_labels)):
        cur_mask = cluster_labels == cur_cluster
        fig.add_trace(
            go.Scattermapbox(
                lat=station_df.loc[sites].loc[cur_mask].lat,
                lon=station_df.loc[sites].loc[cur_mask].lon,
                mode="markers",
                marker=dict(size=10, color=colors[ix]),
            )
        )

    # fig.update_layout(height=600, margin=dict(l=0, r=0, t=0, b=0))
    fig.update_mapboxes(
        accesstoken="pk.eyJ1IjoiY3MyMyIsImEiOiJjbGtpeXIxNnkwbDQ3M25xbDFrZWFnNHo3In0.OD7TJ_1PegpGvCOCxfHsnA",
        center=dict(
            lat=station_df.lat.mean(),
            lon=station_df.lon.mean(),
        ),
        zoom=8,
    )
    fig.write_html(output_dir / "site_clusters.html")


if __name__ == "__main__":
    app()
