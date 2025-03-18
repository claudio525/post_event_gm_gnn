from pathlib import Path
from typing import List, Sequence

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import typer
from tqdm import tqdm
import seaborn as sns

import sim_ranking as sr
import ml_tools as mlt


app = typer.Typer()


@app.command("mag-rrup-scatter")
def mag_rrup_scatter(nzgmdb_ffp: Path, output_ffp: Path):
    """
    Creates a scatter of rrup vs magnitude
    with the marginal distributions on the sides
    """
    obs_data = sr.data.load_obs_nzgmdb(nzgmdb_ffp)

    g = sns.jointplot(
        obs_data.record_df,
        x=sr.ObservedData.EventSiteColEnums.RRUP,
        y=sr.ObservedData.EventColEnums.MAG,
        marker=".",
        marginal_kws=dict(bins=25),
    )
    g.set_axis_labels("$R_{Rup}$ (km)", "Magnitude")
    plt.savefig(output_ffp)


@app.command("sample-weighting")
def sample_weighting(
    nzgmdb_ffp: Path,
    output_dir: Path,
    max_dist: float,
    closest_max_dist: float,
    max_n_obs_sites: int,
    min_n_obs_sites: int,
    min_pga: float = 0.025,
    mag_n_bins: int = 20,
    rrup_n_bins: int = 20,
):
    # Load observed data
    obs_data = sr.data.load_obs_nzgmdb(nzgmdb_ffp)

    events, all_sites = obs_data.events, obs_data.sites
    event_sites = obs_data.event_sites
    print(f"Number of events: {len(events)}")

    # Get the set of valid site-interests per event
    print("Getting valid sites of interest")
    int_sites, valid_event_int_sites, _ = sr.ml.data.get_valid_site_ints(
        event_sites, obs_data.record_df.drop(columns=obs_data.ims), min_pga=min_pga
    )
    events = np.intersect1d(events, np.asarray(list(valid_event_int_sites.keys())))

    # Distance matrix
    dist_matrix = sr.utils.calculate_distance_matrix(all_sites, obs_data.site_df)

    # Loth & Baker spatial correlations
    corr_data = sr.LBSiteCorrelationData.from_dist_matrix(
        dist_matrix, sr.constants.PSA_KEYS
    )

    # Compute available scenarios
    obs_sites = all_sites
    site_combs, event_sites = sr.ml.data.compute_site_combinations(
        event_sites,
        valid_event_int_sites,
        events,
        dist_matrix,
        obs_sites,
        int_sites,
        max_dist,
        closest_max_dist,
        max_n_obs_sites,
        min_n_obs_sites,
    )
    # Create scenario dataframe
    scenario_df = sr.ml.utils.create_scenario_df(
        site_combs,
        event_sites,
        obs_data,
        dist_matrix=dist_matrix,
        lb_corr_data=corr_data,
    )
    print(f"Number of scenarios: {len(scenario_df)}")

    ### Magnitude
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.histplot(scenario_df["mag"], bins=mag_n_bins, ax=ax)
    ax.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.25))

    # Weighting
    weight_func = sr.ml.gnn_gm.get_mag_weight_func(0.0, 6, 4.8, 6.5)
    mag_values = np.linspace(scenario_df.mag.min(), scenario_df.mag.max(), 100)
    weights = np.asarray([weight_func(cur_mag) for cur_mag in mag_values])
    ax_weight = ax.twinx()
    ax_weight.plot(mag_values, weights, color="red", linestyle="--")
    ax_weight.set_ylim(0.0, None)

    ax.set_title("Magnitude Distribution (Original) + Weighting Function")
    fig.tight_layout()

    plt.savefig(output_dir / "mag_weighting.png")
    plt.close(fig)

    ### RRUP
    fig, ax_hist = plt.subplots(1, 1, figsize=(8, 6))
    sns.histplot(scenario_df["rrup"], bins=rrup_n_bins, ax=ax_hist)
    ax_hist.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax_hist.xaxis.set_minor_locator(plt.MultipleLocator(5))
    ax_hist.set_xlim(0.0)
    fig.tight_layout()

    plt.savefig(output_dir / "rrup_hist.png")
    plt.close(fig)

    ### Degree of constraint
    # Distribution
    fig, ax_hist = plt.subplots(1, 1, figsize=(8, 6))
    sns.histplot(scenario_df["constraintness"], bins=20, ax=ax_hist)
    ax_hist.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    ax_hist.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    ax_hist.set_xlim(0.0)

    # Weighting
    doc_weight_fn = sr.ml.gnn_gm.get_doc_weight_func(0.0, 2, 1, 6)
    doc_values = np.linspace(scenario_df.constraintness.min(), scenario_df.constraintness.max(), 100)
    weights = [doc_weight_fn(cur_doc) for cur_doc in doc_values]

    ax_weight = ax_hist.twinx()
    ax_weight.plot(doc_values, weights, color="red", linestyle="--")
    ax_weight.set_ylim(0.0, None)
    ax_hist.set_title("Degree of Constraint Distribution (Original) + Weighting Function")

    fig.tight_layout()
    plt.savefig(output_dir / "doc_weighting.png")


if __name__ == "__main__":
    app()
