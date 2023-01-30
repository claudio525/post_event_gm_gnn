from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
import typer
import matplotlib.pyplot as plt

import gmhazard_calc as gc
import sim_ranking as sr

app = typer.Typer()

# @app.command()
# def plot_sites_historic_events(sites_ffp: Path, source_ffp: Path, map_data_ffp: Path = None):
#     site_df = pd.read_csv(sites_ffp)
#     source_df = pd.read_csv(source_ffp)
#
#     # Only interested in crustal
#     source_df = source_df.loc[source_df.tect_class == "Crustal"]
#
#     # fig = plotting.gen_region_fig("Sites & Faults", "NZ")
#
#     print(f"wtf")


@app.command()
def gen_cMVN_site_plots(
    results_dir: Path,
    sim_imdb_ffp: Path,
    obs_data_ffp: Path,
    sites: List[str] = None,
    show_all_sims: bool = False,
):
    (output_dir := results_dir / "plots").mkdir(exist_ok=True)

    # Load the conditional MVN & site misfit data
    cMVN_result = sr.ConditionalMVNDistribution.load(
        results_dir / "cMVN_distributions.pickle"
    )
    site_misfits_df = pd.read_csv(results_dir / "site_misfits.csv", index_col=0)

    sites = cMVN_result.stations if len(sites) == 0 else np.asarray(sites)

    # Load the observation & simulation data
    obs_df = sr.load_obs_rupture_data(obs_data_ffp, cMVN_result.rupture)
    sim_data = sr.load_sim_data(sim_imdb_ffp, sites)

    # Get relevant periods
    periods = np.sort(
        [
            cur_im.period
            for cur_im in cMVN_result.IMs
            if cur_im.im_type == gc.im.IMType.pSA
        ]
    )
    pSA_keys = [f"pSA_{cur_period}" for cur_period in periods]

    for ix, cur_site in enumerate(sites):
        print(f"Processing site {cur_site}, {ix+1}/{sites.size}")
        if not cur_site in sim_data.keys():
            print(f"No simulation data for site {cur_site}, skipping")
            continue

        cur_best_sim_id = site_misfits_df.index.values[
            site_misfits_df.loc[:, cur_site].argmin()
        ]

        fig = plt.figure(figsize=(16, 10))

        # All other simulations
        if show_all_sims:
            for cur_sim_id, cur_row in sim_data[cur_site].iterrows():
                if cur_sim_id == cur_best_sim_id:
                    continue

                plt.plot(
                    periods,
                    cur_row.loc[pSA_keys].values.astype(float),
                    c="gray",
                    alpha=0.3,
                    linewidth=0.5,
                )

        # Observation
        plt.plot(
            periods,
            obs_df.loc[cur_site, pSA_keys].values.astype(float),
            c="k",
            linewidth=1.2,
            label="Observed",
        )

        # Best Simulation
        plt.plot(
            periods,
            sim_data[cur_site].loc[cur_best_sim_id, pSA_keys].values.astype(float),
            c="r",
            linewidth=1.2,
            label="Simulation",
        )

        # Conditional MVN
        plt.plot(
            periods,
            np.exp(
                cMVN_result.cond_lnIM_mean_df.loc[cur_site, pSA_keys].values.astype(
                    float
                )
            ),
            c="b",
            linewidth=1.2,
            label=r"Conditional MVN",
        )
        plt.plot(
            periods,
            np.exp(
                cMVN_result.cond_lnIM_mean_df.loc[cur_site, pSA_keys].values.astype(
                    float
                ) + cMVN_result.cond_lnIM_std_df.loc[cur_site, pSA_keys].values.astype(
                    float
                )
            ),
            c="b",
            linewidth=1.0,
            linestyle="--",
        )
        plt.plot(
            periods,
            np.exp(
                cMVN_result.cond_lnIM_mean_df.loc[cur_site, pSA_keys].values.astype(
                    float
                ) - cMVN_result.cond_lnIM_std_df.loc[cur_site, pSA_keys].values.astype(
                    float
                )
            ),
            c="b",
            linewidth=1.0,
            linestyle="--",
        )



        plt.semilogx()
        plt.xlim(periods.min(), periods.max())

        plt.title(
            f"{cur_site}, {r'$R_{rup}$'} = {obs_df.loc[cur_site, 'r_rup']:.0f} (km), "
            f"{'$V_{S30}$'} = {obs_df.loc[cur_site, 'Vs30']:.0f} (m/s)"
        )
        plt.xlabel(f"Period")
        plt.ylabel(f"Pseudo-spectral acceleration, Sa (g)")
        plt.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
        plt.legend()
        fig.tight_layout()

        plt.savefig(output_dir / f"{cur_site}_response_spectra.png")
        plt.close()


if __name__ == "__main__":
    app()
