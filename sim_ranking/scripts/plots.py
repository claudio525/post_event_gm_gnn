from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
import typer
import matplotlib
import matplotlib.pyplot as plt

import gmhazard_calc as gc
import sim_ranking as sr

app = typer.Typer()

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
    best_sim_ids = pd.read_csv(results_dir / "best_sim_ids.csv", index_col=0).squeeze()
    # site_misfits_df = pd.read_csv(results_dir / "site_misfits.csv", index_col=0)
    sites = cMVN_result.stations if len(sites) == 0 else np.asarray(sites)

    # Load the observation & simulation data
    obs_df = sr.load_obs_rupture_data(obs_data_ffp, cMVN_result.rupture)
    sim_data = sr.load_sim_data(sim_imdb_ffp, sites)

    # Drop any sites for which there is no simulation data
    mask = np.isin(sites, list(sim_data.keys()))
    if np.any(~mask):
        print(
            f"Dropping the following sites as no simulation data exists:\n{sites[~mask]}"
        )
        sites = sites[mask]

    # Get relevant periods
    periods = np.sort(
        [
            cur_im.period
            for cur_im in cMVN_result.IMs
            if cur_im.im_type == gc.im.IMType.pSA
        ]
    )
    pSA_keys = [f"pSA_{cur_period}" for cur_period in periods]

    # Individual site plots
    (site_output_dir := output_dir / "site_plots").mkdir(exist_ok=True)
    for ix, cur_site in enumerate(sites):
        print(f"Processing site {cur_site}, {ix+1}/{sites.size}")
        if not cur_site in sim_data.keys():
            print(f"No simulation data for site {cur_site}, skipping")
            continue

        # Plot response spectrum
        sr.plot_response_spectrum(
            periods,
            pSA_keys,
            sim_data[cur_site],
            cMVN_result,
            obs_df.loc[cur_site],
            cur_site,
            best_sim_ids.loc[cur_site],
            site_output_dir,
            show_all_sims=show_all_sims,
        )

    # Residual Plots
    best_sim_df = pd.concat(
        [
            sim_data[cur_site].loc[best_sim_ids.loc[cur_site], pSA_keys]
            for cur_site in sites
        ],
        axis=1,
    ).T
    best_sim_df.index = sites

    obs_sim_ln_ratio = np.log(obs_df.loc[sites, pSA_keys]) - np.log(best_sim_df)
    obs_cmvn_ln_ratio = np.log(obs_df.loc[sites, pSA_keys]) - cMVN_result.cond_lnIM_mean_df.loc[sites, pSA_keys]
    cmvn_sim_ln_ratio = cMVN_result.cond_lnIM_mean_df.loc[sites, pSA_keys] - np.log(best_sim_df)

    # All sites
    sr.plot_response_spectrum_residual(
        periods,
        pSA_keys,
        sites,
        obs_sim_ln_ratio,
        output_dir / "obs_sim_residuals.png",
        title="Observation - Simulation Residual",
        ylabel=r"$lnIM_{Obs} - lnIM_{Sim}$",
    )
    sr.plot_response_spectrum_residual(
        periods,
        pSA_keys,
        sites,
        obs_cmvn_ln_ratio,
        output_dir / "obs_cmvn_residuals.png",
        title="Observation - cMVN Residual",
        ylabel=r"$lnIM_{Obs} - lnIM_{cMVN}$",
    )
    sr.plot_response_spectrum_residual(
        periods,
        pSA_keys,
        sites,
        cmvn_sim_ln_ratio,
        output_dir / "cmvn_sim_residuals.png",
        title="cMVN - Simulation Residual",
        ylabel=r"$lnIM_{cMVN} - lnIM_{Sim}$",
    )

    # Rrup bins
    cur_sites = sites[obs_df.loc[sites, "r_rup"].values < 30]
    sr.plot_response_spectrum_residual(
        periods,
        pSA_keys,
        cur_sites,
        obs_sim_ln_ratio,
        output_dir / "obs_sim_residuals_rrup_0_30.png",
        title="Observation - Simulation Residual ($R_{Rup}$ < 30)",
        ylabel=r"$lnIM_{Obs} - lnIM_{Sim}$",
    )
    sr.plot_response_spectrum_residual(
        periods,
        pSA_keys,
        cur_sites,
        obs_cmvn_ln_ratio,
        output_dir / "obs_cmvn_residuals_rrup_0_30.png",
        title="Observation - cMVN Residual ($R_{Rup}$ < 30)",
        ylabel=r"$lnIM_{Obs} - lnIM_{cMVN}$",
    )
    sr.plot_response_spectrum_residual(
        periods,
        pSA_keys,
        cur_sites,
        cmvn_sim_ln_ratio,
        output_dir / "cmvn_sim_residuals_rrup_0_30.png",
        title="cMVN - Simulation Residual ($R_{Rup}$ < 30)",
        ylabel=r"$lnIM_{cMVN} - lnIM_{Sim}$",
    )


    cur_sites = sites[
        (obs_df.loc[sites, "r_rup"].values > 30)
        & (obs_df.loc[sites, "r_rup"].values < 75)
    ]
    sr.plot_response_spectrum_residual(
        periods,
        pSA_keys,
        cur_sites,
        obs_sim_ln_ratio,
        output_dir / "obs_sim_residuals_rrup_30_75.png",
        title="Observation - Simulation Residual (30 < $R_{Rup}$ < 75)",
        ylabel=r"$lnIM_{Obs} - lnIM_{Sim}$",
    )
    sr.plot_response_spectrum_residual(
        periods,
        pSA_keys,
        cur_sites,
        obs_cmvn_ln_ratio,
        output_dir / "obs_cmvn_residuals_rrup_30_75.png",
        title="Observation - cMVN Residual (30 < $R_{Rup}$ < 75)",
        ylabel=r"$lnIM_{Obs} - lnIM_{cMVN}$",
    )
    sr.plot_response_spectrum_residual(
        periods,
        pSA_keys,
        cur_sites,
        cmvn_sim_ln_ratio,
        output_dir / "cmvn_sim_residuals_rrup_30_75.png",
        title="cMVN - Simulation Residual (30 < $R_{Rup}$ < 75)",
        ylabel=r"$lnIM_{cMVN} - lnIM_{Sim}$",
    )

    cur_sites = sites[(obs_df.loc[sites, "r_rup"].values > 75)]
    sr.plot_response_spectrum_residual(
        periods,
        pSA_keys,
        cur_sites,
        obs_sim_ln_ratio,
        output_dir / "obs_sim_residuals_rrup_75.png",
        title="Observation - Simulation Residual (75 < $R_{Rup}$)",
        ylabel=r"$lnIM_{Obs} - lnIM_{Sim}$",
    )
    sr.plot_response_spectrum_residual(
        periods,
        pSA_keys,
        cur_sites,
        obs_cmvn_ln_ratio,
        output_dir / "obs_cmvn_residuals_rrup_75.png",
        title="Observation - cMVN Residual (75 < $R_{Rup}$)",
        ylabel=r"$lnIM_{Obs} - lnIM_{cMVN}$",
    )
    sr.plot_response_spectrum_residual(
        periods,
        pSA_keys,
        cur_sites,
        cmvn_sim_ln_ratio,
        output_dir / "cmvn_sim_residuals_rrup_75.png",
        title="cMVN - Simulation Residual (75 < $R_{Rup}$)",
        ylabel=r"$lnIM_{cMVN} - lnIM_{Sim}$",
    )

    # Vs30 bins
    cur_sites = sites[obs_df.loc[sites, "Vs30"].values < 300]
    sr.plot_response_spectrum_residual(
        periods,
        pSA_keys,
        cur_sites,
        obs_sim_ln_ratio,
        output_dir / "obs_sim_residuals_vs30_0_300.png",
        title="Observation - Simulation Residual ($V_{S30}$ < 300)",
        ylabel=r"$lnIM_{Obs} - lnIM_{Sim}$",
    )
    sr.plot_response_spectrum_residual(
        periods,
        pSA_keys,
        cur_sites,
        obs_cmvn_ln_ratio,
        output_dir / "obs_cmvn_residuals_vs30_0_300.png",
        title="Observation - cMVN Residual ($V_{S30}$ < 300)",
        ylabel=r"$lnIM_{Obs} - lnIM_{cMVN}$",
    )
    sr.plot_response_spectrum_residual(
        periods,
        pSA_keys,
        cur_sites,
        cmvn_sim_ln_ratio,
        output_dir / "cmvn_sim_residuals_vs30_0_300.png",
        title="cMVN - Simulation Residual ($V_{S30}$ < 300)",
        ylabel=r"$lnIM_{cMVN} - lnIM_{Sim}$",
    )


    cur_sites = sites[
        (obs_df.loc[sites, "Vs30"].values > 300)
        & (obs_df.loc[sites, "Vs30"].values < 500)
    ]
    sr.plot_response_spectrum_residual(
        periods,
        pSA_keys,
        cur_sites,
        obs_sim_ln_ratio,
        output_dir / "obs_sim_residuals_vs30_300_500.png",
        title="Observation - Simulation Residual (300 < $V_{S30}$ < 500)",
        ylabel=r"$lnIM_{Obs} - lnIM_{Sim}$",
    )
    sr.plot_response_spectrum_residual(
        periods,
        pSA_keys,
        cur_sites,
        obs_cmvn_ln_ratio,
        output_dir / "obs_cmvn_residuals_vs30_300_500.png",
        title="Observation - cMVN Residual (300 < $V_{S30}$ < 500)",
        ylabel=r"$lnIM_{Obs} - lnIM_{cMVN}$",
    )
    sr.plot_response_spectrum_residual(
        periods,
        pSA_keys,
        cur_sites,
        cmvn_sim_ln_ratio,
        output_dir / "cmvn_sim_residuals_vs30_300_500.png",
        title="cMVN - Simulation Residual (300 < $V_{S30}$ < 500)",
        ylabel=r"$lnIM_{cMVN} - lnIM_{Sim}$",
    )

    cur_sites = sites[(obs_df.loc[sites, "r_rup"].values > 75)]
    sr.plot_response_spectrum_residual(
        periods,
        pSA_keys,
        cur_sites,
        obs_sim_ln_ratio,
        output_dir / "obs_sim_residuals_vs30_500.png",
        title="Observation - Simulation Residual (500 < $V_{S30}$)",
        ylabel=r"$lnIM_{Obs} - lnIM_{Sim}$",
    )
    sr.plot_response_spectrum_residual(
        periods,
        pSA_keys,
        cur_sites,
        obs_cmvn_ln_ratio,
        output_dir / "obs_cmvn_residuals_vs30_500.png",
        title="Observation - cMVN Residual (500 < $V_{S30}$)",
        ylabel=r"$lnIM_{Obs} - lnIM_{cMVN}$",
    )
    sr.plot_response_spectrum_residual(
        periods,
        pSA_keys,
        cur_sites,
        cmvn_sim_ln_ratio,
        output_dir / "cmvn_sim_residuals_vs30_500.png",
        title="cMVN Residual - Simulation (500 < $V_{S30}$)",
        ylabel=r"$lnIM_{cMVN} - lnIM_{Sim}$",
    )



if __name__ == "__main__":
    app()
