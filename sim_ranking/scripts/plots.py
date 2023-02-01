from pathlib import Path
from typing import List, Sequence

import pandas as pd
import numpy as np
import typer
import matplotlib
import matplotlib.pyplot as plt

import gmhazard_calc as gc
import sim_ranking as sr
from qcore.timeseries import BBSeis, read_ascii

app = typer.Typer()


@app.command("cmvn-result-plots")
def gen_cMVN_plots(
    results_dir: Path,
    sim_imdb_ffp: Path,
    obs_data_ffp: Path,
    sites: List[str] = None,
    show_all_sims: bool = False,
):
    """
    Generates
      - response spectrum plot for each site
      - summary residual plots
    """
    (output_dir := results_dir / "plots").mkdir(exist_ok=True)

    # Load the conditional MVN & site misfit data
    cMVN_result = sr.ConditionalMVNDistribution.load(
        results_dir / "cMVN_distributions.pickle"
    )
    best_sim_ids = pd.read_csv(results_dir / "best_sim_ids.csv", index_col=0).squeeze()
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
    obs_cmvn_ln_ratio = (
        np.log(obs_df.loc[sites, pSA_keys])
        - cMVN_result.cond_lnIM_mean_df.loc[sites, pSA_keys]
    )
    cmvn_sim_ln_ratio = cMVN_result.cond_lnIM_mean_df.loc[sites, pSA_keys] - np.log(
        best_sim_df
    )

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


@app.command("cmvn-waveform-plots")
def gen_cMVN_waveform_plots(
    results_dir: Path,
    sim_rupture_dir: Path = typer.Argument(
        ..., help="The event directory in the Runs folder"
    ),
    obs_waveform_dir: Path = typer.Argument(
        ..., help="Path to the acceleration waveform directory for this rupture"
    ),
    obs_data_ffp: Path = typer.Argument(..., help="Path the NZ-GMDB IM flat file"),
):
    """
    Generates waveform plots for each site
    """
    best_sim_ids = pd.read_csv(results_dir / "best_sim_ids.csv", index_col=0).squeeze()
    sites = best_sim_ids.index.values.astype(str)
    rupture = sr.ConditionalMVNDistribution.load(
        results_dir / "cMVN_distributions.pickle"
    ).rupture

    obs_df = sr.load_obs_rupture_data(obs_data_ffp, rupture)

    (output_dir := results_dir / "plots" / "site_plots").mkdir(
        exist_ok=True, parents=True
    )
    for ix, cur_site in enumerate(sites):
        print(f"Processing site {cur_site}, {ix + 1}/{sites.size}")

        cur_sim_id = best_sim_ids.loc[cur_site]

        # Get the BB file
        if not (
            cur_bb_ffp := sim_rupture_dir / cur_sim_id / "BB" / "Acc" / "BB.bin"
        ).exists():
            print(f"Can't find BB file for {cur_site} - {cur_sim_id}")
            continue

        bb = BBSeis(str(cur_bb_ffp))
        sim_acc = bb.acc(cur_site)
        sim_t = bb.dt * np.arange(sim_acc.shape[0])

        if bb.start_sec < 0:
            sim_mask = sim_t > np.abs(bb.start_sec)
            sim_acc = sim_acc[sim_mask, :]
            sim_t = bb.dt * np.arange(sim_acc.shape[0])
        else:
            raise NotImplementedError()

        # Get the observed waveforms
        if not all(
            [
                (obs_waveform_dir / f"{cur_site}.{cur_comp}").exists()
                for cur_comp in sr.constants.COMPONENTS
            ]
        ):
            print(f"Can't find all acceleration waveform files for {cur_site}")
            continue

        obs_acc = []
        meta = None
        for cur_comp in sr.constants.COMPONENTS:
            cur_acc, cur_meta = read_ascii(
                str(obs_waveform_dir / f"{cur_site}.{cur_comp}"), meta=True
            )
            if meta is None:
                meta = cur_meta
            else:
                assert meta["dt"] == cur_meta["dt"]
            obs_acc.append(cur_acc)

        obs_acc = np.stack(obs_acc, axis=1)
        obs_t = meta["dt"] * np.arange(obs_acc.shape[0])

        fig = plt.figure(figsize=(16, 10))

        sr.draw_waveforms(fig, [sim_acc, obs_acc], [sim_t, obs_t], ["r", "k"])

        fig.suptitle(
            f"{cur_site}, {r'$R_{rup}$'} = {obs_df.loc[cur_site, 'r_rup']:.0f} (km), "
            f"{'$V_{S30}$'} = {obs_df.loc[cur_site, 'Vs30']:.0f} (m/s)"
        )
        fig.tight_layout()

        plt.savefig(output_dir / f"{cur_site}_waveform.png")
        plt.close()


if __name__ == "__main__":
    app()
