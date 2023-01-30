from typing import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .conditional_MVN import ConditionalMVNDistribution


def plot_response_spectrum(
    periods: np.ndarray,
    pSA_keys: Sequence[str],
    sim_data: pd.DataFrame,
    cMVN_result: ConditionalMVNDistribution,
    obs_data: pd.Series,
    site: str,
    best_sim_id: str,
    output_dir: Path,
    show_all_sims: bool = False,
):
    """
    Generates a response spectrum plot
    showing the observed, simulated, and
    conditional MVN

    Parameters
    ----------
    periods: array of floats
        The periods to plot
        (must be sorted)
    pSA_keys: sequence of strings
        The pSA keys into the dataframes
        Must be in order (same as periods)
    sim_data: dataframe
        Simulation IM values
        Index = Realisations
        Columns = IMs
    cMVN_result: ConditionalMVNDistribution
    obs_data: series
        The observation data for the
        current site
    site: string
        Site of interest
    best_sim_id: string
        Id of the best simulation realisation
    output_dir: Path
    show_all_sims: bool, optional
        If true, then all simulation
        realisations are plotted, not just
        the best one
    """
    fig = plt.figure(figsize=(16, 10))

    # All other simulations
    if show_all_sims:
        for cur_sim_id, cur_row in sim_data.iterrows():
            if cur_sim_id == best_sim_id:
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
        obs_data.loc[pSA_keys].values.astype(float),
        c="k",
        linewidth=1.2,
        label="Observed",
    )

    # Best Simulation
    plt.plot(
        periods,
        sim_data.loc[best_sim_id, pSA_keys].values.astype(float),
        c="r",
        linewidth=1.2,
        label="Simulation",
    )

    # Conditional MVN
    plt.plot(
        periods,
        np.exp(cMVN_result.cond_lnIM_mean_df.loc[site, pSA_keys].values.astype(float)),
        c="b",
        linewidth=1.2,
        label=r"Conditional MVN",
    )
    plt.plot(
        periods,
        np.exp(
            cMVN_result.cond_lnIM_mean_df.loc[site, pSA_keys].values.astype(float)
            + cMVN_result.cond_lnIM_std_df.loc[site, pSA_keys].values.astype(float)
        ),
        c="b",
        linewidth=1.0,
        linestyle="--",
    )
    plt.plot(
        periods,
        np.exp(
            cMVN_result.cond_lnIM_mean_df.loc[site, pSA_keys].values.astype(float)
            - cMVN_result.cond_lnIM_std_df.loc[site, pSA_keys].values.astype(float)
        ),
        c="b",
        linewidth=1.0,
        linestyle="--",
    )

    plt.semilogx()
    plt.xlim(periods.min(), periods.max())

    plt.title(
        f"{site}, {r'$R_{rup}$'} = {obs_data.loc['r_rup']:.0f} (km), "
        f"{'$V_{S30}$'} = {obs_data.loc['Vs30']:.0f} (m/s)"
    )
    plt.xlabel(f"Period")
    plt.ylabel(f"Pseudo-spectral acceleration, Sa (g)")
    plt.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    plt.legend()
    fig.tight_layout()

    plt.savefig(output_dir / f"{site}_response_spectra.png")
    plt.close()


def plot_response_spectrum_residual(
    periods: np.ndarray,
    pSA_keys: Sequence[str],
    sites: Sequence[str],
    ratio_df: pd.DataFrame,
    output_ffp: Path,
    title: str = None,
    ylabel: str = None
):
    fig = plt.figure(figsize=(16, 10))

    for ix, cur_site in enumerate(sites):
        plt.plot(
            periods,
            ratio_df.loc[cur_site, pSA_keys].values.astype(float),
            linewidth=0.75,
            alpha=0.4,
            c="gray",
            label="Individual Sites" if ix == 0 else None
            # c=cmap(norm(obs_df.loc[cur_site, "Vs30"])),
        )

    plt.plot(
        periods,
        res_mean := ratio_df.loc[sites, pSA_keys].mean(axis=0),
        linewidth=1.2,
        c="r",
        label="Bias",
    )
    plt.plot(
        periods,
        res_mean + (res_std := ratio_df.loc[sites, pSA_keys].std(axis=0)),
        linewidth=1.2,
        c="r",
        linestyle="--",
    )
    plt.plot(periods, res_mean - res_std, linewidth=1.0, c="r", linestyle="--")

    plt.semilogx()
    plt.xlim(periods.min(), periods.max())
    plt.ylim(-2.0, 2.0)

    plt.title(title if title is not None else "")
    plt.xlabel(f"Period (s)")
    plt.ylabel(ylabel if ylabel is not None else "")
    plt.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_ffp)
    plt.close()
