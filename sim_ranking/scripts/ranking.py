from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
import typer


import gmhazard_calc as gc
import sim_ranking as sr

app = typer.Typer()


@app.command("cmvn")
def conditional_mvn_ranking(
    rupture: str,
    gm_params_ffp: Path,
    obs_data_ffp: Path,
    stations_ll_ffp: Path,
    sim_imdb_ffp: Path,
    output_dir: Path,
    IMs: List[str] = None,
):
    """
    Performs simulation ranking based on the
    conditional MVN

    Note: Currently computes it for all given
    observation sites

    To add:
     - Support for specifying sites of interest
     - Support for setting IM weights
    """
    assert len(list(output_dir.iterdir())) == 0, "Output directory has to be empty"

    # Load the station data
    stations_df = pd.read_csv(
        stations_ll_ffp, sep=" ", index_col=2, header=None, names=["lon", "lat"]
    )

    # IMs to use for ranking
    if len(IMs) == 0:
        IMs = [
            gc.im.IM(gc.im.IMType.pSA, period=cur_period)
            for cur_period in sr.constants.PERIODS
        ]
    # Convert to IM type
    else:
        IMs = [gc.im.IM.from_str(cur_im) for cur_im in IMs]
    IMs_str = [str(cur_im) for cur_im in IMs]

    # Get GMM parameters
    gmm_params_df = pd.read_csv(gm_params_ffp, index_col=0, dtype={"event": str})
    gmm_params_df = gmm_params_df.loc[gmm_params_df.event == rupture]
    gmm_params_df = gmm_params_df.set_index("site").sort_index()

    # Loading Observations
    obs_df = sr.data.load_obs_rupture_data(obs_data_ffp, rupture)

    # Use all available observation stations
    int_stations = obs_df.index.values.astype(str)

    # Load the simulation IM data
    sim_data = sr.data.load_sim_data(sim_imdb_ffp, int_stations)

    # Compute the conditional MVN distributions for each IM
    cMVNs_result = sr.cmvn.compute_cond_MVN_distributions(
        IMs, obs_df, gmm_params_df, stations_df, int_stations
    )

    # Compute the misfit for each site of interest
    site_misfits = []
    for cur_site in int_stations:
        if (cur_sim_df := sim_data.get(cur_site)) is None:
            print(f"No simulation data available for site: {cur_site}, skipping")
            continue

        # Compute misfit for each IM
        cur_misfit = (
            cMVNs_result.cond_lnIM_mean_df.loc[cur_site, IMs_str].values
            - np.log(cur_sim_df[IMs_str].values)
        ) ** 2

        # Aggregate along IM axis
        site_misfits.append(
            pd.Series(
                index=cur_sim_df.index, data=cur_misfit.sum(axis=1), name=cur_site
            )
        )

    # Combine
    site_misfits_df = pd.concat(site_misfits, axis=1)

    # Select the best realisation for each site
    best_sim_id = pd.Series(
        data=site_misfits_df.index[np.argmin(site_misfits_df.values, axis=0)],
        index=site_misfits_df.columns,
    )

    # Save the results
    cMVNs_result.save(output_dir / "cMVN_distributions.pickle")
    site_misfits_df.to_csv(output_dir / "site_misfits.csv")
    best_sim_id.to_csv(output_dir / "best_sim_ids.csv")


if __name__ == "__main__":
    app()
