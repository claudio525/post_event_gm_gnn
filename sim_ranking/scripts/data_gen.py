from pathlib import Path

import numpy as np
import pandas as pd
import typer

import sim_ranking as sr

app = typer.Typer()


COLUMN_MAPPING = {
    "Vs30": "vs30",
    "r_rup": "rrup",
    "r_jb": "rjb",
    "Z1.0": "z1pt0",
    "mag": "mag",
    "rake": "rake",
    "dip": "dip",
    "z_tor": "ztor",
    "r_x": "rx",
    "ev_depth": "hypo_depth",
}


RJB_MAX = 200


@app.command("get-emp-gm-params")
def get_emp_gmm_params(
    output_ffp: Path,
    site_dir: Path,
    srf_dir: Path,
    nz_gmdb_source_ffp: Path,
    rjb_max: float = RJB_MAX,
):
    """Computes the GM parameters using empirical GMMs"""
    sr.data.run_emp_gmms(output_ffp, site_dir, srf_dir, nz_gmdb_source_ffp, rjb_max)


@app.command("compute-sim-gm-params-mera")
def get_sim_gm_params_mera(output_dir: Path, simulation_imdb_ffp: Path):
    """
    Computes the GM parameters from the simulation data
    directly using MERA
    """
    sim_gm_params = sr.data.compute_sim_gm_params_mera(simulation_imdb_ffp)

    for cur_params in sim_gm_params:
        (cur_out_dir := output_dir / cur_params.event).mkdir(exist_ok=True)
        cur_params.write(cur_out_dir)


@app.command("compute-sim-gm-params-total")
def get_sim_gm_params_total(output_dir: Path, simulation_imdb_ffp: Path):
    """
    Computes the GM parameters from the simulation data
    directly, assumes between event term is 0, i.e. just uses total residual
    """
    sim_gm_params = sr.data.compute_sim_gm_params_total(simulation_imdb_ffp)

    for cur_params in sim_gm_params:
        (cur_out_dir := output_dir / cur_params.event).mkdir(exist_ok=True)
        cur_params.write(cur_out_dir)


@app.command("compute-sim-site-correlations")
def compute_sim_site_correlations(
    output_dir: Path,
    sim_params_dir: Path,
    smooth: bool = False,
):
    """Computes the site correlations from the simulation data directly"""
    correlation_results = sr.data.compute_sim_site_corrs(sim_params_dir, smooth=smooth)

    for cur_result in correlation_results:
        cur_result.write(output_dir / f"{cur_result.event}.pickle")

@app.command("compute-obs-site-correlations")
def compute_obs_site_correlations(output_dir: Path, db_ffp: Path, site_count_th: int = 20):
    db = sr.db.DB(db_ffp)
    obs_df = db.get_obs_df()

    results = {}
    n_site_pairs = None
    for pSA_key in sr.constants.PSA_KEYS:
        # Get the data into the correct format
        cur_df = obs_df.loc[:, ["event_id", "site_id", pSA_key]]
        cur_df = cur_df.pivot(index="event_id", columns="site_id", values=pSA_key)

        # Compute the correlations
        cur_corrs = cur_df.corr(min_periods=site_count_th)

        # Drop any sites for which we don't have any correlations
        # cur_corrs = cur_corrs.dropna(axis=0, how="all")
        # cur_corrs = cur_corrs.dropna(axis=1, how="all")
        # assert cur_corrs.shape[0] == cur_corrs.shape[1]

        # Sanity checking
        cur_n_site_pairs = (cur_corrs.size - np.count_nonzero(cur_corrs.isna()) - cur_corrs.shape[
                0]) // 2
        n_site_pairs = cur_n_site_pairs if n_site_pairs is None else n_site_pairs
        assert n_site_pairs == cur_n_site_pairs

        # Store the result
        results[pSA_key] = cur_corrs

    # Compute the mean correlations
    corr_values = np.stack([cur_corrs.values for cur_corrs in results.values()], axis=2)
    results["mean"] = pd.DataFrame(
        np.mean(corr_values, axis=2),
        index=cur_corrs.index,
        columns=cur_corrs.columns,
    )

    # Save the results
    pd.to_pickle(results, output_dir / "obs_site_correlations.pickle")


def __run_interp(pSA_values: np.ndarray, periods: np.ndarray):
    return np.interp(sr.constants.PERIODS, periods, pSA_values)

@app.command("process-im-csv-files")
def process_im_csv_files(raw_im_dir: Path, output_dir: Path):
    ffps = list(raw_im_dir.glob("*.csv"))

    for cur_ffp in ffps:
        df = pd.read_csv(cur_ffp, index_col=0)

        # Only interested in rotd50
        df = df.loc[df.component == "rotd50", :]

        # Get the periods
        pSA_keys = [cur_col for cur_col in df.columns if cur_col.startswith("pSA_")]
        periods = np.array([float(cur_col.split("_")[1]) for cur_col in pSA_keys])

        assert np.all(periods == sorted(periods))
        other_cols = [cur_col for cur_col in df.columns if cur_col not in pSA_keys]

        # Run interpolation
        df_interp = pd.DataFrame(
            data=np.apply_along_axis(__run_interp, 1, df.loc[:, pSA_keys].values, periods),
            index=df.index, columns=sr.constants.PSA_KEYS)

        df = pd.concat([df_interp, df.loc[:, other_cols]], axis=1)

        df.to_csv(output_dir / cur_ffp.name)


if __name__ == "__main__":
    app()
