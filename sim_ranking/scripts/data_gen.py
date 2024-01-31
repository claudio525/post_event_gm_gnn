from pathlib import Path

import numpy as np
import pandas as pd
import typer

import sim_ranking as sr
import ml_tools as mlt
import spatial_hazard as sh
import sha_calc as sha

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
    nz_gmdb_source_ffp: Path,
    srf_dir: Path = None,
    nzgmdb_flatfile_ffp: Path = None,
    site_dir: Path = None,
    nzgmdb_site_ffp: Path = None,
    rjb_max: float = RJB_MAX,
    events_ffp: Path = None,
):
    """Computes the GM parameters using empirical GMMs"""
    events = mlt.utils.load_txt(events_ffp) if events_ffp is not None else None

    sr.data.run_emp_gmms(
        output_ffp,
        nz_gmdb_source_ffp,
        rjb_max,
        site_dir=site_dir,
        nzgmdb_site_ffp=nzgmdb_site_ffp,
        srf_dir=srf_dir,
        nzgmdb_flatfile_ffp=nzgmdb_flatfile_ffp,
        events=events,
    )


@app.command("gen-emp-realisations")
def gen_emp_realisations(
    emp_gm_params_ffp: Path, nzgmdb_site_ffp: Path, output_dir: Path, n_rels: int = 25
):
    gm_params = pd.read_csv(emp_gm_params_ffp, index_col=0)
    site_df = pd.read_csv(nzgmdb_site_ffp, index_col="sta")

    results = sr.data.gen_emp_realisations(gm_params, site_df, n_rels=n_rels)

    pd.to_pickle(results, output_dir / "emp_realisations.pickle", compression=None)

    # for cur_event_rel, cur_df in results.items():
    #     cur_df["component"] = "rotd50"
    #     cur_df.to_csv(output_dir / f"{cur_event_rel}.csv")
    # cur_df.to_parquet(output_dir / f"{cur_event_rel}.parquet")


@app.command("gen-emp-synthetic-observed")
def gen_emp_synthetic_observed(
    emp_gm_params_ffp: Path,
    nzgmdb_site_ffp: Path,
    nzgmdb_flat_file: Path,
    syn_obs_ffp: Path,
    syn_gm_params_ffp: Path,
):
    syn_obs_df, mod_gm_params_df = sr.data.gen_emp_synthetic_observed(
        emp_gm_params_ffp, nzgmdb_site_ffp, nzgmdb_flat_file
    )

    syn_obs_df.to_csv(syn_obs_ffp)
    mod_gm_params_df.to_csv(syn_gm_params_ffp)


@app.command("compute-sim-gm-params-mera")
def get_sim_gm_params_mera(
    output_dir: Path, db_ffp: Path, data_source: str = None, n_procs: int = 1
):
    """
    Computes the GM parameters from the simulation data
    directly using MERA
    """
    sim_gm_params = sr.data.compute_sim_gm_params_mera(
        db_ffp, data_source=data_source, n_procs=n_procs
    )

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


@app.command("compute-emp-event-site-correlations")
def compute_emp_event_site_correlations(
    output_dir: Path, emp_gm_params_ffp: Path, nzgmdb_site_ffp: Path
):
    """
    Uses the Loth & Baker model to compute the site correlations

    Note: As correlations only depend on the distance between sites
        there are no differences between events, however it is required
        in this format for the pairwise ranking model
    """
    gm_params_df = pd.read_csv(emp_gm_params_ffp, index_col=0)
    site_df = pd.read_csv(nzgmdb_site_ffp, index_col="sta")

    sites = np.sort(np.unique(gm_params_df.site.values.astype(str)))
    ims = np.asarray(sr.constants.PSA_KEYS)

    # Compute the distance matrix
    print(f"Computing distance matrix")
    dist_matrix = sh.im_dist.calculate_distance_matrix(sites, site_df)

    # Compute the site correlations
    print(f"Computing site correlations")
    corrs = np.full((sites.size, sites.size, len(ims)), fill_value=np.nan)
    upper_mask = np.triu(np.ones_like(corrs[:, :, 0], dtype=bool), k=1)
    for i, cur_im in enumerate(ims):
        t = sha.loth_baker_corr_model.get_correlations(
            cur_im, cur_im, dist_matrix.values[upper_mask]
        )
        corrs[:, :, i][upper_mask] = corrs[:, :, i].T[upper_mask] = t
        np.fill_diagonal(corrs[:, :, i], 1.0)

    # Write the results
    print(f"Writing results")
    for cur_event in gm_params_df.event.unique().astype(str):
        cur_sites = np.sort(
            gm_params_df.loc[gm_params_df.event == cur_event, "site"].values.astype(str)
        )
        site_mask = np.isin(sites, cur_sites)

        assert np.all(sites[site_mask] == cur_sites)

        cur_site_corr = sr.data.SiteCorrelations(
            corrs[site_mask, :, :][:, site_mask, :], cur_sites, ims, cur_event
        )
        cur_site_corr.write(output_dir / f"{cur_event}.pickle")


@app.command("compute-sim-event-site-correlations")
def compute_sim_event_site_correlations(
    output_dir: Path,
    sim_params_dir: Path,
    smooth: bool = False,
):
    """
    Computes the site correlations for each simulated event
    I.e. Produces site-correlations per event using the simulation realisations
    """
    correlation_results = sr.data.compute_sim_event_site_corrs(
        sim_params_dir, smooth=smooth
    )

    for cur_result in correlation_results:
        cur_result.write(output_dir / f"{cur_result.event}.pickle")


@app.command("compute-sim-site-correlations")
def compute_sim_site_correlations(output_dir: Path, sim_params_dir: Path):
    """
    Computes the site-correlations across all simulations
    using the within-event residual
    """
    corr_dfs = sr.data.compute_sim_site_corrs(sim_params_dir)

    for cur_key, cur_df in corr_dfs.items():
        cur_df.to_csv(output_dir / f"{cur_key}.csv")


@app.command("compute-obs-site-correlations")
def compute_obs_site_correlations(
    output_dir: Path, db_ffp: Path, site_count_th: int = 20
):
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
        cur_n_site_pairs = (
            cur_corrs.size - np.count_nonzero(cur_corrs.isna()) - cur_corrs.shape[0]
        ) // 2
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
            data=np.apply_along_axis(
                __run_interp, 1, df.loc[:, pSA_keys].values, periods
            ),
            index=df.index,
            columns=sr.constants.PSA_KEYS,
        )

        df = pd.concat([df_interp, df.loc[:, other_cols]], axis=1)

        df.to_csv(output_dir / cur_ffp.name)


if __name__ == "__main__":
    app()
