from pathlib import Path

import pandas as pd
import typer

import ml_tools as mlt
import post_event_gm_gnn as pg

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




@app.command("get-nzgmdb-emp-gm-params")
def get_nzgmdb_emp_gmm_params(
    output_ffp: Path,
    nzgmdb_flatfile_ffp: Path,
    events_ffp: Path = None,
):
    """Computes the GM parameters using empirical GMMs"""
    events = mlt.utils.load_txt(events_ffp) if events_ffp is not None else None

    obs_data = pg.data.load_obs_nzgmdb(nzgmdb_flatfile_ffp)

    pg.data.compute_nzgmdb_emp_gm_params(
        output_ffp,
        obs_data,
        events=events,
    )

@app.command("get-event-non-uniform-gm-params")
def get_event_non_uniform_sites_gm_params(
    event_id: str,
    non_uniform_sites_dir: Path,
    nzgmdb_ffp: Path,
    srf_ffp: Path,
    output_ffp: Path,
    max_rjb: float = None,
):
    """Computes the GM parameters for a single event for non-uniform grid sites"""
    site_df = pg.data.load_non_uniform_grid(non_uniform_sites_dir)

    pg.data.compute_event_sites_emp_gm_params(
        event_id, site_df, nzgmdb_ffp, srf_ffp, output_ffp, max_rjb=max_rjb
    )

@app.command("get-event-uniform-gm-params")
def get_event_uniform_sites_gm_params(
    event_id: str,
    grid_ffp: Path,
    nzgmdb_ffp: Path,
    srf_ffp: Path,
    output_ffp: Path,
    max_rjb: float = None,
):
    """Computes the GM parameters for a single event for uniform grid sites"""
    site_df = pd.read_parquet(grid_ffp)

    pg.data.compute_event_sites_emp_gm_params(
        event_id, site_df, nzgmdb_ffp, srf_ffp, output_ffp, max_rjb=max_rjb
    )
    

@app.command("gen-uniform-site-grid")
def gen_uniform_site_grid(
    region_key: str,
    resolution: float,
    output_ffp: Path,
):
    """
    Generates a uniform grid of sites for the given region. 
    Resolution is in metres.
    """
    region = pg.constants.REGION_MAPPINGS[region_key]

    pg.data.gen_uniform_site_grid(region, resolution, output_ffp)

    
    


if __name__ == "__main__":
    app()


# @app.command("compute-obs-site-correlations")
# def compute_obs_site_correlations(
#     output_dir: Path, db_ffp: Path, site_count_th: int = 20
# ):
#     """Computes the observed correlation between sites for each IM"""
#     db = sr.db.DB(db_ffp)
#     obs_df = db.get_obs_df()
#
#     results = {}
#     n_site_pairs = None
#     for pSA_key in sr.constants.PSA_KEYS:
#         # Get the data into the correct format
#         cur_df = obs_df.loc[:, ["event_id", "site_id", pSA_key]]
#         cur_df = cur_df.pivot(index="event_id", columns="site_id", values=pSA_key)
#
#         # Compute the correlations
#         cur_corrs = cur_df.corr(min_periods=site_count_th)
#
#         # Sanity checking
#         cur_n_site_pairs = (
#             cur_corrs.size - np.count_nonzero(cur_corrs.isna()) - cur_corrs.shape[0]
#         ) // 2
#         n_site_pairs = cur_n_site_pairs if n_site_pairs is None else n_site_pairs
#         assert n_site_pairs == cur_n_site_pairs
#
#         # Store the result
#         results[pSA_key] = cur_corrs
#
#     # Compute the mean correlations
#     corr_values = np.stack([cur_corrs.values for cur_corrs in results.values()], axis=2)
#     results["mean"] = pd.DataFrame(
#         np.mean(corr_values, axis=2),
#         index=cur_corrs.index,
#         columns=cur_corrs.columns,
#     )
#
#     # Save the results
#     pd.to_pickle(results, output_dir / "obs_site_correlations.pickle")
#
#
# @app.command("compute-emp-event-site-correlations")
# def compute_emp_event_site_correlations(
#     output_dir: Path, emp_gm_params_ffp: Path, nzgmdb_site_ffp: Path
# ):
#     """
#     Uses the Loth & Baker model to compute the site correlations
#
#     Note: As correlations only depend on the distance between sites
#         there are no differences between events, however it is required
#         in this format for the pairwise ranking model
#     """
#     gm_params_df = pd.read_csv(emp_gm_params_ffp, index_col=0)
#     site_df = pd.read_csv(nzgmdb_site_ffp, index_col="sta")
#
#     sites = np.sort(np.unique(gm_params_df.site.values.astype(str)))
#     ims = np.asarray(sr.constants.PSA_KEYS)
#
#     # Compute the distance matrix
#     print(f"Computing distance matrix")
#     dist_matrix = sh.im_dist.calculate_distance_matrix(sites, site_df)
#
#     # Compute the site correlations
#     print(f"Computing site correlations")
#     corrs = np.full((sites.size, sites.size, len(ims)), fill_value=np.nan)
#     upper_mask = np.triu(np.ones_like(corrs[:, :, 0], dtype=bool), k=1)
#     for i, cur_im in enumerate(ims):
#         corrs[:, :, i][upper_mask] = corrs[:, :, i].T[
#             upper_mask
#         ] = sha.loth_baker_corr_model.get_correlations(
#             cur_im, cur_im, dist_matrix.values[upper_mask]
#         )
#         np.fill_diagonal(corrs[:, :, i], 1.0)
#
#     # Write the results
#     print(f"Writing results")
#     for cur_event in gm_params_df.event.unique().astype(str):
#         cur_sites = np.sort(
#             gm_params_df.loc[gm_params_df.event == cur_event, "site"].values.astype(str)
#         )
#         site_mask = np.isin(sites, cur_sites)
#
#         assert np.all(sites[site_mask] == cur_sites)
#
#         cur_site_corr = sr.data.SiteCorrelations(
#             corrs[site_mask, :, :][:, site_mask, :], cur_sites, ims, cur_event
#         )
#         cur_site_corr.write(output_dir / f"{cur_event}.pickle")


# @app.command("compute-gm-params-mera")
# def compute_gm_params_mera(
#     output_dir: Path,
#     db_ffp: Path,
#     data_source: str = None,
# ):
#     """
#     Computes the GM parameters for all events using MERA
#     and all available data (i.e. all events & realisations)
#
#     1) Compute mean for each event from the realisations
#     2) Compute the residuals (realisations - mu)
#     3) Run MERA on the residuals
#     """
#     from mera.mera_pymer4 import run_mera
#
#     db = sr.db.DB(db_ffp)
#     sim_df = db.get_sim_df(log=True)
#
#     ims = [cur_col for cur_col in sim_df.columns if cur_col in sr.constants.IMs]
#
#     print(f"Computing residuals")
#     residual_df = []
#     for cur_im in tqdm.tqdm(ims):
#         cur_residual_df = sim_df[cur_im] - sim_df.groupby(
#             ["event_id", "site_id"], observed=True
#         )[cur_im].transform("mean")
#
#         residual_df.append(cur_residual_df)
#
#     residual_df = pd.concat(residual_df, axis=1)
#     residual_df["event_id"] = sim_df["event_id"]
#     residual_df["site_id"] = sim_df["site_id"]
#
#     # Run MERA
#     event_res_df, rem_res_df, bias_std_df = run_mera(
#         residual_df, ims, "event_id", "site_id", compute_site_term=False
#     )
#
#     event_res_df.to_parquet(output_dir / "event_res_df.parquet")
#     rem_res_df.to_parquet(output_dir / "rem_res_df.parquet")
#     bias_std_df.to_parquet(output_dir / "bias_std_df.parquet")
#     residual_df.to_parquet(output_dir / "residual_df.parquet")
