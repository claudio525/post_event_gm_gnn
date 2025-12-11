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
