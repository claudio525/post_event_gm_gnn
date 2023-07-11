from pathlib import Path

import pandas as pd
import numpy as np
import typer



from IM_calculation.source_site_dist import src_site_dist
from qcore import srf

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
    output_ffp: Path, site_dir: Path, srf_dir: Path, nz_gmdb_source_ffp: Path, rjb_max: float = RJB_MAX
):
    """Computes the GM parameters using empirical GMMs"""
    sr.data.run_emp_gmms(output_ffp, site_dir, srf_dir, nz_gmdb_source_ffp, rjb_max)

@app.command("compute-sim-gm-params")
def get_sim_gmm_params(simulation_imdb_ffp: Path, ):
    """Computes the GM parameters from the simulation data directly"""
    sim_data =  sr.data.compute_sim_gm_parameters(simulation_imdb_ffp)


if __name__ == "__main__":
    app()
