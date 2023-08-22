from pathlib import Path

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


if __name__ == "__main__":
    app()
