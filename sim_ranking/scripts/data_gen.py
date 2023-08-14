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


@app.command("compute-sim-gm-params")
def get_sim_gm_params(output_dir: Path, simulation_imdb_ffp: Path, obs_data_ffp: Path):
    """Computes the GM parameters from the simulation data directly"""
    sim_gm_params = sr.data.compute_sim_gm_parameters(simulation_imdb_ffp, obs_data_ffp)

    for cur_params in sim_gm_params:
        (cur_out_dir := output_dir / cur_params.event).mkdir(exist_ok=True)
        cur_params.write(cur_out_dir)


@app.command("compute-sim-site-correlations")
def compute_sim_site_correlations(
    output_dir: Path, sim_params_dir: Path,
):
    """Computes the site correlations from the simulation data directly"""
    correlation_results = sr.data.compute_sim_site_correlations(
        sim_params_dir
    )

    for cur_result in correlation_results:
        (cur_out_dir := output_dir / cur_result.event).mkdir(exist_ok=True)
        cur_result.write(cur_out_dir)

    # for cur_event, cur_im_dict in site_correlations.items():
    #     (cur_out_dir := output_dir / cur_event).mkdir(exist_ok=True)
    #     for cur_im, cur_site_corr in cur_im_dict.items():
    #         cur_site_corr.to_csv(cur_out_dir / f"{cur_im.replace('.', 'p')}.csv")
    #
    #     (cur_im_res_out_dir := cur_out_dir / "im_residuals").mkdir(exist_ok=True)
    #     for cur_im, cur_im_residuals in im_residuals[cur_event].items():
    #         cur_im_residuals.to_csv(cur_im_res_out_dir / f"{cur_im.replace('.', 'p')}.csv")


if __name__ == "__main__":
    app()
