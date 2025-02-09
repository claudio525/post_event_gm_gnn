import itertools
import os
import shutil
import random
import copy
from pathlib import Path

import torch
import torch.multiprocessing as mp
import pandas as pd
import numpy as np
import typer
import optuna

import ml_tools as mlt
import sim_ranking as sr

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

print(f"Using device: {device.upper()}")

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command("run-holdout")
def run_holdout(
    run_config_ffp: Path,
    holdout_config_ffp: Path,
    n_epochs: int = None,
    id_suffix: str = "",
):
    sr.ml.run_holdout(
        run_config_ffp,
        holdout_config_ffp,
        n_epochs=n_epochs,
        id_suffix=id_suffix,
        device=device,
    )


@app.command("run-cv")
def run_cv(
    run_config_ffp: Path,
    n_event_folds: int,
    n_site_folds: int,
    n_epochs: int = None,
    id_suffix: str = "",
    n_procs: int = mp.cpu_count(),
):
    mp.set_start_method("spawn")

    sr.ml.run_cv(
        run_config_ffp,
        n_event_folds,
        n_site_folds,
        n_epochs=n_epochs,
        id_suffix=id_suffix,
        n_procs=n_procs,
        device=device,
    )

@app.command("continue-hp-opt")
def continue_hp_opt(
    rel_results_dir: str,
    n_trials: int,
):
    mp.set_start_method("spawn")

    sr.ml.gnn_hp.continue_hp_opt(
        rel_results_dir,
        n_trials,
    )

@app.command("run-hp-opt")
def run_hp_opt(
    base_run_config_ffp: Path,
    hp_opt_config_ffp: Path,
    n_event_folds: int,
    n_site_folds: int,
    rel_results_dir: str,
    n_trials: int,
    n_epochs: int,
    n_procs: int = mp.cpu_count(),
):
    mp.set_start_method("spawn")

    sr.ml.gnn_hp.run_hp_opt(
        base_run_config_ffp,
        hp_opt_config_ffp,
        n_event_folds,
        n_site_folds,
        rel_results_dir,
        n_trials,
        n_epochs,
        n_procs,
        device
    )


if __name__ == "__main__":
    app()
