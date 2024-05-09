import os
from pathlib import Path

import torch
import numpy as np
import typer

import sim_ranking as sr
import sim_ranking.ml.sc_prob as sc_prob


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

print(f"Using device: {device.upper()}")


def train_model(
    rel_db_ffp: Path = typer.Argument(
        ...,
        help="Relative path to the database file with "
        "respect to $wdata env variable",
    ),
    hyperparams_ffp: Path = typer.Argument(
        ..., help="Path to the hyperparameters file"
    ),
    rel_corr_dir: Path = typer.Argument(
        ...,
        help="Relative path to the correlation directory "
        "with respect to $wdata env variable",
    ),
    n_epochs: int = typer.Option(10, help="Number of epochs for training"),
    max_dist: float = typer.Option(
        50,
        help="Maximum allowed distance of an observation "
        "site from the site of interest",
    ),
    per_im_prob: bool = typer.Option(
        False, help="If true the realisation probabilities are computed per IM"
    ),
    debug: bool = typer.Option(False, help="Extra debug information"),
    n_rels: int = typer.Option(None, help="Number of realisations to use"),
    id_suffix: str = typer.Option("", help="Suffix to add to the output directory"),
    data_source: str = typer.Option(
        None, help="Data source to use when selecting data from the specified database"
    ),
    im_set: str = typer.Option("all", help="The IM set to use. One of ['all', 'pSA']"),
    quiet: bool = typer.Option(False),
    sample_weighting: sc_prob.SampleWeighting = typer.Option(
        sc_prob.SampleWeighting.LOTH_BAKER,
        help="Sample weighting method to use when "
        "aggregating sample predictions for a scenario",
    ),
    apply_sc_weighting: bool = typer.Option(
        False, help="Whether to apply scenario weighting or not"
    ),
    min_sc_weight: float = typer.Option(0.5, help="Minimum scenario weight"),
    max_sc_weight: float = typer.Option(2.0, help="Maximum scenario weight"),
    l2_prob_penalty: float = typer.Option(
        default=0.0, help="L2 penalty to discourage sparse realisations probabilities"
    ),
    seed: int = typer.Option(None),
    out_dir: Path = typer.Option(..., help="Output directory"),
):
    run_config = sc_prob.RunParamsConfig(
        max_dist,
        n_rels,
        sr.constants.IM_SETS[im_set],
        sr.constants.IM_WEIGTHS_SETS[im_set],
        per_im_prob,
        apply_sc_weighting,
        min_sc_weight,
        max_sc_weight,
        sample_weighting,
        l2_prob_penalty,
        debug,
        device,
        results_dir=out_dir,
    )
    hp_config = sc_prob.HyperParamsConfig.from_yaml(hyperparams_ffp, n_epochs)

    corr_dir = (
        Path(os.path.expandvars("$wdata")) / rel_corr_dir
        if rel_corr_dir is not None
        else None
    )

    ### Data loading
    db_ffp = Path(os.path.expandvars("$wdata")) / rel_db_ffp
    db = sr.db.DB(db_ffp)

    events = db.get_avail_events(data_source=data_source)
    print(f"Number of events: {len(events)}")

    # Get all relevant sites across all events
    all_sites = db.get_avail_sites()

    ### Data setup
    # Get the sites per event
    event_sites = db.get_event_sites()

    # Split into training and validation
    if seed is not None:
        print(f"Using numpy random seed: {seed}")
        np.random.seed(seed)
    val_int_sites = np.random.choice(all_sites, 100, replace=False)
    train_sites = np.setdiff1d(all_sites, val_int_sites)

    val_events = np.random.choice(events, 75, replace=False)
    train_events = np.setdiff1d(events, val_events)

    train_dataset, val_dataset, scalar_features, data_metadata = sc_prob.data_prep(
        event_sites,
        train_events,
        val_events,
        train_sites,
        val_int_sites,
        events,
        run_config,
        hp_config,
        db,
        corr_dir=corr_dir,
    )

    # prob_model = sc_prob.create_IMmodel(hp_config, scalar_features, run_config)
    prob_model = sc_prob.create_indRelModel(hp_config, scalar_features, run_config)
    prob_model.to(device)

    weight_model = sr.ml.models.WeightModel(
        run_config.n_ims, [32], hp_config.weight_model_features.size
    )
    weight_model.to(device)

    print(f"Run training")
    metrics, best_model_state, best_model_epoch = sc_prob.train(
        prob_model,
        weight_model,
        train_dataset,
        val_dataset,
        hp_config,
        run_config,
        data_metadata,
    )

    prob_model.load_state_dict(best_model_state)

    print(
        f"Best model epoch: {best_model_epoch + 1}, "
        f"Validation:\n"
        f"\tLoss: {metrics['loss_hist_val'][best_model_epoch]:.4f}\n"
    )

    print(f"Run post-processing")
    data_metadata["db"] = str(rel_db_ffp)
    sc_prob.post_processing(
        prob_model,
        weight_model,
        train_dataset,
        val_dataset,
        hp_config,
        run_config,
        metrics,
        best_model_epoch,
        scalar_features,
        data_metadata,
        val_int_sites,
        train_sites,
        id_suffix=id_suffix,
    )


if __name__ == "__main__":
    typer.run(train_model)
