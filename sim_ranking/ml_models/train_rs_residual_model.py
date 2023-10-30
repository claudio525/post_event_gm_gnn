"""
Trains a model that given
- Response spectrum from observed GM at the observation site
- Response spectrum from simulation realisation at the observation site
- Response spectrum from simulation realisation at the site of interest
- Site properties of observation site
- Site properties of site of interest
predicts the residual between the simulation and observation at
the site of interest.
"""
from pathlib import Path

import torch
import numpy as np

import typer

import sim_ranking as sr
from sim_ranking.ml import rs_residual as rs

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

print(f"Using device: {device}")


def main(
    hyperparams_ffp: Path,
    n_rels_used: int = 5,
    comment: str = "",
    max_dist: float = 100.0,
    save_best_val_model: bool = True,
    debug: bool = False,
):
    # print(os.environ.get("LD_LIBRARY_PATH"))
    # torch.autograd.set_detect_anomaly(True)

    # Fixing the random seed
    np.random.seed(42)

    run_config = rs.RunParamsConfig(n_rels_used, max_dist, debug, device)
    hp_config = rs.HyperParamsConfig.from_yaml(hyperparams_ffp)

    # Select one of the events for validation
    # val_events = np.asarray(["3468575", "3528839"])
    val_events = np.asarray(
        [
            "3528839",
            "3497857",
            "2017p161601",
            "2013p543121",
            "2016p355041",
            "2017p512943",
        ]
    )

    (
        train_dataset,
        train_dataloader,
        val_dataset,
        val_dataloader,
        scalar_features,
        weight_scalar_features,
        data_metadata,
    ) = rs.prep(val_events, run_config, hp_config)

    n_periods = len(sr.constants.PERIODS)

    # Create the models
    res_model, weight_model = rs.create_models(
        hp_config,
        run_config,
        n_periods,
        scalar_features,
        scalar_features,
        weight_scalar_features,
    )

    print(f"Running training")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of training batches: {len(train_dataloader)}")
    print(f"Number of validation batches: {len(val_dataloader)}")
    metrics, best_res_model_state, best_weight_model_state, best_epoch = rs.train(
        res_model,
        weight_model,
        train_dataloader,
        val_dataloader,
        device,
        hp_config,
    )

    # Load the best model
    if save_best_val_model:
        res_model.load_state_dict(best_res_model_state)
        weight_model.load_state_dict(best_weight_model_state)
        print(
            f"Best model epoch: {best_epoch + 1}, "
            f"Validation:\n"
            f"\tTotal {metrics['loss_hist_val'][best_epoch]:.4f}\n"
            f"\tMisfit {metrics['misfit_hist_val'][best_epoch]:.4f}\n"
            f"\tWeighted Misfit {metrics['weighted_misfit_hist_val'][best_epoch]:.4f}\n"
            f"\tWeight Penalty {metrics['weight_penalty_hist_val'][best_epoch]:.4f}\n"
        )

    # Run post-processing
    rs.post_processing(
        res_model,
        weight_model,
        train_dataset,
        val_dataset,
        hp_config,
        run_config,
        scalar_features,
        weight_scalar_features,
        metrics,
        data_metadata,
        comment,
    )


if __name__ == "__main__":
    typer.run(main)
