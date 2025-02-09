from pathlib import Path
import os

import optuna

import ml_tools as mlt

from . import gnn_gm
from . import gnn_train_cv


def run_hp_opt(
    base_run_config_ffp: Path,
    hp_opt_config_ffp: Path,
    n_event_folds: int,
    n_site_folds: int,
    rel_results_dir: str,
    n_trials: int,
    n_epochs: int,
    n_procs: int,
    device: str,
):
    objective = HPObjective(
        mlt.utils.load_yaml(hp_opt_config_ffp),
        mlt.utils.load_yaml(base_run_config_ffp),
        n_event_folds,
        n_site_folds,
        n_epochs,
        n_procs,
        rel_results_dir,
        device
    )

    # Write HPObjective
    results_dir = Path(os.environ["wdata"]) / rel_results_dir
    objective.to_yaml(results_dir / "hp_objective.yaml")

    # Permanent study storage
    study_storage = f"sqlite:///{results_dir / 'hp_opt.db'}"

    study = optuna.create_study(
        direction="minimize",
        storage=study_storage,
        study_name=Path(rel_results_dir).stem,
        load_if_exists=False,
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1)


def continue_hp_opt(
    rel_results_dir: str,
    n_trials: int,
):
    results_dir = Path(os.environ["wdata"]) / rel_results_dir
    try:
        objective = HPObjective.from_yaml(results_dir / "hp_objective.yaml")
        study_storage = f"sqlite:///{results_dir / 'hp_opt.db'}"
    except FileNotFoundError:
        print(
            "Results directory exists but insufficient "
            "files found for continuation of study."
        )
        return

    study = optuna.create_study(
        direction="minimize",
        storage=study_storage,
        study_name=Path(rel_results_dir).stem,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1)


class HPObjective:

    def __init__(
        self,
        hp_opt_config: dict,
        base_run_config: dict,
        n_event_folds: int,
        n_site_folds: int,
        n_epochs: int,
        n_procs: int,
        rel_results_dir: str,
        device: str,
    ):
        self.hp_opt_config = hp_opt_config
        self.base_run_config = base_run_config

        self.n_event_folds = n_event_folds
        self.n_site_folds = n_site_folds
        self.n_epochs = n_epochs

        self.rel_results_dir = rel_results_dir
        self.n_procs = n_procs
        self.device = device

    def to_yaml(self, ffp: Path):
        mlt.utils.write_to_yaml(
            {
                "hp_opt_config": self.hp_opt_config,
                "base_run_config": self.base_run_config,
                "n_event_folds": self.n_event_folds,
                "n_site_folds": self.n_site_folds,
                "n_epochs": self.n_epochs,
                "n_procs": self.n_procs,
                "rel_results_dir": self.rel_results_dir,
            },
            ffp,
        )

    @classmethod
    def from_yaml(cls, ffp: Path):
        config = mlt.utils.load_yaml(ffp)
        return cls(
            config["hp_opt_config"],
            config["base_run_config"],
            config["n_event_folds"],
            config["n_site_folds"],
            config["n_epochs"],
            config["n_procs"],
            config["rel_results_dir"],
        )

    def __call__(self, trial: optuna.Trial):

        batch_size = trial.suggest_categorical(
            "batch_size", self.hp_opt_config["batch_size"]
        )
        n_gcn_layers = trial.suggest_int(
            "n_gcn_layers",
            self.hp_opt_config["n_gcn_layers"]["min"],
            self.hp_opt_config["n_gcn_layers"]["max"],
        )

        n_obs_node_channels = trial.suggest_categorical(
            "n_obs_node_channels",
            self.hp_opt_config["n_obs_node_channels"],
        )

        n_att_heads = trial.suggest_int(
            "n_att_heads",
            self.hp_opt_config["n_att_heads"]["min"],
            self.hp_opt_config["n_att_heads"]["max"],
        )

        n_int_node_channels = trial.suggest_int(
            "n_int_node_channels",
            self.hp_opt_config["n_int_node_channels"]["min"],
            self.hp_opt_config["n_int_node_channels"]["max"],
            step=self.hp_opt_config["n_int_node_channels"]["step"],
        )

        n_edge_channels = trial.suggest_int(
            "n_edge_channels",
            self.hp_opt_config["n_edge_channels"]["min"],
            self.hp_opt_config["n_edge_channels"]["max"],
            step=self.hp_opt_config["n_edge_channels"]["step"],
        )

        gcn_act_fn = trial.suggest_categorical(
            "gcn_act_fn", self.hp_opt_config["gcn_act_fn"]
        )

        att_n_units = trial.suggest_int(
            "att_n_units",
            self.hp_opt_config["att_n_units"]["min"],
            self.hp_opt_config["att_n_units"]["max"],
            step=self.hp_opt_config["att_n_units"]["step"],
        )

        att_act_fn = trial.suggest_categorical(
            "att_act_fn", self.hp_opt_config["att_act_fn"]
        )

        fc_n_units = trial.suggest_categorical(
            "fc_n_units", self.hp_opt_config["fc_n_units"]
        )
        fc_act_fn = trial.suggest_categorical(
            "fc_act_fn", self.hp_opt_config["fc_act_fn"]
        )

        l2_reg = trial.suggest_categorical(
            "l2_reg",
            self.hp_opt_config["l2_reg"],
        )

        batch_norm = trial.suggest_categorical(
            "batch_norm",
            self.hp_opt_config["batch_norm"],
        )

        dropout_rate = trial.suggest_categorical(
            "dropout_rate",
            self.hp_opt_config["dropout_rate"],
        )

        run_config_dict = self.base_run_config | {
            "batch_size": batch_size,
            "n_att_heads": n_gcn_layers * [n_att_heads],
            "n_int_node_channels": n_gcn_layers * [n_int_node_channels],
            "n_obs_node_channels": n_gcn_layers * [n_obs_node_channels],
            "n_edge_channels": n_gcn_layers * [n_edge_channels],
            "gcn_act_fn": gcn_act_fn,
            "att_n_units": n_gcn_layers * [att_n_units],
            "att_act_fn": att_act_fn,
            "fc_n_units": fc_n_units,
            "fc_act_fn": fc_act_fn,
            "batch_norm": batch_norm,
            "l2_reg": l2_reg,
            "dropout_rate": dropout_rate,
            "rel_results_dir": self.rel_results_dir,
            "n_epochs": self.n_epochs,
            "device": self.device,
        }
        run_config = gnn_gm.RunConfig.from_dict(run_config_dict)

        result_dir, agg_metrics = gnn_train_cv.run_cv(
            run_config,
            self.n_event_folds,
            self.n_site_folds,
            n_epochs=None,
            id_suffix=f"trial_{trial._trial_id}",
            n_procs=self.n_procs,
        )

        # Set the trial user attributes
        for cur_key, cur_value in agg_metrics.items():
            trial.set_user_attr(cur_key, cur_value)
        trial.set_user_attr("result_dir", str(result_dir))
        trial.set_user_attr("id", result_dir.stem)

        return agg_metrics["mean_min_val_w_loss"]
