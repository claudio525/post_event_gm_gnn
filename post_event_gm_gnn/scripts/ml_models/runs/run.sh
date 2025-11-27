#!/usr/bin/env zsh
# Run a series of GNN training jobs with different configurations

python ../run_gnn.py train-cv ./hyper_var/gnn_config_residual_batchLarge.yaml 6 8 --id-suffix "batchLarge"
python ../run_gnn.py train-cv ./hyper_var/gnn_config_residual_batchSmall.yaml 6 8 --id-suffix "batchSmall"
python ../run_gnn.py train-cv ./hyper_var/gnn_config_residual_fcUnitsLarge.yaml 6 8 --id-suffix "fcUnitsLarge"
python ../run_gnn.py train-cv ./hyper_var/gnn_config_residual_fcUnitsSmall.yaml 6 8 --id-suffix "fcUnitsSmall"
python ../run_gnn.py train-cv ./hyper_var/gnn_config_residual_nConvLayersLarge.yaml 6 8 --id-suffix "nConvLayersLarge"
python ../run_gnn.py train-cv ./hyper_var/gnn_config_residual_nConvLayersSmall.yaml 6 8 --id-suffix "nConvLayersSmall"


