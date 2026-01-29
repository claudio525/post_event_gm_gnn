#!/usr/bin/env zsh

# Break if any command fails
set -e

# Inputs
n_procs=12
code_dir="/path/to/post_event_gm_gnn"

scripts_dir="${code_dir}/post_event_gm_gnn/scripts/ml_models"
gnn_only_config_ffp="${code_dir}/post_event_gm_gnn/scripts/ml_models/gnn_configs/gnn_config_v2p9.yaml"
gnn_residual_config_ffp="${code_dir}/post_event_gm_gnn/scripts/ml_models/gnn_configs/gnn_config_v2p10.yaml"
gnn_residual_config_ignore_CCCC_SHLC_ffp="${code_dir}/post_event_gm_gnn/scripts/ml_models/gnn_configs/gnn_config_v2p10_ignore_CCCC_SHLC.yaml"
full_output_dir="${wdata}/to/post_event_gm_gnn/results/gnn/1205_4p3FNZGMDB_v2p10_full"
full_output_dir_ignore_CCCC_SHLC="${wdata}/to/post_event_gm_gnn/results/gnn/1205_4p3FNZGMDB_v2p10_full_ignore_CCCC_SHLC"

# Train the GNN-only CV model
python ${scripts_dir}/run_gnn.py train-cv $gnn_only_config_ffp 6 8 --id-suffix v4p3FNZGMDB_v2p9_6e8s --n-procs $n_procs 

# Train the GNN-residual CV model
python ${scripts_dir}/run_gnn.py train-cv $gnn_residual_config_ffp 6 8 --id-suffix v4p3FNZGMDB_v2p10_6e8s --n-procs $n_procs 

# Train full GNN-residual model (ignoring the 22 February 2011 Magnitude 6.2 Christchurch event)
python ${scripts_dir}/run_gnn.py train-full $full_output_dir $gnn_residual_config_ffp 500

# Train full GNN-residual model (ignoring the 22 February 2011 Magnitude 6.2 Christchurch event)
python ${scripts_dir}/run_gnn.py train-full $full_output_dir_ignore_CCCC_SHLC $gnn_residual_config_ignore_CCCC_SHLC_ffp 500