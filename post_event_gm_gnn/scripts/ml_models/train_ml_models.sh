#!/usr/bin/env zsh

# Break if any command fails
set -e

# Inputs
n_procs=12
code_dir="/home/claudy/dev/work/code/post_event_gm_gnn"
scripts_dir="${code_dir}/post_event_gm_gnn/scripts/ml_models"

gnn_only_config_ffp="${code_dir}/post_event_gm_gnn/scripts/ml_models/gnn_configs/gnn_config_v2p13_base.yaml"
gnn_residual_config_ffp="${code_dir}/post_event_gm_gnn/scripts/ml_models/gnn_configs/gnn_config_v2p14_res.yaml"
gnn_residual_config_ignore_CCCC_SHLC_ffp="${code_dir}/post_event_gm_gnn/scripts/ml_models/gnn_configs/gnn_config_v2p14_res_ignore_CCCC_SHLC.yaml"

only_full_output_dir="${wdata}/post_event_gm_gnn/results/gnn/20260416_final_4p3FNZGMDB_v2p13_base_full"
residual_full_output_dir="${wdata}/post_event_gm_gnn/results/gnn/20260416_final_4p3FNZGMDB_v2p14_res_full"
residual_full_output_dir_ignore_CCCC_SHLC="${wdata}/post_event_gm_gnn/results/gnn/20260416_final_4p3FNZGMDB_v2p14_res_ignore_CCCC_SHLC"

### GNN-Only
# CV model
python ${scripts_dir}/run_gnn.py train-cv $gnn_only_config_ffp 6 8 --id-suffix final_v4p3FNZGMDB_v2p13_base --n-procs $n_procs 
# Full (ignoring the 22 February 2011 Magnitude 6.2 Christchurch event)
python ${scripts_dir}/run_gnn.py train-full $only_full_output_dir $gnn_only_config_ffp 500

### GNN-Residual
# CV model
python ${scripts_dir}/run_gnn.py train-cv $gnn_residual_config_ffp 6 8 --id-suffix final_v4p3FNZGMDB_v2p14_res --n-procs $n_procs 
# Full (ignoring the 22 February 2011 Magnitude 6.2 Christchurch event)
python ${scripts_dir}/run_gnn.py train-full $residual_full_output_dir $gnn_residual_config_ffp 500
# Full ignoring CCCC and SHLC (ignoring the 22 February 2011 Magnitude 6.2 Christchurch event)
python ${scripts_dir}/run_gnn.py train-full $residual_full_output_dir_ignore_CCCC_SHLC $gnn_residual_config_ignore_CCCC_SHLC_ffp 500