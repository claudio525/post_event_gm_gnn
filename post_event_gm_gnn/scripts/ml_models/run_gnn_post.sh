#!/usr/bin/env zsh

# Break if any command fails
set -e

# Shared Inputs
scripts_dir="/path/to/post_event_gm_gnn/post_event_gm_gnn/scripts"
n_procs=16


#### CV Models
cv_gnn_only_dir=${wdata}/post_event_gm_gnn/results/gnn/0416_1137_cv_final_v4p3FNZGMDB_v2p13_base 
cv_gnn_residual_dir=${wdata}/post_event_gm_gnn/results/gnn/0416_1606_cv_final_v4p3FNZGMDB_v2p14_res

### Full Models
full_gnn_only_model_dir=${wdata}/post_event_gm_gnn/results/gnn/0416_final_4p3FNZGMDB_v2p13_base_full
full_gnn_residual_model_dir=${wdata}/post_event_gm_gnn/results/gnn/0416_final_4p3FNZGMDB_v2p14_res_full
full_gnn_residual_model_dir_ignore_CCCC_SHLC=${wdata}/post_event_gm_gnn/results/gnn/20260416_final_4p3FNZGMDB_v2p14_res_ignore_CCCC_SHLC


### ------------------- Compute cIM results for GNN CV folds ------------------------------

nzgmdb_emp_gm_params="${wdata}/post_event_gm_gnn/emp_gm_params/nzgmdb_v4p3_final/emp_gm_params.parquet"

echo "Computing cIM results for GNN CV folds..."
python ${scripts_dir}/run_cim.py run-cIM-for-CV-GNN $cv_gnn_only_dir $nzgmdb_emp_gm_params --n-procs $n_procs
python ${scripts_dir}/run_cim.py run-cIM-for-CV-GNN $cv_gnn_residual_dir $nzgmdb_emp_gm_params --n-procs $n_procs

### ------------------- Full GNN Canterbury Grid Predictions ------------------------------

# Inputs
srf_ffp=${wdata}/post_event_gm_gnn/other/3468575.srf
grid_site_ffp=${wdata}/post_event_gm_gnn/uniform_grid/canterbury_extended_500m/grid_nzgmdbV4p3.parquet
grid_emp_gm_params_ffp=${wdata}/post_event_gm_gnn/emp_gm_params/canterbury_extended_500m_nzgmdbV4p3Final/emp_gm_params.parquet

# Predict for event 3468575 using the GNN-residual model trained using all sites
mkdir -p ${full_gnn_residual_model_dir}/3468575
python ${scripts_dir}/ml_models/run_gnn.py predict-event-3468575 ${full_gnn_residual_model_dir} ${srf_ffp} ${full_gnn_residual_model_dir}/3468575/predictions_noAllowSelf.parquet --grid-site-ffp ${grid_site_ffp} --no-allow-self --batch-size 5000 --emp-gm-params-ffp ${grid_emp_gm_params_ffp}

# Predict for event 3468575 using the GNN-residual model trained ignoring CCCC and SHLC
mkdir -p ${full_gnn_residual_model_dir_ignore_CCCC_SHLC}/3468575
python ${scripts_dir}/ml_models/run_gnn.py predict-event-3468575 ${full_gnn_residual_model_dir_ignore_CCCC_SHLC} ${srf_ffp} ${full_gnn_residual_model_dir_ignore_CCCC_SHLC}/3468575/predictions_noAllowSelf.parquet --grid-site-ffp ${grid_site_ffp} --no-allow-self --batch-size 5000 --emp-gm-params-ffp ${grid_emp_gm_params_ffp}


### ------------------- Compute attention SHAP values ------------------------------
echo "Computing attention model SHAP values"

python ${scripts_dir}/ml_models/run_gnn.py get-att-SHAP-values ${full_gnn_only_model_dir} & 
python ${scripts_dir}/ml_models/run_gnn.py get-att-SHAP-values ${full_gnn_residual_model_dir} &
wait


