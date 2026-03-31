#!/usr/bin/env zsh

# Break if any command fails
set -e

# Shared Inputs
# scripts_dir="/path/to/post_event_gm_gnn/post_event_gm_gnn/scripts"
scripts_dir="/home/claudy/dev/work/code/post_event_gm_gnn/post_event_gm_gnn/scripts"
n_procs=16


#### CV Models
## 2p12
# cv_gnn_only_dir=${wdata}/post_event_gm_gnn/results/gnn/0319_1443_cv_final_v4p3FNZGMDB_v2p12_base 
# cv_gnn_residual_dir=${wdata}/post_event_gm_gnn/results/gnn/0319_1726_cv_final_v4p3FNZGMDB_v2p12_res

## 2p11
cv_gnn_only_dir=${wdata}/post_event_gm_gnn/results/gnn/0318_1632_cv_final_v4p3FNZGMDB_v2p11_base 
cv_gnn_residual_dir=${wdata}/post_event_gm_gnn/results/gnn/0318_1907_cv_final_v4p3FNZGMDB_v2p11_res

### Full Models
## 2p12
# full_gnn_only_model_dir=${wdata}/post_event_gm_gnn/results/gnn/0323_final_4p3FNZGMDB_v2p12_base_full
# full_gnn_residual_model_dir=${wdata}/post_event_gm_gnn/results/gnn/0320_final_4p3FNZGMDB_v2p12_res_full
# full_gnn_residual_model_dir_ignore_CCCC_SHLC=${wdata}/post_event_gm_gnn/results/gnn/0320_final_4p3FNZGMDB_v2p12_res_ignore_CCCC_SHLC

## 2p11
full_gnn_only_model_dir=${wdata}/post_event_gm_gnn/results/gnn/0324_final_4p3FNZGMDB_v2p11_base_full
full_gnn_residual_model_dir=${wdata}/post_event_gm_gnn/results/gnn/0318_final_4p3FNZGMDB_v2p11_res_full
full_gnn_residual_model_dir_ignore_CCCC_SHLC=${wdata}/post_event_gm_gnn/results/gnn/0318_final_4p3FNZGMDB_v2p11_res_ignore_CCCC_SHLC

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


