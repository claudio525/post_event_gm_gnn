#!/usr/bin/env zsh

# Define environment variables for figure configuration
export fig_size="8, 4"
export fig_format="png"
export fig_dpi="900"
export fig_font_size="8"
export fig_linewidth="2.5"
export fig_group_linewidth="2.0"

gnn_only="/Users/claudy/dev/work/data/sim_ranking/results/gnn/final/0725_0929_cv_v4p3FNZGMDB_v2p9_6e8s"
gnn_residual="/Users/claudy/dev/work/data/sim_ranking/results/gnn/final/0725_1117_cv_v4p3FNZGMDB_v2p10_6e8s"
emp_gm_params="/Users/claudy/dev/work/data/sim_ranking/emp_gm_params/nzgmdb_v4p3_final/emp_gm_params.parquet"
# out_dir="/Users/claudy/dev/work/docs/PhD/cond_gm_est_paper/resources/case_study/results"
out_dir="/Users/claudy/dev/work/tmp/tmp_paper_figures"


general_bias_limit=0.75
general_std_limit=1.0

bias_limit=0.4
std_limit=0.75

corr_bias_limit=1.0
corr_std_limit=0.75

### General Bias & Residual Standard Deviation
echo "Generating general bias and residual standard deviation figures..."
python ../gen_paper_figures.py bias-res-std $gnn_only $gnn_residual $gnn_residual/cim_results $emp_gm_params $out_dir --bias-limit $general_bias_limit --std-limit $general_std_limit

### Magnitude -- GNN Only
echo "Generating magnitude bias and residual standard deviation figures for GNN Only..."
python ../gen_paper_figures.py mag-bias-res-std $gnn_only $gnn_only/cim_results $out_dir --plot-labels "a)," --legend-ax 3 --output-name gnn_only_mag_bias_residual_std --bias-limit $bias_limit --std-limit $std_limit
### Magnitude -- GNN Residual
echo "Generating magnitude bias and residual standard deviation figures for GNN Residual..."
python ../gen_paper_figures.py mag-bias-res-std $gnn_residual $gnn_residual/cim_results $out_dir --plot-labels "b)," --legend-ax 3 --output-name gnn_residual_mag_bias_residual_std --bias-limit $bias_limit --std-limit $std_limit

### R_Rup -- GNN Only
echo "Generating R_Rup bias and residual standard deviation figures for GNN Only..."
python ../gen_paper_figures.py rrup-bias-res-std $gnn_only $gnn_only/cim_results $out_dir --plot-labels "a)," --legend-ax 3 --output-name gnn_only_rrup_bias_residual_std --bias-limit $bias_limit --std-limit $std_limit
### R_Rup -- GNN Residual
echo "Generating R_Rup bias and residual standard deviation figures for GNN Residual..."
python ../gen_paper_figures.py rrup-bias-res-std $gnn_residual $gnn_residual/cim_results $out_dir --plot-labels "b)," --legend-ax 3 --output-name gnn_residual_rrup_bias_residual_std --bias-limit $bias_limit --std-limit $std_limit

### DoC -- GNN Only
echo "Generating DoC bias and residual standard deviation figures for GNN Only..."
python ../gen_paper_figures.py doc-bias-res-std $gnn_only $gnn_only/cim_results $out_dir --plot-labels "a)," --legend-ax 3 --output-name gnn_only_doc_bias_residual_std --bias-limit $bias_limit --std-limit $std_limit
### DoC -- GNN Residual
echo "Generating DoC bias and residual standard deviation figures for GNN Residual..."
python ../gen_paper_figures.py doc-bias-res-std $gnn_residual $gnn_residual/cim_results $out_dir --plot-labels "b)," --legend-ax 3 --output-name gnn_residual_doc_bias_residual_std --bias-limit $bias_limit --std-limit $std_limit

### Spatial Correlation
echo "Generating spatial correlation trends figures..."
python ../gen_paper_figures.py spatial-corr-trends $gnn_only $gnn_residual $emp_gm_params $out_dir --plot-labels "a),b)" --bias-limit $corr_bias_limit --std-limit $corr_std_limit





