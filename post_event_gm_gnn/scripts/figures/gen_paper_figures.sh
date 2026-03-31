#!/usr/bin/env zsh

# Break if any command fails
set -e

# Shared Inputs
scripts_dir="/Users/claudy/dev/work/code/post_event_gm_gnn/post_event_gm_gnn/scripts/figures"
nzgmdb_ffp="${wdata}/gm_datasets/nz_gmdb/v4.3_final/custom/mod_ground_motion_im_table_rotd50_flat.csv"

# 2p12
out_dir="/Users/claudy/dev/tmp_share/post_event_gnn/paper_figures_v2p12"
gnn_only="${wdata}/post_event_gm_gnn/results/gnn/0319_1443_cv_final_v4p3FNZGMDB_v2p12_base"
gnn_only_full="${wdata}/post_event_gm_gnn/results/gnn/0323_final_4p3FNZGMDB_v2p12_base_full"
gnn_residual="${wdata}/post_event_gm_gnn/results/gnn/0319_1726_cv_final_v4p3FNZGMDB_v2p12_res"
gnn_residual_full="${wdata}/post_event_gm_gnn/results/gnn/0320_final_4p3FNZGMDB_v2p12_res_full"
gnn_residual_full_no_CCCC_SHLC="${wdata}/post_event_gm_gnn/results/gnn/0320_final_4p3FNZGMDB_v2p12_res_ignore_CCCC_SHLC"

# 2p11
# out_dir="/Users/claudy/dev/tmp_share/post_event_gnn/paper_figures_v2p11"
# gnn_only="${wdata}/post_event_gm_gnn/results/gnn/0318_1632_cv_final_v4p3FNZGMDB_v2p11_base"
# gnn_only_full="${wdata}/post_event_gm_gnn/results/gnn/0324_final_4p3FNZGMDB_v2p11_base_full"
# gnn_residual="${wdata}/post_event_gm_gnn/results/gnn/0318_1907_cv_final_v4p3FNZGMDB_v2p11_res"
# gnn_residual_full="${wdata}/post_event_gm_gnn/results/gnn/0318_final_4p3FNZGMDB_v2p11_res_full"
# gnn_residual_full_no_CCCC_SHLC="${wdata}/post_event_gm_gnn/results/gnn/0318_final_4p3FNZGMDB_v2p11_res_ignore_CCCC_SHLC"


### ------------------- Aggregate Figures ------------------------------

# Inputs
nzgmdb_emp_gm_params="${wdata}/post_event_gm_gnn/emp_gm_params/nzgmdb_v4p3_final/emp_gm_params.parquet"

# Define environment variables for figure configuration
default_fig_size="8, 4"
export fig_size=$default_fig_size
export fig_format="png"
export fig_dpi="900"
export fig_font_size="8"
export fig_linewidth="2.5"
export fig_group_linewidth="2.0"

general_bias_limit=0.75
general_std_limit=1.0

bias_limit=0.4
std_limit=0.75

corr_bias_limit=1.0
corr_std_limit=0.75

# ### General Bias & Residual Standard Deviation
# echo "Generating general bias and residual standard deviation figures..."
# python ${scripts_dir}/gen_paper_figures.py bias-res-std $gnn_only $gnn_residual $gnn_residual/cim_results $nzgmdb_emp_gm_params $out_dir --bias-limit $general_bias_limit --std-limit $general_std_limit

# ### Magnitude -- GNN Only
# echo "Generating magnitude bias and residual standard deviation figures for GNN Only..."
# python ${scripts_dir}/gen_paper_figures.py mag-bias-res-std $gnn_only $gnn_only/cim_results $out_dir --plot-labels "a)," --legend-ax 3 --output-name gnn_only_mag_bias_residual_std --bias-limit $bias_limit --std-limit $std_limit
# ### Magnitude -- GNN Residual
# echo "Generating magnitude bias and residual standard deviation figures for GNN Residual..."
# python ${scripts_dir}/gen_paper_figures.py mag-bias-res-std $gnn_residual $gnn_residual/cim_results $out_dir --plot-labels "b)," --legend-ax 3 --output-name gnn_residual_mag_bias_residual_std --bias-limit $bias_limit --std-limit $std_limit

# ### R_Rup -- GNN Only
# echo "Generating R_Rup bias and residual standard deviation figures for GNN Only..."
# python ${scripts_dir}/gen_paper_figures.py rrup-bias-res-std $gnn_only $gnn_only/cim_results $out_dir --plot-labels "a)," --legend-ax 3 --output-name gnn_only_rrup_bias_residual_std --bias-limit $bias_limit --std-limit $std_limit
# ### R_Rup -- GNN Residual
# echo "Generating R_Rup bias and residual standard deviation figures for GNN Residual..."
# python ${scripts_dir}/gen_paper_figures.py rrup-bias-res-std $gnn_residual $gnn_residual/cim_results $out_dir --plot-labels "b)," --legend-ax 3 --output-name gnn_residual_rrup_bias_residual_std --bias-limit $bias_limit --std-limit $std_limit

# ### DoC -- GNN Only
# echo "Generating DoC bias and residual standard deviation figures for GNN Only..."
# python ${scripts_dir}/gen_paper_figures.py doc-bias-res-std $gnn_only $gnn_only/cim_results $out_dir --plot-labels "a)," --legend-ax 3 --output-name gnn_only_doc_bias_residual_std --bias-limit $bias_limit --std-limit $std_limit
# ### DoC -- GNN Residual
# echo "Generating DoC bias and residual standard deviation figures for GNN Residual..."
# python ${scripts_dir}/gen_paper_figures.py doc-bias-res-std $gnn_residual $gnn_residual/cim_results $out_dir --plot-labels "b)," --legend-ax 3 --output-name gnn_residual_doc_bias_residual_std --bias-limit $bias_limit --std-limit $std_limit

# ## Spatial Correlation
# echo "Generating spatial correlation trends figures..."
# python ${scripts_dir}/gen_paper_figures.py spatial-corr-trends $gnn_only $gnn_residual $nzgmdb_emp_gm_params $out_dir --plot-labels "a),b)" --bias-limit $corr_bias_limit --std-limit $corr_std_limit

# ### Attention Weights
# echo "Generating attention weight figures..."
# # Site-to-site distance
# python ${scripts_dir}/gen_paper_figures.py attention-coefficients dist $gnn_only_full $gnn_residual_full $out_dir 25 --angular-dist 0 --angular-dist 90 --angular-dist 0 --angular-dist 90 --ln-vs30-diff 0 --ln-vs30-diff 0 --ln-vs30-diff 1.0 --ln-vs30-diff 1.0
# # Angular distance
# python ${scripts_dir}/gen_paper_figures.py attention-coefficients angular_dist $gnn_only_full $gnn_residual_full $out_dir 25 --site-dist 1 --site-dist 5 --site-dist 1 --site-dist 5 --ln-vs30-diff 0 --ln-vs30-diff 0 --ln-vs30-diff 1.0 --ln-vs30-diff 1.0
# # ln(Vs30) difference
# python ${scripts_dir}/gen_paper_figures.py attention-coefficients ln_vs30_diff $gnn_only_full $gnn_residual_full $out_dir 25 --site-dist 1 --site-dist 5 --site-dist 1 --site-dist 5 --angular-dist 0 --angular-dist 0 --angular-dist 90 --angular-dist 90

# ### Feature Importance
# python ${scripts_dir}/gen_paper_figures.py att-feature-importance-plot $gnn_only_full $out_dir --prefix gnn_only
# python ${scripts_dir}/gen_paper_figures.py att-feature-importance-plot $gnn_residual_full $out_dir --prefix gnn_residual


### ------------------- Spatial Figures ------------------------------

# Inputs
gnn_residual_full_predictions_file="${gnn_residual_full}/3468575/predictions_noAllowSelf.parquet"
cim_results_file="${wdata}/post_event_gm_gnn/results/cIM/0728_3468575_canterbury_extended_500m_nzgmdbV4p3Final/cim_results_noAllowSelf.parquet"
grid_emp_gm_params="${wdata}/post_event_gm_gnn/emp_gm_params/canterbury_extended_500m_nzgmdbV4p3Final/emp_gm_params.parquet"


# # Define environment variables for figure configuration
# export gmt_fig_font_annot_primary="16p,Helvetica,black"
# export gmt_fig_font_label="20p,Helvetica,black"
# export gmt_show_cb_label="false"

ims="pSA_0.1 pSA_1.0 pSA_5.0"

# GM Maps
# echo "Generating spatial GM maps..."
# python ${scripts_dir}/gen_spatial_plots.py "gen-event-prediction-plots" "--region-key" "chch" "${gnn_residual_full}" "${gnn_residual_full_predictions_file}" "${out_dir}" "${grid_emp_gm_params}" "${out_dir}" "${cim_results_file}" "${out_dir}" "${nzgmdb_ffp}" "3468575" ${=ims}

# Plot event CIM-GNN residuals
echo "Generating spatial CIM-GNN residuals..."
python ${scripts_dir}/gen_spatial_plots.py "plot-event-cim-gnn-residuals" "--region-key" "chch" "${gnn_residual_full}" "${gnn_residual_full_predictions_file}" "${cim_results_file}" "${out_dir}" ${=ims}

# Combine spatial figures
echo "Combining spatial figures..."
python ${scripts_dir}/gen_spatial_plots.py combine-spatial-figures ${out_dir}/cim_3468575_pSA_0p1.png ${out_dir}/gnn_3468575_pSA_0p1.png ${out_dir}/cim_gnn_res_3468575_pSA_0p1.png ${out_dir}/combined_pSA_0p1.png --dpi 900
python ${scripts_dir}/gen_spatial_plots.py combine-spatial-figures ${out_dir}/cim_3468575_pSA_1p0.png ${out_dir}/gnn_3468575_pSA_1p0.png ${out_dir}/cim_gnn_res_3468575_pSA_1p0.png ${out_dir}/combined_pSA_1p0.png --dpi 900
python ${scripts_dir}/gen_spatial_plots.py combine-spatial-figures ${out_dir}/cim_3468575_pSA_5p0.png ${out_dir}/gnn_3468575_pSA_5p0.png ${out_dir}/cim_gnn_res_3468575_pSA_5p0.png ${out_dir}/combined_pSA_5p0.png --dpi 900


### ------------------- Other ------------------------------

# # Magnitude-Rrup Scatter Plot
# echo "Generating magnitude vs R_Rup scatter plot..."
# python ${scripts_dir}/gen_paper_figures.py mag-rrup-scatter ${nzgmdb_ffp} ${out_dir}

# # Event-Site Map 
# echo "Generating event-site map..."
# python ${scripts_dir}/gen_spatial_plots.py event-site-map --emp-gm-params-ffp "${grid_emp_gm_params}" --region-key "chch_extended_we" --site-int-lon "172.67" --site-int-lat "-43.515" --event-lon "172.67131" --event-lat "-43.57222" "3468575" "${nzgmdb_ffp}" "${out_dir}/event_site_map.png"

# 2011 Christchurch event CCCC & SHLC plots
# export fig_size="4, 4"
# gnn_residual_full_no_CCCC_SHLC_predictions_file="${gnn_residual_full_no_CCCC_SHLC}/3468575/predictions_noAllowSelf.parquet"
# python ${scripts_dir}/gen_paper_figures.py ind-scenario-pSA 3468575 ${gnn_residual_full_no_CCCC_SHLC_predictions_file} ${cim_results_file} ${nzgmdb_emp_gm_params} ${nzgmdb_ffp} ${out_dir}
# export fig_size=$default_fig_size


