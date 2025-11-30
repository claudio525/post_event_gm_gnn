#!/usr/bin/env zsh

# Define environment variables for figure configuration
export gmt_fig_font_annot_primary="16p,Helvetica,black"
export gmt_fig_font_label="20p,Helvetica,black"
export gmt_show_cb_label="false"

gnn_results_dir="${wdata}/sim_ranking/results/gnn/final/0728_4p3FNZGMDB_v2p10_full"
gnn_predictions_file="${gnn_results_dir}/3468575/predictions_noAllowSelf.parquet"
cim_results_file="${wdata}/sim_ranking/results/cIM/0728_3468575_canterbury_extended_500m_nzgmdbV4p3Final/cim_results_noAllowSelf.parquet"
emp_gm_params="${wdata}/sim_ranking/emp_gm_params/canterbury_extended_500m_nzgmdbV4p3Final/emp_gm_params.parquet"
nzgmdb_ffp="${wdata}/gm_datasets/nz_gmdb/v4.3_final/custom/mod_ground_motion_im_table_rotd50_flat.csv"
out_dir="/Users/claudy/dev/work/tmp/tmp_paper_figures/3468575"

ims="pSA_0.1 pSA_1.0 pSA_5.0"

# GM Maps
python ../gen_spatial_plots.py "gen-event-prediction-plots" "--region-key" "chch" "${gnn_results_dir}" "${gnn_predictions_file}" "${out_dir}" "${emp_gm_params}" "${out_dir}" "${cim_results_file}" "${out_dir}" "${nzgmdb_ffp}" "3468575" ${=ims}

# Plot event CIM-GNN residuals
python ../gen_spatial_plots.py "plot-event-cim-gnn-residuals" "--region-key" "chch" "${gnn_results_dir}" "${gnn_predictions_file}" "${cim_results_file}" "${out_dir}" ${=ims}

python ../gen_spatial_plots.py combine-spatial-figures ${out_dir}/cim_3468575_pSA_0p1.png ${out_dir}/gnn_3468575_pSA_0p1.png ${out_dir}/cim_gnn_res_3468575_pSA_0p1.png ${out_dir}/combined_pSA_0p1.png --dpi 900
python ../gen_spatial_plots.py combine-spatial-figures ${out_dir}/cim_3468575_pSA_1p0.png ${out_dir}/gnn_3468575_pSA_1p0.png ${out_dir}/cim_gnn_res_3468575_pSA_1p0.png ${out_dir}/combined_pSA_1p0.png --dpi 900
python ../gen_spatial_plots.py combine-spatial-figures ${out_dir}/cim_3468575_pSA_5p0.png ${out_dir}/gnn_3468575_pSA_5p0.png ${out_dir}/cim_gnn_res_3468575_pSA_5p0.png ${out_dir}/combined_pSA_5p0.png --dpi 900
