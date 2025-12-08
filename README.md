# Post-Event Ground Motion Estimation Using Graph Neural Networks

This repository contains the code for reproducing the results from the paper.  
It is advised to use a linux machine, all instructions below assume a linux OS. 

## Installation

In order to install this package and run the relevant scripts, the following libraries have to be available:
- [GMT](https://docs.generic-mapping-tools.org/dev/index.html) version 6.6 

Then clone the repository using `git clone` and install using `pip install ./post_event_gm_gnn`

## Data Setup
Most of the data loading/saving is done relative to the `wdata` environment variable.
So create a data directory for this project, e.g. `base_data_dir`, and add `export wdata=/path/to/base_data_dir` to your bashrc (or equivalent).

Download the data file (TODO) from here (TODO), and unzip. 
This should give you the following directory structure
```
- gm_datasets
    - nz_gmdb
        - v4.3_final
            - custom
                - mod_ground_motion_im_table_rotd50_flat.csv
- post_event_gm_gnn
    - results
        - gnn
            - final
                - 0725_0929_cv_v4p3FNZGMDB_v2p9_6e8s
                - 0725_1117_cv_v4p3FNZGMDB_v2p10_6e8s
                - 0728_4p3FNZGMDB_v2p10_full
                - 0729_v4p3FNZGMDB_v2p10Ignore_CCCC_SHLC_full
        - cim
            - 0728_3468575_canterbury_extended_500m_nzgmdbV4p3Final
                - cim_results_noAllowSelf.parquet
    - emp_gm_params
        - canterbury_extended_500m_nzgmdbV4p3Final
            - emp_gm_params.parquet
        - nzgmdb_v4p3_final
            - emp_gm_params.parquet
    - other
        - 3468575.srf
```

### Data Overview
#### Dataset
The `gm_datasets` folder contains the relevent NZGMDB data file.  

#### Empirical GMM GM Parameters
The `post_event_gm_gnn/emp_gm_params` constains the GM parameters from empirical GMM models, i.e. the marginal distribution for the multivariate normal conditional IM method for the NZGMDB (`nzgmdb_v4p3_final`) and the uniform canterbury grid (`canterbury_extended_500m_nzgmdbV4p3Final`).

#### GNN Results
The `post_event_gm_gnn/results` folder contains the developed GNN models (`gnn/final`), and the results for the multivariate normal conditional IM method (`cim`).  

There are 4 models results folders:
- `0725_0929_cv_v4p3FNZGMDB_v2p9_6e8s`, which contains the cross-validation results for the GNN-only model
- `0725_1117_cv_v4p3FNZGMDB_v2p10_6e8s`, wich contains the cross validation results for the GNN-residual model
- `0728_4p3FNZGMDB_v2p10_full`, which contains the fully trained model, and model predictions for the 22 February Magnitude 6.2 2011 Christchurch earthquake
- `0729_v4p3FNZGMDB_v2p10Ignore_CCCC_SHLC_full`, which contains the fully trained model with the CCCC and SHLC site ignored, and model predictions for the 22 February Magnitude 6.2 2011 Christchurch earthquake

Shared file types across the model results folders:
- `model.pt`, the trained GNN, can be loaded with `torch.load(model.pt, weights_only=False)`
- `run_config.yaml`, the run configuration used to train the model
- `obs_sites.npy`, observation sites used
- `train_int_sites.npy`, training location of interest sites
- `train_results.parquet`, results for training scenarios, containg the event id (`event_id`), location of interest id (`site_int`), observation sites (`obs_sites`), true/observed GM at the location of interest (`pSA_X.XX`), predicted mean residual (only for the GNN-residual models) (`pSA_X.XX_pred_res`), predicted IM (`pSA_X.XX_pred`), predicted IM standard deviation (`pSA_X.XX_pred_std`), number of observation sites (`n_obs_sites`), distance to the closest observation site (`closest_dist`)
- `metadata.yaml`, some metadata
- `agg_metrics.pickle`, aggregate metrics for that model run

The CV model result directories contain:
- `metrics.pickle`, a 3D LabelledDataArray which contains metric values per epoch and CV-fold
- `run_config.yaml`, the input run configuration for each CV model
- `val_results.parquet`, the combined validation results across all the CV-folds. Same columns as `train_results.parquet`, with the additional column `cv_iter` specifying which CV-fold the scenario is from.
- A `cv_XX` folder for each CV-fold, with each one containing (in addition to the shared file types):
    - `val_results.parquet`, validation results for that CV-fold
    - `cim_results/val_results.parquet`, GM estimates from the multivariate normal conditional IM method for that CV-fold
- `cim_results/val_results.parquet`, combined GM estimates from the multivariate normal conditional IM method across the CV-folds

The full model result directories contain (in addition to the shared file types): 
- `metrics.parquet`, which is a Dataframe containing metrics (columns) for each epoch (rows)
- `3468575/predictions_noAllowSelf.parquet`, which contains GNN predictions for the 22 February Magnitude 6.2 2011 Christchurch earthquake


#### Multivariate Normal Conditional IM Method Results
The `cim/0728_3468575_canterbury_extended_500m_nzgmdbV4p3Final` folder contains the parquet file `cim_results_noAllowSelf.parquet` file, which contains the multivariate normal conditional IM method estimates for the 22 February Magnitude 6.2 2011 Christchurch earthquake on a uniform grid (plus recording stations).
So each row corresponds to a location, and the relevant columns are:
- `pSA_X.XX` is the observed IM at that location, nan for almost all locations, unless it corresponds to a observation site   
- `pSA_X.XX_cond_mean` and `pSA_X.XX_cond_std` are the mean and standard deviation estimated by the multivariate normal conditional 


## Model Training
Training of models, can be done using the scrip `train_ml_models.sh` located under `post_event_gm_gnn/scripts/ml_models`.
You have to set the variables defined under `# Input`, and is advisable to use a GPU otherwise it will take a long time. 

## Post-processing
The `run_gnn_post.sh` script, generates the multivariate normal conditional IM results for the CV GNN runs, and runs the 22 February Magnitude 6.2 2011 Christchurch earthquake predictions for the GNNs trained on all data (except the 2011 February event). Again, you'll need to update some of the inputs, specifically: `cv_gnn_only_dir`, `cv_gnn_residual_dir`, `full_gnn_model_dir` and `full_gnn_model_dir_ignore_CCCC_SHLC`; these are just the model directories created `train_ml_models.sh`. Note that this can be quite slow (multiple hours).

## Generation Of Paper Plots
The plots of the paper can be generated using the `gen_paper_figures.sh` script located under `post_event_gm_gnn/scripts/figures/gen_paper_figures.sh` in the repository.
The only thing that is required to make this work is setting the `scripts_dir` and `out_dir` variable (and ensuring that `wdata` is set).


