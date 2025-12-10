# Post-Event Ground Motion Estimation Using Graph Neural Networks

This repository contains the code for reproducing the results from the paper.

**Note:** It is advised to use a Linux machine. All instructions below assume a Linux OS.

If you encounter any problems, please raise an issue on the GitHub repository.

## Installation

### Prerequisites

The following libraries must be available:
- [GMT](https://docs.generic-mapping-tools.org/dev/index.html) version 6.6

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```

2. Install the package:
   ```bash
   pip install ./post_event_gm_gnn
   ```

## Data Setup

### Environment Configuration

Most data loading/saving is done relative to the `wdata` environment variable.

1. Create a data directory for this project (e.g., `base_data_dir`)
2. Add the following to your `.bashrc` (or equivalent):
   ```bash
   export wdata=/path/to/base_data_dir
   ```

### Download Data

Download the data file from [TODO] and unzip it.

This will create the following directory structure:

```
base_data_dir/
├── gm_datasets/
│   └── nz_gmdb/
│       └── v4.3_final/
│           └── custom/
│               └── mod_ground_motion_im_table_rotd50_flat.csv
└── post_event_gm_gnn/
    ├── results/
    │   ├── gnn/
    │   │   └── final/
    │   │       ├── 0725_0929_cv_v4p3FNZGMDB_v2p9_6e8s/
    │   │       ├── 0725_1117_cv_v4p3FNZGMDB_v2p10_6e8s/
    │   │       ├── 0728_4p3FNZGMDB_v2p10_full/
    │   │       └── 0729_v4p3FNZGMDB_v2p10Ignore_CCCC_SHLC_full/
    │   └── cim/
    │       └── 0728_3468575_canterbury_extended_500m_nzgmdbV4p3Final/
    │           └── cim_results_noAllowSelf.parquet
    ├── emp_gm_params/
    │   ├── canterbury_extended_500m_nzgmdbV4p3Final/
    │   │   └── emp_gm_params.parquet
    │   └── nzgmdb_v4p3_final/
    │       └── emp_gm_params.parquet
    └── other/
        └── 3468575.srf
```

## Data Overview

### Dataset

The `gm_datasets` folder contains the relevant NZGMDB data file.

### Empirical GMM Ground Motion Parameters

The `post_event_gm_gnn/emp_gm_params` folder contains GM parameters from empirical GMM models, i.e., the marginal distribution for the multivariate normal conditional IM method for:
- NZGMDB (`nzgmdb_v4p3_final`)
- Uniform Canterbury grid (`canterbury_extended_500m_nzgmdbV4p3Final`)

### GNN Results

The `post_event_gm_gnn/results` folder contains:
- Developed GNN models (`gnn/final`)
- Results for the multivariate normal conditional IM method (`cim`)

#### Model Results Directories

There are 4 model result folders:

1. **`0725_0929_cv_v4p3FNZGMDB_v2p9_6e8s`** - Cross-validation results for the GNN-only model
2. **`0725_1117_cv_v4p3FNZGMDB_v2p10_6e8s`** - Cross-validation results for the GNN-residual model
3. **`0728_4p3FNZGMDB_v2p10_full`** - Fully trained model and predictions for the 22 February 2011 Magnitude 6.2 Christchurch earthquake
4. **`0729_v4p3FNZGMDB_v2p10Ignore_CCCC_SHLC_full`** - Fully trained model (CCCC and SHLC sites ignored) and predictions for the 22 February 2011 Magnitude 6.2 Christchurch earthquake

#### Shared File Types

> **Reading Data Files:**
> - `.pickle` files: Use `pd.read_pickle()` from pandas
> - `.parquet` files: Use `pd.read_parquet()` from pandas
> - `.npy` files: Use `np.load()` from numpy
> - `.pt` files: Use `torch.load(filepath, weights_only=False)` from PyTorch

All model result folders contain:

- **`model.pt`** - Trained GNN model
- **`run_config.yaml`** - Run configuration used to train the model
- **`obs_sites.npy`** - Observation sites used
- **`train_int_sites.npy`** - Training location of interest sites
- **`train_results.parquet`** - Training scenario results with columns:
  - `event_id` - Event ID
  - `site_int` - Location of interest ID
  - `obs_sites` - Observation sites
  - `pSA_X.XX` - True/observed GM at the location of interest
  - `pSA_X.XX_pred_res` - Predicted mean residual (GNN-residual models only)
  - `pSA_X.XX_pred` - Predicted IM
  - `pSA_X.XX_pred_std` - Predicted IM standard deviation
  - `n_obs_sites` - Number of observation sites
  - `closest_dist` - Distance to the closest observation site
- **`metadata.yaml`** - Metadata
- **`agg_metrics.pickle`** - Aggregate metrics for the model run

#### Cross-Validation Model Directories

CV model result directories additionally contain:

- **`metrics.pickle`** - 3D LabelledDataArray with metric values per epoch and CV-fold
- **`val_results.parquet`** - Combined validation results across all CV-folds (same columns as `train_results.parquet` plus `cv_iter` indicating the CV-fold)
- **`cv_XX/`** - Folder for each CV-fold containing:
  - `val_results.parquet` - Validation results for that fold
  - `cim_results/val_results.parquet` - GM estimates from the multivariate normal conditional IM method for that fold
- **`cim_results/val_results.parquet`** - Combined GM estimates from the multivariate normal conditional IM method across all CV-folds

#### Full Model Directories

Full model result directories additionally contain:

- **`metrics.parquet`** - DataFrame with metrics (columns) for each epoch (rows)
- **`3468575/predictions_noAllowSelf.parquet`** - GNN predictions for the 22 February 2011 Magnitude 6.2 Christchurch earthquake

### Multivariate Normal Conditional IM Method Results

The `cim/0728_3468575_canterbury_extended_500m_nzgmdbV4p3Final` folder contains `cim_results_noAllowSelf.parquet`, which contains the multivariate normal conditional IM method estimates for the 22 February 2011 Magnitude 6.2 Christchurch earthquake on a uniform grid (plus recording stations).

Each row corresponds to a location with columns:
- **`pSA_X.XX`** - Observed IM at that location (NaN for most locations unless it's an observation site)
- **`pSA_X.XX_cond_mean`** - Mean estimated by the multivariate normal conditional IM method
- **`pSA_X.XX_cond_std`** - Standard deviation estimated by the multivariate normal conditional IM method

## Model Training

Train models using the `train_ml_models.sh` script located at `post_event_gm_gnn/scripts/ml_models/train_ml_models.sh`.

1. Set the variables defined under the `# Input` section
2. A GPU is highly recommended (training on CPU will be very slow)

## Post-Processing

The `run_gnn_post.sh` script performs the following:
- Generates multivariate normal conditional IM results for CV GNN runs
- Runs 22 February 2011 Magnitude 6.2 Christchurch earthquake predictions for GNNs trained on all data (excluding the 2011 February event)

Update the following input variables:
- `cv_gnn_only_dir`
- `cv_gnn_residual_dir`
- `full_gnn_model_dir`
- `full_gnn_model_dir_ignore_CCCC_SHLC`

These correspond to the model directories created by `train_ml_models.sh`.

**Note:** This process can take multiple hours.

## Generating Paper Figures

Generate the paper plots using the `gen_paper_figures.sh` script located at `post_event_gm_gnn/scripts/figures/gen_paper_figures.sh`.

Required configuration:
1. Set the `scripts_dir` variable
2. Set the `out_dir` variable
3. Ensure the `wdata` environment variable is set


