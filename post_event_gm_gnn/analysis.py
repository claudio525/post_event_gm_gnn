from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from tqdm import tqdm
import ml_tools as mlt

from . import constants
from .data_classes import ObservedData


def get_residuals(
    results: pd.DataFrame,
    ims: Sequence[str] = constants.PSA_KEYS,
    pred_suffix: str = "pred",
    site_col: str = "site_int",
):
    """Computes the residual between the observed and predicted IMs for each scenario"""
    pred_im_keys = mlt.array_utils.numpy_str_join("_", ims, pred_suffix)
    res_df = pd.DataFrame(
        data=results.loc[:, ims].values - results.loc[:, pred_im_keys].values,
        columns=ims,
    )

    res_df.index = results.index
    res_df["event_id"] = results["event_id"]
    res_df[site_col] = results[site_col]
    if "n_obs_sites" in results.columns:
        res_df["n_obs_sites"] = results["n_obs_sites"]
    return res_df


def get_res_mean_std(
    residual_df: pd.DataFrame,
    ims: Sequence[str] = constants.PSA_KEYS,
):
    """Compute the bias and residual standard deviation"""
    res_mean_std_df = pd.concat(
        (residual_df[ims].mean(axis=0), residual_df[ims].std(axis=0)), axis=1
    )
    res_mean_std_df.columns = ["mean", "std"]

    return res_mean_std_df


def load_emp_gm_params_res(emp_gm_params_ffp: Path, obs_data: ObservedData):
    """
    Loads the empirical GM parameters and residuals wrt. observed
    """
    emp_gm_params = pd.read_parquet(emp_gm_params_ffp)
    assert obs_data.record_df.index.isin(
        emp_gm_params.index
    ).all(), "Missing empirical data"
    emp_gm_params = emp_gm_params.loc[obs_data.record_df.index]

    emp_res_df = pd.DataFrame(
        data=np.log(obs_data.record_df[constants.PSA_KEYS].values)
        - emp_gm_params.loc[
            obs_data.record_df.index, constants.GMM_PRED_PSA_KEYS
        ].values,
        index=obs_data.record_df.index,
        columns=constants.PSA_KEYS,
    )
    emp_res_df["event_id"] = emp_gm_params.loc[emp_res_df.index, "event_id"]
    emp_res_df["site_id"] = emp_gm_params.loc[emp_res_df.index, "site_id"]

    return emp_gm_params, emp_res_df


def compute_corr_site_pairs(gnn_results: pd.DataFrame):
    """
    Computes all site of interest and observation site pairs
    with at least 10 predictions & observed records between them

    Parameters
    ----------
    gnn_results : pd.DataFrame
        DataFrame containing the GNN results.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the site-pairs and their counts.
    dict
        Dictionary containing the events for each site-pair.
    """
    # Get the site-pairs with at least 10 predictions/records
    site_pairs_count = {}
    site_pair_events = {}
    for cur_ix, cur_row in gnn_results.iterrows():
        cur_site_int = cur_row.site_int
        cur_obs_sites = cur_row.obs_sites

        cur_site_pairs = mlt.array_utils.numpy_str_join(
            "_", cur_site_int, cur_obs_sites.astype(str)
        )

        for cur_site_pair in cur_site_pairs:
            if cur_site_pair not in site_pairs_count:
                site_pairs_count[cur_site_pair] = 1
                site_pair_events[cur_site_pair] = [cur_row.event_id]
            else:
                site_pairs_count[cur_site_pair] += 1
                site_pair_events[cur_site_pair].append(cur_row.event_id)

    site_pairs_df = pd.DataFrame.from_dict(
        site_pairs_count, orient="index", columns=["count"]
    )
    site_pairs_df["site_int"] = site_pairs_df.index.str.split("_").str[0]
    site_pairs_df["obs_site"] = site_pairs_df.index.str.split("_").str[1]

    # Drop any with less than 10 predictions/recordings
    site_pairs_df = site_pairs_df.loc[site_pairs_df["count"] >= 10]

    return site_pairs_df, site_pair_events


def compute_site_int_obs_correlations(
    site_pairs_df: pd.DataFrame,
    site_pair_events: dict,
    ims: np.ndarray[str],
    site_int_res_df: pd.DataFrame,
    site_obs_res_df: pd.DataFrame,
    verbose: bool = True,
):
    """
    Computes the correlations for each site-pair, with the first
    site always corresponding to a site of interest and the second
    site corresponding to an observation site.

    Parameters
    ----------
    site_pairs_df : pd.DataFrame
        DataFrame containing the site-pairs and their counts
    site_pair_events : dict
        Dictionary containing the events for each site-pair
    ims : np.ndarray[str]
        Array of IM keys
    site_int_res_df : pd.DataFrame
        DataFrame containing the IM residuals with respect to
         an empirical GMM for the site of interest.
        Columns corresponding to the IM keys
    site_obs_res_df : pd.DataFrame
        DataFrame containing the IM residuals with respect to
         an empirical GMM for the observation site.
        Columns corresponding to the IM keys

    Returns
    -------
    pd.DataFrame
        DataFrame containing the correlations for each site-pair
        and IM key
    """
    site_pair_corrs = pd.DataFrame(index=site_pairs_df.index, columns=ims)
    for cur_ix, cur_row in tqdm(
        site_pairs_df.iterrows(),
        desc="Computing correlations",
        total=len(site_pairs_df),
        disable=not verbose,
    ):
        cur_site_int, cur_obs_site = cur_row.site_int, cur_row.obs_site
        cur_events = site_pair_events[cur_row.name]
        assert len(cur_events) == cur_row["count"]

        cur_site_int_keys = mlt.array_utils.numpy_str_join(
            "_", cur_events, cur_site_int
        )
        cur_site_int_im_df = site_int_res_df.loc[cur_site_int_keys, ims]

        cur_obs_site_keys = mlt.array_utils.numpy_str_join(
            "_", cur_events, cur_obs_site
        )
        cur_obs_site_im_df = site_obs_res_df.loc[cur_obs_site_keys, ims]

        for cur_im in ims:
            # Drop NaN values
            cur_nan_mask = (
                cur_obs_site_im_df[cur_im].isna().values
                | cur_site_int_im_df[cur_im].isna().values
            )
            if (~cur_nan_mask).sum() < 10:
                continue

            cur_im_int_df = cur_site_int_im_df.loc[~cur_nan_mask, cur_im]
            cur_im_obs_df = cur_obs_site_im_df.loc[~cur_nan_mask, cur_im]

            # Compute the correlation
            cur_corr = np.mean(
                (cur_im_int_df.values - cur_im_int_df.values.mean())
                * (cur_im_obs_df.values - cur_im_obs_df.values.mean())
            )
            cur_corr /= cur_im_int_df.values.std() * cur_im_obs_df.values.std()

            site_pair_corrs.loc[cur_ix, cur_im] = cur_corr

    return site_pair_corrs.astype("float")


def compute_site_int_obs_correlation_residuals(
    gnn_results: pd.DataFrame,
    obs_data: ObservedData,
    emp_gm_params: pd.DataFrame,
    cim_results: pd.DataFrame = None,
):
    # Get the site-pairs with at least 10 predictions/records
    site_pairs_df, site_pair_events = compute_corr_site_pairs(gnn_results)

    site_int_pred_im_df = gnn_results[constants.GNN_PRED_PSA_KEYS].rename(
        columns=dict(zip(constants.GNN_PRED_PSA_KEYS, constants.PSA_KEYS))
    )
    emp_mean = emp_gm_params[constants.GMM_PRED_PSA_KEYS].rename(
        columns=dict(zip(constants.GMM_PRED_PSA_KEYS, constants.PSA_KEYS))
    )

    # Compute residuals of GNN with respect to empirical GMM
    emp_gnn_res_df = emp_mean.loc[site_int_pred_im_df.index] - site_int_pred_im_df

    # Compute residuals of observed (SoI) with respect to empirical GMM
    obs_im_df = np.log(obs_data.record_df[constants.PSA_KEYS])
    emp_obs_res_df = emp_mean - obs_im_df

    # Compute correlations
    emp_gnn_res_site_pair_corrs = compute_site_int_obs_correlations(
        site_pairs_df,
        site_pair_events,
        constants.PSA_KEYS,
        emp_gnn_res_df,
        emp_obs_res_df,
        verbose=False,
    )
    emp_obs_res_site_pair_corrs = compute_site_int_obs_correlations(
        site_pairs_df,
        site_pair_events,
        constants.PSA_KEYS,
        emp_obs_res_df,
        emp_obs_res_df,
        verbose=False,
    )

    emp_cim_res_site_pair_corrs = None 
    if cim_results is not None:
        cim_pred_im_df = cim_results[constants.CIM_PRED_PSA_KEYS].rename(columns=dict(zip(constants.CIM_PRED_PSA_KEYS, constants.PSA_KEYS)))
        emp_cim_res_df = emp_mean.loc[cim_pred_im_df.index] - cim_pred_im_df
        emp_cim_res_site_pair_corrs = compute_site_int_obs_correlations(site_pairs_df, site_pair_events, constants.PSA_KEYS, emp_cim_res_df, emp_obs_res_df, verbose=False)

    # Sanity checking
    assert np.all(emp_gnn_res_site_pair_corrs.index == emp_obs_res_site_pair_corrs.index)
    assert emp_cim_res_site_pair_corrs is None or np.all(emp_gnn_res_site_pair_corrs.index == emp_cim_res_site_pair_corrs.index)
    assert np.all(emp_gnn_res_site_pair_corrs.index == site_pairs_df.index)

    return emp_gnn_res_site_pair_corrs, emp_obs_res_site_pair_corrs, emp_cim_res_site_pair_corrs, site_pairs_df


def fisher_transform(rho: np.ndarray) -> np.ndarray:
    """
    Fisher transform for correlation coefficients.
    Transforms the correlation coefficient to a z-score.

    Parameters
    ----------
    rho: np.ndarray
        Correlation coefficient

    Returns
    -------
    np.ndarray
        Fisher transformed z-score
    """
    return 0.5 * np.log((1 + rho) / (1 - rho))


def get_fisher_transform_residuals(
    pred_site_pair_corrs: pd.DataFrame,
    obs_site_pair_corrs: pd.DataFrame,
):
    # Apply Fisher transform to the correlations
    pred_site_pair_corrs = pd.DataFrame(
        data=fisher_transform(pred_site_pair_corrs.values),
        index=pred_site_pair_corrs.index,
        columns=pred_site_pair_corrs.columns,
    )
    obs_site_pair_corrs = pd.DataFrame(
        data=fisher_transform(obs_site_pair_corrs.values),
        index=obs_site_pair_corrs.index,
        columns=obs_site_pair_corrs.columns,
    )

    # Compute the residuals
    res_site_pair_corrs = obs_site_pair_corrs - pred_site_pair_corrs
    return res_site_pair_corrs

