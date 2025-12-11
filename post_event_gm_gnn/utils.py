from typing import Sequence, TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances

from . import constants

if TYPE_CHECKING:
    from .data_classes import LBSiteCorrelationData


def reverse_im_filename(im: str):
    if im.startswith("pSA"):
        return im[::-1].replace("p", ".", 1)[::-1]
    return im


def get_im_filename(im: str):
    if im.startswith("pSA"):
        return im.replace(".", "p", 1)
    return im


def get_nice_im_name(im: str, use_latex: bool = False):
    if im.startswith("pSA"):
        return f"pSA({im.split('_')[-1]}s)"

    if use_latex:
        match im.lower():
            case "ds595":
                return "$D_{s595}$"
            case "ds575":
                return "$D_{s575}$"
            case _:
                return im
    return im


def get_pSA_period(im: str):
    if im.startswith("pSA"):
        return float(im.split("_")[-1])
    return None


def get_emp_gm_mean_im_keys(ims: Sequence[str]):
    return [f"{im}_mean" for im in ims]


def calculate_distance_matrix(
    stations: Sequence[str],
    locations_df: pd.DataFrame,
    site_lon_col: str = "lon",
    site_lat_col: str = "lat",
):
    """
    Given a set of stations and their locations (in lat, lon format),
    calculate the matrix containing
    the pairwise distance

    Parameters
    ----------
    stations: Sequence[str]
        List of the station names
    locations_df: pd.DataFrame
        Locations of each of the stations (in lat, lon)
    site_lat_col: str
    site_lon_col: str
    """
    dist_matrix = (
        haversine_distances(
            np.radians(locations_df.loc[stations, [site_lat_col, site_lon_col]].values)
        )
        * constants.R_EARTH
    )
    return pd.DataFrame(data=dist_matrix, index=stations, columns=stations)


def compute_degree_of_constraint(
    result_df: pd.DataFrame, corr_data: "LBSiteCorrelationData"
) -> pd.DataFrame:
    """
    Computes the constraintness for each scenario.
    Constraintness is defined as the sum of the
    correlation coefficients across all observed sites
    and mean over all the IMs.

    Note: Adds new column to the result_df called "constraintness".

    Parameters
    ----------
    result_df: pd.DataFrame
        Result dataframe for which to compute constraintness.
        Required columns: "obs_sites", "site_int"
    corr_data: LBSiteCorrelationData
        Correlation data object. Must contain the
        correlations for all relevant sites and IMs.

    Returns
    -------
    result_df: pd.DataFrame
        Result dataframe with the new column "constraintness".
        All other columns are not modified.
    """

    for cur_key in result_df.index:
        cur_obs_sites = result_df.loc[cur_key, "obs_sites"]
        cur_site_int = result_df.loc[cur_key, "site_int"]

        result_df.loc[cur_key, "doc"] = (
            corr_data.corr_data.sel[cur_site_int, :, :]
            .loc[cur_obs_sites]
            .sum(axis=0)
            .mean()
        )

    return result_df
