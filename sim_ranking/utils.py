from typing import Sequence, NamedTuple, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
from tqdm import tqdm

from qcore.src_site_dist import calc_rrup_rjb

if TYPE_CHECKING:
    from .data_classes import LBSiteCorrelationData


class SourceInfo(NamedTuple):

    rupture_name: str
    hypo_loc: Tuple[float, float]


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
    verbose: bool = False,
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
    distance_matrix = -1 * np.ones((len(stations), len(stations)), dtype=np.float32)
    for i, station in tqdm(enumerate(stations), disable=not verbose, total=len(stations)):
        cur_dist, _ = calc_rrup_rjb(
            np.asarray(
                [
                    [
                        locations_df.loc[station, site_lon_col],
                        locations_df.loc[station, site_lat_col],
                        0,
                    ]
                ]
            ),
            np.stack(
                (
                    locations_df.loc[stations, site_lon_col],
                    locations_df.loc[stations, site_lat_col],
                    np.zeros(len(stations)),
                ),
                axis=1,
            ),
        )
        distance_matrix[i, :] = cur_dist
    return pd.DataFrame(index=stations, data=distance_matrix, columns=stations)


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
