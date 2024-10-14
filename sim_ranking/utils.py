from typing import Sequence, NamedTuple, Tuple

import numpy as np
import pandas as pd

from IM_calculation.source_site_dist.src_site_dist import calc_rrup_rjb


class SourceInfo(NamedTuple):

    rupture_name: str
    hypo_loc: Tuple[float, float]


def reverse_im_filename(im: str):
    if im.startswith("pSA"):
        return im[::-1].replace("p", ".", 1)[::-1]
    return im


def get_nice_im_name(im: str):
    if im.startswith("pSA"):
        return f"pSA({im.split('_')[-1]}s)"
    return im


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
    distance_matrix = -1 * np.ones((len(stations), len(stations)))
    for i, station in enumerate(stations):
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
