import pickle
from pathlib import Path
from typing import Sequence
from dataclasses import dataclass
from collections import defaultdict

import tqdm
import pandas as pd
import numpy as np

from qcore import src_site_dist, grid
from source_modelling import srf
import empirical.util.estimations as emp_estimations
import ml_tools as mlt

from . import constants
from .data_classes import ObservedData
from . import data

SRF_POINTS_PER_KM = 10


OBS_DATA_OQ_COLS_MAPPING = {
    ObservedData.SiteColEnums.VS30: "vs30",
    ObservedData.EventSiteColEnums.RRUP: "rrup",
    ObservedData.EventSiteColEnums.RJB: "rjb",
    ObservedData.SiteColEnums.Z1P0: "z1pt0",
    ObservedData.EventColEnums.MAG: "mag",
    ObservedData.EventColEnums.RAKE: "rake",
    ObservedData.EventColEnums.DIP: "dip",
    ObservedData.EventColEnums.ZTOR: "ztor",
    ObservedData.EventSiteColEnums.RX: "rx",
    ObservedData.EventColEnums.DEPTH: "hypo_depth",
}


@dataclass
class SiteCorrelations:
    corrs: np.ndarray
    sites: np.ndarray
    ims: np.ndarray
    event: str

    def __post_init__(self):
        assert self.corrs.shape == (
            self.sites.size,
            self.sites.size,
            self.ims.size,
        )
        assert self.sites.size == np.unique(self.sites).size
        assert self.ims.size == np.unique(self.ims).size

    def write(self, data_ffp: Path):
        with data_ffp.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, data_ffp: Path):
        with data_ffp.open("rb") as f:
            return pickle.load(f)

    def to_im_dict(self):
        return {
            cur_im: pd.DataFrame(
                data=self.corrs[:, :, self._get_im_ix(cur_im)],
                index=self.sites,
                columns=self.sites,
            )
            for cur_im in self.ims
        }

    def _get_im_ix(self, cur_im: str):
        return np.flatnonzero(cur_im == self.ims)[0]

    def get_im_corrs(self, im: str):
        return pd.DataFrame(
            data=self.corrs[:, :, self._get_im_ix(im)],
            index=self.sites,
            columns=self.sites,
        )

    def get_site_im_corrs(self, site: str, ims: Sequence[str]):
        site_ix = np.flatnonzero(site == self.sites)[0]
        im_ind = [self._get_im_ix(cur_im) for cur_im in ims]

        # Why do I need a transpose here??
        return pd.DataFrame(
            data=self.corrs[site_ix, :, im_ind].T, index=self.sites, columns=ims
        )


def _compute_emp_gm_params(
    rupture_df: pd.DataFrame, periods: Sequence[float], output_ffp: Path
):
    """
    Compute empirical GM parameters for the given rupture data using OQ.

    Parameters
    ----------
    rupture_df : pd.DataFrame
        DataFrame containing rupture information with required columns.
        Columns z1pt0 and z2pt5 have to be in kilometres.
    periods : Sequence[float]
        List of periods for which pSA is to be computed.
    output_ffp : Path
        Output file paht.
    """
    from empirical.util.openquake_wrapper_vectorized import oq_run
    from empirical.util.classdef import TectType, GMM

    ### Constants
    GMM_MAPPING = {
        TectType.ACTIVE_SHALLOW: GMM.Br_13,
        TectType.SUBDUCTION_SLAB: GMM.ZA_06,
        TectType.SUBDUCTION_INTERFACE: GMM.ZA_06,
    }

    TECT_CLASS_MAPPING = {
        constants.TectonicType.CRUSTAL: TectType.ACTIVE_SHALLOW,
        constants.TectonicType.SUBDUCTION_SLAB: TectType.SUBDUCTION_SLAB,
        constants.TectonicType.SUBDUCTION_INTERFACE: TectType.SUBDUCTION_INTERFACE,
        constants.TectonicType.UNKNOWN: TectType.ACTIVE_SHALLOW,
        constants.TectonicType.OUTER_RISE: TectType.SUBDUCTION_SLAB,
    }

    ### GM prediction
    print("Running predictions")
    dfs = []
    sites = np.unique(rupture_df[ObservedData.SiteColEnums.SITE_ID])
    for cur_site in tqdm.tqdm(sites):
        cur_site_mask = rupture_df[ObservedData.SiteColEnums.SITE_ID].values == cur_site

        for cur_tect_class in np.unique(
            rupture_df.loc[cur_site_mask, ObservedData.EventColEnums.TECT_TYPE]
        ):
            cur_tect_mask = cur_site_mask & (
                rupture_df[ObservedData.EventColEnums.TECT_TYPE].values
                == cur_tect_class
            )

            if cur_tect_class not in TECT_CLASS_MAPPING:
                continue

            cur_tect_type = TECT_CLASS_MAPPING[cur_tect_class]
            pga_result = oq_run(
                GMM_MAPPING[cur_tect_type],
                cur_tect_type,
                rupture_df.loc[cur_tect_mask, constants.OQ_INPUT_COLUMNS],
                "PGA",
            )

            psa_result = oq_run(
                GMM_MAPPING[cur_tect_type],
                cur_tect_type,
                rupture_df.loc[cur_tect_mask, constants.OQ_INPUT_COLUMNS],
                "pSA",
                periods,
            )

            cur_df = pd.concat((pga_result, psa_result), axis=1)
            cur_df.index = rupture_df.loc[cur_tect_mask].index
            cur_df[
                [ObservedData.EventColEnums.EVENT_ID, ObservedData.SiteColEnums.SITE_ID]
            ] = rupture_df[
                [ObservedData.EventColEnums.EVENT_ID, ObservedData.SiteColEnums.SITE_ID]
            ]

            dfs.append(cur_df)

    result_df = pd.concat(dfs, axis=0)
    result_df.to_parquet(output_ffp)


def compute_event_non_uniform_sites_emp_gm_params(
    event_id: str,
    non_uniform_sites_dir: Path,
    nzgmdb_ffp: Path,
    srf_ffp: Path,
    max_rjb: float,
    output_ffp: Path,
):
    """
    Compute empirical ground motion parameters
    for non-uniform sites for a given event.

    Parameters
    ----------
    event_id : str
        The identifier for the event.
    non_uniform_sites_dir : Path
        Directory containing non-uniform site data.
    nzgmdb_ffp : Path
        File path to the NZGMDB data file.
    srf_ffp : Path
        File path to the SRF file.
    max_rjb : float
        Maximum RJB distance.
    output_ffp : Path
        File path to save the output empirical ground motion parameters.
    """
    # NZGMBD data
    obs_data = load_obs_nzgmdb(nzgmdb_ffp)
    obs_event_data = obs_data.get_event_data(event_id)

    # Load non-uniform grid data
    site_df = data.load_non_uniform_grid(non_uniform_sites_dir)
    rupture_df = data.add_srf_site_to_source_distances(
        site_df,
        srf_ffp,
        event_id,
    )

    # Use NZGMDB data for shared sites
    shared_sites = rupture_df.index.values[
        np.isin(rupture_df.site_id, obs_event_data.index)
    ]
    cols = [
        ObservedData.SiteColEnums.VS30,
        ObservedData.SiteColEnums.Z1P0,
        ObservedData.SiteColEnums.Z2P5,
        ObservedData.SiteColEnums.SITE_LON,
        ObservedData.SiteColEnums.SITE_LAT,
        ObservedData.EventSiteColEnums.RRUP,
        ObservedData.EventSiteColEnums.RJB,
        ObservedData.EventSiteColEnums.RX,
    ]
    rupture_df.loc[shared_sites, cols] = obs_event_data.loc[
        shared_sites, cols
    ].values.astype(np.float32)

    # Populate event fields
    rupture_df[ObservedData.EventColEnums.EVENT_ID] = event_id
    rupture_df[ObservedData.EventColEnums.TECT_TYPE] = obs_event_data.iloc[0][
        ObservedData.EventColEnums.TECT_TYPE
    ]
    rupture_df[ObservedData.EventColEnums.MAG] = obs_event_data.iloc[0][
        ObservedData.EventColEnums.MAG
    ]
    rupture_df[ObservedData.EventColEnums.RAKE] = obs_event_data.iloc[0][
        ObservedData.EventColEnums.RAKE
    ]
    rupture_df[ObservedData.EventColEnums.DIP] = obs_event_data.iloc[0][
        ObservedData.EventColEnums.DIP
    ]
    rupture_df[ObservedData.EventColEnums.ZTOR] = obs_event_data.iloc[0][
        ObservedData.EventColEnums.ZTOR
    ]
    rupture_df[ObservedData.EventColEnums.DEPTH] = obs_event_data.iloc[0][
        ObservedData.EventColEnums.DEPTH
    ]

    # Filter out sites with rjb > max_rjb
    rupture_df = rupture_df.loc[rupture_df.rjb <= max_rjb]

    # Convert Z1.0 to kilometres
    rupture_df[ObservedData.SiteColEnums.Z1P0] /= 1000

    # Rename columns for OQ
    rupture_df = rupture_df.rename(columns=OBS_DATA_OQ_COLS_MAPPING)
    rupture_df["vs30measured"] = False

    rupture_df.index = mlt.array_utils.numpy_str_join("_", event_id, rupture_df.site_id.values.astype(str))

    # Compute the empirical GM parameters
    _compute_emp_gm_params(rupture_df, constants.PERIODS, output_ffp)


def compute_nzgmdb_emp_gm_params(
    output_ffp: Path,
    nzgmdb_flat_ffp: Path,
    rjb_max: float,
    events: Sequence[str] = None,
    periods: Sequence[float] = constants.PERIODS,
):
    """
    Computes the empirical GMM parameters for all
    specified sites and sources, based on inputs
    from NZGMDB

    Parameters
    ----------
    output_ffp: Path
        Output file path
    nzgmdb_flat_ffp: Path
        Path to the NZGMDB flat file
    rjb_max: float
        Maximum Rjb distance for which to
        to compute the empirical GMM parameters

    Returns
    -------
    result_df: DataFrame
        The empirical GMM parameters for PGA
        and the default set of pSA periods
    """

    ### Data loading
    obs_data = load_obs_nzgmdb(nzgmdb_flat_ffp)

    # Create rupture dataframe
    columns = [
        ObservedData.EventColEnums.EVENT_ID,
        ObservedData.SiteColEnums.SITE_ID,
        ObservedData.SiteColEnums.SITE_LON,
        ObservedData.SiteColEnums.SITE_LAT,
        ObservedData.EventColEnums.TECT_TYPE,
    ] + list(OBS_DATA_OQ_COLS_MAPPING.keys())
    # rupture_df = pd.read_csv(nzgmdb_flat_ffp, index_col=0)[columns]
    rupture_df = obs_data.record_df[columns].copy(True)

    # Filter events
    if events is not None:
        rupture_df = rupture_df.loc[rupture_df.event.isin(events)]

    # Apply rjb filter
    rupture_df = rupture_df.loc[rupture_df.rjb <= rjb_max]

    # Convert Z1.0 to kilometres
    rupture_df[ObservedData.SiteColEnums.Z1P0] /= 1000

    # Rename columns for OQ
    rupture_df = rupture_df.rename(columns=OBS_DATA_OQ_COLS_MAPPING)
    rupture_df["vs30measured"] = False

    _compute_emp_gm_params(rupture_df, periods, output_ffp)


def load_obs_nzgmdb(nzgmdb_ffp: Path):
    """
    Load the observed data from NZGMDB and performs the
    necessary preparation steps, depending
    on the version.

    Parameters
    ----------
    nzgmdb_ffp: Path
        Path to the NZGMDB flat file

    Returns
    -------
    obs_data: ObservedData
        Observed data object
    """
    obs_data = ObservedData.from_nzgmdb_flat(nzgmdb_ffp)
    assert obs_data.data_source is constants.ObsDataSource.NZGMDB

    # Filter out nan values
    obs_data = obs_data.drop_nan()

    # Some basic filtering
    if obs_data.nzgmdb_version is constants.NZGMDBVersion.v3p4:
        obs_data = obs_data.metadata_filter(dict(rrup=(0, 250)))
    elif obs_data.nzgmdb_version in [
        constants.NZGMDBVersion.v4p0,
        constants.NZGMDBVersion.v4p1,
        constants.NZGMDBVersion.v4p2,
    ]:
        obs_data = obs_data.metadata_filter(dict(rrup=(0, 250), is_ground_level=True))
    else:
        raise NotImplementedError("Invalid NZGMDB version")

    # Drop duplicates
    # Use the one with the smaller fmin
    obs_data = obs_data.drop_duplicates(
        ["event_id", "site_id"],
        sort_key=ObservedData.OtherColEnums.FMIN,
        ascending=True,
    )

    # Convert to event_site index
    obs_data = obs_data.to_event_site_index()

    # Apply fmin
    obs_data = obs_data.apply_fmin_filter(ObservedData.OtherColEnums.FMIN)

    return obs_data


def load_non_uniform_grid(non_uniform_site_dir: Path):
    """
    Loads the non-uniform grid sites

    Parameters
    ----------
    non_uniform_site_dir : Path
        Path to the directory containing the non-uniform site files.

    Returns
    -------
    pd.DataFrame
        DataFrame containing site information with columns:
            - 'lon': Longitude
            - 'lat': Latitude
            - 'site_id': Site identifier
            - 'vs30': Vs30 value
            - 'z1p0': Z1.0 value in meters
            - 'z2p5': Z2.5 value
    """
    ll_df = pd.read_csv(
        non_uniform_site_dir
        / "non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.ll",
        sep=" ",
        header=None,
        index_col="site_id",
        names=[
            ObservedData.SiteColEnums.SITE_LON,
            ObservedData.SiteColEnums.SITE_LAT,
            ObservedData.SiteColEnums.SITE_ID,
        ],
    )
    vs30_df = pd.read_csv(
        non_uniform_site_dir
        / "non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.vs30",
        header=None,
        sep=" ",
        index_col="site_id",
        names=[ObservedData.SiteColEnums.SITE_ID, ObservedData.SiteColEnums.VS30],
    )
    z_df = pd.read_csv(
        non_uniform_site_dir
        / "non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.z",
        index_col="Station_Name",
    )

    site_df = ll_df.copy(deep=True)
    site_df[ObservedData.SiteColEnums.VS30] = vs30_df.loc[ll_df.index, "vs30"]
    site_df[ObservedData.SiteColEnums.Z1P0] = z_df.loc[ll_df.index, "Z_1.0(km)"] * 1000
    site_df[ObservedData.SiteColEnums.Z2P5] = z_df.loc[ll_df.index, "Z_2.5(km)"]

    return site_df


def add_srf_site_to_source_distances(
    site_df: pd.DataFrame, srf_ffp: Path, event_id: str
):
    """
    Adds SRF site-to-source distances to the given site DataFrame.

    Parameters
    ----------
    site_df : pd.DataFrame
        DataFrame containing site information with columns 'lon' and 'lat'.
    srf_ffp : Path
        File path to the SRF file.
    event_id : str
        Identifier for the seismic event.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional columns for site-to-source distances:
        - 'site_id': Identifier for the site.
        - 'event_id': Identifier for the event.
        - 'rrup': Rupture distance.
        - 'rjb': Joyner-Boore distance.
        - 'rx': Distance from the surface projection of the rupture plane.
    """
    record_df = site_df.copy(True)
    record_df[ObservedData.SiteColEnums.SITE_ID] = record_df.index.values.astype(str)
    record_df[ObservedData.EventColEnums.EVENT_ID] = event_id

    srf_model = srf.read_srf(srf_ffp)

    # Read the srf file to determine the nodal plane information
    nodal_plane_info = defaultdict(lambda: None)
    nodal_plane_info["f_type"] = "ff"

    # Find the total slip and average rake for each subfault
    total_slip = [np.sum(plane_points["slip"]) for plane_points in srf_model.segments]
    avg_rake = [np.average(plane_points["rake"]) for plane_points in srf_model.segments]

    # Calculate the average strike, dip and rake based on weighted average of slip
    (
        nodal_plane_info["strike"],
        nodal_plane_info["dip"],
        nodal_plane_info["rake"],
    ) = emp_estimations.calculate_avg_strike_dip_rake(
        srf_model.planes, avg_rake, total_slip
    )

    # Recompute the srf points to have a consistent resolution
    srf_points = []
    for plane in srf_model.planes:
        corner_0, corner_1, corner_2, _ = plane.corners
        # Utilise grid functions from qcore to get the mesh grid
        plane_points = grid.coordinate_meshgrid(
            corner_0, corner_1, corner_2, 1000 / SRF_POINTS_PER_KM
        )
        # Reshape to (n, 3)
        plane_points = plane_points.reshape(-1, 3)
        srf_points.append(plane_points)
    srf_points = np.vstack(srf_points)
    # Convert depth to km
    srf_points[:, 2] /= 1000
    # Swap the lat and lon for the srf points
    srf_points = nodal_plane_info["srf_points"] = srf_points[:, [1, 0, 2]]

    # Generate the srf header
    nodal_plane_info["srf_header"] = (
        srf_model.header[["nstk", "ndip", "stk", "len", "wid"]]
        .rename(
            columns={
                "nstk": "nstrike",
                "ndip": "ndip",
                "stk": "strike",
                "len": "length",
                "wid": "width",
            }
        )
        .to_dict(orient="records")
    )
    nodal_plane_info["length"] = sum([plane.length for plane in srf_model.planes])

    loc_values = record_df[
        [ObservedData.SiteColEnums.SITE_LON, ObservedData.SiteColEnums.SITE_LAT]
    ].values
    loc_values = np.hstack((loc_values, np.zeros((loc_values.shape[0], 1))))

    # Compute rjb, rrup
    rrup, rjb = src_site_dist.calc_rrup_rjb(srf_points, loc_values)
    record_df[ObservedData.EventSiteColEnums.RRUP] = rrup
    record_df[ObservedData.EventSiteColEnums.RJB] = rjb

    # Compute rx
    rx, _ = src_site_dist.calc_rx_ry_GC2(
        srf_points, nodal_plane_info["srf_header"], loc_values
    )
    record_df[ObservedData.EventSiteColEnums.RX] = rx

    return record_df


def load_emp_gmm_params(emp_gmm_ffp: Path):
    """
    Load the empirical GMM parameters

    Parameters
    ----------
    emp_gmm_ffp: Path
        Path to the empirical GMM parameters

    Returns
    -------
    emp_gmm_params: DataFrame
        Empirical GMM parameters
    """
    return pd.read_parquet(emp_gmm_ffp)


# def load_obs_waveform(obs_waveform_dir: Path, site: str):
#     """
#     Loads the observation waveform data from the
#     NZ-GMDB waveforms

#     Note: Does not perform any time-shifting

#     Parameters
#     ----------
#     obs_waveform_dir: path
#         Path to the accBB folder in the
#         NZ-GMDB waveforms
#     site: string

#     Returns
#     -------
#     obs_t: array of floats
#         The time values
#     obs_acc: array of floats
#         Acceleration data,
#         shape [nt, 3] with the components
#         in the order 090, 000, Ver
#     """
#     if not all(
#         [
#             (obs_waveform_dir / f"{site}.{cur_comp}").exists()
#             for cur_comp in constants.COMPONENTS
#         ]
#     ):
#         print(f"Can't find all acceleration waveform files for {site}")
#         return None, None

#     obs_acc = []
#     meta = None
#     for cur_comp in constants.COMPONENTS:
#         cur_acc, cur_meta = read_ascii(
#             str(obs_waveform_dir / f"{site}.{cur_comp}"), meta=True
#         )
#         if meta is None:
#             meta = cur_meta
#         else:
#             assert meta["dt"] == cur_meta["dt"]
#         obs_acc.append(cur_acc)

#     obs_acc = np.stack(obs_acc, axis=1)
#     obs_t = meta["dt"] * np.arange(obs_acc.shape[0])

#     return obs_t, obs_acc

# def load_obs_nga_west2(
#     nga_west2_ffp: Path,
# ):
#     """
#     Load the observed data from NGA-West2 and performs the
#     necessary preparation steps.

#     Parameters
#     ----------
#     nga_west2_ffp: Path
#         Path to the NGA-West2 flat file

#     Returns
#     -------
#     obs_data: ObservedData
#         Observed data object
#     """
#     # Load
#     obs_data = ObservedData.from_nga_west2_flat(nga_west2_ffp)

#     # Drop nan values
#     obs_data.drop_nan()

#     # Drop duplicates
#     obs_data.drop_duplicates(
#         [ObservedData.EventColEnums.EVENT_ID, ObservedData.SiteColEnums.SITE_ID]
#     )

#     # Distance filtering
#     obs_data = obs_data.metadata_filter(dict(rrup=(0, 250)))

#     # Apply fmin
#     obs_data = obs_data.apply_fmin_filter(ObservedData.OtherColEnums.FMIN)

#     return obs_data


# def load_obs_nga_subduction(nga_sub_ffp: Path):
#     """
#     Load the observed data from NGA-Subduction and performs the
#     necessary preparation steps.

#     Parameters
#     ----------
#     nga_sub_ffp: Path
#         Path to the NGA-Subduction flat file

#     Returns
#     -------
#     obs_data: ObservedData
#         Observed data object

#     """
#     # Load
#     obs_data = ObservedData.from_nga_subduction_flat(nga_sub_ffp)

#     # Drop nan rows
#     obs_data = obs_data.drop_nan()

#     # Drop duplicates
#     obs_data = obs_data.drop_duplicates(
#         [ObservedData.EventColEnums.EVENT_ID, ObservedData.SiteColEnums.SITE_ID]
#     )

#     # Distance filtering
#     obs_data = obs_data.metadata_filter(dict(rrup=(0, 250)))

#     return obs_data
