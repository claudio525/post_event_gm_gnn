import pickle
from pathlib import Path
from typing import Sequence
from dataclasses import dataclass

import tqdm
import pandas as pd
import numpy as np

from qcore.timeseries import BBSeis, read_ascii

from . import constants
from .data_classes import ObservedData


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


def run_emp_gmms(
    output_ffp: Path,
    nzgmdb_flat_ffp: Path,
    rjb_max: float,
    events: Sequence[str] = None,
):
    """
    Computes the empirical GMM parameters for all
    specified sites and sources, based on inputs
    from NZGMDB

    Parameters
    ----------


    Returns
    -------
    result_df: DataFrame
        The empirical GMM parameters for PGA
        and the default set of pSA periods
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

    OQ_INPUT_COLUMNS = [
        "vs30",
        "rrup",
        "rjb",
        "z1pt0",
        "mag",
        "rake",
        "dip",
        "vs30measured",
        "ztor",
        "rx",
        "hypo_depth",
    ]

    OBS_DATA_COLS_MAPPING = {
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

    ### Data loading
    obs_data = load_obs_nzgmdb(nzgmdb_flat_ffp)

    # Create rupture dataframe
    columns = [
        ObservedData.EventColEnums.EVENT_ID,
        ObservedData.SiteColEnums.SITE_ID,
        ObservedData.SiteColEnums.SITE_LON,
        ObservedData.SiteColEnums.SITE_LAT,
        ObservedData.EventColEnums.TECT_TYPE,
    ] + list(OBS_DATA_COLS_MAPPING.keys())
    rupture_df = obs_data.record_df[columns].copy(True)

    # Filter events
    if events is not None:
        rupture_df = rupture_df.loc[rupture_df.event.isin(events)]

    # Convert Z1.0 to kilometres
    rupture_df[ObservedData.SiteColEnums.Z1P0] /= 1000

    # Rename columns for OQ
    rupture_df = rupture_df.rename(columns=OBS_DATA_COLS_MAPPING)
    rupture_df["vs30measured"] = False

    # Apply rjb filter
    rupture_df = rupture_df.loc[rupture_df.rjb <= rjb_max]

    ### GM prediction
    print(f"Running predictions")
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
                rupture_df.loc[cur_tect_mask, OQ_INPUT_COLUMNS],
                "PGA",
            )

            psa_result = oq_run(
                GMM_MAPPING[cur_tect_type],
                cur_tect_type,
                rupture_df.loc[cur_tect_mask, OQ_INPUT_COLUMNS],
                "pSA",
                constants.PERIODS,
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


def load_sim_waveform(sim_rupture_dir: Path, rel_id: str, site: str):
    """
    Loads the acceleration time-series data
    for the specified simulation id and site

    Parameters
    ----------
    sim_rupture_dir: Path
        Path to the event simulation directory
        i.e. Runs/{event_id}
    rel_id: string
    site: string

    Returns
    -------
    sim_t: array of floats
        The time values
    sim_acc: array of floats
        Acceleration data,
        shape [nt, 3] with the components
        in the order 090, 000, Ver
    """
    if not (cur_bb_ffp := sim_rupture_dir / rel_id / "BB" / "Acc" / "BB.bin").exists():
        print(f"Can't find BB file for {site} - {rel_id}")
        return None, None

    bb = BBSeis(str(cur_bb_ffp))
    sim_acc = bb.acc(site)
    sim_t = bb.dt * np.arange(sim_acc.shape[0])

    if bb.start_sec < 0:
        sim_mask = sim_t > np.abs(bb.start_sec)
        sim_acc = sim_acc[sim_mask, :]
        sim_t = bb.dt * np.arange(sim_acc.shape[0])
    else:
        raise NotImplementedError()

    return sim_t, sim_acc


def load_obs_waveform(obs_waveform_dir: Path, site: str):
    """
    Loads the observation waveform data from the
    NZ-GMDB waveforms

    Note: Does not perform any time-shifting

    Parameters
    ----------
    obs_waveform_dir: path
        Path to the accBB folder in the
        NZ-GMDB waveforms
    site: string

    Returns
    -------
    obs_t: array of floats
        The time values
    obs_acc: array of floats
        Acceleration data,
        shape [nt, 3] with the components
        in the order 090, 000, Ver
    """
    if not all(
        [
            (obs_waveform_dir / f"{site}.{cur_comp}").exists()
            for cur_comp in constants.COMPONENTS
        ]
    ):
        print(f"Can't find all acceleration waveform files for {site}")
        return None, None

    obs_acc = []
    meta = None
    for cur_comp in constants.COMPONENTS:
        cur_acc, cur_meta = read_ascii(
            str(obs_waveform_dir / f"{site}.{cur_comp}"), meta=True
        )
        if meta is None:
            meta = cur_meta
        else:
            assert meta["dt"] == cur_meta["dt"]
        obs_acc.append(cur_acc)

    obs_acc = np.stack(obs_acc, axis=1)
    obs_t = meta["dt"] * np.arange(obs_acc.shape[0])

    return obs_t, obs_acc


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


def load_obs_nga_west2(
    nga_west2_ffp: Path,
):
    """
    Load the observed data from NGA-West2 and performs the
    necessary preparation steps.

    Parameters
    ----------
    nga_west2_ffp: Path
        Path to the NGA-West2 flat file

    Returns
    -------
    obs_data: ObservedData
        Observed data object
    """
    # Load
    obs_data = ObservedData.from_nga_west2_flat(nga_west2_ffp)

    # Drop nan values
    obs_data.drop_nan()

    # Drop duplicates
    obs_data.drop_duplicates(
        [ObservedData.EventColEnums.EVENT_ID, ObservedData.SiteColEnums.SITE_ID]
    )

    # Distance filtering
    obs_data = obs_data.metadata_filter(dict(rrup=(0, 250)))

    # Apply fmin
    obs_data = obs_data.apply_fmin_filter(ObservedData.OtherColEnums.FMIN)

    return obs_data


def load_obs_nga_subduction(nga_sub_ffp: Path):
    """
    Load the observed data from NGA-Subduction and performs the
    necessary preparation steps.

    Parameters
    ----------
    nga_sub_ffp: Path
        Path to the NGA-Subduction flat file

    Returns
    -------
    obs_data: ObservedData
        Observed data object

    """
    # Load
    obs_data = ObservedData.from_nga_subduction_flat(nga_sub_ffp)

    # Drop nan rows
    obs_data = obs_data.drop_nan()

    # Drop duplicates
    obs_data = obs_data.drop_duplicates(
        [ObservedData.EventColEnums.EVENT_ID, ObservedData.SiteColEnums.SITE_ID]
    )

    # Distance filtering
    obs_data = obs_data.metadata_filter(dict(rrup=(0, 250)))

    return obs_data


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
