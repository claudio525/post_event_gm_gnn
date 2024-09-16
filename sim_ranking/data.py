import pickle
from pathlib import Path
from typing import Sequence
from dataclasses import dataclass

import tqdm
import pandas as pd
import numpy as np

from qcore.timeseries import BBSeis, read_ascii
import ml_tools as mlt
import spatial_hazard as sh
import sha_calc as sha

from . import constants


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


def gen_emp_synthetic_observed(
    emp_gm_params_ffp: Path, nzgmdb_site_ffp: Path, nzgmdb_flat_file: Path
):
    """
    Generates synthetic observed data
    based on the (perturbed) empirical
    GMM parameters correlated using the
    Loth and Baker spatial correlation model

    The nzgmdb_flat_file is only used to get
    r_rup and r_x
    """
    # Load data
    gm_params = pd.read_csv(emp_gm_params_ffp, index_col=0)
    site_df = pd.read_csv(nzgmdb_site_ffp, index_col="sta")

    nzgmdb_df = pd.read_csv(nzgmdb_flat_file, dtype={"evid": str})
    nzgmdb_df.index = mlt.array_utils.numpy_str_join(
        "_", nzgmdb_df.evid.values.astype(str), nzgmdb_df.sta.values.astype(str)
    )

    # Shift the mean
    mean_cols = [f"{cur_im}_mean" for cur_im in constants.PSA_KEYS]
    gm_params.loc[:, mean_cols] = gm_params.loc[:, mean_cols].values + 0.5 * 0.6

    # Generate the realisation
    rels = gen_emp_synthethic_realisations(gm_params, site_df, n_rels=1)

    # Combine
    dfs = []
    for cur_key, cur_df in rels.items():
        cur_event = cur_key.split("_")[0]
        cur_df["event"] = cur_event
        cur_df["site"] = cur_df.index.values.astype(str)
        cur_df.index = mlt.array_utils.numpy_str_join(
            "_", cur_df.event.values.astype(str), cur_df.site.values.astype(str)
        )
        dfs.append(cur_df)

    obs_df = pd.concat(dfs, axis=0)
    obs_df["r_rup"] = nzgmdb_df.loc[obs_df.index, "r_rup"].values
    obs_df["r_x"] = nzgmdb_df.loc[obs_df.index, "r_x"].values
    obs_df = obs_df.rename(columns={"event": "evid", "site": "sta"})

    return obs_df, gm_params


def gen_emp_synthethic_realisations(
    gm_params: pd.DataFrame, site_df: pd.DataFrame, n_rels: int = 25
):
    """
    Generates correlated synthetic observed data
    based on the empirical GM parameters
    and the Loth and Baker spatial correlation
    model

    Parameters
    ----------
    gm_params: DataFrame
        The empirical GM parameters
    site_df: DataFrame
        The NZGMDB site data
    n_rels: int, optional
        The number of realisations to generate

    Returns
    -------
    dict[str, DataFrame]
        The generated realisations
        Keys are {event}_{rel}
    """
    events = gm_params.event.unique().astype(str)

    # Calculate the distance matrix
    dist_matrix = sh.im_dist.calculate_distance_matrix(
        site_df.index.values.astype(str), site_df
    )
    ims = constants.PSA_KEYS

    # Generate the realisations
    result_dict = {}
    for event_ix, cur_event in enumerate(tqdm.tqdm(events)):
        # print(f"Processing event: {cur_event}, {event_ix + 1}/{events.size}")
        cur_df = gm_params.loc[gm_params.event == cur_event, :].set_index("site")
        cur_sites = cur_df.index.values.astype(str)

        im_values, between_event, within_event = sh.im_dist.gen_im_rels(
            cur_df,
            dist_matrix.loc[cur_sites, cur_sites],
            ims,
            n_rels,
            corr_fn=sha.models.loth_baker_corr_model.get_correlations,
        )

        # Save the results
        for ix in range(n_rels):
            cur_df = pd.DataFrame(
                data=np.exp(im_values[:, :, ix]),
                index=cur_sites,
                columns=ims,
            )
            result_dict[f"{cur_event}_REL{ix + 1:02}"] = cur_df

    return result_dict


def run_emp_gmms(
    output_ffp: Path,
    nzgmdb_source_ffp: Path,
    rjb_max: float,
    srf_dir: Path = None,
    nzgmdb_flatfile_ffp: Path = None,
    site_dir: Path = None,
    nzgmdb_site_ffp: Path = None,
    events: Sequence[str] = None,
):
    """
    Computes the empirical GMM parameters for all
        specified sites and sources

    Note I: One of srf_dir or nzgmdb_flatfile_ffp must be specified
    Note II: One of site_dir or nzgmdb_site_ffp must be specified

    Parameters
    ----------
    output_ffp: Path
    site_dir: Path
        Directory that contains all the site
        information files (i.e. vs30, ll, and z)
    nzgmdb_source_ffp: Path
        Path to the NZ-GMDB source file
    rjb_max: float
        RJB distance threshold
    srf_dir: Path, optional
        Directory that contains the srf files
    nzgmdb_flatfile_ffp: Path, optional
        Path to the NZ-GMDB flat file

    Returns
    -------
    result_df: DataFrame
        The empirical GMM parameters for PGA
        and the default set of pSA periods
    """

    from empirical.util.openquake_wrapper_vectorized import oq_run
    from empirical.util.classdef import TectType, GMM
    from IM_calculation.source_site_dist import src_site_dist

    ### Constants
    GMM_MAPPING = {
        TectType.ACTIVE_SHALLOW: GMM.Br_10,
        TectType.SUBDUCTION_SLAB: GMM.ZA_06,
        TectType.SUBDUCTION_INTERFACE: GMM.ZA_06,
    }

    TECT_CLASS_MAPPING = {
        "Crustal": TectType.ACTIVE_SHALLOW,
        "Slab": TectType.SUBDUCTION_SLAB,
        "Interface": TectType.SUBDUCTION_INTERFACE,
        "Undetermined": TectType.ACTIVE_SHALLOW,
        "Outer-rise": TectType.SUBDUCTION_SLAB,
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

    # Input sanity checking
    assert (srf_dir is not None) or (nzgmdb_flatfile_ffp is not None)
    assert not ((srf_dir is not None) and (nzgmdb_flatfile_ffp is not None))

    ### Data loading
    # Get all srf files
    if srf_dir is not None:
        from qcore import srf

        srf_ffps = list(srf_dir.rglob("*.srf"))
        srf_events = [cur_ffp.stem for cur_ffp in srf_ffps]

        # Load srf data
        srf_points, plane_infos = {}, {}
        for cur_srf_ffp in srf_ffps:
            srf_points[cur_srf_ffp.stem] = srf.read_srf_points(str(cur_srf_ffp))
            plane_infos[cur_srf_ffp.stem] = srf.read_header(str(cur_srf_ffp), idx=True)

        events = (
            np.intersect1d(events, srf_events) if events is not None else srf_events
        )
    else:
        nzgmdb_flatfile = pd.read_csv(
            nzgmdb_flatfile_ffp, index_col=0, dtype={"evid": str}
        )
        nzgmdb_events = np.unique(nzgmdb_flatfile.evid).astype(str)

        events = (
            np.intersect1d(events, nzgmdb_events)
            if events is not None
            else nzgmdb_events
        )

    # Load source info
    source_df = pd.read_csv(nzgmdb_source_ffp, index_col=0)

    # Load the site_data
    if site_dir is not None:
        stations_df = pd.read_csv(
            site_dir / f"{constants.STATION_FN_NAME}.ll",
            sep=" ",
            index_col=2,
            header=None,
            names=["lon", "lat"],
        )
        vs30_df = pd.read_csv(
            site_dir / f"{constants.STATION_FN_NAME}.vs30",
            sep=" ",
            index_col=0,
            header=None,
            names=["vs30"],
        )
        z_df = pd.read_csv(site_dir / f"{constants.STATION_FN_NAME}.z", index_col=0)

        ### Data merging/re-naming and tidy up
        assert np.all(stations_df.index == vs30_df.index) and np.all(
            stations_df.index == z_df.index
        )
        site_df = pd.concat([stations_df, vs30_df, z_df], axis=1)
        site_df = site_df.rename(columns={"Z_1.0(km)": "z1pt0"})
        del stations_df, vs30_df, z_df
    else:
        site_df = pd.read_csv(nzgmdb_site_ffp, index_col="sta")[
            ["lat", "lon", "Vs30", "Z1.0"]
        ]
        site_df = site_df.rename(columns={"Vs30": "vs30", "Z1.0": "z1pt0"})
        site_df["z1pt0"] = site_df["z1pt0"] / 1000


    ### Data prep
    site_locs = np.concatenate(
        (site_df[["lon", "lat"]].values, np.zeros((site_df.shape[0], 1))), axis=1
    )
    print(f"Creation of rupture dataframe")
    data_dfs = []
    for cur_event in tqdm.tqdm(events):
        if srf_dir is not None:
            cur_data_df = site_df.copy(True)
            cur_data_df["rrup"], cur_data_df["rjb"] = src_site_dist.calc_rrup_rjb(
                srf_points[cur_event], site_locs
            )

            cur_data_df["rx"], cur_data_df["ry"] = src_site_dist.calc_rx_ry(
                srf_points[cur_event], plane_infos[cur_event], site_locs
            )
        else:
            cur_nzgmdb_flatfile = nzgmdb_flatfile.loc[
                nzgmdb_flatfile.evid == cur_event
            ].copy(True)
            cur_nzgmdb_flatfile = cur_nzgmdb_flatfile.set_index("sta")
            cur_sites = np.intersect1d(
                cur_nzgmdb_flatfile.index.values.astype(str),
                site_df.index.values.astype(str),
            )
            if cur_sites.size == 0:
                continue

            cur_data_df = site_df.loc[cur_sites].copy(True)
            cur_data_df.loc[:, "rrup"] = cur_nzgmdb_flatfile.loc[cur_sites, "r_rup"].values
            cur_data_df.loc[:, "rjb"] = cur_nzgmdb_flatfile.loc[cur_sites, "r_jb"].values
            cur_data_df.loc[:, "rx"] = cur_nzgmdb_flatfile.loc[cur_sites, "r_x"].values
            cur_data_df.loc[:, "ry"] = cur_nzgmdb_flatfile.loc[cur_sites, "r_y"].values

        cur_data_df.loc[:, "site"] = cur_data_df.index.values
        cur_data_df.loc[:, "event"] = str(cur_event)
        cur_data_df.index = np.add(f"{cur_event}_", cur_data_df.index.values)

        # Enforce distance threshold
        cur_data_df = cur_data_df.loc[cur_data_df.rjb <= rjb_max]

        if cur_data_df.shape[0] > 0:
            # Add event data
            cur_data_df = cur_data_df.merge(
                source_df[["mag", "tect_class", "z_tor", "rake", "dip", "depth"]],
                left_on="event",
                right_index=True,
            )
            data_dfs.append(cur_data_df)

    data_df = pd.concat(data_dfs, axis=0)
    data_df["vs30measured"] = False

    data_df = data_df.rename(columns={"z_tor": "ztor", "depth": "hypo_depth"})

    ### GM prediction
    print(f"Running predictions")
    dfs = []
    sites = np.unique(data_df.site)
    for cur_site in tqdm.tqdm(sites):

        cur_site_mask = data_df.site.values == cur_site

        for cur_tect_class in np.unique(data_df.loc[cur_site_mask].tect_class):
            cur_tect_mask = cur_site_mask & (data_df.tect_class == cur_tect_class)

            if cur_tect_class not in TECT_CLASS_MAPPING:
                continue

            cur_tect_type = TECT_CLASS_MAPPING[cur_tect_class]
            pga_result = oq_run(
                GMM_MAPPING[cur_tect_type],
                cur_tect_type,
                data_df.loc[cur_tect_mask, OQ_INPUT_COLUMNS],
                "PGA",
            )

            psa_result = oq_run(
                GMM_MAPPING[cur_tect_type],
                cur_tect_type,
                data_df.loc[cur_tect_mask, OQ_INPUT_COLUMNS],
                "pSA",
                constants.PERIODS,
            )

            cur_df = pd.concat((pga_result, psa_result), axis=1)
            cur_df.index = data_df.loc[cur_tect_mask].index
            cur_df[["event", "site"]] = data_df[["event", "site"]]

            dfs.append(cur_df)

    result_df = pd.concat(dfs, axis=0)
    result_df.to_csv(output_ffp, index_label="id")


def load_obs_data(obs_ffp: Path):
    """Loads the observation data from the NZ-GMDB IM flat file"""
    return pd.read_csv(obs_ffp, index_col=0, low_memory=False, dtype={"evid": str})


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


def load_correlations(data_dir: Path):
    return {
        cur_ffp.stem: SiteCorrelations.load(cur_ffp)
        for cur_ffp in data_dir.iterdir()
        if cur_ffp.is_file() and not cur_ffp.stem.startswith("_")
    }
