import os
import pickle
from pathlib import Path
from typing import Sequence, NamedTuple, Dict, List, Union
from dataclasses import dataclass

import gmhazard_calc as gc
import pandas as pd
import numpy as np


from qcore.timeseries import BBSeis, read_ascii
import ml_tools as mlt

from . import constants
from . import utils


def run_emp_gmms(
    output_ffp: Path,
    site_dir: Path,
    srf_dir: Path,
    nz_gmdb_source_ffp: Path,
    rjb_max: float,
):
    """
    Computes the empirical GMM parameters for all
        specified sites and sources

    Parameters
    ----------
    output_ffp: Path
    site_dir: Path
        Directory that contains all the site
        information files (i.e. vs30, ll, and z)
    srf_dir: Path
        Directory that contains the srf files
    nz_gmdb_source_ffp: Path
        Path to the NZ-GMDB source file
    rjb_max: float
        RJB distance threshold

    Returns
    -------
    result_df: DataFrame
        The empirical GMM parameters for PGA
        and the default set of pSA periods
    """
    from qcore import srf
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

    ### Data loading
    # Get all srf files
    srf_ffps = list(srf_dir.rglob("*.srf"))
    events = [cur_ffp.stem for cur_ffp in srf_ffps]

    # Load source info
    source_df = pd.read_csv(nz_gmdb_source_ffp, index_col=0)

    # Load srf data
    srf_points, plane_infos = {}, {}
    for cur_srf_ffp in srf_ffps:
        srf_points[cur_srf_ffp.stem] = srf.read_srf_points(str(cur_srf_ffp))
        plane_infos[cur_srf_ffp.stem] = srf.read_header(str(cur_srf_ffp), idx=True)

    # Load the site_data
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

    ### Distance calculation
    site_locs = np.concatenate(
        (site_df[["lon", "lat"]].values, np.zeros((site_df.shape[0], 1))), axis=1
    )
    data_dfs = []
    for cur_event in events:
        cur_data_df = site_df.copy(True)
        cur_data_df["rrup"], cur_data_df["rjb"] = src_site_dist.calc_rrup_rjb(
            srf_points[cur_event], site_locs
        )

        cur_data_df["rx"], cur_data_df["ry"] = src_site_dist.calc_rx_ry(
            srf_points[cur_event], plane_infos[cur_event], site_locs
        )
        # Enforce distance threshold
        cur_data_df = cur_data_df.loc[cur_data_df.rjb <= rjb_max]
        cur_data_df["site"] = cur_data_df.index.values
        cur_data_df["event"] = str(cur_event)
        cur_data_df.index = np.add(f"{cur_event}_", cur_data_df.index.values)

        # Add event data
        cur_data_df[
            ["mag", "tect_class", "ztor", "rake", "dip", "hypo_depth"]
        ] = source_df.loc[
            cur_event, ["mag", "tect_class", "z_tor", "rake", "dip", "depth"]
        ]

        data_dfs.append(cur_data_df)

    data_df = pd.concat(data_dfs, axis=0)
    data_df["vs30measured"] = False

    ### GM prediction
    dfs = []
    sites = np.unique(data_df.site)
    for site_ix, cur_site in enumerate(sites):
        print(f"Processing site {cur_site}, {site_ix + 1}/{len(sites)}")

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


class SimWithinEventSiteCorrelations(NamedTuple):

    event: str
    ims: List[str]
    sites: List[str]
    correlations: Dict[str, pd.DataFrame]

    def write(self, data_dir: Path):
        mlt.utils.write_to_yaml(
            dict(event=self.event, ims=self.ims, sites=self.sites),
            data_dir / "meta.yaml",
        )
        for cur_im, cur_df in self.correlations.items():
            cur_df.to_csv(data_dir / f"{cur_im.replace('.', 'p')}.csv")

    @classmethod
    def load(cls, data_dir: Path):
        meta = mlt.utils.load_yaml(data_dir / "meta.yaml")
        correlations = {}
        for cur_ffp in data_dir.iterdir():
            correlations[utils.reverse_im_filename(cur_ffp.stem)] = pd.read_csv(
                cur_ffp, index_col=0
            )
        return cls(meta["event"], meta["ims"], meta["sites"], correlations)


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


def compute_sim_site_corrs(sim_params_dir: Path, smooth: bool = False):
    """Computes the site correlations based on the simulation data"""
    events = [
        cur_ffp.stem
        for cur_ffp in sim_params_dir.iterdir()
        if cur_ffp.is_dir() and not cur_ffp.stem.startswith("_")
    ]

    results = []
    for cur_event in events:
        cur_sim_gm_params = SimGMParams.load(sim_params_dir / cur_event)
        cur_ims = np.asarray(cur_sim_gm_params.ims)
        cur_sites = np.asarray(cur_sim_gm_params.sites)

        cur_site_corrs = np.full(
            (len(cur_sites), len(cur_sites), len(cur_ims)), fill_value=np.nan
        )

        for i, cur_im in enumerate(cur_ims):
            cur_within_residuals = cur_sim_gm_params.within_residuals[
                [cur_im, "site", "rel"]
            ]
            cur_within_residuals = cur_within_residuals.pivot(
                index="rel", columns="site", values=cur_im
            )

            cur_corrs = cur_within_residuals.loc[:, cur_sites].corr(method="pearson")
            assert np.all(cur_corrs.index.values.astype(str) == cur_sites)

            cur_site_corrs[:, :, i] = cur_corrs.values

        if smooth:
            # If this is ever changed, ensure that the
            # boundaries are handled correctly
            smoothing_kernel = np.array([1 / 3, 1 / 3, 1 / 3])

            # Get the indices of the pSA IMs
            pSA_keys = cur_ims[np.char.startswith(cur_ims, "pSA")]
            periods, pSA_keys = utils.get_periods(pSA_keys)
            pSA_ind = [
                np.flatnonzero(cur_pSA_key == cur_ims)[0] for cur_pSA_key in pSA_keys
            ]

            # This is sub-optimal
            # Check https://stackoverflow.com/questions/62442341/vectorizing-2d-convolutions-in-numpy
            # specifically the last answer
            for i in range(len(cur_sites)):
                for j in range(len(cur_sites)):
                    # Run the convolution
                    cur_corrs = np.convolve(
                        cur_site_corrs[i, j, pSA_ind], smoothing_kernel, mode="same"
                    )

                    # Deal with incorrect boundaries values
                    cur_corrs[0] = cur_site_corrs[i, j, pSA_ind[0]]
                    cur_corrs[-1] = cur_site_corrs[i, j, pSA_ind[-1]]

                    # Save
                    cur_site_corrs[i, j, pSA_ind] = cur_corrs

        results.append(
            SiteCorrelations(
                cur_site_corrs, np.asarray(cur_sites), np.asarray(cur_ims), cur_event
            )
        )

        # correlations = {}
        # for cur_im in sim_gm_params.ims:
        #     cur_within_residuals = sim_gm_params.within_residuals[
        #         [cur_im, "site", "rel"]
        #     ]
        #     cur_within_residuals = cur_within_residuals.pivot(
        #         index="rel", columns="site", values=cur_im
        #     )
        #
        #     correlations[cur_im] = cur_within_residuals.corr(method="pearson")
        #
        # results.append(
        #     SimWithinEventSiteCorrelations(
        #         cur_event, sim_gm_params.ims, sim_gm_params.sites, correlations
        #     )
        # )

    return results


class SimGMParams(NamedTuple):

    event: str
    ims: List[str]
    sites: List[str]
    gm_params: pd.DataFrame
    residuals: Union[pd.DataFrame, None] = None
    event_residuals: Union[pd.DataFrame, None] = None
    within_residuals: Union[pd.DataFrame, None] = None
    bias_std: Union[pd.DataFrame, None] = None

    def write(self, data_dir: Path):
        mlt.utils.write_to_yaml(
            dict(event=self.event, ims=self.ims, sites=self.sites),
            data_dir / "meta.yaml",
        )

        self.gm_params.to_csv(data_dir / "gm_params.csv")

        if self.residuals is not None:
            self.residuals.to_csv(data_dir / "residuals.csv")

        if self.event_residuals is not None:
            self.event_residuals.to_csv(data_dir / "event_residuals.csv")

        if self.within_residuals is not None:
            self.within_residuals.to_csv(data_dir / "rem_residuals.csv")

        if self.bias_std is not None:
            self.bias_std.to_csv(data_dir / "bias_std.csv")

    @classmethod
    def load(cls, data_dir: Path):
        meta = mlt.utils.load_yaml(data_dir / "meta.yaml")
        return cls(
            meta["event"],
            meta["ims"],
            meta["sites"],
            pd.read_csv(data_dir / "gm_params.csv", index_col=0),
            pd.read_csv(cur_path, index_col=0)
            if (cur_path := data_dir / "residuals.csv").exists()
            else None,
            pd.read_csv(cur_path, index_col=0)
            if (cur_path := data_dir / "event_residuals.csv").exists()
            else None,
            pd.read_csv(cur_path, index_col=0)
            if (cur_path := data_dir / "rem_residuals.csv").exists()
            else None,
            pd.read_csv(cur_path, index_col=0)
            if (cur_path := data_dir / "bias_std.csv").exists()
            else None,
        )


def compute_sim_gm_params_total(simulation_imdb_ffp: Path):
    """
    Computes the parametric IM distribution based
    on the simulation data.
    Does not use MERA, i.e. uses total residual only
    """
    # Get the simulation data
    sim_data = load_site_sim_data(simulation_imdb_ffp)

    sites = list(sim_data.keys())
    events = np.unique(sim_data[sites[0]].index.get_level_values(0))
    ims = [str(cur_im) for cur_im in sim_data[sites[0]].columns.values.astype(str)]

    results = []
    for cur_event in events:
        print(f"Processing event {cur_event}")
        gm_params = {}
        residual_df = []
        sites = []
        for cur_site, cur_sim_data in sim_data.items():
            # No simulation data for the current site
            if cur_event not in cur_sim_data.index:
                continue
            sites.append(cur_site)

            # Get the simulation IM data
            cur_sim_data = cur_sim_data.loc[cur_event]

            # Compute the log mean
            cur_mean = np.log(cur_sim_data[ims]).mean(axis=0)

            # Compute the residual
            cur_residual = np.log(cur_sim_data[ims].values) - cur_mean.values
            cur_residual = pd.DataFrame(
                index=np.char.add(
                    cur_sim_data.index.values.astype(str), f"_{cur_site}"
                ),
                columns=ims,
                data=cur_residual,
            )
            cur_sigma_total = cur_residual.std(axis=0)

            cur_residual["rel"] = [
                cur_i.split("_")[1] for cur_i in cur_residual.index.values.astype(str)
            ]
            cur_residual["site"] = cur_site
            residual_df.append(cur_residual)

            # Put GM params in correct format
            cur_mean.index = np.char.add(cur_mean.index.values.astype(str), "_mean")
            cur_phi = cur_sigma_total.copy()
            cur_phi.index = np.char.add(cur_phi.index.values.astype(str), "_std_Intra")

            cur_tau = cur_sigma_total.copy()
            cur_tau.loc[:] = 0.0
            cur_tau.index = np.char.add(cur_tau.index.values.astype(str), "_std_Inter")

            cur_sigma_total.index = np.char.add(
                cur_sigma_total.index.values.astype(str), "_std_Total"
            )

            gm_params[cur_site] = {
                **cur_mean.to_dict(),
                **cur_tau.to_dict(),
                **cur_phi.to_dict(),
                **cur_sigma_total.to_dict(),
            }

        # Combine residuals
        residual_df = pd.concat(residual_df)

        gm_params = pd.DataFrame(gm_params).T
        results.append(
            SimGMParams(
                cur_event,
                ims,
                sites,
                gm_params,
                residual_df,
                None,
                residual_df,
                None,
            )
        )

    return results


def compute_sim_gm_params_mera(simulation_imdb_ffp: Path):
    """
    Computes the parametric IM distributions based
    on the simulation data using mixed effects regression
    """
    from mera.mera_pymer4 import run_mera

    # Get the simulation data
    sim_data = load_site_sim_data(simulation_imdb_ffp)

    sites = list(sim_data.keys())
    events = np.unique(sim_data[sites[0]].index.get_level_values(0))
    ims = [str(cur_im) for cur_im in sim_data[sites[0]].columns.values.astype(str)]

    results = []
    for cur_event in events:
        print(f"Processing event {cur_event}")
        gm_params = {}
        residual_df = []
        sites = []
        for cur_site, cur_sim_data in sim_data.items():
            # No simulation data for the current site
            if cur_event not in cur_sim_data.index:
                continue
            sites.append(cur_site)

            # Get the simulation IM data
            cur_sim_data = cur_sim_data.loc[cur_event]

            # Compute the log mean
            cur_mean = np.log(cur_sim_data[ims]).mean(axis=0)
            cur_mean.index = np.char.add(cur_mean.index.values.astype(str), "_mean")

            # Compute the residual
            cur_residual = np.log(cur_sim_data[ims].values) - cur_mean.values
            cur_residual = pd.DataFrame(
                index=np.char.add(
                    cur_sim_data.index.values.astype(str), f"_{cur_site}"
                ),
                columns=ims,
                data=cur_residual,
            )
            cur_sigma_total = cur_residual.std(axis=0)
            cur_residual["rel"] = [
                cur_i.split("_")[1] for cur_i in cur_residual.index.values.astype(str)
            ]
            cur_residual["site"] = cur_site

            gm_params[cur_site] = (cur_mean, cur_sigma_total)
            residual_df.append(cur_residual)

        # Combine
        residual_df = pd.concat(residual_df)

        # Run the mixed-effects regression
        # This treats each realisation as an event
        event_res_df, within_res, bias_std_df = run_mera(
            residual_df, ims, "rel", "site", compute_site_term=False
        )

        # Add site and rel column to within event residuals
        assert np.all(within_res.index == residual_df.index)
        within_res["site"] = residual_df["site"]
        within_res["rel"] = residual_df["rel"]

        # Get the GM params
        for cur_site, (cur_mean, cur_sigma_total) in gm_params.items():
            # Use tau from ME-regression
            tau = bias_std_df.tau.copy()

            assert np.all(tau.index == cur_sigma_total.index)

            # Want site-specific phi, so compute it as
            # sqrt(sigma_total^2 - tau^2), where sigma_total is
            # computed from the realisations
            ## TODO: How to handle negative values??
            phi = np.sqrt(np.abs(cur_sigma_total ** 2 - tau ** 2))

            # Update the indices
            tau.index = np.char.add(tau.index.values.astype(str), "_std_Inter")
            phi.index = np.char.add(phi.index.values.astype(str), "_std_Intra")

            sigma = cur_sigma_total
            sigma.index = np.char.add(sigma.index.values.astype(str), "_std_Total")

            gm_params[cur_site] = {
                **cur_mean.to_dict(),
                **tau.to_dict(),
                **phi.to_dict(),
                **sigma.to_dict(),
            }

        gm_params = pd.DataFrame(gm_params).T
        results.append(
            SimGMParams(
                cur_event,
                ims,
                sites,
                gm_params,
                residual_df,
                event_res_df,
                within_res,
                bias_std_df,
            )
        )

    return results


def load_site_sim_data(sim_imdb_ffp: Path, sites: Sequence[str] = None, event: str = None):
    """Loads the simulation IM values for the specified sites"""
    sim_data = {}
    with gc.dbs.IMDB.get_imdb(str(sim_imdb_ffp)) as db:
        sites = sites if sites is not None else db.get_stored_stations()

        for cur_site in sites:
            if (cur_im_df := db.im_data(cur_site)) is not None:
                if event is not None:
                    # Not data for this event/site combination
                    if event not in cur_im_df.index:
                        continue

                    cur_im_df = cur_im_df.loc[event]

                sim_data[cur_site] = cur_im_df

    return sim_data

def load_avail_sim_events(sim_imdb_ffp: Path):
    """Loads the available simulations in the specified IMDB"""
    with gc.dbs.IMDB.get_imdb(str(sim_imdb_ffp)) as db:
        return db.rupture_names()

def load_obs_data(obs_ffp: Path):
    """Loads the observation data from the NZ-GMDB IM flat file"""
    return pd.read_csv(obs_ffp, index_col=0, low_memory=False, dtype={"evid": str})

def get_obs_rupture_data(obs_df: pd.DataFrame, rupture: str):
    """
    Loads the observation data for the specified
    data from the NZ-GMDB IM flat file
    """
    obs_df = obs_df.loc[obs_df.evid == rupture]
    obs_df = obs_df.set_index("sta").sort_index()

    return obs_df


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


# def load_correlations(data_dir: Path):
#     return {
#         utils.reverse_im_filename(cur_ffp.stem): pd.read_csv(cur_ffp, index_col=0)
#         for cur_ffp in data_dir.iterdir()
#         if cur_ffp.is_file()
#     }


def load_correlations(data_dir: Path):
    return {
        cur_ffp.stem: SiteCorrelations.load(cur_ffp)
        for cur_ffp in data_dir.iterdir()
        if cur_ffp.is_file() and not cur_ffp.stem.startswith("_")
    }


def load_ll_file(ffp: Path):
    return pd.read_csv(ffp, sep=" ", index_col=2, header=None, names=["lon", "lat"])


def load_vs30_file(ffp: Path):
    return pd.read_csv(ffp, sep=" ", index_col=0, header=None, names=["vs30"])


def load_sim_gm_params(data_dir: Path):
    return SimGMParams.load(data_dir)


def load_emp_gm_params(gm_params_ffp: Path, event: str):
    gm_params = pd.read_csv(gm_params_ffp, index_col=0)

    gm_params.event = gm_params.event.values.astype(str)
    gm_params = gm_params.loc[gm_params.event == event]
    gm_params = gm_params.set_index("site")
    return gm_params


def get_method_type(results_dir: Path):
    return constants.RankingMethod(get_meta(results_dir)["method_type"])


def get_meta(results_dir: Path):
    meta = mlt.utils.load_yaml(results_dir / "meta.yaml")
    meta = {
        key: os.path.expandvars(val)
        if isinstance(val, str) and (key.endswith("_ffp") or key.endswith("_dir"))
        else val
        for key, val in meta.items()
    }
    return meta


def get_gm_params(results_dir: Path):
    method_type = get_method_type(results_dir)
    meta = get_meta(results_dir)
    if method_type is constants.RankingMethod.emp_cMVN:
        return load_emp_gm_params(meta["gm_params_ffp"], meta["rupture"])
    else:
        sim_gm_params = load_sim_gm_params(Path(meta["sim_gm_params_dir"]))
        return sim_gm_params.gm_params
