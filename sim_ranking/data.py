import os
import pickle
import warnings
import multiprocessing as mp
from pathlib import Path
from typing import Sequence, NamedTuple, Dict, List, Union
from dataclasses import dataclass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tqdm
from qcore.timeseries import BBSeis, read_ascii
import ml_tools as mlt
import spatial_hazard as sh
import sha_calc as sha

from . import constants
from . import conditional
from .db import DB


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
            dict(event=str(self.event), ims=self.ims, sites=self.sites),
            data_dir / "meta.yaml",
        )

        self.gm_params.to_csv(data_dir / "gm_params.csv")

        if self.residuals is not None:
            self.residuals.to_parquet(data_dir / "residuals.parquet")

        if self.event_residuals is not None:
            self.event_residuals.to_csv(data_dir / "event_residuals.csv")

        if self.within_residuals is not None:
            self.within_residuals.to_parquet(data_dir / "rem_residuals.parquet")

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
            pd.read_parquet(cur_path)
            if (cur_path := data_dir / "residuals.parquet").exists()
            else None,
            pd.read_csv(cur_path, index_col=0)
            if (cur_path := data_dir / "event_residuals.csv").exists()
            else None,
            pd.read_parquet(cur_path)
            if (cur_path := data_dir / "rem_residuals.parquet").exists()
            else None,
            pd.read_csv(cur_path, index_col=0)
            if (cur_path := data_dir / "bias_std.csv").exists()
            else None,
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
    gm_params.loc[:, mean_cols] = (
        gm_params.loc[:, mean_cols].values
        + 0.5 * 0.6
    )

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

    ### Distance calculation
    site_locs = np.concatenate(
        (site_df[["lon", "lat"]].values, np.zeros((site_df.shape[0], 1))), axis=1
    )
    data_dfs = []
    for cur_event in events:
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

            cur_data_df = site_df.loc[cur_sites].copy(True)
            cur_data_df["rrup"] = cur_nzgmdb_flatfile.loc[cur_sites, "r_rup"].values
            cur_data_df["rjb"] = cur_nzgmdb_flatfile.loc[cur_sites, "r_jb"].values
            cur_data_df["rx"] = cur_nzgmdb_flatfile.loc[cur_sites, "r_x"].values
            cur_data_df["ry"] = cur_nzgmdb_flatfile.loc[cur_sites, "r_y"].values

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

        if cur_data_df.shape[0] > 0:
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


def compute_sim_site_corrs(sim_params_dir: Path):
    """Generates site-correlations using all available simulations"""
    events = [
        cur_ffp.stem
        for cur_ffp in sim_params_dir.iterdir()
        if cur_ffp.is_dir() and not cur_ffp.stem.startswith("_")
    ]

    im_dfs = None
    print(f"Combining event GM parameters")
    for ix, cur_event in enumerate(tqdm.tqdm(events)):
        cur_sim_gm_params = SimGMParams.load(sim_params_dir / cur_event)
        cur_within_residuals = cur_sim_gm_params.within_residuals

        if ix == 0:
            im_dfs = {cur_im: [] for cur_im in cur_sim_gm_params.ims}

        for cur_im in cur_sim_gm_params.ims:
            cur_df = cur_within_residuals[[cur_im, "site", "rel"]].copy()
            cur_df["sim_id"] = np.char.add(
                f"{cur_event}_", cur_df.rel.values.astype(str)
            )
            cur_df = cur_df.drop(columns=["rel"])

            im_dfs[cur_im].append(cur_df)

    corr_dfs = {}
    print(f"Computing correlations for each IM")
    for cur_im, cur_dfs in tqdm.tqdm(im_dfs.items()):
        cur_df = pd.concat(cur_dfs, axis=0)
        cur_df = cur_df.pivot(index="sim_id", columns="site", values=cur_im)
        cur_corr_df = cur_df.corr(method="pearson")

        cur_corr_df = pd.melt(
            cur_corr_df.reset_index(),
            id_vars=["site"],
            var_name="site_2",
            value_name="corr",
        )
        cur_corr_df = cur_corr_df.rename(columns={"site": "site_1"}).dropna()
        cur_corr_df.index = mlt.array_utils.numpy_str_join(
            "_",
            cur_corr_df.site_1.values.astype(str),
            cur_corr_df.site_2.values.astype(str),
        )

        corr_dfs[cur_im] = cur_corr_df

    return corr_dfs


def compute_event_site_corrs_from_rels(sim_params_dir: Path):
    """
    Computes the site correlations for each event using the
    within-event residuals from the realisations
    """
    events = [
        cur_ffp.stem
        for cur_ffp in sim_params_dir.iterdir()
        if cur_ffp.is_dir() and not cur_ffp.stem.startswith("_")
    ]

    results = []
    for ix, cur_event in enumerate(tqdm.tqdm(events)):
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

        results.append(
            SiteCorrelations(
                cur_site_corrs, np.asarray(cur_sites), np.asarray(cur_ims), cur_event
            )
        )

    return results


def compute_event_gm_params_rel_total(
    db_ffp: Path, ims: List[str], data_source: str = None
):
    """
    Computes the parametric IM distribution based
    on the simulation data.
    Does not use MERA, i.e. uses total residual only
    """
    db = DB(db_ffp)
    events = db.get_avail_events(data_source)
    sites = db.get_avail_sites().tolist()

    results = []
    for ix, cur_event in enumerate(events):
        print(f"Processing event {cur_event}, {ix + 1}/{len(events)}")
        # Get the simulation data
        sim_data = db.get_sim_data(cur_event, sites)

        gm_params = {}
        residual_df = []
        for cur_site, cur_sim_data in sim_data.groupby("site_id"):
            # Compute the log mean
            cur_mean = np.log(cur_sim_data[ims]).mean(axis=0)

            # Compute the residual
            cur_residual = np.log(cur_sim_data[ims].values) - cur_mean.values
            cur_residual = pd.DataFrame(
                index=cur_sim_data.index,
                columns=ims,
                data=cur_residual,
            )
            cur_sigma_total = cur_residual.std(axis=0)

            cur_residual["rel"] = [
                cur_i.split("_")[-1] for cur_i in cur_residual.index.values.astype(str)
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
                sim_data.site_id.unique().astype(str).tolist(),
                gm_params,
                residual_df,
                None,
                residual_df,
                None,
            )
        )

    return results


def _process_sim_gm_params_mera_event(event: str, db_ffp: Path, ims: List[str]):
    from mera.mera_pymer4 import run_mera

    print(f"Processing event {event}")

    db = DB(db_ffp)
    avail_sites = db.get_avail_sites()

    gm_params = {}
    residual_df = []
    sites = []

    ### Compute the residuals
    cur_sim_data = db.get_sim_data(event, avail_sites)
    cur_sim_data[ims] = np.log(cur_sim_data[ims])

    site_mean_values = cur_sim_data[["site_id"] + ims].groupby("site_id").mean()
    site_mean_values.columns = np.char.add(
        site_mean_values.columns.values.astype(str), "_mean"
    )
    for cur_site in cur_sim_data.site_id.unique().astype(str):
        sites.append(str(cur_site))

        # Compute the residual
        cur_mask = cur_sim_data.site_id.values == cur_site
        cur_residual = (
            cur_sim_data.loc[cur_mask, ims].values
            - site_mean_values.loc[cur_site].values
        )
        cur_residual = pd.DataFrame(
            index=cur_sim_data.loc[cur_mask].index,
            columns=ims,
            data=cur_residual,
        )
        cur_sigma_total = cur_residual.std(axis=0)
        cur_residual["rel"] = cur_sim_data.rel_id
        cur_residual["site"] = cur_site

        gm_params[cur_site] = (site_mean_values.loc[cur_site], cur_sigma_total)
        residual_df.append(cur_residual)

    # Combine
    residual_df = pd.concat(residual_df)

    # Run the mixed-effects regression
    # This treats each realisation as an event
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        event_res_df, within_res, bias_std_df = run_mera(
            residual_df, ims, "rel", "site", compute_site_term=False, verbose=False
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
    return SimGMParams(
        event,
        ims,
        sites,
        gm_params,
        residual_df,
        event_res_df,
        within_res,
        bias_std_df,
    )


def compute_event_gm_params_rel_mera(
    db_ffp: Path, ims: List[str], data_source: str = None, n_procs: int = 1
):
    """
    Computes the parametric IM distributions based
    on the simulation data using mixed effects regression
    """
    db = DB(db_ffp)
    events = db.get_avail_events(data_source=data_source)

    if n_procs > 1:
        with mp.Pool(n_procs) as p:
            results = p.starmap(
                _process_sim_gm_params_mera_event,
                [(cur_event, db_ffp, ims) for cur_event in events],
            )
    else:
        results = []
        for cur_event in events:
            results.append(_process_sim_gm_params_mera_event(cur_event, db_ffp, ims))

    return results


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


def get_overlap_emp_ml_data(emp_results_dir: Path, ml_sc_sum_df: pd.DataFrame):
    """
    Loads & combines the conditional mean & standard deviation data for
    the overlapping scenarios for the given ML scenario summary dataframe

    Parameters
    ----------
    emp_results_dir: Path
    ml_sc_sum_df: DataFrame

    Returns
    -------
    emp_mean_df: DataFrame
    emp_std_df: DataFrame
        Mean & standard deviation dataframes
        for the empirical conditional IM distributions
    ml_sc_sum_df: DataFrame
        Updated ML scenario summary dataframe
    """
    events = ml_sc_sum_df["event_id"].unique()

    emp_mean_df, emp_std_df = [], []
    for cur_event in events:
        cur_cim = conditional.load_emp_cim_data(
            emp_results_dir, cur_event, constants.RankingMethod.emp_cMVN
        )
        if cur_cim is not None:
            assert cur_cim.cond_lnIM_mean_df.index.equals(
                cur_cim.cond_lnIM_std_df.index
            ), f"Index mismatch for {cur_event}"
            cur_index = mlt.array_utils.numpy_str_join("_", cur_event, cur_cim.cond_lnIM_mean_df.index)
            emp_mean_df.append(cur_cim.cond_lnIM_mean_df.set_index(cur_index))
            emp_std_df.append(cur_cim.cond_lnIM_std_df.set_index(cur_index))

    emp_mean_df = pd.concat(emp_mean_df, axis=0).sort_index()
    emp_std_df = pd.concat(emp_std_df, axis=0).sort_index()

    # Get overlapping scenarios
    ml_in_emp = np.isin(ml_sc_sum_df.index.values.astype(str), emp_mean_df.index.values.astype(str))
    print(f"Overlapping scenarios: {np.sum(ml_in_emp)}/{ml_sc_sum_df.shape[0]}")

    ids = ml_sc_sum_df.index[ml_in_emp]

    return emp_mean_df.loc[ids], emp_std_df.loc[ids], ml_sc_sum_df.loc[ids]