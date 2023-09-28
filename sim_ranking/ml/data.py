from typing import Dict, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from . import similarity_score as ss
from .. import db


@dataclass
class ScalarFeatures:
    site_features_data: pd.DataFrame
    site_feature_keys: Sequence[str]
    site_to_site_features_data: Dict[str, pd.DataFrame]
    site_to_site_feature_keys: Sequence[str]
    event_site_features_data: Dict[str, pd.DataFrame]
    event_site_feature_keys: Sequence[str]
    event_site_to_site_features_data: Dict[str, Dict[str, pd.DataFrame]]
    event_site_to_site_feature_keys: Sequence[str]

    def __post_init__(self):
        self.n_scalar_features = (
            len(self.site_feature_keys) * 2
            + len(self.site_to_site_feature_keys)
            + len(self.event_site_feature_keys) * 2
            + len(self.event_site_to_site_feature_keys)
        )


def compute_site_combinations(
    sites: Dict[str, np.ndarray],
    events: Sequence[str],
    dist_matrix: pd.DataFrame,
    sites_to_use: np.ndarray = None,
    max_dist: float = 100,
):
    """
    Compute the site combinations for each event

    Returns allowed sites and site-combination
    indices (for allowed sites) for each event
    """
    site_combs, used_sites = {}, {}
    for cur_event in events:
        cur_sites = sites[cur_event]
        cur_sites = (
            cur_sites
            if sites_to_use is None
            else cur_sites[np.isin(cur_sites, sites_to_use)]
        )

        # Need at least two sites for the event
        if len(cur_sites) < 2:
            continue

        # Filter for the current event sites
        # and site-combinations less than max_dist km apart
        cur_dist_matrix = dist_matrix.loc[cur_sites, cur_sites]
        cur_dist_mask = (cur_dist_matrix.values < max_dist) & (
            cur_dist_matrix.values > 0
        )
        cur_row_ind, cur_col_ind = np.nonzero(cur_dist_mask)

        # Need at least one site combination within the
        # specified distance requirements
        if cur_row_ind.size == 0:
            continue

        # Get the site combinations
        # First is the site of interest, second is the observation site
        # Indices into the sites to use for the current event
        cur_site_combs = np.stack((cur_row_ind, cur_col_ind), axis=1)

        site_combs[cur_event] = cur_site_combs
        used_sites[cur_event] = cur_sites

    return site_combs, used_sites


def _get_event_sim_obs_pSA_data(
    event_sites: Dict[str, np.ndarray],
    site_combs: Dict[str, np.ndarray],
    db: db.DB,
    pSA_keys: np.ndarray,
    n_rels: int,
):
    """
    Gets the simulation and observation pSA data
    for each event, in addition to the number of
    samples per event and the realisations used
    """
    sim_im_dfs, obs_im_dfs = {}, {}
    n_samples_event, rels = [], {}
    for cur_event, cur_sites in event_sites.items():
        cur_sim_data = db.get_sim_data(cur_event, cur_sites)
        # Only use a subset of the available realisations to
        # prevent over-fitting to these events
        if np.any(cur_sim_data.data_source == "specific"):
            rels[cur_event] = np.random.choice(
                cur_sim_data.rel_id.unique(), n_rels, replace=False
            )
            cur_mask = (cur_sim_data.data_source.values == "specific") & np.isin(
                cur_sim_data.rel_id.values, rels[cur_event]
            )
            cur_sim_data = cur_sim_data.loc[cur_mask]

            n_samples_event.append(site_combs[cur_event].shape[0] * n_rels)
        else:
            rels[cur_event] = None
            n_samples_event.append(site_combs[cur_event].shape[0])

        # Get observation data
        cur_obs_data = db.get_obs_data(cur_event, cur_sites)

        # Sanity checks
        assert np.all(cur_obs_data.columns == pSA_keys)
        assert (
            cur_sim_data.shape[0] == cur_obs_data.shape[0]
            or cur_sim_data.shape[0] == cur_obs_data.shape[0] * n_rels
        )

        sim_im_dfs[cur_event] = cur_sim_data
        obs_im_dfs[cur_event] = cur_obs_data

    return sim_im_dfs, obs_im_dfs, n_samples_event, rels


def _get_event_n_sampels(n_samples_event: Sequence[int], rels: Dict[str, np.ndarray]):
    """Computes the number of samples per event"""
    n_samples_event = np.asarray(n_samples_event)
    cum_n_samples = np.cumsum(n_samples_event)
    n_rels_used = {
        cur_event: 1 if cur_rels is None else cur_rels.size
        for cur_event, cur_rels in rels.items()
    }

    return n_samples_event, cum_n_samples, n_rels_used


def _station_df_sanity_check(station_df: pd.DataFrame, site_features: Sequence[str]):
    """Checks that the station_df has been normalised"""
    assert all(
        [np.isclose(station_df[cur_feature].mean(), 0) for cur_feature in site_features]
    )
    assert all(
        [np.isclose(station_df[cur_feature].std(), 1) for cur_feature in site_features]
    )


class BaseDataset(Dataset):
    def __init__(
        self,
        event_sites: Dict[str, np.ndarray],
        site_combs: Dict[str, np.ndarray],
        db: db.DB,
        n_rels: int,
        station_df: pd.DataFrame,
        periods: np.ndarray,
        pSA_keys: np.ndarray,
        scalar_features: ScalarFeatures,
    ):
        self.site_combs = site_combs

        self.station_df = station_df
        self.scalar_features = scalar_features

        self.event_sites = event_sites
        self.events = np.asarray(list(self.event_sites.keys()))
        self.n_rels = n_rels

        # Get all relevant sites for this dataset
        self.sites = np.asarray(
            list(
                set().union(
                    *[list(cur_sites) for cur_sites in self.event_sites.values()]
                )
            )
        )
        self.sites_ix_lookup = {cur_site: ix for ix, cur_site in enumerate(self.sites)}

        self.pSA_keys = pSA_keys
        self.periods = periods

        # Get the simulation and observation data
        (
            self.sim_im_dfs,
            self.obs_im_dfs,
            self.n_samples_event,
            self.rels,
        ) = _get_event_sim_obs_pSA_data(
            self.event_sites, self.site_combs, db, self.pSA_keys, self.n_rels
        )

        # Compute the number of samples per event
        (
            self.n_samples_event,
            self._cum_n_samples,
            self.n_rels_used,
        ) = _get_event_n_sampels(self.n_samples_event, self.rels)

        # Create feature matrix for all site combinations
        # of shape [n_sites, n_sites, n_features]
        # per event, where i/axis 0 = site of interest
        # and j/axis 1 = observation site
        # Order of the features is:
        # 1) Site of interest - site features
        # 2) Observation site - site features
        # 3) Site of interest - event site features
        # 4) Observation site - event site features
        # 5) Site to site features
        # 6) Event site to site features
        self.scalar_features_tensors = {}
        for cur_event in self.events:
            cur_sites = self.event_sites[cur_event]
            cur_tensor = np.full(
                (
                    cur_sites.size,
                    cur_sites.size,
                    self.scalar_features.n_scalar_features,
                ),
                fill_value=np.nan,
            )

            # Set the site features
            n_site_features = len(self.scalar_features.site_feature_keys)
            for i, feature_i in enumerate(self.scalar_features.site_feature_keys):
                for j, site_j in enumerate(cur_sites):
                    # Site of interest/observation site
                    cur_tensor[j, :, i] = cur_tensor[
                        :, j, i + n_site_features
                    ] = self.scalar_features.site_features_data.loc[site_j, feature_i]
            # Set the event site features
            cur_f_ix = n_site_features * 2
            n_event_site_features = len(self.scalar_features.event_site_feature_keys)
            for i, feature_i in enumerate(self.scalar_features.event_site_feature_keys):
                for j, site_j in enumerate(cur_sites):
                    # Site of interest/observation site
                    cur_tensor[j, :, cur_f_ix + i] = cur_tensor[
                        :, j, cur_f_ix + i + n_event_site_features
                    ] = self.scalar_features.event_site_features_data[cur_event].loc[
                        site_j, feature_i
                    ]
            # Set the site to site features
            cur_f_ix += n_event_site_features * 2
            for i, feature_i in enumerate(
                self.scalar_features.site_to_site_feature_keys
            ):
                cur_tensor[
                    :, :, cur_f_ix + i
                ] = self.scalar_features.site_to_site_features_data[feature_i].loc[
                    cur_sites, cur_sites
                ]
            # Set the event site to site features
            cur_f_ix += len(self.scalar_features.site_to_site_feature_keys)
            for i, feature_i in enumerate(
                self.scalar_features.event_site_to_site_feature_keys
            ):
                cur_tensor[
                    :, :, cur_f_ix + i
                ] = self.scalar_features.event_site_to_site_features_data[feature_i][
                    cur_event
                ].loc[
                    cur_sites, cur_sites
                ]

            self.scalar_features_tensors[cur_event] = cur_tensor

    @property
    def n_samples(self):
        return self._cum_n_samples[-1]

    def get_metadata(self, idx: int):
        """Get the metadata for a specific sample"""
        event, event_ix, site_ix, rel_ix = self.get_indices(idx)

        # Get the site of interest and observation site
        site_int_ix = self.site_combs[event][site_ix, 0]
        site_obs_ix = self.site_combs[event][site_ix, 1]

        site_int = self.event_sites[event][site_int_ix]
        site_obs = self.event_sites[event][site_obs_ix]
        rel = "NA" if self.rels[event] is None else self.rels[event][rel_ix]

        return (
            event,
            rel,
            site_int,
            site_obs,
        )

    def __len__(self):
        return self.n_samples

    def get_indices(self, idx: int):
        # Have to it this way, as some events may not have samples
        event_ix = np.flatnonzero(idx - self._cum_n_samples < 0)[0]

        event = self.events[event_ix]
        n_rels = self.n_rels_used[event]

        site_ix = (idx - self._cum_n_samples[max(event_ix - 1, 0)]) // n_rels
        rel_ix = idx % n_rels

        return event, event_ix, site_ix, rel_ix

    def __getitem__(self, idx: int):
        raise NotImplementedError()


class ResponseSpectrumResidualDataset(BaseDataset):
    def __init__(
        self,
        event_sites: Dict[str, np.ndarray],
        site_combs: Dict[str, np.ndarray],
        db: db.DB,
        n_rels: int,
        station_df: pd.DataFrame,
        periods: np.ndarray,
        pSA_keys: np.ndarray,
        scalar_features: ScalarFeatures,
    ):
        # Base Class
        super().__init__(
            event_sites,
            site_combs,
            db,
            n_rels,
            station_df,
            periods,
            pSA_keys,
            scalar_features,
        )

        # Compute the residuals
        # and organize into the format (per event):
        # [n_sites, n_sites, n_periods, n_rels]
        # where the last dimension corresponds to the
        self.obs_sim_residuals = {}
        self.sim_sim_residuals = {}
        for cur_event in self.events:
            cur_sites = self.event_sites[cur_event]
            cur_sim_df = self.sim_im_dfs[cur_event]
            cur_obs_df = self.obs_im_dfs[cur_event]

            n_rels = 1 if self.rels[cur_event] is None else len(self.rels[cur_event])
            cur_obs_sim_residuals = np.full(
                (cur_sites.size, cur_sites.size, len(self.pSA_keys), n_rels),
                fill_value=np.nan,
                dtype=float,
            )
            cur_sim_sim_residuals = cur_obs_sim_residuals.copy()

            for rel_ix in range(n_rels):
                cur_rel_sim_df = (
                    cur_sim_df.set_index("site_id")
                    if n_rels == 1
                    else cur_sim_df.loc[
                        cur_sim_df.rel_id == self.rels[cur_event][rel_ix]
                    ].set_index("site_id")
                )

                # Have to do it per IM as .outer does not support
                # an axis argument
                for im_ix, im in enumerate(self.pSA_keys):
                    # Obs - Sim
                    cur_obs_sim_residuals[:, :, im_ix, rel_ix] = np.subtract.outer(
                        np.log(cur_obs_df.loc[cur_sites, im].values),
                        np.log(cur_rel_sim_df.loc[cur_sites, im].values),
                    )
                    # Sim - Sim
                    cur_sim_sim_residuals[:, :, im_ix, rel_ix] = np.subtract.outer(
                        np.log(cur_rel_sim_df.loc[cur_sites, im].values),
                        np.log(cur_rel_sim_df.loc[cur_sites, im].values),
                    )

            assert ~np.any(np.isnan(cur_sim_sim_residuals))
            assert ~np.any(np.isnan(cur_obs_sim_residuals))

            self.obs_sim_residuals[cur_event] = cur_obs_sim_residuals
            self.sim_sim_residuals[cur_event] = cur_sim_sim_residuals

    def __getitem__(self, idx: int):
        # Break the index down
        event, event_ix, site_ix, rel_ix = self.get_indices(idx)

        # Get the site of interest and observation site
        site_int_ix = self.site_combs[event][site_ix, 0]
        site_obs_ix = self.site_combs[event][site_ix, 1]

        site_int = self.event_sites[event][site_int_ix]
        site_obs = self.event_sites[event][site_obs_ix]

        # Features
        obs_obs_obs_sim_res = self.obs_sim_residuals[event][
            site_obs_ix, site_obs_ix, :, rel_ix
        ]
        obs_obs_int_sim_res = self.obs_sim_residuals[event][
            site_obs_ix, site_int_ix, :, rel_ix
        ]
        obs_sim_int_sim_rel = self.sim_sim_residuals[event][
            site_obs_ix, site_int_ix, :, rel_ix
        ]

        site_features = self.scalar_features_tensors[event][site_int_ix, site_obs_ix, :]

        # site_int_all_ix = self.all_sites_ix_lookup[site_int]
        # site_obs_all_ix = self.all_sites_ix_lookup[site_obs]
        # site_features = self.feature_tensor[site_int_all_ix, site_obs_all_ix, :]

        # Labels
        int_obs_int_sim_res = self.obs_sim_residuals[event][
            site_int_ix, site_int_ix, :, rel_ix
        ]

        return (
            idx,
            obs_obs_obs_sim_res,
            obs_obs_int_sim_res,
            obs_sim_int_sim_rel,
            site_features,
            int_obs_int_sim_res,
        )


class ResponseSpectrumDataset(BaseDataset):
    def __init__(
        self,
        event_sites: Dict[str, np.ndarray],
        site_combs: Dict[str, np.ndarray],
        db: db.DB,
        n_rels: int,
        station_df: pd.DataFrame,
        periods: np.ndarray,
        pSA_keys: np.ndarray,
        dist_matrix: pd.DataFrame,
        site_features: Sequence[str],
        max_site_to_site_dist: float = 100,
    ):

        # Base Class
        super().__init__(
            event_sites,
            site_combs,
            db,
            n_rels,
            station_df,
            periods,
            pSA_keys,
            dist_matrix,
            site_features,
            max_site_to_site_dist,
        )

        # Organize the sim response spectra such that it is
        # in the format [n_rels, n_periods, n_sites]
        # per event
        self.sim_im_data = {}
        # And observed response spectra in the format
        # [n_periods, n_sites]
        self.obs_im_data = {}
        for cur_event in self.events:
            cur_sites = self.event_sites[cur_event]
            cur_sim_df = self.sim_im_dfs[cur_event]

            if self.rels[cur_event] is None:
                self.sim_im_data[cur_event] = (
                    cur_sim_df.set_index("site_id")
                    .loc[cur_sites, pSA_keys]
                    .values.T[np.newaxis, ...]
                )
            else:
                assert cur_sim_df.shape[0] == cur_sites.size * self.n_rels
                self.sim_im_data[cur_event] = np.stack(
                    [
                        cur_sim_df.loc[cur_sim_df.rel_id == cur_rel]
                        .set_index("site_id")
                        .loc[cur_sites, pSA_keys]
                        .T.values
                        for cur_rel in self.rels[cur_event]
                    ],
                    axis=0,
                )

            self.obs_im_data[cur_event] = (
                self.obs_im_dfs[cur_event].loc[cur_sites, pSA_keys].values.T
            )

        # Some more sanity checking
        for cur_event in self.events:
            assert self.sim_im_data[cur_event].shape[0] in [1, self.n_rels]
            assert (
                self.sim_im_data[cur_event].shape[2] == self.event_sites[cur_event].size
            )

    def __getitem__(self, idx: int):
        # Break the index down
        event, event_ix, site_ix, rel_ix = self.get_indices(idx)

        # Get the site of interest and observation site
        site_int_ix = self.site_combs[event][site_ix, 0]
        site_obs_ix = self.site_combs[event][site_ix, 1]

        site_int = self.event_sites[event][site_int_ix]
        site_obs = self.event_sites[event][site_obs_ix]

        # Features
        site_int_sim = self.sim_im_data[event][rel_ix, :, site_int_ix]
        site_obs_sim = self.sim_im_data[event][rel_ix, :, site_obs_ix]
        site_obs_obs = self.obs_im_data[event][:, site_obs_ix]

        site_int_all_ix = self.all_sites_ix_lookup[site_int]
        site_obs_all_ix = self.all_sites_ix_lookup[site_obs]
        site_features = self.feature_tensor[site_int_all_ix, site_obs_all_ix, :]

        # Labels
        site_int_obs = self.obs_im_data[event][:, site_int_ix]

        return (
            np.log(site_int_sim),
            np.log(site_obs_sim),
            np.log(site_obs_obs),
            site_features,
            np.log(site_int_obs),
            self.dist_matrix.iat[site_int_all_ix, site_obs_all_ix],
        )
