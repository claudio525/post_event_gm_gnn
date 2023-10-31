from typing import Dict, Sequence

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from . import data
from ..db import DB


class BaseDataset(Dataset):
    def __init__(
        self,
        event_sites: Dict[str, np.ndarray],
        site_combs: Dict[str, np.ndarray],
        db: DB,
        n_rels: int,
        station_df: pd.DataFrame,
        periods: np.ndarray,
        pSA_keys: np.ndarray,
        scalar_features: data.ScalarFeatures,
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
        ) = data._get_event_sim_obs_pSA_data(
            self.event_sites, self.site_combs, db, self.pSA_keys, self.n_rels
        )

        # Compute the number of samples per event
        (
            self.n_samples_event,
            self._cum_n_samples,
            self.n_rels_used,
        ) = data._get_event_n_sampels(self.n_samples_event, self.rels)

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

        if event_ix == 0:
            site_ix = idx // n_rels
        else:
            site_ix = (idx - self._cum_n_samples[event_ix - 1]) // n_rels
        rel_ix = idx % n_rels

        return event, event_ix, site_ix, rel_ix

    def __getitem__(self, idx: int):
        raise NotImplementedError()


class ResponseSpectrumResidualDataset(BaseDataset):
    def __init__(
        self,
        event_sites: Dict[str, np.ndarray],
        site_combs: Dict[str, np.ndarray],
        db: DB,
        n_rels: int,
        station_df: pd.DataFrame,
        periods: np.ndarray,
        pSA_keys: np.ndarray,
        scalar_features: data.ScalarFeatures,
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

    def _get_data(self, idx: int, event: str, site_ix: int, rel_ix: int):
        # Get the site of interest and observation site
        site_int_ix = self.site_combs[event][site_ix, 0]
        site_obs_ix = self.site_combs[event][site_ix, 1]

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

        scalar_features = self.scalar_features_tensors[event][
            site_int_ix, site_obs_ix, :
        ]

        # Labels
        int_obs_int_sim_res = self.obs_sim_residuals[event][
            site_int_ix, site_int_ix, :, rel_ix
        ]

        return (
            idx,
            obs_obs_obs_sim_res,
            obs_obs_int_sim_res,
            obs_sim_int_sim_rel,
            scalar_features,
            int_obs_int_sim_res,
        )

    def __getitem__(self, idx: int):
        # Break the index down
        event, event_ix, site_ix, rel_ix = self.get_indices(idx)

        return self._get_data(idx, event, site_ix, rel_ix)


class WeightRSResidualDataset(ResponseSpectrumResidualDataset):
    def __init__(
        self,
        event_sites: Dict[str, np.ndarray],
        site_combs: Dict[str, np.ndarray],
        db: DB,
        n_rels: int,
        station_df: pd.DataFrame,
        periods: np.ndarray,
        pSA_keys: np.ndarray,
        scalar_features: data.ScalarFeatures,
        weight_scalar_features: data.WeightScalarFeatures,
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

        self.weight_scalar_features = weight_scalar_features
        self.weight_scalar_feature_tensor = {}
        for cur_event in self.events:
            cur_sites = self.event_sites[cur_event]
            cur_tensor = np.full(
                (
                    cur_sites.size,
                    cur_sites.size,
                    self.weight_scalar_features.n_scalar_features,
                ),
                fill_value=np.nan,
            )
            # Site to site features
            for i, feature_i in enumerate(
                self.weight_scalar_features.site_to_site_feature_keys
            ):
                cur_tensor[
                    :, :, i
                ] = self.weight_scalar_features.site_to_site_features_data[
                    feature_i
                ].loc[
                    cur_sites, cur_sites
                ]
            cur_f_ix = len(self.weight_scalar_features.site_to_site_feature_keys)
            # Event site to site features
            for i, feature_i in enumerate(
                self.weight_scalar_features.event_site_to_site_feature_keys
            ):
                cur_tensor[
                    :, :, cur_f_ix + i
                ] = self.weight_scalar_features.event_site_to_site_features_data[
                    feature_i
                ][
                    cur_event
                ].loc[
                    cur_sites, cur_sites
                ]

            self.weight_scalar_feature_tensor[cur_event] = cur_tensor

    def __getitem__(self, idx: int):
        # Break the index down
        event, event_ix, site_ix, rel_ix = self.get_indices(idx)

        # Get the site of interest and observation site
        site_int_ix = self.site_combs[event][site_ix, 0]
        site_obs_ix = self.site_combs[event][site_ix, 1]

        weight_scalar_features = self.weight_scalar_feature_tensor[event][
            site_int_ix, site_obs_ix, :
        ]

        return *self._get_data(idx, event, site_ix, rel_ix), weight_scalar_features
