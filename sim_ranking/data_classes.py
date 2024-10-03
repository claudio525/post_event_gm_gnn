from typing import Sequence, Optional
from pathlib import Path
from enum import StrEnum, Enum, auto

import pandas as pd
import numpy as np

import ml_tools as mlt
import sha_calc as sha
from labelled_data_array import LabelledDataArray

from . import conditional
from . import constants


class ObservedData:

    class EventColEnums(StrEnum):
        event_id = "event_id"
        mag = "mag"
        depth = "depth"
        rake = "rake"
        strike = "strike"
        dip = "dip"
        ztor = "ztor"
        tect_type = "tect_type"
        event_lat = "event_lat"
        event_lon = "event_lon"

    class SiteColEnums(StrEnum):
        site_id = "site_id"
        vs30 = "vs30"
        tsite = "tsite"
        z1p0 = "z1p0"
        z2p5 = "z2p5"
        site_lat = "site_lat"
        site_lon = "site_lon"

    class EventSiteColEnums(StrEnum):
        rjb = "rjb"
        rrup = "rrup"
        rx = "rx"

    class OtherColEnums(StrEnum):
        fhp = "fhp"
        fmin = "fmin"
        fmin_h1 = "fmin_h1"
        fmin_h2 = "fmin_h2"
        fmin_v = "fmin_v"
        quality_score_h1 = "score_h1"
        quality_score_h2 = "score_h2"
        quality_score_v = "score_v"
        is_ground_level = "is_ground_level"
        channel = "channel"

    SITE_COLS = list(SiteColEnums)
    EVENT_COLS = list(EventColEnums)
    EVENT_SITE_COLS = list(EventSiteColEnums)
    IM_COLUMNS = constants.PSA_KEYS + ["PGA", "PGV"]
    OTHER_COLUMNS = list(OtherColEnums)
    COLUMNS = SITE_COLS + EVENT_COLS + EVENT_SITE_COLS + IM_COLUMNS + OTHER_COLUMNS

    def __init__(
        self,
        record_df: pd.DataFrame,
        data_ffp: Path,
        data_source: constants.ObsDataSource,
        nzgmdb_version: constants.NZGMDBVersion = None,
    ):
        """
        Class for storing and handling observed data.

        Note: Instantiation of this class should be done through the class methods!

        Parameters
        ----------
        record_df: pd.DataFrame
            DataFrame containing the observed data, event and site information.
        data_ffp: Path
            File path to the data file.
        data_source: constants.ObsDataSource
            Source of the observed data.
        nzgmdb_version: constants.NZGMDBVersion
            Version of the NZGMDB data.
            Only applicable if the data source is NZGMDB, otherwise None.
        """
        self.record_df = record_df
        """DataFrame containing the observed data, event and site information"""
        self.data_ffp = data_ffp
        """Path to the data file"""

        self.data_source = data_source
        """Source of the observed data"""
        self.nzgmdb_version = nzgmdb_version
        """Version of the NZGMDB data. 
        Only applicable if the data source is NZGMDB, otherwise None."""

        # Cache variables
        self._sites = None
        self._events = None
        self._site_df = None
        self._event_df = None
        self._event_sites = None

    def __hash__(self):
        return hash(self.data_ffp)

    def __reset_cache(self):
        self._sites = None
        self._events = None
        self._site_df = None
        self._event_df = None
        self._event_sites = None

    def get_event_data(
        self, event_id: str, sites: Optional[Sequence[str]] = None
    ) -> pd.DataFrame:
        """Gets data for the specified event and sites."""
        result_df = self.record_df[self.record_df["event_id"] == event_id]

        if sites is not None:
            result_df = result_df[result_df["site_id"].isin(sites)]

        return result_df.set_index("site_id")

    def __setitem__(self, column: str, value: np.ndarray | pd.Series):
        """Support adding of columns"""
        self.record_df[column] = value

    @property
    def n_records(self):
        return self.record_df.shape[0]

    @property
    def sites(self):
        """All sites in the observed data."""
        if self._sites is None:
            self._sites = self.record_df.site_id.unique().astype(str)
        return self._sites

    @property
    def events(self):
        """All events in the observed data."""
        if self._events is None:
            self._events = self.record_df.event_id.unique().astype(str)
        return self._events

    @property
    def site_df(self):
        if self._site_df is None:
            self._site_df = (
                self.record_df[self.SITE_COLS]
                .drop_duplicates("site_id")
                .set_index("site_id")
                .rename(columns={"site_lat": "lat", "site_lon": "lon"})
            )
        return self._site_df

    @property
    def event_df(self):
        if self._event_df is None:
            self._event_df = (
                self.record_df[self.EVENT_COLS]
                .drop_duplicates("event_id")
                .set_index("event_id")
            ).rename(columns={"event_lat": "lat", "event_lon": "lon"})
        return self._event_df

    @property
    def event_sites(self):
        if self._event_sites is None:
            self._event_sites = {}
            for cur_event, cur_group in self.record_df.groupby("event_id"):
                self._event_sites[cur_event] = cur_group.site_id.unique().astype(str)
        return self._event_sites

    def drop_nan(self):
        """Drops any rows with NaN values."""
        nan_mask = self.record_df.isna().any(axis=1)
        self.record_df = self.record_df[~nan_mask]
        print(f"Dropped {nan_mask.sum()}/{nan_mask.shape[0]} rows with NaN values.")

        self.__reset_cache()
        return self

    def drop_duplicates(self, subset: Sequence[str] = None):
        """Drops any duplicate rows."""
        self.record_df = self.record_df.drop_duplicates(subset=subset)

        self.__reset_cache()
        return self

    def metadata_filter(
        self,
        filter_dict: dict[str, tuple[float, float]] = None,
        record_ids: Sequence[str] = None,
    ):
        """
        Performs filtering on the record metadata.
        Does not return anything, but modifies the observed data instance in place.

        Parameters
        ----------
        filter_dict: dict
            Dictionary of key (column name) and
            value (tuple, bool) to filter on.
            E.g. {"mag": (5.0, 6.0), "rrup": (0.0, 10.0),
            "is_ground_level": True}
        record_ids: array of strings
            Record IDs to keep.
        """
        if filter_dict is not None:
            for cur_key, cur_filter in filter_dict.items():
                if isinstance(cur_filter, tuple):
                    self.record_df = self.record_df[
                        (self.record_df[cur_key] >= cur_filter[0])
                        & (self.record_df[cur_key] <= cur_filter[1])
                    ]
                elif isinstance(cur_filter, bool):
                    self.record_df = self.record_df.loc[
                        self.record_df[cur_key] == cur_filter
                    ]
                else:
                    raise ValueError(f"Unknown filter type: {type(cur_filter)}")

        if record_ids is not None:
            self.record_df = self.record_df[self.record_df.index.isin(record_ids)]

        self.__reset_cache()
        return self

    def apply_fmin_filter(self, fmin_col: str):
        """Applies fmin filtering to pSA"""
        max_usable_period = 1 / self.record_df[fmin_col]
        pSA_cols = [
            cur_col for cur_col in self.record_df.columns if cur_col.startswith("pSA")
        ]

        for cur_pSA_col in pSA_cols:
            cur_period = float(cur_pSA_col.split("_")[1])
            self.record_df[cur_pSA_col] = np.where(
                cur_period > max_usable_period, np.nan, self.record_df[cur_pSA_col]
            )

        return self

    @classmethod
    def from_nzgmdb_flat(
        cls,
        nzgmdb_flat_ffp: Path,
        version: constants.NZGMDBVersion = None,
        event_site_id_index: bool = True,
    ):
        site_cols_map = {
            "sta": cls.SiteColEnums.site_id,
            "Vs30": cls.SiteColEnums.vs30,
            "Tsite": cls.SiteColEnums.tsite,
            "T0": cls.SiteColEnums.tsite,
            "Z1.0": cls.SiteColEnums.z1p0,
            "Z2.5": cls.SiteColEnums.z2p5,
            "sta_lat": cls.SiteColEnums.site_lat,
            "sta_lon": cls.SiteColEnums.site_lon,
        }
        event_map = {
            "evid": cls.EventColEnums.event_id,
            "mag": cls.EventColEnums.mag,
            "rake": cls.EventColEnums.rake,
            "strike": cls.EventColEnums.strike,
            "dip": cls.EventColEnums.dip,
            "tect_class": cls.EventColEnums.tect_type,
            "ev_depth": cls.EventColEnums.depth,
            "z_tor": cls.EventColEnums.ztor,
            "ev_lat": cls.EventColEnums.event_lat,
            "ev_lon": cls.EventColEnums.event_lon,
        }
        event_site_map = {
            "r_jb": cls.EventSiteColEnums.rjb,
            "r_rup": cls.EventSiteColEnums.rrup,
            "r_x": cls.EventSiteColEnums.rx,
        }
        other_map = {
            "fmin_X": cls.OtherColEnums.fmin_h1,
            "fmin_Y": cls.OtherColEnums.fmin_h2,
            "fmin_Z": cls.OtherColEnums.fmin_v,
            "score_X": cls.OtherColEnums.quality_score_h1,
            "score_Y": cls.OtherColEnums.quality_score_h2,
            "score_Z": cls.OtherColEnums.quality_score_v,
            "fmin_mean_X": cls.OtherColEnums.fmin_h1,
            "fmin_mean_Y": cls.OtherColEnums.fmin_h2,
            "fmin_mean_Z": cls.OtherColEnums.fmin_v,
            "score_mean_X": cls.OtherColEnums.quality_score_h1,
            "score_mean_Y": cls.OtherColEnums.quality_score_h2,
            "score_mean_Z": cls.OtherColEnums.quality_score_v,
            "is_ground_level": cls.OtherColEnums.is_ground_level,
            "chan": cls.OtherColEnums.channel,
        }
        mapping_dict = site_cols_map | event_map | event_site_map | other_map

        # Determine the version if not provided
        if version is None:
            try:
                version = constants.NZGMDBVersion(nzgmdb_flat_ffp.parent.parent.name)
            except ValueError as e:
                raise ValueError(
                    f"Could not determine version of NZGMDB "
                    f"from {nzgmdb_flat_ffp.parent.parent.name}"
                ) from e

        # Load
        if version in [constants.NZGMDBVersion.v3p4, constants.NZGMDBVersion.v3p0]:
            record_df = pd.read_csv(
                nzgmdb_flat_ffp, dtype={"evid": str}, index_col="gmid", engine="c"
            ).sort_index()

            # Renaming
            record_df = record_df.rename(columns=mapping_dict)
            record_df.index.name = "record_id"

            # Convert index
            if event_site_id_index:
                index = mlt.array_utils.numpy_str_join(
                    "_",
                    record_df["event_id"].values.astype(str),
                    record_df["site_id"].values.astype(str),
                )
                record_df.index = index
                record_df = record_df.sort_index()

            # The GMC fmin in version 3.4 is used to select fHP, hence
            # the actual fmin is not the GMC fmin values
            if (
                cls.OtherColEnums.fmin_h1 in record_df.columns
                and cls.OtherColEnums.fmin_h2 in record_df.columns
            ):
                # Rotd50 does not contain vertical GMC fmin, hack this in...
                fmin_v = pd.read_csv(
                    nzgmdb_flat_ffp.parent
                    / nzgmdb_flat_ffp.name.replace("rotd50", "ver"),
                    dtype={"evid": str},
                    index_col="gmid",
                    engine="c",
                ).sort_index()

                # Convert index, as gmid is incorrect in 3.4
                index = mlt.array_utils.numpy_str_join(
                    "_",
                    fmin_v["evid"].values.astype(str),
                    fmin_v["sta"].values.astype(str),
                )
                fmin_v.index = index
                fmin_v = fmin_v.sort_index()
                assert fmin_v.index.equals(record_df.index)

                record_df[cls.OtherColEnums.fhp] = np.max(
                    np.stack(
                        (
                            record_df[cls.OtherColEnums.fmin_h1].values,
                            record_df[cls.OtherColEnums.fmin_h2].values,
                            fmin_v["fmin_mean_Z"].values,
                        ),
                        axis=1,
                    ),
                    axis=1,
                )

                record_df[cls.OtherColEnums.fmin] = (
                    record_df[cls.OtherColEnums.fhp] / 1.25
                )
                record_df = record_df.drop(
                    columns=[
                        cls.OtherColEnums.fmin_h1,
                        cls.OtherColEnums.fmin_h2,
                        cls.OtherColEnums.fmin_v,
                    ],
                    errors="ignore",
                )
        else:
            record_df = pd.read_csv(
                nzgmdb_flat_ffp,
                dtype={"evid": str, "loc": str},
                engine="c",
                index_col="record_id",
            ).sort_index()

            # Renaming
            record_df = record_df.rename(columns=mapping_dict)

        # Drop any columns not of interest
        cols = record_df.columns[record_df.columns.isin(cls.COLUMNS)]
        record_df = record_df[cols]

        return cls(record_df, nzgmdb_flat_ffp, constants.ObsDataSource.NZGMDB, version)

    @classmethod
    def from_nga_west2_flat(
        cls, nga_west2_flat_ffp: Path, event_site_id_index: bool = True
    ):
        site_cols_map = {
            "Station Name": "site_id",
            "Vs30 (m/s) selected for analysis": "vs30",
            "Station Latitude": "site_lat",
            "Station Longitude": "site_lon",
        }
        event_map = {
            "EQID": "event_id",
            "Earthquake Magnitude": "mag",
            "ev_depth": "depth",
            "z_tor": "ztor",
            "ev_lat": "event_lat",
            "ev_lon": "event_lon",
        }
        event_site_map = {
            "r_jb": "rjb",
            "r_rup": "rrup",
            "r_x": "rx",
        }

        record_df = pd.read_excel(nga_west2_flat_ffp, index_col=0)


class CIMResults:

    def __init__(
        self, emp_cim_results: dict[str, conditional.ConditionalMVNDistribution]
    ):
        self.emp_cim_results = emp_cim_results
        self.ims = np.asarray(
            [str(cur_im) for cur_im in list(emp_cim_results.values())[0].IMs]
        )

        mean_dfs, std_dfs = [], []
        for cur_event in self.emp_cim_results.keys():
            cur_result = self.emp_cim_results[cur_event]

            cur_mean_df = cur_result.cond_lnIM_mean_df.copy()
            cur_mean_df["site_int"] = cur_mean_df.index
            cur_mean_df["event_id"] = cur_event
            cur_mean_df.index = mlt.array_utils.numpy_str_join(
                "_", cur_event, cur_mean_df.site_int
            )

            cur_std_df = cur_result.cond_lnIM_std_df.copy()
            cur_std_df["site_int"] = cur_std_df.index
            cur_std_df["event_id"] = cur_event
            cur_std_df.index = mlt.array_utils.numpy_str_join(
                "_", cur_event, cur_std_df.site_int
            )

            mean_dfs.append(cur_mean_df)
            std_dfs.append(cur_std_df)

        self.mean_df = pd.concat(mean_dfs, axis=0)
        self.std_df = pd.concat(std_dfs, axis=0)

    @property
    def events(self):
        return np.asarray(list(self.emp_cim_results.keys())).astype(str)

    def get_residual_df(self, obs_data: ObservedData):
        residuals = []
        obs_df = obs_data.record_df
        for cur_event in self.events:
            cur_emp_cim = self.emp_cim_results[cur_event]

            cur_emp_mean_df = cur_emp_cim.cond_lnIM_mean_df

            cur_obs_df = obs_df.loc[obs_df.event_id == cur_event].set_index("site_id")
            cur_residual = pd.DataFrame(
                data=np.log(cur_obs_df.loc[cur_emp_mean_df.index, self.ims].values)
                - cur_emp_mean_df.loc[cur_emp_mean_df.index, self.ims].values,
                columns=self.ims,
                index=cur_emp_mean_df.index,
            )
            cur_residual["event_id"] = cur_event
            cur_residual["site_int"] = cur_residual.index
            cur_residual.index = mlt.array_utils.numpy_str_join(
                "_", cur_event, cur_residual.index.values.astype(str)
            )

            residuals.append(cur_residual)

        residual_df = pd.concat(residuals, axis=0)
        return residual_df

    @classmethod
    def from_dir(cls, data_dir: Path, events: Sequence[str]):
        emp_cim_results = {}
        no_data = []
        for event in events:
            cur_result = conditional.load_emp_cim_data(data_dir, event)
            if cur_result is None:
                no_data.append(event)
                continue

            emp_cim_results[event] = cur_result

        if len(no_data) > 0:
            print(f"No data for events: {no_data}")
        return cls(emp_cim_results)


class LBSiteCorrelationData:

    def __init__(self, corr_data: LabelledDataArray):
        self.corr_data = corr_data

    @classmethod
    def from_dist_matrix(cls, dist_matrix: pd.DataFrame, ims: Sequence[str]):
        sites = dist_matrix.index
        corr_values = []
        for cur_im in ims:
            r = sha.loth_baker_corr_model.get_correlations(
                cur_im, cur_im, dist_matrix.values.ravel()
            )
            cur_corr_matrix = r.reshape(dist_matrix.shape)
            np.fill_diagonal(cur_corr_matrix, 1.0)
            corr_values.append(cur_corr_matrix)

        corr_values = np.stack(corr_values, axis=-1)
        lda = LabelledDataArray(
            corr_values, (sites, sites, ims), ("site1", "site2", "im")
        )

        return cls(lda)
