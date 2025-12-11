from typing import Sequence, Optional
from pathlib import Path
from enum import StrEnum

import pandas as pd
import numpy as np

import ml_tools as mlt
from labelled_data_array import LabelledDataArray

from . import constants
from . import loth_baker_2013_corr_model as lb13
from . import utils



class ObservedData:

    class EventColEnums(StrEnum):
        EVENT_ID = "event_id"
        MAG = "mag"
        DEPTH = "depth"
        RAKE = "rake"
        STRIKE = "strike"
        DIP = "dip"
        ZTOR = "ztor"
        TECT_TYPE = "tect_type"
        EVENT_LAT = "event_lat"
        EVENT_LON = "event_lon"

    class SiteColEnums(StrEnum):
        SITE_ID = "site_id"
        VS30 = "vs30"
        TSITE = "tsite"
        Z1P0 = "z1p0"
        Z2P5 = "z2p5"
        SITE_LAT = "site_lat"
        SITE_LON = "site_lon"

    class EventSiteColEnums(StrEnum):
        RJB = "rjb"
        RRUP = "rrup"
        RX = "rx"

    class OtherColEnums(StrEnum):
        FHP_HORIZONTAL = "HPF_h"
        """High-pass filter frequency for horizontal components"""
        FHP_VERTICAL = "HPF_v"
        """High-pass filter frequency for vertical components"""
        FHP = "fhp"
        """High-pass filter frequency"""
        TMAX = "tmax"
        """Maximum usable period"""
        FMIN = "fmin"
        """Minimum usable frequency"""
        FMIN_H1 = "fmin_h1"
        """Minimum usable frequency for horizontal component 1"""
        FMIN_H2 = "fmin_h2"
        """Minimum usable frequency for horizontal component 2"""
        FMIN_V = "fmin_v"
        """Minimum usable frequency for vertical component"""
        QUALITY_SCORE_H1 = "score_h1"
        QUALITY_SCORE_H2 = "score_h2"
        QUALITY_SCORE_V = "score_v"
        IS_GROUND_LEVEL = "is_ground_level"
        CHANNEL = "channel"

    SITE_COLS = list(SiteColEnums)
    EVENT_COLS = list(EventColEnums)
    EVENT_SITE_COLS = list(EventSiteColEnums)
    IM_COLUMNS = ["PGA", "PGV"]
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
        self._ims = None

    def __hash__(self):
        return hash(self.data_ffp)

    def __reset_cache(self):
        self._sites = None
        self._events = None
        self._site_df = None
        self._event_df = None
        self._event_sites = None
        self._ims = None

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
    def ims(self):
        if self._ims is None:
            pSA_vals = [cur_col for cur_col in self.record_df.columns if cur_col.startswith("pSA")]
            other_ims = [cur_im for cur_im in self.IM_COLUMNS if cur_im in self.record_df.columns]
            self._ims = np.asarray(pSA_vals + other_ims)

        return self._ims


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

    def drop_nan(self, subset: Sequence[str] = None, verbose: bool = False):
        """Drops any rows with NaN values."""
        cols = subset if subset else self.record_df.columns
        nan_mask = self.record_df.loc[:, cols].isna().any(axis=1)
        if np.count_nonzero(nan_mask) > 0:
            if verbose:
                print(f"Dropping {np.count_nonzero(nan_mask)} records with NaN values")
            self.record_df = self.record_df[~nan_mask]

            self.__reset_cache()
        return self

    def drop_events(self, events: Sequence[str]):
        """Drops all records associated with the specified events."""
        self.record_df = self.record_df[~self.record_df[self.EventColEnums.EVENT_ID].isin(events)]

        self.__reset_cache()
        return self
    
    def drop_sites(self, sites: Sequence[str]):
        """Drops all records associated with the specified sites."""
        self.record_df = self.record_df[~self.record_df[self.SiteColEnums.SITE_ID].isin(sites)]

        self.__reset_cache()
        return self

    def drop_duplicates(self, subset: Sequence[str] = None, sort_key: str = None, ascending: bool = True):
        """
        Drops any duplicate rows.
        Allows for sorting of the dataframe first.
        """
        if sort_key is not None:
            self.record_df = self.record_df.sort_values(sort_key, ascending=ascending)
        self.record_df = self.record_df.drop_duplicates(subset=subset)

        self.__reset_cache()
        return self

    def metadata_filter(
        self,
        filter_dict: dict[str, tuple[float, float]],
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
        """
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

        self.__reset_cache()
        return self
    
    def filter_record_ids(
        self,
        record_ids: np.ndarray[str],
    ):
        """
        Filters the observed data based on the provided record IDs.

        Parameters
        ----------
        record_ids: array of strings
            Record IDs to keep.
        """
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

    def to_event_site_index(self):
        """
        Updates the index to be {event_id}_{site_id}

        Note: This will cause if there are duplicate event_id & site_id pairs.
        """
        index = mlt.array_utils.numpy_str_join(
            "_",
            self.record_df["event_id"].values.astype(str),
            self.record_df["site_id"].values.astype(str),
        )
        self.record_df.index = index
        self.record_df = self.record_df.sort_index()

        return self

    @classmethod
    def from_nzgmdb_flat(
        cls,
        nzgmdb_flat_ffp: Path,
        version: constants.NZGMDBVersion = None,
    ):
        site_cols_map = {
            "sta": cls.SiteColEnums.SITE_ID,
            "Vs30": cls.SiteColEnums.VS30,
            "Tsite": cls.SiteColEnums.TSITE,
            "T0": cls.SiteColEnums.TSITE,
            "Z1.0": cls.SiteColEnums.Z1P0,
            "Z2.5": cls.SiteColEnums.Z2P5,
            "sta_lat": cls.SiteColEnums.SITE_LAT,
            "sta_lon": cls.SiteColEnums.SITE_LON,
        }
        event_map = {
            "evid": cls.EventColEnums.EVENT_ID,
            "mag": cls.EventColEnums.MAG,
            "rake": cls.EventColEnums.RAKE,
            "strike": cls.EventColEnums.STRIKE,
            "dip": cls.EventColEnums.DIP,
            "tect_class": cls.EventColEnums.TECT_TYPE,
            "ev_depth": cls.EventColEnums.DEPTH,
            "z_tor": cls.EventColEnums.ZTOR,
            "ev_lat": cls.EventColEnums.EVENT_LAT,
            "ev_lon": cls.EventColEnums.EVENT_LON,
        }
        event_site_map = {
            "r_jb": cls.EventSiteColEnums.RJB,
            "r_rup": cls.EventSiteColEnums.RRUP,
            "r_x": cls.EventSiteColEnums.RX,
        }
        other_map = {
            "HPF_h": cls.OtherColEnums.FHP_HORIZONTAL,
            "HPF_v": cls.OtherColEnums.FHP_VERTICAL, 
            "fmin_X": cls.OtherColEnums.FMIN_H1,
            "fmin_Y": cls.OtherColEnums.FMIN_H2,
            "fmin_Z": cls.OtherColEnums.FMIN_V,
            "score_X": cls.OtherColEnums.QUALITY_SCORE_H1,
            "score_Y": cls.OtherColEnums.QUALITY_SCORE_H2,
            "score_Z": cls.OtherColEnums.QUALITY_SCORE_V,
            "fmin_mean_X": cls.OtherColEnums.FMIN_H1,
            "fmin_mean_Y": cls.OtherColEnums.FMIN_H2,
            "fmin_mean_Z": cls.OtherColEnums.FMIN_V,
            "score_mean_X": cls.OtherColEnums.QUALITY_SCORE_H1,
            "score_mean_Y": cls.OtherColEnums.QUALITY_SCORE_H2,
            "score_mean_Z": cls.OtherColEnums.QUALITY_SCORE_V,
            "is_ground_level": cls.OtherColEnums.IS_GROUND_LEVEL,
            "chan": cls.OtherColEnums.CHANNEL,
            "fmin_max": cls.OtherColEnums.FMIN,
        }
        mapping_dict = site_cols_map | event_map | event_site_map | other_map

        tect_type_mapping = {
            "Crustal": constants.TectonicType.CRUSTAL,
            "Interface": constants.TectonicType.SUBDUCTION_INTERFACE,
            "Slab": constants.TectonicType.SUBDUCTION_SLAB,
            "Outer-rise": constants.TectonicType.OUTER_RISE,
            "Undetermined": constants.TectonicType.UNKNOWN,
            np.nan: constants.TectonicType.UNKNOWN,
        }

        # Determine the version if not provided
        if version is None:
            try:
                version = constants.NZGMDBVersion(nzgmdb_flat_ffp.parent.parent.name)
            except ValueError as e:
                raise ValueError(
                    f"Could not determine version of NZGMDB "
                    f"from {nzgmdb_flat_ffp.parent.parent.name}"
                ) from e

        record_df = pd.read_csv(
            nzgmdb_flat_ffp,
            dtype={"evid": str, "loc": str},
            engine="c",
            index_col="record_id",
        ).sort_index()

        if version is constants.NZGMDBVersion.v4p3_final:
            # Add fmin column
            record_df[cls.OtherColEnums.FMIN] = record_df[cls.OtherColEnums.FHP_HORIZONTAL] * 1.25

        # Renaming
        record_df = record_df.rename(columns=mapping_dict)

        # Tectonic type
        record_df[cls.EventColEnums.TECT_TYPE] = record_df[cls.EventColEnums.TECT_TYPE].map(
            tect_type_mapping
        )

        # Drop any columns not of interest
        im_cols = constants.IMs
        cols = record_df.columns[record_df.columns.isin(cls.COLUMNS + im_cols)]
        record_df = record_df[cols]

        return cls(record_df, nzgmdb_flat_ffp, constants.ObsDataSource.NZGMDB, version)


class DynamicLBSiteCorrelationsData:
    
    def __init__(self, dist_matrix: pd.DataFrame):
        self.dist_matrix = dist_matrix
        self.sites = dist_matrix.index.values.astype(str)

    def get_im_corr(self, im: str, sites: np.ndarray = None):
        """
        Gets the correlation for the specified IM
        and the specified sites. If sites are not specified,
        all sites in the distance matrix are used.
        """
        T = utils.get_pSA_period(im)
        
        sites = sites if sites is not None else self.sites
        cur_dist_matrix = self.dist_matrix.loc[sites, sites]

        corr_matrix = lb13.get_correlations(T, T, cur_dist_matrix.values)
        corr_matrix = corr_matrix.reshape(cur_dist_matrix.shape)
        # np.fill_diagonal(corr_matrix, 1.0)
        assert np.allclose(np.diagonal(corr_matrix), 1.0)
        return pd.DataFrame(
            index=cur_dist_matrix.index,
            data=corr_matrix,
            columns=cur_dist_matrix.columns,
        )

    def get_ims_corr(self, ims: Sequence[str], sites: np.ndarray = None):
        """
        Gets the correlations for the specified IMs
        and the specified sites. If sites are not specified,
        all sites in the distance matrix are used.
        """
        T = np.array([utils.get_pSA_period(im) for im in ims])
        sites = sites if sites is not None else self.sites
        cur_dist_matrix = self.dist_matrix.loc[sites, sites]

        corr_matrix = lb13.get_correlations_vec(T, T, cur_dist_matrix.values.ravel())
        corr_matrix = corr_matrix.T.reshape(sites.size, sites.size, T.size)
        
        return LabelledDataArray(corr_matrix, (sites, sites, ims), ("site1", "site2", "im"))
        

class LBSiteCorrelationData:

    def __init__(self, corr_data: LabelledDataArray):
        self.corr_data = corr_data

    @classmethod
    def from_dist_matrix(cls, dist_matrix: pd.DataFrame, ims: Sequence[str]):
        lda = DynamicLBSiteCorrelationsData(dist_matrix).get_ims_corr(ims, dist_matrix.index)
        return cls(lda)
