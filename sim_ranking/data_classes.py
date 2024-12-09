from typing import Sequence, Optional
from pathlib import Path
from enum import StrEnum, Enum, auto

import pandas as pd
import numpy as np

import ml_tools as mlt
import sha_calc as sha
from labelled_data_array import LabelledDataArray

from . import constants


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
        event_site_id_index: bool = True,
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
                cls.OtherColEnums.FMIN_H1 in record_df.columns
                and cls.OtherColEnums.FMIN_H2 in record_df.columns
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

                record_df[cls.OtherColEnums.FHP] = np.max(
                    np.stack(
                        (
                            record_df[cls.OtherColEnums.FMIN_H1].values,
                            record_df[cls.OtherColEnums.FMIN_H2].values,
                            fmin_v["fmin_mean_Z"].values,
                        ),
                        axis=1,
                    ),
                    axis=1,
                )

                record_df[cls.OtherColEnums.FMIN] = (
                        record_df[cls.OtherColEnums.FHP] / 1.25
                )
                record_df = record_df.drop(
                    columns=[
                        cls.OtherColEnums.FMIN_H1,
                        cls.OtherColEnums.FMIN_H2,
                        cls.OtherColEnums.FMIN_V,
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

        # Tectonic type
        record_df[cls.EventColEnums.TECT_TYPE] = record_df[cls.EventColEnums.TECT_TYPE].map(
            tect_type_mapping
        )

        # Drop any columns not of interest
        im_cols = [col for col in record_df.columns if col.startswith("pSA")]
        cols = record_df.columns[record_df.columns.isin(cls.COLUMNS + im_cols)]
        record_df = record_df[cols]

        return cls(record_df, nzgmdb_flat_ffp, constants.ObsDataSource.NZGMDB, version)

    @classmethod
    def from_nga_west2_flat(cls, nga_west2_flat_ffp: Path):
        """Creates an observed data object from the NGA West 2 flat file."""
        site_cols_map = {
            "Station Sequence Number": cls.SiteColEnums.SITE_ID,
            "Vs30 (m/s) selected for analysis": cls.SiteColEnums.VS30,
            "Station Latitude": cls.SiteColEnums.SITE_LAT,
            "Station Longitude": cls.SiteColEnums.SITE_LON,
        }
        event_map = {
            "EQID": cls.EventColEnums.EVENT_ID,
            "Earthquake Magnitude": cls.EventColEnums.MAG,
            "Hypocenter Depth (km)": cls.EventColEnums.DEPTH,
            "Depth to Top Of Fault Rupture Model": cls.EventColEnums.ZTOR,
            "Hypocenter Latitude (deg)": cls.EventColEnums.EVENT_LAT,
            "Hypocenter Longitude (deg)": cls.EventColEnums.EVENT_LON,
        }
        event_site_map = {
            "Joyner-Boore Dist. (km)": cls.EventSiteColEnums.RJB,
            "ClstD (km)": cls.EventSiteColEnums.RRUP,
            "Rx": cls.EventSiteColEnums.RX,
        }
        other_map = {
            "Lowest Usable Freq - H1 (Hz)": cls.OtherColEnums.FMIN_H1,
            "Lowest Usable Freq - H2 (Hz)": cls.OtherColEnums.FMIN_H2,
            "Lowest Usable Freq - Ave. Component (Hz)": cls.OtherColEnums.FMIN,
        }
        im_map = {
            "PGA (g)": "PGA",
            "PGV (cm/s)": "PGV",
        }

        def _is_pSA(col: str):
            return col.startswith("T") and col.endswith("S")

        def _get_pSA_key(col: str):
            return f"pSA_{float(col[1:-1])}"

        # Load
        if nga_west2_flat_ffp.name.endswith(".parquet"):
            record_df = pd.read_parquet(nga_west2_flat_ffp)
        else:
            record_df = pd.read_excel(nga_west2_flat_ffp, index_col=0)

        # Renaming
        im_map = im_map | {
            cur_col: _get_pSA_key(cur_col)
            for cur_col in record_df.columns
            if _is_pSA(cur_col)
        }
        mapping_dict = site_cols_map | event_map | event_site_map | other_map | im_map
        record_df = record_df.rename(columns=mapping_dict)

        # Tectonic Type
        record_df[cls.EventColEnums.TECT_TYPE] = constants.TectonicType.CRUSTAL

        # Drop any columns not of interest
        im_cols = list(im_map.values())
        cols = record_df.columns[record_df.columns.isin(cls.COLUMNS + im_cols)]
        record_df = record_df[cols]

        # Drop any records with invalid event or site id
        drop_mask = (record_df[cls.EventColEnums.EVENT_ID] == -999) | (
                record_df[cls.SiteColEnums.SITE_ID] == -999
        )
        record_df = record_df[~drop_mask]

        # Replace -999 with nan values
        record_df = record_df.replace(-999.0, np.nan)

        # Update index
        assert (record_df.dtypes[cls.EventColEnums.EVENT_ID] == np.int64) and (
                record_df.dtypes[cls.SiteColEnums.SITE_ID] == np.int64
        )
        record_df.index = mlt.array_utils.numpy_str_join(
            "_",
            record_df["event_id"].values.astype(str),
            record_df["site_id"].values.astype(str),
        )

        return cls(
            record_df,
            nga_west2_flat_ffp,
            constants.ObsDataSource.NGAWest2,
        )

    @classmethod
    def from_nga_subduction_flat(cls, nga_sub_ffp: Path):
        site_cols_map = {
            "NGAsubSSN": cls.SiteColEnums.SITE_ID,
            "Vs30_Selected_for_Analysis_m_s": cls.SiteColEnums.VS30,
            "Station_Latitude_deg": cls.SiteColEnums.SITE_LAT,
            "Station_Longitude_deg": cls.SiteColEnums.SITE_LON,
        }
        event_map = {
            "NGAsubEQID": cls.EventColEnums.EVENT_ID,
            "Earthquake_Magnitude": cls.EventColEnums.MAG,
            "Hypocenter_Depth_km": cls.EventColEnums.DEPTH,
            "Ztor_km": cls.EventColEnums.ZTOR,
            "Hypocenter_Latitude_deg": cls.EventColEnums.EVENT_LAT,
            "Hypocenter_Longitude_deg": cls.EventColEnums.EVENT_LON,
            "Intra_Inter_Flag": cls.EventColEnums.TECT_TYPE
        }
        event_site_map = {
            "Rjb_km": cls.EventSiteColEnums.RJB,
            "ClstD_km": cls.EventSiteColEnums.RRUP,
            "Rx_km": cls.EventSiteColEnums.RX,
        }
        other_map = {
            "Longest_Usable_Period_for_PSa_Ave_Component_sec": cls.OtherColEnums.TMAX,
        }
        im_map = {
            "PGA (g)": "PGA",
            "PGV (cm/s)": "PGV",
        }

        tect_type_mapping = {
            0: constants.TectonicType.SUBDUCTION_INTERFACE,
            1: constants.TectonicType.SUBDUCTION_SLAB,
            2: constants.TectonicType.CRUSTAL,
            3: constants.TectonicType.MANTLE,
            4: constants.TectonicType.OUTER_RISE,
            -444: constants.TectonicType.OUTER_RISE,
            5: constants.TectonicType.SUBDUCTION_SLAB,
            -666: constants.TectonicType.CRUSTAL,
            -777: constants.TectonicType.SUBDUCTION_SLAB,
            -888: constants.TectonicType.SUBDUCTION_INTERFACE,
            -999: constants.TectonicType.UNKNOWN,
        }

        def _is_pSA(col: str):
            return col.startswith("T") and col.endswith("S")

        def _get_pSA_key(col: str):
            return f"pSA_{float(col[1:-1].replace('pt', '.'))}"

        # Load
        if nga_sub_ffp.name.endswith(".parquet"):
            record_df = pd.read_parquet(nga_sub_ffp)
        else:
            record_df = pd.read_excel(nga_sub_ffp, index_col=0)

        # Renaming
        im_map = im_map | {
            cur_col: _get_pSA_key(cur_col)
            for cur_col in record_df.columns
            if _is_pSA(cur_col)
        }
        mapping_dict = site_cols_map | event_map | event_site_map | other_map | im_map
        record_df = record_df.rename(columns=mapping_dict)

        # Tectonic type
        record_df[cls.EventColEnums.TECT_TYPE] = record_df[cls.EventColEnums.TECT_TYPE].map(
            tect_type_mapping
        )

        # Drop any columns not of interest
        im_cols = list(im_map.values())
        cols = record_df.columns[record_df.columns.isin(cls.COLUMNS + im_cols)]
        record_df = record_df[cols]

        # Replace -999 with nan values
        record_df = record_df.replace(-999, np.nan)

        # Update index
        record_df.index = mlt.array_utils.numpy_str_join(
            "_",
            record_df["event_id"].values.astype(str),
            record_df["site_id"].values.astype(str),
        )

        return cls(
            record_df,
            nga_sub_ffp,
            constants.ObsDataSource.NGASubduction,
        )


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
