from typing import Sequence, Optional
from pathlib import Path

import pandas as pd
import numpy as np

from . import constants

class ObservedData:

    SITE_COLS = ["site_id", "site_lat", "site_lon", "vs30", "tsite", "z1p0", "z2p5"]
    EVENT_COLS = [
        "event_id",
        "mag",
        "depth",
        "rake",
        "strike",
        "dip",
        "ztor",
        "tect_type",
        "event_lat",
        "event_lon",
    ]
    EVENT_SITE_COLS = ["rrup", "rjb"]
    IM_COLUMNS = constants.PSA_KEYS
    COLUMNS = SITE_COLS + EVENT_COLS + EVENT_SITE_COLS + IM_COLUMNS

    def __init__(self, record_df: pd.DataFrame, data_source: Path):
        self.record_df = record_df
        self.data_source = data_source

        # Cache variables
        self._all_sites = None
        self._site_df = None
        self._event_df = None

    def __hash__(self):
        return hash(self.data_source)

    def get_event_data(
        self, event_id: str, sites: Optional[Sequence[str]] = None
    ) -> pd.DataFrame:
        result_df = self.record_df[self.record_df["event_id"] == event_id]

        if sites is not None:
            result_df = result_df[result_df["site_id"].isin(sites)]

        return result_df

    @property
    def all_sites(self):
        if self._all_sites is None:
            self._all_sites = self.record_df.site_id.unique().astype(str)
        return self._all_sites

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

    @classmethod
    def from_nzgmdb_flat(cls, nzgmdb_flat_ffp: Path):
        site_cols_map = {
            "sta": "site_id",
            "Vs30": "vs30",
            "Tsite": "tsite",
            "Z1.0": "z1p0",
            "Z2.5": "z2p5",
            "sta_lat": "site_lat",
            "sta_lon": "site_lon",
        }
        event_map = {
            "evid": "event_id",
            "tect_class": "tect_type",
            "ev_depth": "depth",
            "z_tor": "ztor",
            "ev_lat": "event_lat",
            "ev_lon": "event_lon",
        }
        event_site_map = {
            "r_jb": "rjb",
            "r_rup": "rrup",
        }
        mapping_dict = site_cols_map | event_map | event_site_map

        # Load
        record_df = pd.read_csv(
            nzgmdb_flat_ffp, dtype={"evid": str}, index_col="gmid", engine="c"
        )

        # Renaming
        record_df = record_df.rename(columns=mapping_dict)
        record_df.index.name = "record_id"

        # Drop any columns not of interest
        record_df = record_df[cls.COLUMNS]

        return cls(record_df, nzgmdb_flat_ffp)
