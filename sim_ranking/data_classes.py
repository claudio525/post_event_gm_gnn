from typing import Sequence, Optional
from pathlib import Path

import pandas as pd
import numpy as np

import ml_tools as mlt

from . import conditional
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


class CIMResults:

    def __init__(
        self, emp_cim_results: dict[str, conditional.ConditionalMVNDistribution]
    ):
        self.emp_cim_results = emp_cim_results
        self.ims = np.asarray([str(cur_im) for cur_im in list(emp_cim_results.values())[0].IMs])

        mean_dfs, std_dfs = [], []
        for cur_event in self.emp_cim_results.keys():
            cur_result = self.emp_cim_results[cur_event]

            cur_mean_df = cur_result.cond_lnIM_mean_df.copy()
            cur_mean_df["site_int"] = cur_mean_df.index
            cur_mean_df["event_id"] = cur_event
            cur_mean_df.index = mlt.array_utils.numpy_str_join("_", cur_event, cur_mean_df.site_int)

            cur_std_df = cur_result.cond_lnIM_std_df.copy()
            cur_std_df["site_int"] = cur_std_df.index
            cur_std_df["event_id"] = cur_event
            cur_std_df.index = mlt.array_utils.numpy_str_join("_", cur_event, cur_std_df.site_int)

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
    def from_dir(
        cls, data_dir: Path, events: Sequence[str], method: constants.RankingMethod
    ):
        emp_cim_results = {}
        no_data = []
        for event in events:
            cur_result = conditional.load_emp_cim_data(
                data_dir, event, method
            )
            if cur_result is None:
                no_data.append(event)
                continue

            emp_cim_results[event] = cur_result

        if len(no_data) > 0:
            print(f"No data for events: {no_data}")
        return cls(emp_cim_results)
