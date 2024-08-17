from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import sqlite3
import tqdm

import ml_tools as mlt

from . import data
from . import constants


class DB:
    def __init__(self, db_ffp: Path):
        self.con = sqlite3.connect(db_ffp)
        self.cur = self.con.cursor()

        self.ffp = db_ffp

    def add_data(
        self,
        sim_im_dir: Path,
        obs_ffp: Path,
        site_ffp: Path,
        source_ffp: Path,
        data_source: str,
        ims: Sequence[str],
    ):
        """Adds the simulation and observation data to the database"""
        site_df = pd.read_csv(site_ffp, index_col="sta")
        event_df = pd.read_csv(source_ffp, index_col=0)
        obs_df = data.load_obs_data(obs_ffp)

        # Load the data
        im_csvs = list(sim_im_dir.rglob("*.csv"))
        if len(im_csvs) == 0:
            im_data = pd.read_pickle(sim_im_dir / "emp_realisations.pickle")
        else:
            im_data = {
                cur_ffp.stem: pd.read_csv(cur_ffp, index_col=0) for cur_ffp in im_csvs
            }

        # Process the data
        events, sites = set(), set()
        for ix, (cur_id, cur_im_df) in enumerate(tqdm.tqdm(im_data.items())):
            if "REL" in cur_id:
                cur_event_id, cur_rel_id = cur_id.split("_")
            else:
                cur_event_id, cur_rel_id = cur_id, "NA"

            # Only interested in events with observation data
            if cur_event_id not in event_df.index:
                print(f"Skipping {cur_event_id} as no observation data is available")
                continue

            ### Add the simulation IM data
            # Read simulation IM data
            # cur_im_df = pd.read_csv(cur_ffp, index_col=0)
            if "component" in cur_im_df.columns:
                cur_im_df = cur_im_df.loc[cur_im_df["component"] == "rotd50"]
            else:
                if ix == 0:
                    print(f"No component column, assuming data is rotd50")
            cur_df = cur_im_df.loc[:, ims]

            # Add extra columns and update index
            cur_df["event_id"] = cur_event_id
            if cur_rel_id is None:
                cur_df["record_id"] = np.char.add(
                    f"{cur_event_id}_", cur_df.index.values.astype(str)
                )
            else:
                cur_df["record_id"] = np.char.add(
                    np.char.add(f"{cur_event_id}_", cur_df.index.values.astype(str)),
                    f"_{cur_rel_id}",
                )
            cur_df["rel_id"] = cur_rel_id
            cur_df["site_id"] = cur_df.index.values.astype(str)
            cur_df["data_source"] = data_source
            cur_df = cur_df.set_index("record_id")

            # Only interested in real sites
            cur_df = cur_df.loc[cur_df.site_id.isin(site_df.index)]

            # Re-order and add to db
            cur_df = cur_df.loc[
                :,
                ["event_id", "rel_id", "site_id", "data_source"] + ims,
            ]
            cur_df.to_sql("sim_im_data", self.con, if_exists="append")

            sites.update(cur_df.site_id.values)

            ### Add the observed IM data and record data
            # Only need to add observed IM data first time
            if cur_event_id not in events:
                # Get current event data
                cur_obs_df = obs_df.loc[obs_df["evid"] == cur_event_id]

                # Split into observed IM and record data
                cur_record_df = cur_obs_df.loc[:, ["evid", "sta", "r_rup", "r_x"]]
                cur_obs_df = cur_obs_df[["evid", "sta"] + ims]

                # Write observed IM data
                cur_obs_df = cur_obs_df.rename(
                    columns={"evid": "event_id", "sta": "site_id"}
                )
                cur_obs_df.to_sql(
                    "obs_im_data",
                    self.con,
                    if_exists="append",
                    index=True,
                    index_label="record_id",
                )

                # Write record data
                cur_record_df = cur_record_df.rename(
                    columns={"evid": "event_id", "sta": "site_id"}
                )
                cur_record_df.to_sql(
                    "records",
                    self.con,
                    if_exists="append",
                    index=True,
                    index_label="record_id",
                )

            events.add(cur_event_id)

        # Add the events
        existing_events = self.get_avail_events()
        events = list(events.difference(existing_events))

        cur_event_df = event_df.loc[events, ["lat", "lon", "mag"]]
        cur_event_df.to_sql(
            "events", self.con, if_exists="append", index=True, index_label="event_id"
        )

        # Add the sites
        existing_sites = self.get_avail_sites()
        sites = list(sites.difference(existing_sites))

        cur_site_df = site_df.loc[
            sites, ["lat", "lon", "Vs30", "Z1.0", "Z2.5", "Tsite"]
        ]
        cur_site_df = cur_site_df.rename(
            columns={"Vs30": "vs30", "Z1.0": "z1.0", "Z2.5": "z2.5", "Tsite": "tsite"}
        )
        cur_site_df.to_sql(
            "sites", self.con, if_exists="append", index=True, index_label="site_id"
        )

    def get_site_df(self):
        """Gets the site data"""
        return pd.read_sql("SELECT * FROM sites", self.con, index_col="site_id")

    def get_event_df(self):
        """Gets the event data"""
        return pd.read_sql("SELECT * FROM events", self.con, index_col="event_id")

    def get_record_df(self):
        """Gets the record data"""
        return pd.read_sql("SELECT * FROM records", self.con, index_col="record_id")

    def get_sim_df(self, log: bool = False):
        """Gets the simulation data"""
        result_df = pd.read_sql(
            "SELECT * FROM sim_im_data",
            self.con,
            index_col="record_id",
            dtype={
                "event_id": "category",
                "rel_id": "category",
                "site_id": "category",
                "data_source": "category",
            },
        )

        if log:
            ims = [
                cur_col
                for cur_col in result_df.columns
                if cur_col in constants.PSA_KEYS
            ]
            result_df[ims] = np.log(result_df[ims])

        return result_df

    def get_obs_df(self, log: bool = False, fix_index: bool = False):
        """Gets the observation IM data"""
        result_df = pd.read_sql(
            "SELECT * FROM obs_im_data", self.con, index_col="record_id"
        )

        if log:
            ims = [
                cur_col
                for cur_col in result_df.columns
                if cur_col in constants.PSA_KEYS
            ]
            result_df[ims] = np.log(result_df[ims])

        if fix_index:
            result_df["record_id"] = result_df.index.values
            result_df.index = mlt.array_utils.numpy_str_join(
                "_",
                result_df.event_id.values.astype(str),
                result_df.site_id.values.astype(str),
            )

        return result_df

    def get_full_obs_df(self):
        """Gets the observation IM data, event and site information"""
        query = (
            "SELECT * FROM obs_im_data "
            "  LEFT JOIN events ON obs_im_data.event_id == events.event_id "
            "  INNER JOIN sites ON obs_im_data.site_id == sites.site_id"
        )

        result_df = pd.read_sql(query, self.con, index_col="record_id")

        # Drop duplicated columns due to the join
        result_df = result_df.loc[:, ~result_df.columns.duplicated()]
        return result_df

    def get_sim_obs_df(self):
        """Gets the simulation and observation IM data"""
        sim_df = self.get_sim_df()
        sim_df["record_id"] = sim_df.index.values
        obs_df = self.get_obs_df()

        sim_obs_df = pd.merge(
            sim_df,
            obs_df,
            how="inner",
            on=["event_id", "site_id"],
            suffixes=("_sim", "_obs"),
        )
        sim_obs_df = sim_obs_df.set_index("record_id")

        return sim_obs_df

    def get_avail_events(self, data_source: str = None):
        """Gets the available events in the database"""
        query = (
            "SELECT event_id FROM events"
            if data_source is None
            else f"SELECT DISTINCT event_id FROM sim_im_data WHERE data_source = '{data_source}'"
        )
        return pd.read_sql(query, self.con, index_col="event_id").index.values.astype(
            str
        )

    def get_avail_sites(self):
        """Gets the available sites in the database"""
        return pd.read_sql(
            "SELECT site_id FROM sites", self.con, index_col="site_id"
        ).index.values.astype(str)

    def get_event_sites(self):
        """
        Retrieves the available sites in both
        simulated and observed data for each event
        """
        event_sites = {}
        events = self.get_avail_events()
        sim_df = self.get_sim_df()
        for cur_event in events:
            cur_obs_sites = pd.read_sql(
                f"SELECT site_id FROM obs_im_data WHERE event_id = (?)",
                self.con,
                params=(cur_event,),
                index_col="site_id",
            ).index.values.astype(str)
            event_sites[cur_event] = np.intersect1d(
                sim_df.loc[sim_df.event_id == cur_event]["site_id"].values.astype(str),
                cur_obs_sites,
            )

        return event_sites

    def get_sim_data(self, event: str, sites: Sequence[str]):
        """Retrieves the simulation data for the given event and sites"""
        query = f"SELECT * FROM sim_im_data WHERE event_id = (?) AND site_id IN ({','.join(['?']*len(sites))})"
        return pd.read_sql(
            query,
            self.con,
            params=(event, *sites),
            index_col="record_id",
        ).drop(columns=["event_id"])

    def get_obs_data(self, event: str, sites: Sequence[str]):
        """Retrieves the observation data for the given event and sites"""
        query = f"SELECT * FROM obs_im_data WHERE event_id = (?) AND site_id IN ({','.join(['?']*len(sites))})"
        return pd.read_sql(
            query, self.con, params=(event, *sites), index_col="site_id"
        ).drop(columns=["event_id", "record_id"])

    @classmethod
    def create(cls, db_ffp: Path, ims: Sequence[str]):
        if db_ffp.exists():
            raise ValueError(f"DB already exists at {db_ffp}")

        con = sqlite3.connect(db_ffp)
        cur = con.cursor()

        # Create the event and site tables
        cur.execute(
            "CREATE TABLE sites (site_id TEXT PRIMARY KEY, lat REAL, lon REAL, vs30 REAL, [z1.0] REAL, [z2.5] REAL, tsite REAL)"
        )
        cur.execute(
            "CREATE TABLE events (event_id TEXT PRIMARY KEY, lat REAL, lon REAL, mag REAL)"
        )

        # Create the simulation table
        cur.execute(
            "CREATE TABLE sim_im_data (record_id TEXT PRIMARY KEY, event_id TEXT, rel_id TEXT, site_id TEXT, data_source TEXT, "
            "FOREIGN KEY(event_id) REFERENCES events(event_id), FOREIGN KEY(site_id) REFERENCES sites(site_id))"
        )
        for cur_im in ims:
            cur.execute(f"ALTER TABLE sim_im_data ADD COLUMN [{cur_im}] REAL")
        # Create indices
        cur.execute(f"CREATE INDEX sim_im_data_event_idx ON sim_im_data (event_id)")
        cur.execute(f"CREATE INDEX sim_im_data_site_idx ON sim_im_data (site_id)")

        # Create the observed table
        cur.execute(
            "CREATE TABLE obs_im_data (record_id TEXT PRIMARY KEY, event_id TEXT, site_id TEXT, "
            "FOREIGN KEY(event_id) REFERENCES events(event_id), FOREIGN KEY(site_id) REFERENCES sites(site_id))"
        )
        for cur_im in ims:
            cur.execute(f"ALTER TABLE obs_im_data ADD COLUMN [{cur_im}] REAL")
        # Create indices
        cur.execute(f"CREATE INDEX obs_im_data_event_idx ON obs_im_data (event_id)")
        cur.execute(f"CREATE INDEX obs_im_data_site_idx ON obs_im_data (site_id)")

        # Create the records table
        cur.execute(
            "CREATE TABLE records (record_id TEXT PRIMARY KEY, event_id TEXT, "
            "site_id TEXT, r_rup REAL, r_x REAL, "
            "FOREIGN KEY(event_id) REFERENCES events(event_id), FOREIGN KEY(site_id) REFERENCES sites(site_id))"
        )

        return cls(db_ffp)
