from pathlib import Path

import numpy as np
import pandas as pd
import sqlite3

from . import data
from . import constants


class DB:
    def __init__(self, db_ffp: Path):
        self.con = sqlite3.connect(db_ffp)
        self.cur = self.con.cursor()

    def add_data(
        self,
        sim_im_dir: Path,
        obs_ffp: Path,
        site_ffp: Path,
        source_ffp: Path,
        data_source: str,
    ):
        site_df = pd.read_csv(site_ffp, index_col="sta")
        event_df = pd.read_csv(source_ffp, index_col=0)
        obs_df = data.load_obs_data(obs_ffp)

        events, sites = set(), set()
        im_csvs = list(sim_im_dir.rglob("*.csv"))
        for ix, cur_ffp in enumerate(im_csvs):
            print(f"Processing {ix+1}/{len(im_csvs)}")

            if "REL" in cur_ffp.stem:
                cur_event_id, cur_rel_id = cur_ffp.stem.split("_")
            else:
                cur_event_id, cur_rel_id = cur_ffp.stem, None

            # Only interested in events with observation data
            if cur_event_id not in event_df.index:
                print(f"Skipping {cur_event_id} as no observation data is available")
                continue

            ### Add the simulation IM data
            # Read simulation IM data
            cur_im_df = pd.read_csv(cur_ffp, index_col=0)
            cur_im_df = cur_im_df.loc[cur_im_df["component"] == "rotd50"]
            cur_df = cur_im_df.loc[:, constants.PERIOD_KEYS]

            # Add extra columns and update index
            cur_df["event_id"] = cur_event_id
            if cur_rel_id is None:
                cur_df["record_id"] = np.char.add(
                    f"{cur_event_id}_", cur_df.index.astype(str)
                )
            else:
                cur_df["record_id"] = np.char.add(
                    np.char.add(f"{cur_event_id}_", cur_df.index.astype(str)),
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
                ["event_id", "rel_id", "site_id", "data_source"]
                + constants.PERIOD_KEYS,
            ]
            cur_df.to_sql("sim_im_data", self.con, if_exists="append")

            sites.update(cur_df.site_id.values)

            ### Add the observed IM data
            # Only need to add observed IM data first time
            if cur_event_id not in events:
                cur_obs_df = obs_df.loc[obs_df["evid"] == cur_event_id]
                cur_obs_df = cur_obs_df[["evid", "sta"] + constants.PERIOD_KEYS]
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

        cur_site_df = site_df.loc[sites, ["lat", "lon", "Vs30", "Z1.0", "Z2.5"]]
        cur_site_df = cur_site_df.rename(
            columns={"Vs30": "vs30", "Z1.0": "z1.0", "Z2.5": "z2.5"}
        )
        cur_site_df.to_sql(
            "sites", self.con, if_exists="append", index=True, index_label="site_id"
        )

    def get_avail_events(self):
        return pd.read_sql(
            "SELECT event_id FROM events", self.con, index_col="event_id"
        ).index.values.astype(str)

    def get_avail_sites(self):
        return pd.read_sql(
            "SELECT site_id FROM sites", self.con, index_col="site_id"
        ).index.values.astype(str)

    @classmethod
    def create(cls, db_ffp: Path):
        if db_ffp.exists():
            raise ValueError(f"DB already exists at {db_ffp}")

        con = sqlite3.connect(db_ffp)
        cur = con.cursor()

        # Create the event and site tables
        cur.execute(
            "CREATE TABLE sites (site_id TEXT PRIMARY KEY, lat REAL, lon REAL, vs30 REAL, [z1.0] REAL, [z2.5] REAL)"
        )
        cur.execute(
            "CREATE TABLE events (event_id TEXT PRIMARY KEY, lat REAL, lon REAL, mag REAL)"
        )

        # Create the simulation table
        cur.execute(
            "CREATE TABLE sim_im_data (record_id TEXT PRIMARY KEY, event_id TEXT, rel_id TEXT,"
            "site_id TEXT, data_source TEXT, FOREIGN KEY(event_id) REFERENCES events(event_id), FOREIGN KEY(site_id) REFERENCES sites(site_id))"
        )
        for cur_period in constants.PERIODS:
            cur.execute(f"ALTER TABLE sim_im_data ADD COLUMN [pSA_{cur_period}] REAL")

        # Create the observed table
        cur.execute(
            "CREATE TABLE obs_im_data (record_id TEXT PRIMARY KEY, event_id TEXT, "
            "site_id TEXT, FOREIGN KEY(event_id) REFERENCES events(event_id), FOREIGN KEY(site_id) REFERENCES sites(site_id))"
        )
        for cur_period in constants.PERIODS:
            cur.execute(f"ALTER TABLE obs_im_data ADD COLUMN [pSA_{cur_period}] REAL")

        return cls(db_ffp)
