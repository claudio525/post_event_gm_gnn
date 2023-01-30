from typing import Sequence
from pathlib import Path

import pandas as pd

import gmhazard_calc as gc

def load_sim_data(sim_imdb_ffp: Path, sites: Sequence[str]):
    sim_data = {}
    with gc.dbs.IMDB.get_imdb(sim_imdb_ffp) as db:
        for cur_site in sites:
            if (cur_im_df := db.im_data(cur_site)) is not None:
                sim_data[cur_site] = cur_im_df.droplevel(0, 0)

    return sim_data

def load_obs_rupture_data(obs_data_ffp: Path, rupture: str):
    obs_df = pd.read_csv(obs_data_ffp, index_col=0, low_memory=False)
    obs_df = obs_df.loc[obs_df.evid == rupture]
    obs_df = obs_df.set_index("sta").sort_index()

    return obs_df