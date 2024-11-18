from pathlib import Path
from typing import List, Sequence

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import typer
from tqdm import tqdm
import seaborn as sns

import sim_ranking as sr
import ml_tools as mlt


app = typer.Typer()


@app.command("mag-rrup-scatter")
def mag_rrup_scatter(nzgmdb_ffp: Path, output_ffp: Path):
    """
    Creates a scatter of rrup vs magnitude
    with the marginal distributions on the sides
    """
    obs_data = sr.data.load_obs_nzgmdb(nzgmdb_ffp)

    g = sns.jointplot(
        obs_data.record_df,
        x=sr.ObservedData.EventSiteColEnums.RRUP,
        y=sr.ObservedData.EventColEnums.MAG,
        marker=".",
        marginal_kws=dict(bins=25),
    )
    g.set_axis_labels("$R_{Rup}$ (km)", "Magnitude")
    plt.savefig(output_ffp)


@app.command("test")
def test():
    pass


if __name__ == "__main__":
    app()
