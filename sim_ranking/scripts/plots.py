from pathlib import Path

import pandas as pd
import numpy as np
import typer

from pygmt_helper import plotting

app = typer.Typer()

@app.command()
def plot_sites_historic_events(sites_ffp: Path, source_ffp: Path, map_data_ffp: Path = None):
    site_df = pd.read_csv(sites_ffp)
    source_df = pd.read_csv(source_ffp)

    # Only interested in crustal
    source_df = source_df.loc[source_df.tect_class == "Crustal"]

    # fig = plotting.gen_region_fig("Sites & Faults", "NZ")

    print(f"wtf")




if __name__ == '__main__':
    app()


