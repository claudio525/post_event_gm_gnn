from pathlib import Path

import pandas as pd
import typer
import sqlite3

import sim_ranking as sr

app = typer.Typer()


@app.command("create-db")
def create_db(db_ffp: Path):
    sr.db.DB.create(db_ffp)


@app.command("add-data")
def add_data(
    db_ffp: Path,
    sim_im_dir: Path,
    obs_ffp: Path,
    site_ffp: Path,
    source_ffp: Path,
    data_source: str,
):
    db = sr.db.DB(db_ffp)
    db.add_data(sim_im_dir, obs_ffp, site_ffp, source_ffp, data_source)


if __name__ == "__main__":
    app()
