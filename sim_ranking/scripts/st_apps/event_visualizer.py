from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import typer
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

import sim_ranking as sr


# @st.cache_resource
def get_db(db_ffp: Path):
    return sr.db.DB(db_ffp)


@st.cache_data
def get_site_df(db_ffp: Path):
    return get_db(db_ffp).get_site_df()


@st.cache_data
def get_event_df(db_ffp: Path):
    return get_db(db_ffp).get_event_df()


@st.cache_data
def get_avail_events(db_ffp: Path, data_source: str):
    return get_db(db_ffp).get_avail_events(data_source=data_source)


@st.cache_data
def get_event_sites(db_ffp: Path):
    return get_db(db_ffp).get_event_sites()


@st.cache_data
def get_avail_sites(db_ffp: Path):
    return get_db(db_ffp).get_avail_sites()


@st.cache_data
def get_sites_for_events(db_ffp: Path, events: Sequence[str] = None):
    if events is None or len(events) == 0:
        return get_avail_sites(db_ffp)
    else:
        event_sites = get_event_sites(db_ffp)
        return list(
            set.intersection(*[set(event_sites[cur_event]) for cur_event in events])
        )


def main(db_ffp: Path):
    st.set_page_config(layout="wide")

    site_df = get_site_df(db_ffp)

    data_source = st.selectbox("Data Source", ["all", "specific", "validation"])
    data_source = None if data_source == "all" else data_source

    avail_events = get_avail_events(db_ffp, data_source)
    sel_events = st.multiselect("Event", avail_events, default=None)

    avail_sites = get_sites_for_events(db_ffp, sel_events)
    st.write(f"Number of available sites: {len(avail_sites)}")
    sel_sites = st.multiselect("Site", avail_sites, default=None)

    if sel_sites is None or len(sel_sites) == 0:
        sel_sites = avail_sites

    fig = go.Figure(
        data=go.Scattermapbox(
            lat=site_df.loc[sel_sites].lat,
            lon=site_df.loc[sel_sites].lon,
            mode="markers",
            marker=dict(size=10),
            hovertext=sel_sites,
            hoverinfo="text",
        )
    )
    # fig.update_mapboxes(fitbounds="locations")
    fig.update_layout(height=900, width=1400)
    fig.update_mapboxes(
        accesstoken="pk.eyJ1IjoiY3MyMyIsImEiOiJjbGtpeXIxNnkwbDQ3M25xbDFrZWFnNHo3In0.OD7TJ_1PegpGvCOCxfHsnA",
        center=dict(
            lat=-43.534,
            lon=172.622,
        ),
        zoom=8,
    )
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    typer.run(main)
