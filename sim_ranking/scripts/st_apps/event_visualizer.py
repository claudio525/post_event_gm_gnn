from pathlib import Path

import pandas as pd
import numpy as np
import pydeck as pdk

import streamlit as st
import matplotlib.pyplot as plt

from ml_tools.st_tools import utils as st_utils


@st.cache_data
def get_source_df(sources_ffp: Path):
    # Load the data
    source_df = pd.read_csv(sources_ffp, index_col=0)
    source_df = source_df.loc[:, ["lat", "lon", "mag", "tect_class", "datetime", "depth"]]
    source_df = source_df.loc[source_df.mag > 5]

    source_df["event_id"] = source_df.index.values
    source_df["mag_radius"] = source_df.mag * 500

    # Only interested in South island events
    source_df = source_df.loc[(source_df.lon > 165) & (source_df.lon < 178)]

    return source_df


@st.cache_data
def get_site_df(sites_ffp: Path):
    # Load the data
    site_df = pd.read_csv(sites_ffp, index_col="sta")

    site_df = site_df.loc[:, ["lat", "lon", "Vs30", "Z1.0", "Z2.5"]]
    site_df["site_id"] = site_df.index.values
    # site_df = site_df.loc[:, ["lat", "lon"]]

    return site_df


@st.cache_data
def get_gm_records_df(gm_records_ffp: Path):
    # Load the data
    gm_records_df = pd.read_csv(gm_records_ffp, index_col=0, low_memory=False)

    return gm_records_df

@st.cache_data
def get_pSA_df(gm_records_df: pd.DataFrame):
    period_cols = np.asarray([col for col in gm_records_df.columns if col.startswith("pSA")])
    periods = np.asarray([float(col.split("_")[1]) for col in period_cols])
    sort_ind = np.argsort(periods)

    pSA_df = gm_records_df.loc[:, period_cols[sort_ind]]
    pSA_df.columns = periods[sort_ind]

    return pSA_df



if __name__ == "__main__":
    def on_event_change():
        if len(st.session_state.event_ids) > 0:
            st.session_state.sites = gm_records_df.loc[
                gm_records_df.evid == st.session_state.event_ids[0], "sta"].values

            st.session_state.event_mask = source_df.event_id.values == st.session_state.event_ids[0]
            st.session_state.site_mask = np.isin(site_df.site_id.values, st.session_state.sites)
        else:
            st.session_state.sites = site_df.index.values

            st.session_state.event_mask = np.ones(len(source_df), dtype=bool)
            st.session_state.site_mask = np.ones(len(site_df), dtype=bool)

    sources_ffp = Path(
        "/Users/claudy/dev/work/data/gm_datasets/nz_gmdb/v3.0/Tables/earthquake_source_table.csv"
    )
    source_df = get_source_df(sources_ffp)

    sites_ffp = Path(
        "/Users/claudy/dev/work/data/gm_datasets/nz_gmdb/v3.0/Tables/site_table.csv"
    )
    site_df = get_site_df(sites_ffp)

    gm_records_ffp = Path(
        "/Users/claudy/dev/work/data/gm_datasets/nz_gmdb/v3.0/Tables/ground_motion_im_table_rotd50_flat.csv"
    )
    gm_records_df = get_gm_records_df(gm_records_ffp)

    pSA_df = get_pSA_df(gm_records_df)

    st_utils.update_st_width(1600, 2, 0, 1, 1)

    # Initialize the session state
    if "event_mask" not in st.session_state:
        st.session_state.event_mask = np.ones(source_df.shape[0], dtype=bool)
    if "site_mask" not in st.session_state:
        st.session_state.site_mask = np.ones(site_df.shape[0], dtype=bool)
    if "sites" not in st.session_state:
        st.session_state.sites = site_df.index.values
    if "event_ids" not in st.session_state:
        st.session_state.event_ids = []

    event_layer = pdk.Layer(
        "ScatterplotLayer",
        source_df.loc[st.session_state.event_mask, :],
        opacity=0.5,
        auto_highlight=True,
        get_position=["lon", "lat"],
        get_radius="mag_radius",
        get_fill_color=[255, 0, 0],
        radius_min_pixels=5,
        radius_max_pixels=25,
        pickable=True,
        stroked=True,
        filled=True,
    )

    site_layer = pdk.Layer(
        "ScatterplotLayer",
        site_df.loc[st.session_state.site_mask, :],
        opacity=0.8,
        auto_highlight=True,
        get_position=["lon", "lat"],
        get_radius=1000,
        radius_min_pixels=2,
        radius_max_pixels=10,
        get_fill_color=[0, 0, 0],
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=-43.5, longitude=172.5, zoom=5, bearing=0, pitch=0
    )

    deck = pdk.Deck(
        layers=[event_layer, site_layer],
        initial_view_state=view_state,
        map_style=pdk.map_styles.LIGHT,
        tooltip={
            "text": "Event Id: {event_id}\nMagnitude: {mag}\nTectonic Type: {tect_class}"
            "\n------------"
            "\nSite Id: {site_id}\nVs30: {Vs30}\nZ1.0: {Z1.0}\nZ2.5: {Z2.5}\n"
        },
    )
    st.pydeck_chart(deck)


    col1, col2 = st.columns(2)
    with col1:
        st.multiselect(
            "**Events**", source_df.event_id, key="event_ids", max_selections=1, on_change=on_event_change
        )

        if len(st.session_state.event_ids) > 0:

            # st.markdown(f"**Sites**: {st.session_state.sites}")

            # cur_event_ids = st.session_state.event_ids.values()
            cur_series = source_df.loc[st.session_state.event_ids[0], :]
            st.markdown(f"**Event Id**: {cur_series.event_id}")
            st.markdown(f"**Magnitude**: {cur_series.mag:.2f}")
            st.markdown(f"**Tectonic Class**: {cur_series.tect_class}")
            st.markdown(f"**Depth**: {cur_series.depth}")
            st.markdown(f"**Date/Time**: {cur_series.datetime}")
            st.markdown(f"**Number of sites**: {gm_records_df.loc[gm_records_df.evid == cur_series.event_id, :].shape[0]}")

            # st.markdown(.to_markdown())
            # st.markdown(source_df.loc[source_df.event_id == st.session_state.event_id, :].to_markdown())
    with col2:
        st.selectbox("**Site**", st.session_state.sites, key="site")

        st.markdown(f"**Lat/Lon**: {site_df.loc[st.session_state.site, 'lat']:.2f}, {site_df.loc[st.session_state.site, 'lon']:.2f}")
        st.markdown(f"**Vs30**: {site_df.loc[st.session_state.site, 'Vs30']}")
        st.markdown(f"**Z1.0**: {site_df.loc[st.session_state.site, 'Z1.0']}")
        st.markdown(f"**Z2.5**: {site_df.loc[st.session_state.site, 'Z2.5']}")

        if len(st.session_state.event_ids) > 0:
            cur_mask = (gm_records_df.evid == st.session_state.event_ids[0]) & (
                    gm_records_df.sta == st.session_state.site)
            st.markdown(f"**Site-to-Source**: {gm_records_df.loc[cur_mask, 'r_rup'].values[0]:.2f} km")

    if len(st.session_state.event_ids) > 0:
        gm_id = gm_records_df.loc[cur_mask].index.values[0]

        # Create the response spectrum plot
        fig = plt.figure(figsize=(16, 10))

        plt.loglog(pSA_df.columns.values, pSA_df.loc[gm_id, :].values, "k-")

        plt.title(f"GM Id: {gm_id}")
        plt.xlabel(f"Period, T (s)")
        plt.ylabel(f"pSA (g)")
        plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")
        plt.tight_layout()

        st.pyplot(fig)





