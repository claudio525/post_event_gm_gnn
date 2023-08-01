from pathlib import Path

import pandas as pd
import numpy as np

import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from ml_tools.st_tools import utils as st_utils
import spatial_hazard as sh
from gmhazard_calc.im import IM
import sha_calc as sha


@st.cache_data
def get_avail_ims(sim_site_corr_dir: Path, event: str):
    return np.sort(
        [
            cur_ffp.stem
            for cur_ffp in (sim_site_corr_dir / event).iterdir()
            if cur_ffp.is_file()
        ]
    )


@st.cache_data
def load_sim_site_correlations(sim_site_corr_dir: Path, event: str, im: str):
    return pd.read_csv(sim_site_corr_dir / event / f"{im}.csv", index_col=0)


@st.cache_data
def load_sim_im_residuals(sim_site_corr_dir: Path, event, im: str):
    return pd.read_csv(
        sim_site_corr_dir / event / "im_residuals" / f"{im}.csv", index_col=0
    )


@st.cache_data
def load_site_df(ll_ffp: Path):
    return pd.read_csv(
        ll_ffp,
        delim_whitespace=True,
        header=None,
        names=["lon", "lat", "site"],
        index_col="site",
    )


if __name__ == "__main__":
    st_utils.update_st_width(1600, 2, 0, 1, 1)

    sim_site_corr_dir = Path("/Users/claudy/dev/work/data/sim_ranking/sim_correlations")

    ll_ffp = Path(
        "/Users/claudy/dev/work/data/gm_hazard/sites/23p1/non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.ll"
    )
    site_df = load_site_df(ll_ffp)

    avail_events = [
        cur_ffp.stem for cur_ffp in sim_site_corr_dir.iterdir() if cur_ffp.is_dir()
    ]

    # Event & IM selection
    col_1, col_2 = st.columns(2)
    with col_1:
        event = st.selectbox("Event", options=avail_events)

    avail_ims = get_avail_ims(sim_site_corr_dir, event)
    with col_2:
        im = st.selectbox("IM", options=avail_ims)

    sim_site_corrs = load_sim_site_correlations(sim_site_corr_dir, event, im)
    sites = sim_site_corrs.index.values.astype(str)
    im_residuals = load_sim_im_residuals(sim_site_corr_dir, event, im)

    st.markdown(f"# {event} - {im}")

    corr_tab, site_tab = st.tabs(["Correlations", "Site"])

    ### Correlation
    with corr_tab:
        st.header("Correlations")
        if not im.startswith("pSA"):
            st.markdown(f"No empirical model for {im} available")
            st.stop()

        # Compute the empirical site correlation
        dist_matrix = sh.im_dist.calculate_distance_matrix(sites, site_df)
        emp_site_corr = sh.im_dist.get_corr_matrix(sites, dist_matrix, IM.from_str(im))

        # Correlation diff
        corr_diff = sim_site_corrs - emp_site_corr

        st.markdown("### Within-event site correlation comparison")
        lower_tri_mask = np.tril(dist_matrix.values).astype(bool)

        # Get the model values
        dist = np.linspace(0, 300, 100)
        loth_baker_vals = sha.loth_baker_corr_model.get_correlations(im, im, dist)

        fig = plt.figure(figsize=(10, 6))
        plt.scatter(
            dist_matrix.values[lower_tri_mask], sim_site_corrs.values[lower_tri_mask]
        )
        plt.plot(dist, loth_baker_vals, c="k", linewidth=1.0)

        plt.xlabel(f"Distance (km)")
        plt.ylabel(f"Site-Correlation")
        plt.ylim(-1.0, 1.0)
        plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")
        plt.tight_layout()

        st.pyplot(fig)

        with st.expander("**Difference**"):
            st.markdown(
                r"$\Delta_{\hat{\rho}_{i, j, e}} = "
                r"\hat{\rho}_{i, j, e}^{(Simulation)} - "
                r"\hat{\rho}_{i, j}^{(Empirical)}$"
            )

            fig = plt.figure(figsize=(8, 6))

            plt.scatter(
                dist_matrix.values[lower_tri_mask], corr_diff.values[lower_tri_mask]
            )
            plt.xlabel(f"Site-to-Site Distance (km)")
            plt.ylabel(r"Correlation difference, $\Delta_{\hat{\rho}_{i, j, e}}$")
            plt.ylim(-1.0, 1.0)
            plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")
            plt.tight_layout()

            plt.tight_layout()
            st.pyplot(fig)

    ### Site specific
    with site_tab:
        fig = go.Figure(
            data=go.Scattermapbox(
                lat=site_df.loc[sites].lat,
                lon=site_df.loc[sites].lon,
                mode="markers",
                hovertext=sites,
                hoverinfo="text",
            )
        )
        # fig.update_mapboxes(fitbounds="locations")
        fig.update_layout(height=900, width=1400)
        fig.update_mapboxes(
            accesstoken="pk.eyJ1IjoiY3MyMyIsImEiOiJjbGtpeXIxNnkwbDQ3M25xbDFrZWFnNHo3In0.OD7TJ_1PegpGvCOCxfHsnA",
            center=dict(
                lat=site_df.loc[sites].lat.mean(), lon=site_df.loc[sites].lon.mean()
            ),
            zoom=8,
        )
        st.plotly_chart(fig)

        col_1, col_2 = st.columns(2)

        with col_1:
            site_1 = st.selectbox("Site 1", sites)
        with col_2:
            site_2 = st.selectbox("Site 2", sites)

        st.markdown(r"$\rho_{\delta W_{i, j}}=$" + f"{sim_site_corrs.loc[site_1, site_2]:.2f},\t"
                                                   f"Site-to-site distance: {dist_matrix.loc[site_1, site_2]:.2f}")


        fig = plt.figure(figsize=(8, 4.5))

        plt.plot([-3.5, 3.5], [-3.5, 3.5], c="k", linewidth=0.75, alpha=1.0, zorder=0)
        plt.scatter(im_residuals.loc[site_1], im_residuals.loc[site_2], s=15, zorder=1)

        plt.title(f"")
        plt.xlabel(f"Within-Event {site_1}")
        plt.ylabel(f"Within-Event {site_2}")
        plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")

        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)

        plt.tight_layout()


        st.pyplot(fig, use_container_width=False)

