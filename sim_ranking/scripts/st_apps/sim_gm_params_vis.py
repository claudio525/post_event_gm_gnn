import os
from pathlib import Path

import pandas as pd
import numpy as np

import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from ml_tools.st_tools import utils as st_utils
import spatial_hazard as sh
from gmhazard_calc.im import IM
import sha_calc as sha
import sim_ranking as sr


@st.cache_data
def get_avail_ims(sim_site_corr_dir: Path, event: str):
    return np.sort(
        [
            sr.utils.reverse_im_filename(cur_ffp.stem)
            for cur_ffp in (sim_site_corr_dir / event).iterdir()
            if cur_ffp.is_file()
        ]
    )


@st.cache_data
def load_sim_site_correlations(sim_site_corr_dir: Path, event: str):
    sim_site_corrs = sr.data.SimWithinEventSiteCorrelations.load(
        sim_site_corr_dir / event
    )

    return sim_site_corrs


@st.cache_data
def load_sim_gm_params(sim_gm_params_dir: Path, event: str):
    sim_gm_params = sr.data.SimGMParams.load(sim_gm_params_dir / event)

    return sim_gm_params


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

    sim_site_corr_dir = Path(os.path.expandvars("$wdata/sim_ranking/sim_correlations"))
    sim_gm_params_dir = Path(os.path.expandvars("$wdata/sim_ranking/sim_gm_params"))

    ll_ffp = Path(
        os.path.expandvars("$wdata/gm_hazard/sites/23p1/non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.ll")
    )
    site_df = load_site_df(ll_ffp)

    avail_events = [
        cur_ffp.stem
        for cur_ffp in sim_site_corr_dir.iterdir()
        if cur_ffp.is_dir() and not cur_ffp.stem.startswith("_")
    ]

    # Event & IM selection
    col_1, col_2 = st.columns(2)
    with col_1:
        event = st.selectbox("Event", options=avail_events)

    avail_ims = get_avail_ims(sim_site_corr_dir, event)
    with col_2:
        im = st.selectbox("IM", options=avail_ims)

    sim_site_corrs = load_sim_site_correlations(sim_site_corr_dir, event)
    cur_sim_site_corrs = sim_site_corrs.correlations[im]
    sim_gm_params = load_sim_gm_params(sim_gm_params_dir, event)

    sites = cur_sim_site_corrs.index.values.astype(str)
    dist_matrix = sh.im_dist.calculate_distance_matrix(sites, site_df)

    st.markdown(f"# {event}")
    corr_tab, site_tab, residuals_tab = st.tabs(["Correlations General", "Site-Specific Correlations", "Residuals"])

    ### Correlation
    with corr_tab:
        st.header("Correlations")
        if im.startswith("pSA"):
            # Compute the empirical site correlation

            emp_site_corr = sh.im_dist.get_corr_matrix(
                sites, dist_matrix, IM.from_str(im)
            )

            # Correlation diff
            corr_diff = cur_sim_site_corrs - emp_site_corr

            st.markdown("### Within-event site correlation comparison")
            lower_tri_mask = np.tril(dist_matrix.values).astype(bool)

            # Get the model values
            dist = np.linspace(0, 300, 100)
            loth_baker_vals = sha.loth_baker_corr_model.get_correlations(im, im, dist)

            fig = plt.figure(figsize=(10, 6))
            plt.scatter(
                dist_matrix.values[lower_tri_mask],
                cur_sim_site_corrs.values[lower_tri_mask],
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
        else:
            st.markdown(f"No empirical model for {im} available")

    ### Site specific correlations
    with site_tab:
        with st.expander("Site Map"):
            fig = go.Figure(
                data=go.Scattermapbox(
                    lat=site_df.loc[sites].lat,
                    lon=site_df.loc[sites].lon,
                    mode="markers",
                    marker=dict(size=10),
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

        st.markdown(
            r"$\rho_{\delta W_{i, j}}=$"
            + f"{cur_sim_site_corrs.loc[site_1, site_2]:.2f},\t"
            f"Site-to-site distance: {dist_matrix.loc[site_1, site_2]:.2f}"
        )

        # Get the residuals
        site_1_residuals = sim_gm_params.within_residuals.loc[
            sim_gm_params.within_residuals.site == site_1
        ]
        site_2_residuals = sim_gm_params.within_residuals.loc[
            sim_gm_params.within_residuals.site == site_2
        ]
        assert np.all(site_1_residuals.rel.values == site_2_residuals.rel.values)

        ### Site-specific within-event residuals
        with st.expander("IM specific"):
            st.markdown(f"""
            ### Within-event residuals for two sites
            Figure shows the {im} within-event residuals for the two selected sites {site_1} and {site_2}.
            """)
            fig = plt.figure(figsize=(8, 4.5))

            plt.plot([-3.5, 3.5], [-3.5, 3.5], c="k", linewidth=0.75, alpha=1.0, zorder=0)
            plt.scatter(
                site_1_residuals[im],
                site_2_residuals[im],
                s=15,
                zorder=1,
            )

            plt.title(im)
            plt.xlabel(f"Within-Event {site_1}")
            plt.ylabel(f"Within-Event {site_2}")
            plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")

            plt.xlim(-1.5, 1.5)
            plt.ylim(-1.5, 1.5)
            plt.tight_layout()

            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

        ### Correlation coefficient as function of period
        with st.expander("pSA"):
            st.markdown(f"""
            ### Correlation coefficient as function of period
            Figure shows the correlation coefficient as a function of period for the two selected sites {site_1} and {site_2}.
            """)
            pSA_keys = [
                cur_im
                for cur_im in sim_site_corrs.correlations.keys()
                if cur_im.startswith("pSA")
            ]
            periods = [float(cur_key.split("_")[-1]) for cur_key in pSA_keys]
            sort_ind = np.argsort(periods)
            periods = np.array(periods)[sort_ind]
            pSA_keys = np.array(pSA_keys)[sort_ind]

            period_sim_corr_values = [
                sim_site_corrs.correlations[cur_im].loc[site_1, site_2]
                for cur_im in pSA_keys
            ]
            period_emp_corr_values = loth_baker_vals = [
                sha.loth_baker_corr_model.get_correlations(
                    cur_im, cur_im, np.asarray([dist_matrix.loc[site_1, site_2]])
                )
                for cur_im in pSA_keys
            ]

            fig = plt.figure(figsize=(8, 4.5))
            plt.semilogx(
                periods,
                period_sim_corr_values,
                label="Simulation-based",
            )
            plt.semilogx(
                periods,
                period_emp_corr_values,
                label="Loth & Baker (2013)",
            )

            plt.xlabel(f"Period (s)")
            plt.ylabel(f"Within-event Site-Correlation")
            plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")
            plt.legend()
            plt.tight_layout()

            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

    with residuals_tab:
        with st.expander("Summary"):

            # Between-event residuals
            st.markdown("""
            ## Between-Event Residuals
            Shows the between-event residuals for each realisation (gray lines), the mean, and $\\tau$.
            """)
            fig = plt.figure(figsize=(8, 4.5))

            for cur_rel, cur_row in sim_gm_params.event_residuals.iterrows():
                plt.semilogx(periods, cur_row.loc[pSA_keys], c="k", alpha=0.1, linewidth=0.5)

            plt.semilogx(
                periods,
                sim_gm_params.event_residuals.loc[:, pSA_keys].mean(),
                c="b",
                linewidth=1.0,
            )
            plt.semilogx(periods, sim_gm_params.event_residuals.loc[:, pSA_keys].mean() +
                         sim_gm_params.bias_std.tau.loc[pSA_keys], c="b", linewidth=0.75, linestyle="--")
            plt.semilogx(periods, sim_gm_params.event_residuals.loc[:, pSA_keys].mean() -
                            sim_gm_params.bias_std.tau.loc[pSA_keys], c="b", linewidth=0.75, linestyle="--")

            plt.xlabel(f"Period (s)")
            plt.ylabel(r"$\delta B$")
            plt.xlim([0.01, 10])
            plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")
            plt.tight_layout()

            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

            # Within-event residuals
            st.markdown("""
            ## Within-Event Residuals
            Shows the within-event residuals for each realisation and site (gray lines), the mean, and $\\phi$.
            Note: Only showing every 3rd record.
            """)
            fig = plt.figure(figsize=(8, 4.5))

            for i, (cur_record, cur_row) in enumerate(sim_gm_params.within_residuals.iterrows()):
                if i % 3 == 0:
                    plt.semilogx(periods, cur_row.loc[pSA_keys], c="k", alpha=0.1, linewidth=0.5)

            plt.semilogx(
                periods,
                sim_gm_params.within_residuals.loc[:, pSA_keys].mean(),
                c="b",
                linewidth=1.0,
            )
            plt.semilogx(
                periods,
                sim_gm_params.within_residuals.loc[:, pSA_keys].mean() + sim_gm_params.bias_std.phi_w.loc[pSA_keys],
                c="b",
                linewidth=0.75,
                linestyle="--",
            )
            plt.semilogx(
                periods,
                sim_gm_params.within_residuals.loc[:, pSA_keys].mean() - sim_gm_params.bias_std.phi_w.loc[pSA_keys],
                c="b",
                linewidth=0.75,
                linestyle="--",
            )


            plt.xlabel(f"Period (s)")
            plt.ylabel(f"$\delta W$")
            plt.xlim([0.01, 10])
            plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")
            plt.tight_layout()

            st.pyplot(fig, use_container_width=False)
            plt.close(fig)


            print(f"wtf")


        with st.expander("IM specific"):
            # Between-event residuals
            fig = plt.figure(figsize=(8, 4.5))

            plt.scatter(
                np.arange(sim_gm_params.event_residuals.shape[0]),
                sim_gm_params.event_residuals[im],
            )

            plt.xlabel(f"Realisation")
            plt.ylabel(f"{im} Between-realisation residuals")
            plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")
            plt.ylim([-2, 2])
            plt.tight_layout()

            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

            # Within-event residuals
            fig = plt.figure(figsize=(8, 4.5))

            plt.scatter(
                np.arange(sim_gm_params.within_residuals.shape[0]),
                sim_gm_params.within_residuals[im],
            )

            plt.xlabel(f"Record")
            plt.ylabel(f"{im} Within-record residuals")
            plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")
            plt.ylim([-2, 2])
            plt.tight_layout()

            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

            st.markdown("Each blob is a site")
