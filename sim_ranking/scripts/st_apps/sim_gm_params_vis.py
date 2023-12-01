from pathlib import Path

import pandas as pd
import numpy as np

import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import typer

from ml_tools.st_tools import utils as st_utils
import spatial_hazard as sh
from gmhazard_calc.im import IM
import sha_calc as sha
import sim_ranking as sr


@st.cache_data
def get_avail_ims(sim_site_corr_dir: Path, event: str):
    return sr.constants.PSA_KEYS
    # sim_site_corrs = load_sim_site_correlations(sim_site_corr_dir, event)
    # return sim_site_corrs.ims


@st.cache_data
def load_sim_site_correlations(sim_site_corr_dir: Path, event: str):
    sim_site_corrs = sr.data.SiteCorrelations.load(
        sim_site_corr_dir / f"{event}.pickle"
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


def get_site_df(db_ffp: Path):
    db = sr.db.DB(db_ffp)
    return db.get_site_df()


def main(sim_site_corr_dir: Path, sim_gm_params_dir: Path, db_ffp: Path):

    st_utils.update_st_width(1600, 2, 0, 1, 1)

    site_df = get_site_df(db_ffp)

    avail_events = [
        cur_ffp.stem
        for cur_ffp in sim_site_corr_dir.iterdir()
        if not cur_ffp.stem.startswith("_")
    ]

    # Event & IM selection
    col_1, col_2 = st.columns(2)
    with col_1:
        event = st.selectbox("Event", options=avail_events)

    avail_ims = get_avail_ims(sim_site_corr_dir, event)
    with col_2:
        im = st.selectbox("IM", options=avail_ims)

    sim_site_corrs = load_sim_site_correlations(sim_site_corr_dir, event)
    cur_sim_site_corrs = sim_site_corrs.to_im_dict()[im]
    sim_gm_params = load_sim_gm_params(sim_gm_params_dir, event)

    sites = cur_sim_site_corrs.index.values.astype(str)
    sites = np.intersect1d(sites, site_df.index.values.astype(str))

    cur_sim_site_corrs = cur_sim_site_corrs.loc[sites, sites]
    dist_matrix = sh.im_dist.calculate_distance_matrix(sites, site_df)

    assert np.all(dist_matrix.index == cur_sim_site_corrs.index) and np.all(
        dist_matrix.columns == cur_sim_site_corrs.columns
    )

    st.markdown(f"# {event}")
    corr_tab, site_tab, residuals_tab = st.tabs(
        ["Correlations General", "Site-Specific Correlations", "Residuals"]
    )

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

            # Get modified Loth & Baker values
            # based on eq 11.8 and tau and phi (estimates) from Ask14
            lb_tau = 0.5
            cp = IM.from_str(im).period
            if cp < 0.1:
                lb_phi = 0.65
            elif cp < 1.0:
                lb_phi = 0.6
            else:
                lb_phi = 0.55
            lb_updated = (loth_baker_vals * lb_phi ** 2 + lb_tau ** 2) / np.sqrt(
                lb_phi ** 2 + lb_tau ** 2
            )

            # Compute the moving average
            sim_avg_values = []
            sim_std_values = []
            n_bins = 10
            bins = np.logspace(np.log10(1), np.log10(200), n_bins)
            bin_inds = np.digitize(dist_matrix.values[lower_tri_mask], bins)
            for ix in np.unique(bin_inds):
                if ix == 0 or ix == n_bins:
                    continue

                cur_mask = bin_inds == ix
                sim_avg_values.append(
                    np.mean(cur_sim_site_corrs.values[lower_tri_mask][cur_mask])
                )
                sim_std_values.append(
                    np.std(cur_sim_site_corrs.values[lower_tri_mask][cur_mask])
                )

            sim_avg_values = np.asarray(sim_avg_values)
            sim_std_values = np.asarray(sim_std_values)
            bin_centres = np.asarray(
                [np.mean(bins[i : i + 2]) for i in range(n_bins - 1)]
            )

            fig = plt.figure(figsize=(10, 6))
            plt.scatter(
                dist_matrix.values[lower_tri_mask],
                cur_sim_site_corrs.values[lower_tri_mask],
                s=1.0,
                alpha=0.75,
            )
            plt.semilogx(dist, loth_baker_vals, c="k", linewidth=1.0)
            plt.semilogx(dist, lb_updated, c="k", linestyle="--", linewidth=1.0)

            plt.semilogx(
                bin_centres,
                sim_avg_values,
                c="r",
                linewidth=1.0,
                label="Simulation Average & Standard Deviation",
            )
            plt.semilogx(
                bin_centres,
                sim_avg_values + sim_std_values,
                c="r",
                linewidth=1.0,
                linestyle="--",
            )
            plt.semilogx(
                bin_centres,
                sim_avg_values - sim_std_values,
                c="r",
                linewidth=1.0,
                linestyle="--",
            )

            plt.xlabel(f"Distance (km)")
            plt.ylabel(f"Site-Correlation")
            plt.ylim(-1.0, 1.0)
            plt.xlim(1.0, 200)
            plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")
            plt.tight_layout()

            st.pyplot(fig)

        else:
            st.markdown(f"No empirical model for {im} available")

        st.markdown("### pSA correlation")
        max_dist = st.slider("Max Distance (km)", 0, 300,10, 1)

        mean_values = {}
        std_values = {}
        for cur_im in avail_ims:
            cur_corr = sim_site_corrs.to_im_dict()[cur_im].loc[sites, sites]
            cur_corr = cur_corr.values[
                (dist_matrix <= max_dist).values & lower_tri_mask
            ]

            mean_values[cur_im] = np.mean(cur_corr)
            std_values[cur_im] = np.std(cur_corr)

        mean_df = pd.Series(mean_values)
        std_df = pd.Series(std_values)

        fig = plt.figure(figsize=(10, 6))

        plt.semilogx(
            sr.constants.PERIODS,
            mean_df,
            label=f"{max_dist} km",
            c="b"
        )
        plt.semilogx(
            sr.constants.PERIODS,
            mean_df.loc[sr.constants.PSA_KEYS]
            + std_df.loc[sr.constants.PSA_KEYS],
            c="b",
            linestyle="--",
        )
        plt.semilogx(
            sr.constants.PERIODS,
            mean_df.loc[sr.constants.PSA_KEYS]
            - std_df.loc[sr.constants.PSA_KEYS],
            c="b",
            linestyle="--",
        )

        plt.xlabel("Period (s)")
        plt.ylabel("Site-Correlation")
        plt.ylim(-1.0, 1.0)
        plt.xlim(0.01, 10)
        plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")
        plt.legend()
        plt.tight_layout()

        st.pyplot(fig, use_container_width=False)

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
            st.markdown(
                f"""
            ### Within-event residuals for two sites
            Figure shows the {im} within-event residuals for the two selected sites {site_1} and {site_2}.
            """
            )
            fig = plt.figure(figsize=(8, 4.5))

            plt.plot(
                [-3.5, 3.5], [-3.5, 3.5], c="k", linewidth=0.75, alpha=1.0, zorder=0
            )
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
            st.markdown(
                f"""
            ### Correlation coefficient as function of period for **{site_1}** and **{site_2}**
            Figure shows the correlation coefficient as a function of period for the two selected sites {site_1} and {site_2}.
            """
            )
            pSA_keys = [
                cur_im for cur_im in sim_site_corrs.ims if cur_im.startswith("pSA")
            ]

            periods = [float(cur_key.split("_")[-1]) for cur_key in pSA_keys]
            sort_ind = np.argsort(periods)
            periods = np.array(periods)[sort_ind]
            pSA_keys = np.array(pSA_keys)[sort_ind]

            period_sim_corr_values = sim_site_corrs.get_site_im_corrs(
                site_1, pSA_keys
            ).loc[site_2, :]

            # period_sim_corr_values = [
            #     sim_site_corrs.corrs[cur_im].loc[site_1, site_2]
            #     for cur_im in pSA_keys
            # ]

            period_emp_corr_values = [
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
            plt.ylim(-1, 1)
            plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")
            plt.legend()
            plt.tight_layout()

            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

    with residuals_tab:
        with st.expander("Summary"):

            if sim_gm_params.event_residuals is not None:
                # Between-event residuals
                st.markdown(
                    """
                ## Between-Event Residuals
                Shows the between-event residuals for each realisation (gray lines), the mean, and $\\tau$.
                """
                )
                fig = plt.figure(figsize=(8, 4.5))

                for cur_rel, cur_row in sim_gm_params.event_residuals.iterrows():
                    plt.semilogx(
                        periods, cur_row.loc[pSA_keys], c="k", alpha=0.1, linewidth=0.5
                    )

                plt.semilogx(
                    periods,
                    sim_gm_params.event_residuals.loc[:, pSA_keys].mean(),
                    c="b",
                    linewidth=1.0,
                )
                plt.semilogx(
                    periods,
                    sim_gm_params.event_residuals.loc[:, pSA_keys].mean()
                    + sim_gm_params.bias_std.tau.loc[pSA_keys],
                    c="b",
                    linewidth=0.75,
                    linestyle="--",
                )
                plt.semilogx(
                    periods,
                    sim_gm_params.event_residuals.loc[:, pSA_keys].mean()
                    - sim_gm_params.bias_std.tau.loc[pSA_keys],
                    c="b",
                    linewidth=0.75,
                    linestyle="--",
                )

                plt.xlabel(f"Period (s)")
                plt.ylabel("Between-event residual")
                plt.xlim([0.01, 10])
                plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")
                plt.tight_layout()

                st.pyplot(fig, use_container_width=False)
                plt.close(fig)
            else:
                st.markdown("No between-event residuals available")

            # Within-event residuals
            st.markdown(
                """
            ## Within-Event Residuals
            Shows the within-event residuals for each realisation and site (gray lines), the mean, and $\\phi$.
            Note: Only showing every 3rd record.
            """
            )
            fig = plt.figure(figsize=(8, 4.5))

            for i, (cur_record, cur_row) in enumerate(
                sim_gm_params.within_residuals.iterrows()
            ):
                if i % 3 == 0:
                    plt.semilogx(
                        periods, cur_row.loc[pSA_keys], c="k", alpha=0.1, linewidth=0.5
                    )

            plt.semilogx(
                periods,
                sim_gm_params.within_residuals.loc[:, pSA_keys].mean(),
                c="b",
                linewidth=1.0,
            )
            if sim_gm_params.bias_std is not None:
                plt.semilogx(
                    periods,
                    sim_gm_params.within_residuals.loc[:, pSA_keys].mean()
                    + sim_gm_params.bias_std.phi_w.loc[pSA_keys],
                    c="b",
                    linewidth=0.75,
                    linestyle="--",
                )
                plt.semilogx(
                    periods,
                    sim_gm_params.within_residuals.loc[:, pSA_keys].mean()
                    - sim_gm_params.bias_std.phi_w.loc[pSA_keys],
                    c="b",
                    linewidth=0.75,
                    linestyle="--",
                )

            plt.xlabel(f"Period (s)")
            plt.ylabel("Within-event residual")
            plt.xlim([0.01, 10])
            plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")
            plt.tight_layout()

            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

        with st.expander("IM specific"):
            if sim_gm_params.event_residuals is not None:
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


if __name__ == "__main__":
    typer.run(main)
