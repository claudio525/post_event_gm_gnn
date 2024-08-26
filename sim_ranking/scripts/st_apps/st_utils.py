import os
from pathlib import Path
from typing import Sequence

import streamlit as st
import numpy as np
import pandas as pd

import sim_ranking as sr
import ml_tools as mlt


@st.cache_data
def cim_load_cmvn_result(results_dir: Path):
    ffp = results_dir / "cMVN_distributions.pickle"
    if ffp.exists():
        return sr.conditional.ConditionalMVNDistribution.load(ffp)
    return None


def scatter_options_form(df: pd.DataFrame, cols: Sequence[str], key_prefix: str):
    COLORS = ["green", "blue", "red", "black"]
    with st.form(key=f"{key_prefix}_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            ## X-axis
            st.markdown("### X-axis")
            x_axis = st.selectbox("Select X-axis", cols)

            # Axis limits
            c1, c2 = st.columns(2)
            with c1:
                x_min_use_qt = st.checkbox(
                    "Use Quantiles", key=f"{key_prefix}_x_min_use_qt"
                )
                x_min = st.number_input(
                    "Min", key=f"{key_prefix}_x_min", value=float(df[x_axis].min())
                )
                st.text(f"Min data value: {float(df[x_axis].min()):.4f}")
            with c2:
                x_max_use_qt = st.checkbox(
                    "Use Quantiles", key=f"{key_prefix}_x_max_use_qt"
                )
                x_max = st.number_input(
                    "Max", key=f"{key_prefix}_x_max", value=float(df[x_axis].max())
                )
                st.text(f"Max data value: {float(df[x_axis].max()):.4f}")

            st.divider()

            ## Other Options
            st.markdown("### Other Options")
            alpha = st.number_input("Alpha", 0.0, 1.0, 1.0, key=f"{key_prefix}_alpha")
            marker_size = st.number_input(
                "Marker Size", 1, 100, 10, key=f"{key_prefix}_marker_size"
            )

        with col2:
            ## Y-axis
            st.markdown("### Y-axis")
            y_axis = st.selectbox("Select Y-axis", cols, key=f"{key_prefix}_y_axis")

            # Axis limits
            c1, c2 = st.columns(2)
            with c1:
                y_min_use_qt = st.checkbox(
                    "Use Quantiles", key=f"{key_prefix}_y_min_use_qt"
                )
                y_min = st.number_input(
                    "Min", key=f"{key_prefix}_y_min", value=float(df[y_axis].min())
                )
                st.text(f"Min data value: {float(df[y_axis].min()):.4f}")
            with c2:
                y_max_use_qt = st.checkbox(
                    "Use Quantiles", key=f"{key_prefix}_y_max_use_qt"
                )
                y_max = st.number_input(
                    "Max", key=f"{key_prefix}_y_max", value=float(df[y_axis].max())
                )
                st.text(f"Max data value: {float(df[y_axis].max()):.4f}")

            st.divider()

            ## Trend line
            st.markdown("### Binned trend line")
            c1, c2 = st.columns(2)
            with c1:
                show_trend_mean_line = st.checkbox(
                    "Show Binned Trend Line",
                    key=f"{key_prefix}_show_trend_mean_line",
                )
            with c2:
                show_trend_std_line = st.checkbox(
                    "Show Binned Trend Std Line",
                    key=f"{key_prefix}_show_trend_std_line",
                )
            trend_n_bins = st.number_input(
                "Number of Bins", 1, 100, 10, key=f"{key_prefix}_trend_n_bins"
            )

            trend_line_style = st.selectbox(
                "Trend Line Style",
                ["-", "--", "-.", ":"],
                key=f"{key_prefix}_trend_line_style",
            )
            trend_line_width = st.number_input(
                "Trend Line Width",
                0.5,
                10.0,
                1.0,
                key=f"{key_prefix}_trend_line_width",
            )
            trend_color = st.selectbox(
                "Trend Line Color", COLORS, key=f"{key_prefix}_trend_color"
            )

        with col3:
            ## Color
            st.markdown("### Color")
            st.markdown("#### Fixed Color")
            fixed_color = st.checkbox(
                "Use Fixed Color", key=f"{key_prefix}_use_fixed_color"
            )
            color = st.selectbox("Select Color:", COLORS, key=f"{key_prefix}_color")

            st.divider()

            st.markdown("#### Colormap")
            color_axis = st.selectbox(
                "Select Color Axis", cols, key=f"{key_prefix}_color_axis"
            )
            cmap = st.selectbox(
                "Select Color Map",
                [
                    "viridis",
                    "plasma",
                    "inferno",
                    "magma",
                    "cividis",
                    "Blues",
                    "Blues_r",
                ],
                key=f"{key_prefix}_cmap",
            )

            # Axis limits
            c1, c2 = st.columns(2)
            with c1:
                vmin_use_qt = st.checkbox(
                    "Use Quantiles", key=f"{key_prefix}_color_min_use_qt"
                )
                vmin = st.number_input(
                    "Min",
                    key=f"{key_prefix}_color_min",
                    value=float(df[color_axis].min()),
                )
                st.text(f"Min data value: {float(df[color_axis].min()):.4f}")
            with c2:
                vmax_use_qt = st.checkbox(
                    "Use Quantiles", key=f"{key_prefix}_color_max_use_qt"
                )
                vmax = st.number_input(
                    "Max",
                    key=f"{key_prefix}_color_max",
                    value=float(df[color_axis].max()),
                )
                st.text(f"Max data value: {float(df[color_axis].max()):.4f}")

        submitted = st.form_submit_button("Submit")
        if submitted:
            scatter_options = mlt.plotting.ScatterOptions(
                x_axis=x_axis,
                x_min_use_qt=x_min_use_qt,
                x_max_use_qt=x_max_use_qt,
                x_min=x_min,
                x_max=x_max,
                y_axis=y_axis,
                y_min_use_qt=y_min_use_qt,
                y_max_use_qt=y_max_use_qt,
                y_min=y_min,
                y_max=y_max,
                use_fixed_color=fixed_color,
                color=color,
                color_axis=color_axis,
                cmap=cmap,
                vmin_use_qt=vmin_use_qt,
                vmax_use_qt=vmax_use_qt,
                vmin=vmin,
                vmax=vmax,
                alpha=alpha,
                marker_size=marker_size,
                show_trend_mean_line=show_trend_mean_line,
                show_trend_std_line=show_trend_std_line,
                trend_n_bins=trend_n_bins,
                trend_line_style=trend_line_style,
                trend_line_width=trend_line_width,
                trend_color=trend_color,
            )
            return scatter_options
    return None
