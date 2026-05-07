import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def get_pSA_bias_residual_fig(
    figsize: tuple[float, float] = (16, 6),
    fig_dpi: int | None = None,
    left: float = 0.05,
    right: float = 0.98,
    top: float = 0.98,
    bottom: float = 0.1,
    main_wspace: float = 0.1,
    bias_y_axis_limits: tuple[float, float] = (-1.0, 1.0),
    std_y_axis_limits: tuple[float, float] = (0.0, 1.0),
):
    """
    Create a figure for pSA bias and residual standard deviation plots.

    Parameters
    ----------
    figsize : tuple of float, optional
        Size of the figure.
    bias_y_axis_limits : tuple of float, optional
        Y-axis limits for the bias plot.
    std_y_axis_limits : tuple of float, optional
        Y-axis limits for the residual standard deviation plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax1 : matplotlib.axes.Axes
        Axis for the bias plot.
    ax2 : matplotlib.axes.Axes
        Axis for the residual standard deviation plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=fig_dpi)

    ax1.set_xlabel("Vibration Period, T(s)")
    ax1.set_ylabel("Model bias")
    ax1.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax1.set_xscale("log")
    ax1.axhline(0, color="black", zorder=0)
    ax1.set_ylim(*bias_y_axis_limits)
    ax1.set_xlim(0.01, 10.0)

    ax2.set_xlabel("Vibration Period, T(s)")
    ax2.set_ylabel("Residual standard deviation")
    ax2.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax2.set_xscale("log")
    ax2.set_ylim(*std_y_axis_limits)
    ax2.set_xlim(0.01, 10.0)

    fig.subplots_adjust(
        left=left, right=right, top=top, bottom=bottom, wspace=main_wspace
    )
    return fig, ax1, ax2


def get_bias_residual_fig(
    figsize: tuple[float, float] = (16, 6),
    fig_dpi: int | None = None,
    left: float = 0.05,
    right: float = 0.98,
    top: float = 0.98,
    bottom: float = 0.1,
    main_wspace: float = 0.1,
    sub_wspace: float = 0.03,
    bias_y_axis_limits: tuple[float, float] = (-1.0, 1.0),
    std_y_axis_limits: tuple[float, float] = (0.0, 1.0),
):
    """
    Create a figure a bias and residual plots for
    pSA and non-pSA IMs

    Parameters
    ----------
    figsize : tuple of float, optional
        Size of the total figure.
    left : float, optional
        Left margin of the figure.
    right : float, optional
        Right margin of the figure.
    top : float, optional
        Top margin of the figure.
    bottom : float, optional
        Bottom margin of the figure.
    main_wspace : float, optional
        Space between the main plots.
        I.e. space between ax2 and ax3
    sub_wspace : float, optional
        Space between the subplots.
        I.e. space between ax1 and ax2
        and between ax3 and ax4

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax1 : matplotlib.axes.Axes
        Axis for the pSA bias plot.
    ax2 : matplotlib.axes.Axes
        Axis for the non-pSA bias plot.
    ax3 : matplotlib.axes.Axes
        Axis for the pSA residual standard deviation plot.
    ax4 : matplotlib.axes.Axes
        Axis for the non-pSA residual standard deviation plot.
    """
    fig = plt.figure(figsize=figsize, dpi=fig_dpi)

    main_grid = gridspec.GridSpec(1, 2, figure=fig, wspace=main_wspace)

    grid_bias = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=main_grid[0], wspace=sub_wspace, width_ratios=[5, 1]
    )

    ax1 = fig.add_subplot(grid_bias[0])
    ax1.set_xlabel("Vibration Period, T(s)")
    ax1.set_ylabel("Model bias")
    ax1.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax1.set_xscale("log")
    ax1.axhline(0, color="black", zorder=0)
    ax1.set_ylim(*bias_y_axis_limits)
    ax1.set_xlim(0.01, 10.0)

    ax2 = fig.add_subplot(grid_bias[1])
    ax2.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax2.set_yticklabels([])
    ax2.set_ylim(*bias_y_axis_limits)
    ax2.axhline(0, color="black", zorder=0)
    ax2.tick_params(axis="y", which="both", length=0)

    grid_residual = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=main_grid[1], wspace=sub_wspace, width_ratios=[5, 1]
    )

    ax3 = fig.add_subplot(grid_residual[0])
    ax3.set_xlabel("Vibration Period, T(s)")
    ax3.set_ylabel("Residual standard deviation")
    ax3.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax3.set_xscale("log")
    ax3.set_ylim(*std_y_axis_limits)
    ax3.set_xlim(0.01, 10.0)

    ax4 = fig.add_subplot(grid_residual[1])
    ax4.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax4.set_yticklabels([])
    ax4.set_ylim(*std_y_axis_limits)
    ax4.tick_params(axis="y", which="both", length=0)

    # Remove general figure padding
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)

    return fig, ax1, ax2, ax3, ax4


def get_single_pSA_otherIMs_fig(
    figsize: tuple[float, float] = (16, 6),
    fig_dpi: int | None = None,
    left: float = 0.0,
    right: float = 1.0,
    top: float = 1.0,
    bottom: float = 0.0,
    wspace: float = 0.05,
):
    """
    Create figure for pSA and non-pSA IMs plots.

    Parameters
    ----------
    figsize : tuple of float, optional
        Size of the figure.
    top : float, optional
        Top margin of the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax1 : matplotlib.axes.Axes
        Axis for the single pSA plot.
    ax2 : matplotlib.axes.Axes
        Axis for the other IMs plot.
    """
    fig = plt.figure(figsize=figsize, dpi=fig_dpi)

    grid = gridspec.GridSpec(1, 2, figure=fig, wspace=0.05, width_ratios=[5, 1])

    ax1 = fig.add_subplot(grid[0])
    ax1.set_xlabel("Period (s)")
    ax1.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax1.set_xscale("log")
    ax1.set_xlim(0.01, 10.0)

    ax2 = fig.add_subplot(grid[1])
    ax2.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax2.set_yticklabels([])
    ax2.set_ylim(-1.0, 1.0)

    # Remove general figure padding
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=wspace)

    return fig, ax1, ax2
