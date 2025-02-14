import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def get_bias_residual_fig(figsize: tuple[float, float] = (16, 6), left: float = 0.05, right: float = 0.98, top: float = 0.98, bottom: float = 0.1):
    fig = plt.figure(figsize=figsize)

    main_grid = gridspec.GridSpec(1, 2, figure=fig, wspace=0.1)

    grid_bias = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=main_grid[0], wspace=0.03, width_ratios=[5, 1]
    )

    ax1 = fig.add_subplot(grid_bias[0])
    ax1.set_xlabel(f"Period (s)")
    ax1.set_ylabel(f"Bias")
    ax1.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax1.set_xscale("log")
    ax1.axhline(0, color="black")
    ax1.set_ylim(-1.0, 1.0)
    ax1.set_xlim(0.01, 10.0)

    ax2 = fig.add_subplot(grid_bias[1])
    ax2.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax2.set_yticklabels([])
    ax2.set_ylim(-1.0, 1.0)
    ax2.axhline(0, color="black")

    grid_residual = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=main_grid[1], wspace=0.03, width_ratios=[5, 1]
    )

    ax3 = fig.add_subplot(grid_residual[0])
    ax3.set_xlabel(f"Period (s)")
    ax3.set_ylabel(f"Residual Standard Deviation")
    ax3.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax3.set_xscale("log")
    ax3.set_ylim(0.0, 1.0)
    ax3.set_xlim(0.01, 10.0)

    ax4 = fig.add_subplot(grid_residual[1])
    ax4.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax4.set_yticklabels([])
    ax4.set_ylim(0.0, 1.0)

    # Remove general figure padding
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)

    return fig, ax1, ax2, ax3, ax4


def get_single_pSA_otherIMs_fig(figsize: tuple[float, float] = (16, 6), top: float = 1.0):
    fig = plt.figure(figsize=figsize)

    grid = gridspec.GridSpec(1, 2, figure=fig, wspace=0.05, width_ratios=[5, 1])

    ax1 = fig.add_subplot(grid[0])
    ax1.set_xlabel(f"Period (s)")
    ax1.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax1.set_xscale("log")
    ax1.set_xlim(0.01, 10.0)

    ax2 = fig.add_subplot(grid[1])
    ax2.grid(which="both", linewidth=0.5, alpha=0.5, linestyle="--")
    ax2.set_yticklabels([])
    ax2.set_ylim(-1.0, 1.0)

    # Remove general figure padding
    fig.subplots_adjust(left=0, right=1, top=top, bottom=0)

    return fig, ax1, ax2
