from pathlib import Path

import pygmt
import matplotlib.pyplot as plt

import torch
import torch.nn as nn




# fig_1_ffp = Path("/Users/claudy/dev/work/tmp/tmp_paper_figures/3468575/gmm_3468575_pSA_0p01.png")
# fig_2_ffp = Path("/Users/claudy/dev/work/tmp/tmp_paper_figures/3468575/cim_3468575_pSA_0p01.png")
# fig_3_ffp = Path("/Users/claudy/dev/work/tmp/tmp_paper_figures/3468575/gnn_3468575_pSA_0p01.png")


# figsize = (8.25, 2.8)

# fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=figsize, dpi=300)

# ax1.imshow(plt.imread(fig_1_ffp), aspect="equal")
# ax1.axis("off")
# # if title_1 is not None:
#     # ax1.set_title(title_1)

# ax2.imshow(plt.imread(fig_2_ffp), aspect="equal")
# # ax2.set_xticks([])  # Remove x-axis ticks
# # ax2.set_yticks([])  # Remove y-axis ticks
# # ax2.set_xticklabels([])  # Remove x-axis labels
# # ax2.set_yticklabels([])  # Remove y-axis labels
# ax2.axis("off")
# # if title_2 is not None:
#     # ax2.set_title(title_2)

# ax3.imshow(plt.imread(fig_3_ffp), aspect="equal")
# # ax3.set_xticks([])  # Remove x-axis ticks
# # ax3.set_yticks([])  # Remove y-axis ticks
# # ax3.set_xticklabels([])  # Remove x-axis labels
# # ax3.set_yticklabels([])  # Remove y-axis labels
# ax3.axis("off")
# # if title_3 is not None:
#     # ax3.set_title(title_3)

# plt.subplots_adjust(wspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0) 

# # fig.tight_layout()

# fig.savefig("/Users/claudy/dev/work/tmp/tmp_paper_figures/3468575/combined.png")
