import os
from enum import Enum, auto, StrEnum

import numpy as np

R_EARTH = 6378.139

class ObsDataSource(StrEnum):
    NZGMDB = "NZGMDB"
    NGAWest2 = "NGAWest2"
    NGASubduction = "NGASubduction"


class NZGMDBVersion(StrEnum):
    v3p0 = "v3.0"
    v3p4 = "v3.4"
    v4p0 = "v4.0"
    v4p1 = "v4.1"
    v4p2 = "v4.2"
    v4p3 = "v4.3"
    v4p3_final = "v4.3_final"

class TectonicType(StrEnum):
    CRUSTAL = "crustal"
    SUBDUCTION_INTERFACE = "subduction_interface"
    SUBDUCTION_SLAB = "subduction_slab"
    OUTER_RISE = "outer_rise"
    MANTLE = "mantle"
    UNKNOWN = "unknown"

class IMSet(StrEnum):
    pSA = "pSA"
    all = "all"


class RankingMethod(Enum):
    emp_cMVN = 1
    sim_cMVN = 2

    # Same as sim_cMVN but uses correlation coefficients
    # from the empirical model
    sim_cMVN_emp_corr = 3

    ml_prob = 4
    ml_prob_per_im = 5

OQ_INPUT_COLUMNS = [
    "vs30",
    "rrup",
    "rjb",
    "z1pt0",
    "mag",
    "rake",
    "dip",
    "vs30measured",
    "ztor",
    "rx",
    "hypo_depth",
]


PERIODS = [
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.075,
    0.1,
    0.12,
    0.15,
    0.17,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.75,
    0.8,
    0.9,
    1.0,
    # 1.25,
    1.2,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    6.0,
    7.5,
    10.0,
]
PSA_KEYS = [f"pSA_{x}" for x in PERIODS]

NON_PSA_IMs = ["PGV", "PGA", "AI", "CAV", "Ds575", "Ds595"]
IMs = NON_PSA_IMs + PSA_KEYS

GNN_PRED_NON_PSA_KEYS = [
    f"{cur_key}_pred" for cur_key in NON_PSA_IMs
]
GNN_PRED_PSA_KEYS = [f"{cur_key}_pred" for cur_key in PSA_KEYS]
GNN_PRED_IM_KEYS = GNN_PRED_NON_PSA_KEYS + GNN_PRED_PSA_KEYS

GNN_PRED_STD_PSA_KEYS = [f"{cur_key}_pred_std" for cur_key in PSA_KEYS]

CIM_PRED_PSA_KEYS = [f"{cur_key}_cond_mean" for cur_key in PSA_KEYS]
CIM_PRED_STD_PSA_KEYS = [f"{cur_key}_cond_std" for cur_key in PSA_KEYS]

GMM_PRED_PSA_KEYS = [f"{cur_key}_mean" for cur_key in PSA_KEYS]


IM_SETS = {IMSet.pSA: PSA_KEYS, IMSet.all: IMs}

IM_WEIGTHS_SETS = {
    # 1/3 of weights for other IMs and 2/3 for pSA (inc. PGA)
    "all": np.asarray(
        [(1 / 3) / 5] * 5 + [(2 / 3) / (len(PSA_KEYS) + 1)] * (len(PSA_KEYS) + 1)
    ),
    "pSA": np.ones(len(PSA_KEYS)) * (1 / len(PSA_KEYS)),
}

COMPONENTS = ["090", "000", "ver"]

SCALAR_FEATURE_KEYS = {
    "event": ["mag", "is_subduction"],
    "site": ["vs30", "z1p0", "z2p5", "tsite"],
    "site_to_site": ["dist", "vs30_diff", "z1p0_diff", "z2p5_diff", "tsite_diff"],
    "event_site": ["rrup"],
    "event_site_to_site": ["angular_dist", "rrup_diff"],
}


PRE_PROCESS_CONFIG = {
    "mag": (2, 9),
    "vs30": (100, 1500),
    "z1p0": (0, 1500),
    "z2p5": (0, 11),
    "tsite": (0, 10),
    "rrup": (0, 200),
    "rx": (-200, 200),
    "vs30_diff": (-1400, 1400),
    "z1p0_diff": (-1350, 1350),
    "z2p5_diff": (-10.5, 10.5),
    "tsite_diff": (-7.10, 7.10),
}

MAG_BINS = [3.5, 4.5, 5.5, 8]
MAG_BIN_LABELS = ["$M_w$ 3.5 - 4.5", "$M_w$ 4.5 - 5.5", "$M_w$: 5.5 - 8"]
MAG_COLORS = ["blue", "purple", "red"]
# RRUP_BINS = [0, 30, 100, 250, 500]
# RRUP_BIN_LABELS = ["$R_{Rup}$ 0 - 30", "$R_{Rup}$ 30 - 100", "$R_{Rup}$ 100 - 250", "$R_{Rup}$ 250 - 500"]
RRUP_BINS = [0, 30, 100, 500]
RRUP_BIN_LABELS = ["$R_{Rup}$ 0 - 30", "$R_{Rup}$ 30 - 100", "$R_{Rup}$ 100 - 500"]
RRUP_COLORS = ["blue", "purple", "red"]
DOC_BINS = [0, 2.5, 4, 8]
DOC_BIN_LABELS = ["DoC 0 - 2.5", "DoC 2.5 - 4", "DoC 4 - 8"]
DOC_COLORS = ["blue", "purple", "red"]

SITE_TO_SITE_DIST_BINS = [0, 5, 10, 30]
SITE_TO_SITE_DIST_BIN_LABELS = ["0 - 5", "5 - 10", "10 - 30"]
SITE_TO_SITE_DIST_COLORS = ["blue", "purple", "red"]

LN_VS30_DIFF_BINS = [0, 0.25, 0.75, 2.5]
LN_VS30_DIFF_BIN_LABELS = ["0 - 0.25", "0.25 - 0.75", "0.75 - 2.5"]
LN_VS30_DIFF_COLORS = ["blue", "purple", "red"]


CHCH_REGION_EXTENDED_NS = [172.334220, 172.821381, -43.669494, -43.275246]
CHCH_REGION_EXTENDED_WE = [172.45, 172.78288300933386, -43.6, -43.45]
CHCH_REGION = [172.5, 172.821381, -43.63, -43.42]
CANTERBURY_REGION = [171.54, 173.12, -43.95, -43.22]
CANTERBURY_REGION_EXTENDED = [171.57297981348268, 173.21996004180335, -43.95952294853966, -42.88754722262727]
WELLINGTON_REGION = [172.639, 176.35, -42.427, -40.475]

REGION_MAPPINGS = {
    "canterbury": CANTERBURY_REGION,
    "canterbury_extended": CANTERBURY_REGION_EXTENDED,
    "wellington": WELLINGTON_REGION,
    "chch_extended_ns": CHCH_REGION_EXTENDED_NS,
    "chch_extended_we": CHCH_REGION_EXTENDED_WE,
    "chch": CHCH_REGION,
}

STATION_FN_NAME = "non_uniform_whole_nz_with_real_stations-hh400_v20p3_land"

FIG_SIZE = (16, 10)
if (env_figsize := os.environ.get("fig_size")) is not None:
    FIG_SIZE = [float(x) for x in env_figsize.split(",")]

FIG_FORMAT = "png"
if (env_fig_format := os.environ.get("fig_format")) is not None:
    FIG_FORMAT = env_fig_format

FIG_DPI = 300
if (env_fig_dpi := os.environ.get("fig_dpi")) is not None:
    FIG_DPI = int(env_fig_dpi)

FIG_FONT_SIZE = None
if (env_fig_font_size := os.environ.get("fig_font_size")) is not None:
    FIG_FONT_SIZE = int(env_fig_font_size)

FIG_LINEWIDTH = None
if (env_fig_linewidth := os.environ.get("fig_linewidth")) is not None:
    FIG_LINEWIDTH = float(env_fig_linewidth)

FIG_GROUP_LINEWIDTH = None
if (env_fig_group_linewidth := os.environ.get("fig_group_linewidth")) is not None:
    FIG_GROUP_LINEWIDTH = float(env_fig_group_linewidth)

GMT_FIG_FONT_LABEL = "14p,Helvetica,black"
if (env_gmt_fig_font_label := os.environ.get("gmt_fig_font_label")) is not None:
    GMT_FIG_FONT_LABEL = env_gmt_fig_font_label

GMT_FIG_FONT_ANNOT_PRIMARY = "11p,Helvetica,black"
if (env_gmt_fig_font_annot_primary := os.environ.get("gmt_fig_font_annot_primary")) is not None:
    GMT_FIG_FONT_ANNOT_PRIMARY = env_gmt_fig_font_annot_primary

GMT_SHOW_CB_LABEL = True
if (env_gmt_show_cb_label := os.environ.get("gmt_show_cb_label")) is not None:
    GMT_SHOW_CB_LABEL = env_gmt_show_cb_label.lower() in ("1", "true", "yes")

MW_RRUP_LIMITS = np.array([
    [3.5, 96.001584],
    [3.6, 95.9631833664],
    [3.7, 98.0],
    [3.8, 102.0],
    [3.9, 108.026902688],
    [4.0, 114.868566765],
    [4.1, 123.445238634],
    [4.2, 128.689599653],
    [4.3, 134.586833832],
    [4.4, 145.681117253],
    [4.5, 157.689926419],
    [4.6, 170.688647664],
    [4.7, 188.45405921],
    [4.8, 192.223140394],
    [4.9, 203.988734372],
    [5.0, 216.474476825],
    [5.1, 233.196675756],
    [5.2, 248.661128941],
    [5.3, 258.190038239],
    [5.4, 268.728407146],
    [5.5, 280.032818545],
    [5.6, 297.173067302],
    [5.7, 315.362436406],
    [5.8, 334.665140413],
    [5.9, 361.817781899],
    [6.0, 384.425050255],
    [6.1, 407.954938731],
    [6.2, 439.468601405],
    [6.3, 468.049881681],
    [6.4, 497.294793922],
    [6.5, 517.385503596],
    [6.6, 538.287877942],
    [6.7, 560.03470821],
    [6.8, 589.576829967],
    [6.9, 615.854723363],
    [7.0, 643.04652117],
    [7.1, 669.293317953],
    [7.2, 696.332767998],
    [7.3, 724.464611825],
    [7.4, 753.732982143],
    [7.5, 793.492788462],
    [7.6, 845.098386021],
    [7.7, 897.18404165],
    [7.8, 949.816128972],
    [7.9, 994.534662958],
    [8.0, 1055.40814061]
])
