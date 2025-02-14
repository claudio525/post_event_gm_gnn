import os
from enum import Enum, auto, StrEnum

import numpy as np

class ObsDataSource(StrEnum):
    NZGMDB = "NZGMDB"
    NGAWest2 = "NGAWest2"
    NGASubduction = "NGASubduction"


class NZGMDBVersion(StrEnum):
    v3p0 = "v3.0"
    v3p4 = "v3.4"
    v4p0 = "v4.0"
    v4p1 = "v4.1"


class TectonicType(StrEnum):
    CRUSTAL = "crustal"
    SUBDUCTION_INTERFACE = "subduction_interface"
    SUBDUCTION_SLAB = "subduction_slab"
    OUTER_RISE = "outer_rise"
    MANTLE = "mantle"
    UNKNOWN = "unknown"


class RankingMethod(Enum):
    emp_cMVN = 1
    sim_cMVN = 2

    # Same as sim_cMVN but uses correlation coefficients
    # from the empirical model
    sim_cMVN_emp_corr = 3

    ml_prob = 4
    ml_prob_per_im = 5


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

GNN_PRED_PSA_KEYS = [f"{cur_key}_pred" for cur_key in PSA_KEYS]
GNN_PRED_STD_PSA_KEYS = [f"{cur_key}_pred_std" for cur_key in PSA_KEYS]

CIM_PRED_PSA_KEYS = [f"{cur_key}_cond_mean" for cur_key in PSA_KEYS]
CIM_PRED_STD_PSA_KEYS = [f"{cur_key}_cond_std" for cur_key in PSA_KEYS]

NON_PSA_IMs = ["PGV", "AI", "CAV", "Ds575", "Ds595", "PGA"]
IMs = NON_PSA_IMs + PSA_KEYS

IM_SETS = {"pSA": PSA_KEYS, "all": IMs}

IM_WEIGTHS_SETS = {
    # 1/3 of weights for other IMs and 2/3 for pSA (inc. PGA)
    "all": np.asarray(
        [(1 / 3) / 5] * 5 + [(2 / 3) / (len(PSA_KEYS) + 1)] * (len(PSA_KEYS) + 1)
    ),
    "pSA": np.ones(len(PSA_KEYS)) * (1 / len(PSA_KEYS)),
}

COMPONENTS = ["090", "000", "ver"]

SCALAR_FEATURE_KEYS = {
    "event": ["mag"],
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


CANTERBURY_REGION = [171.54, 173.12, -43.95, -43.22]
WELLINGTON_REGION = [172.639, 176.35, -42.427, -40.475]

STATION_FN_NAME = "non_uniform_whole_nz_with_real_stations-hh400_v20p3_land"

FIG_SIZE = (16, 10)
if (env_figsize := os.environ.get("fig_size")) is not None:
    FIG_SIZE = [float(x) for x in env_figsize.split(",")]

FIG_FORMAT = "png"
if (env_fig_format := os.environ.get("fig_format")) is not None:
    FIG_FORMAT = env_fig_format
