import os
from enum import Enum

import numpy as np


class RankingMethod(Enum):
    emp_cMVN = 1
    sim_cMVN = 2
    # Same as sim_cMVN but uses correlation coefficients
    # from the empirical model
    sim_cMVN_emp_corr = 3

METHOD_RESULT_DIR_NAME_MAPPING = {
    RankingMethod.emp_cMVN: "empirical_cMVN",
    RankingMethod.sim_cMVN: "sim_cMVN",
    RankingMethod.sim_cMVN_emp_corr: "sim_cMVN_emp_corr",
}

RESULTS_DIR_NAME_METHOD_MAPPING = {v: k for k, v in METHOD_RESULT_DIR_NAME_MAPPING.items()}



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
    1.25,
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

NON_PSA_IMs = ["PGV", "AI", "CAV", "Ds575", "Ds595", "PGA"]
IMs = NON_PSA_IMs  + PSA_KEYS


# 1/3 of weights for other IMs and 2/3 for pSA (inc. PGA)
IM_weights = np.asarray([(1/3) / 5] * 5 + [(2/3) / (len(PSA_KEYS) + 1)] * (len(PSA_KEYS) + 1))

COMPONENTS = ["090", "000", "ver"]

CANTERBURY_REGION = [171.54, 173.12, -43.95, -43.22]

STATION_FN_NAME = "non_uniform_whole_nz_with_real_stations-hh400_v20p3_land"

FIG_SIZE = (16, 10)
if (env_figsize := os.environ.get("fig_size")) is not None:
    FIG_SIZE = [float(x) for x in env_figsize.split(",")]

FIG_FORMAT = "png"
if (env_fig_format := os.environ.get("fig_format")) is not None:
    FIG_FORMAT = env_fig_format

