import os
from enum import Enum

import numpy as np


class RankingMethod(Enum):
    emp_cMVN = 1
    sim_cMVN = 2

    # Same as sim_cMVN but uses correlation coefficients
    # from the empirical model
    sim_cMVN_emp_corr = 3

    ml_prob = 4
    ml_prob_per_im = 5


METHOD_RESULT_DIR_NAME_MAPPING = {
    RankingMethod.emp_cMVN: "empirical_cMVN",
    RankingMethod.sim_cMVN: "sim_cMVN",
    RankingMethod.sim_cMVN_emp_corr: "sim_cMVN_emp_corr",
}

RESULTS_DIR_NAME_METHOD_MAPPING = {
    v: k for k, v in METHOD_RESULT_DIR_NAME_MAPPING.items()
}


class ScalarFeatureSetKey(str, Enum):
    # All available scalar features
    all = "all"

    # Only the scalar features used by the
    # empirical models for the generation
    # of the synthetic data
    emp_gen = "emp_gen"


ALL_SCALAR_FEATURE_KEYS = {
    "event": ["mag"],
    "site": ["vs30", "z1.0", "z2.5", "tsite"],
    "site_to_site": ["dist"],
    "event_site": ["r_rup"],
    "event_site_to_site": ["angular_dist"],
}

EMP_GEN_SCALAR_FEATURE_KEYS = {
    "event": ["mag"],
    "site": ["vs30", "z1.0", "z2.5"],
    "site_to_site": ["dist"],
    "event_site": ["r_rup"],
    "event_site_to_site": [],
}

SCALAR_FEATURE_SET_LOOKUP = {
    ScalarFeatureSetKey.all: ALL_SCALAR_FEATURE_KEYS,
    ScalarFeatureSetKey.emp_gen: EMP_GEN_SCALAR_FEATURE_KEYS,
}


WEIGHT_MODEL_SCALAR_FEATURE_SET_LOOKUP = {
    ScalarFeatureSetKey.emp_gen: ["vs30_site_int", "vs30_site_obs", "dist"],
    # ScalarFeatureSet.emp_gen: ["dist"],
    ScalarFeatureSetKey.all: [
        "vs30_site_int", "vs30_site_obs",
        "dist",
        "z1.0_site_int", "z1.0_site_obs",
        "z2.5_site_int", "z2.5_site_obs",
        "tsite_site_int", "tsite_site_obs"
    ],
}


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


CANTERBURY_REGION = [171.54, 173.12, -43.95, -43.22]
WELLINGTON_REGION = [172.639, 176.35, -42.427, -40.475]

STATION_FN_NAME = "non_uniform_whole_nz_with_real_stations-hh400_v20p3_land"

FIG_SIZE = (16, 10)
if (env_figsize := os.environ.get("fig_size")) is not None:
    FIG_SIZE = [float(x) for x in env_figsize.split(",")]

FIG_FORMAT = "png"
if (env_fig_format := os.environ.get("fig_format")) is not None:
    FIG_FORMAT = env_fig_format
