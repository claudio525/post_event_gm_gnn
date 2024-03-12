from typing import Sequence, NamedTuple, Tuple


import numpy as np
import pandas as pd


class SourceInfo(NamedTuple):

    rupture_name: str
    hypo_loc: Tuple[float, float]

def reverse_im_filename(im: str):
    if im.startswith("pSA"):
        return im[::-1].replace("p", ".", 1)[::-1]
    return im


def get_periods(columns: Sequence[str]):
    pSA_keys = np.asarray([cur_c for cur_c in columns if cur_c.startswith("pSA")])
    periods = np.asarray(
        [float(cur_c.rsplit("_", maxsplit=1)[-1]) for cur_c in pSA_keys]
    )
    sort_ind = np.argsort(periods)
    return periods[sort_ind], pSA_keys[sort_ind]


def get_nice_im_name(im: str):
    if im.startswith("pSA"):
        return f"pSA({im.split('_')[-1]}s)"
    return im

