from pathlib import Path

import pandas as pd

def get_im_filename(im: str):
    if im.startswith("pSA"):
        return im[::-1].replace('p', '.', 1)[::-1]
    return im

def load_ll_file(ffp: Path):
    return pd.read_csv(
        ffp, sep=" ", index_col=2, header=None, names=["lon", "lat"]
    )


