from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import ml_tools as mlt
import sim_ranking as sr

if __name__ == '__main__':
    obs_data_ffp = Path("/home/claudy/dev/work/data/gm_datasets/nz_gmdb/v3.4/Tables/ground_motion_im_table_rotd50_flat.csv")

    obs_data = sr.ObservedData.from_nzgmdb_flat(obs_data_ffp, sr.constants.NZGMDBVersion.v3p4)
    obs_data.drop_nan()
    obs_data.apply_fmin_filter(sr.ObservedData.OtherColEnums.FMIN)

    inputs = ["mag", "vs30", "rrup"]
    ims = sr.constants.PSA_KEYS

    X = obs_data.record_df[inputs].values
    y = np.log(obs_data.record_df[ims].values)

    # Normalise the data
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    model = nn.Sequential(
        nn.Linear(X.shape[1], 32),
        nn.LeakyReLU(),
        nn.Linear(32, 32),
        nn.LeakyReLU(),
        nn.Linear(32, y.shape[1]),
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    n_epochs = 100

    dataloader = mlt.torch.TabularDataLoader(X, y, batch_size=32, shuffle=True)

    for epoch in range(n_epochs):

        batch_loss = 0
        for i, (batch_X, batch_y, _) in enumerate(dataloader):
            mask = torch.isnan(batch_y)

            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = criterion(y_pred[~mask], batch_y[~mask])
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()

        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {batch_loss / len(dataloader)}")

    print(f"wtf")



