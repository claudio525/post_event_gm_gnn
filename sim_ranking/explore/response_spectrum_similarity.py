"""
Script to investigate the value range of the response spectrum differences
for a specific distance function. Needed to convert the distance scores into
a similarity score that is between 0 and 1.

Also looks at range of pSA values (max across all periods) using observed records,
needed for scaling of response spectra
"""
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sim_ranking as sr
import spatial_hazard as sh

### Load the data
obs_data_ffp = Path("/Users/claudy/dev/work/data/gm_datasets/nz_gmdb/v3.0/Tables/ground_motion_im_table_rotd50_flat.csv")
obs_df = pd.read_csv(obs_data_ffp, index_col=0, low_memory=False)
periods, pSA_keys = sr.utils.get_periods(obs_df.columns)

site_ffp = Path("/Users/claudy/dev/work/data/gm_datasets/nz_gmdb/v3.0/Tables/site_table.csv")
site_df =pd.read_csv(site_ffp, index_col="sta", low_memory=False, usecols=["sta", "lat", "lon"])

simulation_ffp=  Path("/Users/claudy/dev/work/data/sim_ranking/sim_im_data/simulations.imdb")
sim_data = sr.data.load_site_sim_data(simulation_ffp)
sites = list(sim_data.keys())
events = sim_data[sites[0]].index.get_level_values(0).unique().values.astype(str)

### Histogram of pSA values across all periods
max_pSA_df = pd.concat((obs_df.loc[:, pSA_keys].max(axis=1), obs_df.mag), axis=1)
max_pSA_df["mag_class"] = pd.cut(max_pSA_df.mag, bins=[0, 4, 6, 7], labels=["small", "moderate", "large"])
# max_pSA_df["mag_class"] =

fig, ax = plt.subplots(figsize=(16, 10))
# plt.hist(obs_df.loc[:, pSA_keys].values.ravel(), log=True, bins=25)

sns.histplot(data=max_pSA_df, x=0, hue="mag_class", bins=25, log_scale=(False, True), ax=ax, multiple="dodge")

plt.title("Histogram of pSA values across all periods")
plt.xlabel(f"pSA (g)")
plt.ylabel(f"Count")
plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")
plt.tight_layout()
# plt.show()

# Compute the distance between simulation realisations and the observed GM
# for each site
dists = []
residuals = []
res_area = []
res_sum = []
for cur_site in sites:
    cur_sim_df = sim_data[cur_site]
    cur_obs_df = obs_df.loc[obs_df.sta == cur_site, :]

    for cur_event in events:
        if cur_event not in cur_sim_df.index.get_level_values(0):
            continue

        if cur_event not in cur_obs_df.evid.values.astype(str):
            continue

        cur_obs_pSA = cur_obs_df.loc[cur_obs_df.evid == cur_event, pSA_keys].values

        cur_dists = np.linalg.norm(cur_obs_pSA - cur_sim_df.loc[cur_event, pSA_keys].values, axis=1)
        dists.append(cur_dists)

        cur_res = np.log(cur_obs_pSA) - np.log(cur_sim_df.loc[cur_event, pSA_keys].values)
        residuals.append(cur_res)
        res_area.append(np.trapz(np.abs(cur_res), axis=1))

dists = np.concatenate(dists)
residuals = np.concatenate(residuals, axis=0)
res_area = np.concatenate(res_area, axis=0)

#### Using residual


# Histogram of residual area
fig = plt.figure(figsize=(16, 10))

plt.hist(res_area, bins=25)

plt.xlabel(f"Residual Area")
plt.ylabel(f"Count")
plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")
plt.tight_layout()



### Residual at different areas
area = [5.0, 8.0, 12, 16, 20, 30, 40]

for cur_area in area:
    fig = plt.figure(figsize=(16, 10))

    cur_ind = np.argsort(np.abs(res_area - cur_area))[:25]

    plt.semilogx(periods, residuals[cur_ind].T, c="gray", linewidth=1.0)

    plt.title(f"{cur_area}")
    plt.xlabel(f"Period")
    plt.ylabel(f"Residual, Obs - Sim")
    plt.xlim(0.01, 10.0)
    plt.ylim(-2.5, 2.5)
    plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    plt.tight_layout()



score = np.where(res_area < 30, res_area * (-1/30) + 1, 0)

fig = plt.figure(figsize=(16, 10))

plt.hist(score, bins=25)

plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")
plt.tight_layout()

plt.show()

print(f"wtf")





























###### Using distance between response spectra, not ideal as this is a function of distance

### Histogram of obs - simulation realisations distances
fig = plt.figure(figsize=(16, 10))

# plt.hist(dists, bins=25)
plt.hist(dists[dists < 2.0], bins=25)

plt.xlabel(f"Distance")
plt.ylabel(f"Count")
plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")
plt.tight_layout()

### Cumulative version
fig = plt.figure(figsize=(16, 10))

sort_ind = np.argsort(dists)

plt.step(dists[sort_ind], np.arange(dists.size))

plt.xlabel(f"Distance")
plt.ylabel(f"Count")
plt.xlim(0, 2.0)
plt.ylim(0, None)
plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")
plt.tight_layout()

# plt.show()

### Plot some example residuals for the different distances
distances = [0, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0]

for cur_distance in distances:
    fig = plt.figure(figsize=(16, 10))

    cur_ind = np.argsort(np.abs(dists - cur_distance))[:25]

    plt.semilogx(periods, residuals[cur_ind].T, c="gray", linewidth=1.0)

    plt.title(f"{cur_distance}")
    plt.xlabel(f"Period")
    plt.ylabel(f"Residual, Obs - Sim")
    plt.xlim(0.01, 10.0)
    plt.ylim(-1, 1)
    plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")
    plt.tight_layout()


### Compute the similarity score
# scores = np.where(dists < 0.5, -(0.75 / 0.5) * dists + 1, - (0.25 / 1.5) * dists + 1 / 3)
# scores = np.where(scores < 0, 0, scores)



### Plot the score
fig = plt.figure(figsize=(16, 10))

plt.hist(scores, bins=25)

plt.title(f"Score")
plt.xlabel(f"Score")
plt.ylabel(f"Count")
plt.grid(linewidth=0.5, alpha=0.5, linestyle="--")
plt.tight_layout()

plt.show()

print(f"wtf")


