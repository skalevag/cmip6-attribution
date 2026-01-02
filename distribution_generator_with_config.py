#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 12:56:57 2023

This script calculates probability distributions of observed monthly temperatures
in Helsinki Kaisaniemi and Sodankylä Tähtelä in observed and future climates.

The methdology is documented in Rantanen et al. (2024):
A method for estimating the effect of climate change on monthly mean temperatures:
September 2023 and other recent record-warm months in Helsinki, Finland
https://doi.org/10.1002/asl.1216

The main aspects have been also documented in these two papers

Räisänen, J. and L. Ruokolainen, 2008: Estimating present climate in a
warming world: a model-based approach. Climate Dynamics, 31, 573-585

and

Räisänen, J. and L. Ruokolainen, 2008: Ongoing global warming and local
warm extremes: a case study of winter 2006-2007 in Helsinki, Finland.
Geophysica, 44, 45-65.

@author: rantanem
"""

import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml

import subroutines

"""
First, give the input parameters: place, month, year, and scenario
"""

if len(sys.argv) < 2:
    print("Usage: python dev_distribution_generator.py <path_to_yaml_file>")
    try:
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print("Error: config.yaml not found.")
        config = {}  # Provide a default empty config or handle error appropriately
        sys.exit(1)
else:
    yaml_file_path = sys.argv[1]
    with open(yaml_file_path, "r") as file:
        config = yaml.safe_load(file)

if config["station"]["organisation"] == "FMI":
    fmisid = int(config["station"]["id"])
elif config["station"]["organisation"] == "MetNo":
    fmisid = 0
    metnosid = str(config["station"]["id"])
    frost_client_id = config["station"]["frost_client_id"]
    homogenised = config["data"]["homogenised"]
    allow_missing_in_annual = config["data"]["allow_missing_in_annual"]

target_mon = config["analysis"]["target_mon"]
y_target = config["analysis"]["y_target"]
y_climate = config["analysis"]["y_climate"]
y_preind = config["analysis"]["y_preind"]
pwarm = config["analysis"]["pwarm"]
ssp = config["analysis"]["ssp"]

colors = config["plotting"]["colors"]
"""
################################################################
Next, specify some additional parameters used in the calculation
################################################################
"""

y1base = config["general_settings"]["y1base"]
y2base = config["general_settings"]["y2base"]
valmin = config["general_settings"]["valmin"]
valmax = config["general_settings"]["valmax"]
nbins = config["general_settings"]["nbins"]
n_bts = config["general_settings"]["n_bts"]
input_path = config["general_settings"]["input_path"]
figurePath = config["general_settings"]["figurePath"]
output_path = config["general_settings"]["output_path"]

# create directory
Path(figurePath).mkdir(parents=True, exist_ok=True)
if config["general_settings"]["save_results"]:
    Path(config["general_settings"]["output_path"]).mkdir(parents=True, exist_ok=True)

x = np.linspace(valmin, valmax, nbins)

# lat/lon coordinates + station name
if fmisid != 0:
    metadata = subroutines.get_station_metadata(fmisid)
else:
    metadata = subroutines.get_station_metadata_frost(metnosid, frost_client_id)
latlon = metadata[:2]
sname = metadata[2]

if config["general_settings"]["add_station_name_to_filename"]:
    sname_stem = str(config["station"]["id"]).strip() + "_" + sname.strip().replace(" ", "-").replace("(", "").replace(")", "")
    figurePath = figurePath + sname_stem + "_"
    output_path = output_path + sname_stem + "_"

"""
################################################################
Read observed and simulated global temperatures, merge them and
smooth with 11-year rolling average
################################################################
"""
## READ observed global temperature
glob_obs_ds = xr.open_dataset(input_path + "input_data/HadCRUT.5.0.2.0.analysis.summary_series.global.annual.nc")
glob_obs_temp = glob_obs_ds["tas_mean"].squeeze().sel(time=slice("1850-01-01", f"{y_target}-12-31"))
glob_obs_temp["time"] = glob_obs_temp.time.dt.year

## READ simulated temperature, merge it with observed temperature and smooth with 11-year moving average
glob_temp_single = subroutines.read_sim_temp_single_models(input_path, ssp, glob_obs_temp)
glob_temp_mean = subroutines.read_sim_temp_model_mean(input_path, ssp, glob_obs_temp)

# Number of models used in the calculation
n_mod = len(glob_temp_single.columns)

"""
################################################################
Finally, read observed local temperatures and the coefficients 
for the Tglob-regressed changes in mean and variability
################################################################
"""
#### A) Observed local temperatures
if fmisid != 0:
    obs_temp = subroutines.read_obs_temp(input_path, fmisid, target_mon).loc[y1base:]
else:
    obs_temp = subroutines.read_obs_temp_frost(frost_client_id, metnosid, target_mon, homogenised=homogenised, allow_missing_in_annual=allow_missing_in_annual).loc[y1base:]


### B) The coefficients for the Tglob-regressed changes in mean and variability
coeffs_single = subroutines.read_coeffs_single_models(input_path, ssp, target_mon, latlon[0], latlon[1])
coeffs_mean = subroutines.read_coeffs_model_mean(input_path, ssp, target_mon, latlon[0], latlon[1])
"""
################################################################
Everything has been read in. Next, recalculate the observed time series    
for the target year, future year and pre-industrial climate year. 
################################################################
"""
# Calculate single models
pseudo_obs_preind_year_single_models = pd.DataFrame(index=obs_temp.index, columns=glob_temp_single.columns)
pseudo_obs_target_year_single_models = pd.DataFrame(index=obs_temp.index, columns=glob_temp_single.columns)
pseudo_obs_future_year_single_models = pd.DataFrame(index=obs_temp.index, columns=glob_temp_single.columns)


for m in glob_temp_single.columns:

    pseudo_obs_preind_year_single_models[m] = subroutines.modify_obs(obs_temp, glob_temp_single[m], coeffs_single.sel(model=m), y_preind)
    pseudo_obs_target_year_single_models[m] = subroutines.modify_obs(obs_temp, glob_temp_single[m], coeffs_single.sel(model=m), y_target)
    pseudo_obs_future_year_single_models[m] = subroutines.modify_obs(obs_temp, glob_temp_single[m], coeffs_single.sel(model=m), y_climate)

# Calculate multi-model-mean
pseudo_obs_preind_year_model_mean = subroutines.modify_obs(obs_temp, glob_temp_mean, coeffs_mean, y_preind)
pseudo_obs_target_year_model_mean = subroutines.modify_obs(obs_temp, glob_temp_mean, coeffs_mean, y_target)
pseudo_obs_future_year_model_mean = subroutines.modify_obs(obs_temp, glob_temp_mean, coeffs_mean, y_climate)

if config["general_settings"]["save_results"]:
    pseudo_obs_preind_year_single_models.to_csv(output_path + "pseudo_obs_preind_year_single_models.csv")
    pseudo_obs_target_year_single_models.to_csv(output_path + "pseudo_obs_target_year_single_models.csv")
    pseudo_obs_future_year_single_models.to_csv(output_path + "pseudo_obs_future_year_single_models.csv")
    pseudo_obs_model_mean = (
        pseudo_obs_preind_year_model_mean.rename(f"preind_{y_preind}")
        .to_frame()
        .join(pseudo_obs_target_year_model_mean.rename(f"present_{y_target}"))
        .join(pseudo_obs_future_year_model_mean.rename(f"future_{y_climate}"))
    )
    pseudo_obs_model_mean.to_csv(output_path + "pseudo_obs_model_mean.csv")

"""
################################################################
Then, calculate the stochastically generated skewed (SGS) distributions
for observed climate, present climate and future climate. The calculation
of the distributions is based on Sardeshmukh et al. (2015): 
https://doi.org/10.1175/JCLI-D-15-0020.1
################################################################
"""
# Select the first year (either y1base or the first year of the observations)
y1base = np.maximum(y1base, obs_temp.index[0])

# SGS distributions for observed values (real observations)
f_obs, cp_obs, _ = subroutines.calculate_sgs(obs_temp.loc[y1base:y2base], valmax, valmin, nbins)

# SGS distributions for pseudo observations (pre-industrial, target and future climates)
# based on multi-model mean coefficients
f_preind, cp_preind, _ = subroutines.calculate_sgs(pseudo_obs_preind_year_model_mean.loc[y1base:y2base], valmax, valmin, nbins)
f_target, cp_target, moments = subroutines.calculate_sgs(pseudo_obs_target_year_model_mean.loc[y1base:y2base], valmax, valmin, nbins)
f_future, cp_future, _ = subroutines.calculate_sgs(pseudo_obs_future_year_model_mean.loc[y1base:y2base], valmax, valmin, nbins)

if config["general_settings"]["save_results"]:
    df_prob_density = pd.DataFrame([x, f_preind, f_target, f_future], index=["tmon", f"prob_density_{y_preind}", f"prob_density_{y_target}", f"prob_density_{y_climate}"]).T
    df_prob_density.to_csv(output_path + "prob_density.csv")

"""
################################################################
Perform the bootstrapping. This procedure gives the uncertainty related to
internal climate variability. 
################################################################
"""

# Allocate arrays for SGS distributions
# the first index includes the actual observations (not randomly sampled),
# thus +1 in size
cp_obs_arr = np.empty((nbins, n_bts + 1))
f_obs_arr = np.empty((nbins, n_bts + 1))
cp_preind_3d_arr = np.empty((nbins, n_mod, n_bts + 1))
f_preind_3d_arr = np.empty((nbins, n_mod, n_bts + 1))
cp_target_3d_arr = np.empty((nbins, n_mod, n_bts + 1))
f_target_3d_arr = np.empty((nbins, n_mod, n_bts + 1))
cp_future_3d_arr = np.empty((nbins, n_mod, n_bts + 1))
f_future_3d_arr = np.empty((nbins, n_mod, n_bts + 1))

# The years which are resampled in the bootsrapping
the_list = obs_temp.loc[y1base:y2base].index

# bootstrapping loop
for b in np.arange(0, n_bts + 1):
    if b % 50 == 0:
        print("Bootstrapping number", b + 1, "/", n_bts)

    # select randomly 100 years
    res = random.choices(the_list, k=len(the_list))
    ## IMPORTANT: the first bootstrapping round is done with actual observations
    if b == 0:
        res = the_list
    # select temperatures of those years
    temp = obs_temp.loc[res]

    # First, calculate the SGS distribution for bootstrapped actual observations
    f_obs_arr[:, b], cp_obs_arr[:, b], _ = subroutines.calculate_sgs(temp, valmax, valmin, nbins)

    # Then, calculate the SGS distribution for bootstrapped pseudo observations
    f_preind_3d_arr[:, :, b], cp_preind_3d_arr[:, :, b], _ = subroutines.calculate_sgs(pseudo_obs_preind_year_single_models.loc[res], valmax, valmin, nbins)

    f_target_3d_arr[:, :, b], cp_target_3d_arr[:, :, b], _ = subroutines.calculate_sgs(pseudo_obs_target_year_single_models.loc[res], valmax, valmin, nbins)

    f_future_3d_arr[:, :, b], cp_future_3d_arr[:, :, b], _ = subroutines.calculate_sgs(pseudo_obs_future_year_single_models.loc[res], valmax, valmin, nbins)


# Reshape 3D arrays into 2D arrays
f_preind_arr = f_preind_3d_arr.reshape(*f_preind_3d_arr.shape[:-2], -1)
cp_preind_arr = cp_preind_3d_arr.reshape(*cp_preind_3d_arr.shape[:-2], -1)
f_target_arr = f_target_3d_arr.reshape(*f_target_3d_arr.shape[:-2], -1)
cp_target_arr = cp_target_3d_arr.reshape(*cp_target_3d_arr.shape[:-2], -1)
f_future_arr = f_future_3d_arr.reshape(*f_future_3d_arr.shape[:-2], -1)
cp_future_arr = cp_future_3d_arr.reshape(*cp_future_3d_arr.shape[:-2], -1)


"""
################################################################
Print the probabilities of warmer/colder temperatures
################################################################
"""
# The observed temperature (i.e. the target value)
target_value = obs_temp.loc[y_target].round(2)
# The corresponding index
i = int((nbins - 1.0) * (target_value - valmin) / (valmax - valmin))

# calculate probabilites and their upper + lower condidence intervals using 5-95 percentiles
if pwarm:
    prob_in_obs = 1 - cp_obs[i]
    prob_in_obs_up = 1 - np.nanpercentile(cp_obs_arr, 95, 1)[i]
    prob_in_obs_low = 1 - np.nanpercentile(cp_obs_arr, 5, 1)[i]
    prob_in_preind = 1 - cp_preind[i]
    prob_in_preind_up = 1 - np.nanpercentile(cp_preind_arr, 95, 1)[i]
    prob_in_preind_low = 1 - np.nanpercentile(cp_preind_arr, 5, 1)[i]
    prob_in_target = 1 - cp_target[i]
    prob_in_target_up = 1 - np.nanpercentile(cp_target_arr, 95, 1)[i]
    prob_in_target_low = 1 - np.nanpercentile(cp_target_arr, 5, 1)[i]
    prob_in_future = 1 - cp_future[i]
    prob_in_future_up = 1 - np.nanpercentile(cp_future_arr, 95, 1)[i]
    prob_in_future_low = 1 - np.nanpercentile(cp_future_arr, 5, 1)[i]

    pr_ratio = prob_in_target / prob_in_preind
    pr_ratio_low = np.nanpercentile((1 - cp_target_arr[i, :]) / (1 - cp_preind_arr[i, :]), 5)
    pr_ratio_up = np.nanpercentile((1 - cp_target_arr[i, :]) / (1 - cp_preind_arr[i, :]), 95)

    far = 1 - (prob_in_preind / prob_in_target)
    far_low = np.nanpercentile(1 - ((1 - cp_preind_arr[i, :]) / (1 - cp_target_arr[i, :])), 5)
    far_up = np.nanpercentile(1 - ((1 - cp_preind_arr[i, :]) / (1 - cp_target_arr[i, :])), 95)

    pr_ratio_future = prob_in_future / prob_in_target
    pr_ratio_future_low = np.nanpercentile((1 - cp_future_arr[i, :]) / (1 - cp_target_arr[i, :]), 5)
    pr_ratio_future_up = np.nanpercentile((1 - cp_future_arr[i, :]) / (1 - cp_target_arr[i, :]), 95)

else:
    prob_in_obs = cp_obs[i]
    prob_in_obs_up = np.nanpercentile(cp_obs_arr, 5, 1)[i]
    prob_in_obs_low = np.nanpercentile(cp_obs_arr, 95, 1)[i]
    prob_in_preind = cp_preind[i]
    prob_in_preind_up = np.nanpercentile(cp_preind_arr, 5, 1)[i]
    prob_in_preind_low = np.nanpercentile(cp_preind_arr, 95, 1)[i]
    prob_in_target = cp_target[i]
    prob_in_target_up = np.nanpercentile(cp_target_arr, 5, 1)[i]
    prob_in_target_low = np.nanpercentile(cp_target_arr, 95, 1)[i]
    prob_in_future = cp_future[i]
    prob_in_future_up = np.nanpercentile(cp_future_arr, 5, 1)[i]
    prob_in_future_low = np.nanpercentile(cp_future_arr, 95, 1)[i]

    pr_ratio = prob_in_preind / prob_in_target
    pr_ratio_low = np.nanpercentile((cp_preind_arr[i, :]) / (cp_target_arr[i, :]), 5)
    pr_ratio_up = np.nanpercentile((cp_preind_arr[i, :]) / (cp_target_arr[i, :]), 95)

    far = 1 - (prob_in_target / prob_in_preind)
    far_low = np.nanpercentile(1 - ((1 - cp_target_arr[i, :]) / (1 - cp_preind_arr[i, :])), 5)
    far_up = np.nanpercentile(1 - ((1 - cp_target_arr[i, :]) / (1 - cp_preind_arr[i, :])), 95)

    pr_ratio_future = prob_in_target / prob_in_future
    pr_ratio_future_low = np.nanpercentile((cp_target_arr[i, :]) / (cp_future_arr[i, :]), 5)
    pr_ratio_future_up = np.nanpercentile((cp_target_arr[i, :]) / (cp_future_arr[i, :]), 95)


# Percentile in today's climate
prob = cp_target[i]

# Calculate corresponding temperatures in other climates
t_1901 = np.round(np.squeeze(x[np.where(cp_obs == subroutines.find_nearest(cp_obs, prob))[0]]), 1)
t_1900 = np.round(np.squeeze(x[np.where(cp_preind == subroutines.find_nearest(cp_preind, prob))[0]]), 1)
t_2050 = np.round(np.squeeze(x[np.where(cp_future == subroutines.find_nearest(cp_future, prob))[0]]), 1)

t1900_lower, t1900_upper, _ = subroutines.find_intensity_interval(x, cp_target_arr, cp_preind_arr, i)
t2050_lower, t2050_upper, _ = subroutines.find_intensity_interval(x, cp_target_arr, cp_future_arr, i)
tdiff_lower, tdiff_upper, cp_df = subroutines.find_difference_interval(x, cp_target_arr, cp_preind_arr, i)
tdiff_lower_future, tdiff_upper_future, cp_df_future = subroutines.find_difference_interval(x, cp_future_arr, cp_target_arr, i)

print("\nRESULTS\n")

print("Annual probabilities:")
print(str(y_preind) + ":", np.round(prob_in_preind * 100, 2), "%", np.round(prob_in_preind_up * 100, 2), np.round(prob_in_preind_low * 100, 2), "%")

print(str(y_target) + ":", np.round(prob_in_target * 100, 2), "%", np.round(prob_in_target_up * 100, 2), np.round(prob_in_target_low * 100, 2), "%")

print(str(y_climate) + ":", np.round(prob_in_future * 100, 2), "%", np.round(prob_in_future_up * 100, 2), np.round(prob_in_future_low * 100, 2), "%")


print("\nProbability ratio: ", np.round(pr_ratio, 1), "(" + str(np.round(pr_ratio_low, 1)) + "-" + str(np.round(pr_ratio_up, 1)) + ")")

print("Change in intensity: " + str(np.round(target_value - t_1900, 1)) + "°C", "(" + str(np.round(tdiff_lower, 1)) + "-" + str(np.round(tdiff_upper, 1)) + ")")

if config["general_settings"]["save_results"]:
    key_results = pd.concat(
        [
            pd.Series([prob_in_preind, prob_in_preind_up, prob_in_preind_low], name=f"prob_{y_preind}", index=["best_estimate", "lower", "upper"]),
            pd.Series([prob_in_target, prob_in_target_up, prob_in_target_low], name=f"prob_{y_target}", index=["best_estimate", "lower", "upper"]),
            pd.Series([prob_in_future, prob_in_future_up, prob_in_future_low], name=f"prob_{y_climate}", index=["best_estimate", "lower", "upper"]),
            pd.Series([1 / prob_in_preind, 1 / prob_in_preind_low, 1 / prob_in_preind_up], name=f"rp_{y_preind}", index=["best_estimate", "lower", "upper"]),
            pd.Series([1 / prob_in_target, 1 / prob_in_target_low, 1 / prob_in_target_up], name=f"rp_{y_target}", index=["best_estimate", "lower", "upper"]),
            pd.Series([1 / prob_in_future, 1 / prob_in_future_low, 1 / prob_in_future_up], name=f"rp_{y_climate}", index=["best_estimate", "lower", "upper"]),
            pd.Series([t_1900, t1900_lower, t1900_upper], name=f"intensity_{y_preind}", index=["best_estimate", "lower", "upper"]),
            pd.Series([target_value, target_value, target_value], name=f"intensity_{y_target}", index=["best_estimate", "lower", "upper"]),
            pd.Series([t_2050, t2050_lower, t2050_upper], name=f"intensity_{y_climate}", index=["best_estimate", "lower", "upper"]),
            pd.Series([pr_ratio, pr_ratio_low, pr_ratio_up], name="PR_past_present", index=["best_estimate", "lower", "upper"]),
            pd.Series([far, far_low, far_up], name="FAR_past_present", index=["best_estimate", "lower", "upper"]),
            pd.Series([target_value - t_1900, tdiff_lower, tdiff_upper], name="dI_past_present", index=["best_estimate", "lower", "upper"]),
            pd.Series([pr_ratio_future, pr_ratio_future_low, pr_ratio_future_up], name="PR_present_future", index=["best_estimate", "lower", "upper"]),
            pd.Series([t_2050 - target_value, tdiff_lower_future, tdiff_upper_future], name="dI_present_future", index=["best_estimate", "lower", "upper"]),
        ],
        axis=1,
    ).T
    key_results.to_csv(output_path + "attribution_results.csv")
"""
################################################################
Finally, plot the results: first the time series plot and then the
distribution plot
################################################################
"""

# Define the default font
font = {"weight": "normal", "size": 16}
import matplotlib

matplotlib.rc("font", **font)
# Define texts for the plots
place_text = sname
target_text = subroutines.get_target_text(target_mon) + " " + str(y_target)


### 1) Time series plot

## Show time series only up to target year
obs_temp_plot = obs_temp.loc[slice(str(y1base), str(y_target))]
target_mean = pseudo_obs_target_year_model_mean.loc[slice(str(y1base), str(y_target))]
target_up = pseudo_obs_target_year_single_models.quantile(0.95, axis=1).loc[slice(str(y1base), str(y_target))]
target_low = pseudo_obs_target_year_single_models.quantile(0.05, axis=1).loc[slice(str(y1base), str(y_target))]
preind_mean = pseudo_obs_preind_year_model_mean.loc[slice(str(y1base), str(y_target))]
preind_up = pseudo_obs_preind_year_single_models.quantile(0.95, axis=1).loc[slice(str(y1base), str(y_target))]
preind_low = pseudo_obs_preind_year_single_models.quantile(0.05, axis=1).loc[slice(str(y1base), str(y_target))]

# Define error bars
y_l = np.round(np.array(target_mean) - np.array(target_low), 6)
y_u = np.round(np.array(target_up) - np.array(target_mean), 6)
errors = [y_l, y_u]


fig = plt.figure(figsize=(10, 7), dpi=300)
ax = plt.gca()

ax.plot(obs_temp_plot.index, obs_temp_plot, color=colors["obs"], linewidth=1.5, label="Observations")

ax.scatter(target_mean.index, target_mean, label="Pseudo-observations", color=colors["pseudo_obs"])
ax.errorbar(target_mean.index, target_mean, yerr=errors, fmt="o", mfc=colors["pseudo_obs"], mec=colors["pseudo_obs"], ecolor=colors["pseudo_obs_uncertainty"])

ax.axhline(y=obs_temp_plot.loc[y_target], linestyle="--", color=colors["target_magnitude"])

ax.set_ylabel("Temperature [°C]")
# ax.set_ylim(7,17.2)
ax.grid(True)
ax.tick_params(
    axis="both",
    which="major",
)
ax.legend(
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.09),
    ncol=2,
)
# ax.set_title('a) '+subroutines.get_target_text(target_mon)+' temperatures' , loc='left',
#              fontsize=14)
ax.set_xlim(y1base, y_target + 1)

figureName = "timeser_plot.png"

plt.savefig(figurePath + figureName, dpi=300, bbox_inches="tight")


### 2) distribution plot

hist, bin_edges = np.histogram(pseudo_obs_target_year_model_mean.loc[y1base:y2base].values.squeeze(), density=True, bins=np.arange(valmin, valmax, 0.5) + 0.25)


fig, axlist = plt.subplots(nrows=1, ncols=2, figsize=(14, 7), dpi=300, sharey=False)
ax1 = axlist[0]
ax2 = axlist[1]

ax1.plot(x, f_target, color=colors["target"], zorder=7, linewidth=2.5)

# ax2.plot(x,f_obs, color='blue',  zorder=7, linewidth=2)
ax2.plot(x, f_preind, color=colors["preind"], label="'" + str(y_preind) + "'", zorder=9, linewidth=2.5)
ax2.plot(x, f_target, color=colors["target"], label="'" + str(y_target) + "'", zorder=8, linewidth=2.5)
ax2.plot(x, f_future, color=colors["future"], label="'" + str(y_climate) + "'", zorder=9, linewidth=2.5)


ax1.bar(bin_edges[:-1] + 0.25, hist, width=0.4, edgecolor="k", facecolor=colors["target_hist"], zorder=2, label="Pseudo-observations '" + str(y_target) + "'", alpha=0.7)

ax2.axvline(x=target_value, zorder=10, color="k")
ax2.annotate(f"{np.round(target_value,1):.1f}°C", xy=(target_value - 0.1, ax2.get_ylim()[1] + 0.1), xycoords="data", rotation=90, ha="right", va="top")

if pwarm:
    ax2.fill_between(x, f_preind, where=x >= target_value, color=colors["preind"], zorder=5, alpha=0.7)
    ax2.fill_between(x, f_target, where=x >= target_value, color=colors["target"], zorder=4, alpha=0.7)
    ax2.fill_between(x, f_future, where=x >= target_value, color=colors["future"], zorder=3, alpha=0.7)
    probtext = "Probability of T ≥ " + f"{np.round(target_value,1):.1f}°C"
    returntext = "Return period of T ≥ " + f"{np.round(target_value,1):.1f}°C"
else:
    ax2.fill_between(x, f_preind, where=x <= target_value, color=colors["preind"], zorder=3, alpha=0.7)
    ax2.fill_between(x, f_target, where=x <= target_value, color=colors["target"], zorder=4, alpha=0.7)
    ax2.fill_between(x, f_future, where=x <= target_value, color=colors["future"], zorder=5, alpha=0.7)
    probtext = "Probability of T ≤ " + f"{np.round(target_value,1):.1f}°C"
    returntext = "Return period of T ≤ " + f"{np.round(target_value,1):.1f}°C"

# ax.tick_params(axis='both', which='major')
for ax in axlist:
    ax.set_xticks(np.arange(valmin, valmax, 4))
    ax.set_xlim(np.floor(np.nanmin(obs_temp.loc[y1base:y2base].values.squeeze())) - 3, np.ceil(np.nanmax(obs_temp)) + 4)
    ax.set_xlabel("Temperature [°C]")
    ax.grid(True, zorder=1)
    ax.set_ylim(0, max([f_preind.max(), f_target.max(), f_future.max()]) + 0.1)
ax1.set_ylabel("Relative frequency / probability density [1/°C]")
ax1.legend(loc="upper right", frameon=False, ncol=3, bbox_to_anchor=(0.92, 1.1))
ax2.legend(loc="upper right", frameon=False, ncol=3, bbox_to_anchor=(1, 1.1))


howtext = "Intensity  "

textstr = "\n".join(
    (
        place_text,
        target_text,
        "",
        r"$\bf{" + probtext.replace(" ", r"\ ") + "}$",
        str(y1base) + "-" + str(y2base) + ": " + f"{np.round(prob_in_obs*100,1):.1f} (" + f"{np.round(prob_in_obs_up*100,1):.1f}-" + f"{np.round(prob_in_obs_low*100,1):.1f}) %",
        "",
        "'" + str(y_preind) + "'" + ": " + f"{np.round(prob_in_preind*100,1):.1f} (" + f"{np.round(prob_in_preind_up*100,1):.1f}-" + f"{np.round(prob_in_preind_low*100,1):.1f}) %",
        "'" + str(y_target) + "'" + ": " + f"{np.round(prob_in_target*100,1):.1f} (" + f"{np.round(prob_in_target_up*100,1):.1f}-" + f"{np.round(prob_in_target_low*100,1):.1f}) %",
        "'" + str(y_climate) + "'" + ": " + f"{np.round(prob_in_future*100,1):.1f} (" + f"{np.round(prob_in_future_up*100,1):.1f}-" + f"{np.round(prob_in_future_low*100,1):.1f}) %",
        "",
        r"$\bf{" + returntext.replace(" ", r"\ ") + "}$",
        # str(y1base)+'-'+str(y2base)+': '+f'{np.round(1/prob_in_obs,0):.0f} ('+\
        # f'{np.round(1/prob_in_obs_low,0):.0f}-'+f'{np.round(1/prob_in_obs_up,0):.0f}) years',
        "",
        "'" + str(y_preind) + "'" + ": " + f"{np.round(1/prob_in_preind,0):.0f} (" + f"{np.round(1/prob_in_preind_low,0):.0f}-" + f"{np.round(1/prob_in_preind_up,0):.0f}) years",
        "'" + str(y_target) + "'" + ": " + f"{np.round(1/prob_in_target,0):.0f} (" + f"{np.round(1/prob_in_target_low,0):.0f}-" + f"{np.round(1/prob_in_target_up,0):.0f}) years",
        "'" + str(y_climate) + "'" + ": " + f"{np.round(1/prob_in_future,0):.0f} (" + f"{np.round(1/prob_in_future_low,0):.0f}-" + f"{np.round(1/prob_in_future_up,0):.0f}) years",
        "",
        r"$\bf{" + howtext + "}$",
        # str(y1base)+'-'+str(y2base)+': '+f'{t_1901:.1f}°C',
        "",
        "'" + str(y_preind) + "'" + ": " + f"{t_1900:.1f}°C ({t1900_lower:.1f}-{t1900_upper:.1f})°C",
        "'" + str(y_target) + "'" + ": " + f"{target_value:.1f}°C",
        "'" + str(y_climate) + "'" + ": " + f"{t_2050:.1f}°C ({t2050_lower:.1f}-{t2050_upper:.1f})°C",
        "",
        "Scenario for future climate: \n" + subroutines.get_scenario_text(ssp),
    )
)


props = dict(boxstyle="round", facecolor=colors["textbox"], alpha=0.5)
if config["plotting"]["textbox_in_dist_plot"]:
    # place a text box in upper left in axes coords
    ax2.text(1.04, 0.99, textstr, transform=ax.transAxes, fontsize=13, verticalalignment="top", bbox=props)

# ax.set_title("b) " + subroutines.get_target_text(target_mon) + " temperature distributions", loc="left", fontsize=14)

ax1.annotate("μ=" + str(np.round(moments[0], 2)), (0.03, 0.87), xycoords="axes fraction")
ax1.annotate("σ²=" + str(np.round(moments[1], 2)), (0.03, 0.82), xycoords="axes fraction")
ax1.annotate("γ=" + str(np.round(moments[2], 2)), (0.03, 0.77), xycoords="axes fraction")
ax1.annotate("κ=" + str(np.round(moments[3], 2)), (0.03, 0.72), xycoords="axes fraction")


ax1.annotate("a", (0.03, 0.95), xycoords="axes fraction", fontweight="bold", fontsize=18)
ax2.annotate("b", (0.03, 0.95), xycoords="axes fraction", fontweight="bold", fontsize=18)

figureName = "dist_plot.png"

plt.savefig(figurePath + figureName, dpi=300, bbox_inches="tight")


### 3) Boxplots with model spread
medianprops = dict(linestyle="-", linewidth=2.5, color=colors["boxplot_median"])
medianprops_mm = dict(
    linestyle=None,
    linewidth=0,
)
boxprops = dict(facecolor=colors["boxplot_face"], color=colors["boxplot_edge"])
model_names = list(coeffs_single.model.values)

fig, axlist = plt.subplots(nrows=1, ncols=2, figsize=(10, 12), dpi=300, sharey=True)
plt.subplots_adjust(wspace=0.7)

ax = axlist[0]

PRratios = np.transpose((1 - cp_target_3d_arr[i, :, :]) / (1 - cp_preind_3d_arr[i, :, :]))
# Filter data using np.isnan
mask = ~np.isnan(PRratios)
filtered_data = [d[m] for d, m in zip(PRratios.T, mask.T)]

pr_ratio_bp = np.array(pr_ratio)
pr_ratio_bp = pr_ratio_bp.reshape(pr_ratio_bp.shape + (1,))

a = ax.boxplot(filtered_data, positions=np.arange(0, n_mod), tick_labels=model_names, vert=False, showfliers=False, medianprops=medianprops, whis=(5, 95), widths=0.7)

a2 = ax.boxplot(
    np.concatenate(filtered_data), positions=[n_mod], tick_labels=["MMM"], vert=False, showfliers=False, patch_artist=True, medianprops=medianprops_mm, whis=(5, 95), widths=0.7, boxprops=boxprops
)
a2b = ax.boxplot(pr_ratio_bp, positions=[n_mod], tick_labels=[""], vert=False, showfliers=False, medianprops=medianprops, whis=(5, 95), widths=0.7)

ax.yaxis.tick_right()
ax.set_xlim(1, 10000)

ax.set_xlabel("Probability ratio")
ax.set_xscale("log")
ax.set_xticks(ticks=[1, 10, 100, 1000, 10000, 100000])
ax.invert_yaxis()


ax = axlist[1]

deltaI = cp_df[:: n_bts + 1].values
deltaI = np.transpose(deltaI.reshape(deltaI.shape + (1,)))
deltaI_mm = np.round(target_value - t_1900, 1)
deltaI_mm = deltaI_mm.reshape(deltaI_mm.shape + (1,))


b = ax.boxplot(deltaI, positions=np.arange(0, n_mod), tick_labels=model_names, vert=False, showfliers=False, medianprops=medianprops, widths=0.7)
b2 = ax.boxplot(deltaI_mm, positions=[n_mod], tick_labels=[""], vert=False, showfliers=False, medianprops=medianprops, widths=0.7)
b3 = ax.boxplot(
    np.concatenate(deltaI), positions=[n_mod], tick_labels=["MMM"], vert=False, showfliers=False, patch_artist=True, medianprops=medianprops_mm, whis=(5, 95), widths=0.7, boxprops=boxprops
)

ax.set_xlim(0, 4)
ax.set_xlabel("Change in intensity [°C]")

figureName = "model_stats.png"

plt.savefig(figurePath + figureName, dpi=300, bbox_inches="tight")


"""
################################################################
Finally, calculate various percentiles for the distribution
################################################################
"""


def calculate_percentiles(f, arr, x):

    preind_mmm = pd.Series(f, index=x).rename("MMM")
    preind_50 = pd.DataFrame(arr, index=x).median(axis=1).rename("50")
    preind_25 = pd.DataFrame(arr, index=x).quantile(0.25, axis=1).rename("25")
    preind_75 = pd.DataFrame(arr, index=x).quantile(0.75, axis=1).rename("75")
    preind_05 = pd.DataFrame(arr, index=x).quantile(0.05, axis=1).rename("5")
    preind_95 = pd.DataFrame(arr, index=x).quantile(0.95, axis=1).rename("95")
    preind_025 = pd.DataFrame(arr, index=x).quantile(0.025, axis=1).rename("2.5")
    preind_975 = pd.DataFrame(arr, index=x).quantile(0.975, axis=1).rename("97.5")
    out = pd.concat([preind_mmm, preind_50, preind_25, preind_75, preind_05, preind_95, preind_025, preind_975], axis=1)

    return out


out_preind = calculate_percentiles(f_preind, f_preind_arr, x)
out_target = calculate_percentiles(f_target, f_target_arr, x)
out_future = calculate_percentiles(f_future, f_future_arr, x)


out_future = calculate_percentiles(f_future, f_future_arr, x)
