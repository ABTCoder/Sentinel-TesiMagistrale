import pandas as pd
from pygam import LinearGAM, LogisticGAM, PoissonGAM, GammaGAM, InvGaussGAM, GAM, s, f, l
import statsmodels.api as sm

import numpy as np


lowess = sm.nonparametric.lowess


# Apply a smoothing algorithm to the series
def smoothing(pd_series, x, method):
    frac = 0.06
    y = None
    if method == "lowess-gam":
        y = lowess(pd_series, x, xvals=x, frac=frac, is_sorted=True, return_sorted=False)
        lams = np.logspace(-3, 5, 5)
        pd_series = pd.Series(y).dropna()
        gam = LinearGAM(s(0, n_splines=40, basis="ps")).gridsearch(pd_series.index[:, None], pd_series, lam=lams)
        print(gam.summary())
        y = gam.predict(x)
        method = "LOWESS " + str(frac) + " LinearGAM"
    elif method == "gam":
        lams = np.logspace(-3, 5, 5)
        gam = LinearGAM(s(0, n_splines=40, basis="ps")).gridsearch(pd_series.index[:, None], pd_series, lam=lams)
        print(gam.summary())
        y = gam.predict(x)
        method = "LinearGAM"
    elif method == "lowess":
        y = lowess(pd_series, x, xvals=x, frac=frac, is_sorted=True, return_sorted=False)
        method = "LOWESS " + str(frac)
    return pd.Series(y), method


# Apply smoothing to multiple time series (same x)
def smoothing_multi_ts(series_list, x, method):
    smoothed = []
    for series in series_list:
        y, method_t = smoothing(series, x, method)
        smoothed.append(y)
    return smoothed, method_t


# Extract one pixel time series
def single_time_series(data, slots, x, y, channel):
    series = []
    x_inc = []
    x_dates = []
    for idx, image in enumerate(data):
        series.append(image[x, y, channel])
        x_inc.append(idx)
        x_dates.append(slots[idx][0])
    return x_inc, x_dates, pd.Series(series)


# Extract multiple time series
def multi_time_series(data, slots, coord_file, channel):
    healthy_series = []
    modified_series = []
    x_inc = []
    x_dates = []
    for idx, image in enumerate(data):
        x_inc.append(idx)
        x_dates.append(slots[idx][0])
    with open(coord_file, "r") as file:
        lines = file.readlines()
        for line in lines:
            series = []
            vals = line.split(",")
            for idx, image in enumerate(data):
                series.append(image[int(vals[1]), int(vals[0]), channel])
            series = pd.Series(series)
            if vals[2].strip() == "a":
                healthy_series.append(series)
            else:
                modified_series.append(series)
    return x_inc, x_dates, healthy_series, modified_series