import matplotlib.pyplot as plt
import pandas as pd
from pygam import LinearGAM, LogisticGAM, PoissonGAM, GammaGAM, InvGaussGAM, GAM, s, f, l
import statsmodels.api as sm

import seaborn as sns
import numpy as np


lowess = sm.nonparametric.lowess


def residual_analysis(y1, y2, plot=False):
    resid = pd.Series(y1 - y2)
    q25 = resid.quantile(q=0.25)
    q75 = resid.quantile(q=0.75)
    if plot:
        plt.title("BOXPLOT RESIDUI")
        resid.plot.box()
        plt.show()
    iqr = q75-q25
    lower_lim = q25 - (iqr * 0.2)
    upper_lim = q75 + (iqr * 0.2)
    outlier_removed = pd.DataFrame(y1)
    outlier_removed.dropna(inplace=True)
    for index, row in outlier_removed.iterrows():
        if resid[index] < lower_lim or resid[index] > upper_lim:
            outlier_removed.drop(index, inplace=True)
    return outlier_removed


# Apply a smoothing algorithm to the series
def smoothing(pd_series, x, method, plot=False):
    frac = 0.04
    n_splines = 72
    lam = 0.1
    lams = np.logspace(-3, 5, 5)
    y = None
    if method == "lowess-gam":
        y = lowess(pd_series, x, xvals=x, frac=frac, is_sorted=True, return_sorted=False)
        pd_series = pd.Series(y).dropna()
        gam = LinearGAM(s(0, n_splines=n_splines, basis="ps", lam=lam)).fit(pd_series.index[:, None], pd_series)
        # gam = LinearGAM(s(0, n_splines=n_splines, basis="ps")).gridsearch(pd_series.index[:, None], pd_series, lam=lams)
        y = gam.predict(x)
        method = "LOWESS " + str(frac) + " -> LinearGAM n_splines: " + str(n_splines) + ", lam: "+str(lam)
    elif method == "lowess-gam2":
        y = lowess(pd_series, x, xvals=x, frac=frac, is_sorted=True, return_sorted=False)
        outlier_removed = residual_analysis(pd_series, y, plot)
        if plot:
            plt.scatter(outlier_removed.index, outlier_removed, color="r")
        gam = LinearGAM(s(0, n_splines=n_splines, basis="ps", lam=lam)).fit(outlier_removed.index[:, None], outlier_removed)
        y = gam.predict(x)
        method = "LOWESS " + str(frac) + " -> Outlier Removal -> LinearGAM n_splines: " + str(n_splines) + ", lam: "+str(lam)
    elif method == "gam":
        pd_series = pd_series.dropna()
        gam = LinearGAM(s(0, n_splines=n_splines, basis="ps", lam=lam)).fit(pd_series.index[:, None], pd_series)
        print(gam.summary())
        y = gam.predict(x)
        method = "LinearGAM n_splines: " + str(n_splines) + ", lam: "+str(lam)
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
        series.append(image[y, x, channel])
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


def full_image_time_series(data, slots, channel):
    image_series = []
    x_inc = []
    x_dates = []
    height, width = data[0].shape[:-1]
    print(height, width)
    for idx, image in enumerate(data):
        x_inc.append(idx)
        x_dates.append(slots[idx][0])

    for y in range(height):
        for x in range(width):
            series = []
            for idx, image in enumerate(data):
                series.append(image[y, x, channel])
            series = pd.Series(series)
            image_series.append(series)
    return x_inc, x_dates, image_series

