import datetime
import os
import statsmodels.nonparametric.smoothers_lowess
import utils
from pygam import LinearGAM, LogisticGAM, PoissonGAM, GammaGAM, InvGaussGAM, GAM, s, f, l
from sentinelhub import SHConfig
import json
import pandas as pd
from evalscripts import evalscript_raw
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
lowess = sm.nonparametric.lowess
from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    DownloadRequest,
    MimeType,
    MosaickingOrder,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    bbox_to_dimensions,
    Geometry
)

config = SHConfig()

if not config.sh_client_id or not config.sh_client_secret:
    print("Warning! To use Process API, please provide the credentials (OAuth client ID and client secret).")

# MAIN PART

# LOADING GEOMETRY
resolution = 10
geom, rss_size = utils.load_geometry("Buffer3.geojson", resolution)

# Create list of days
slots = utils.all_days("2017-01-01", "2022-10-17", "14D", 14)
# create a list of requests
list_of_requests = [utils.get_request(config, evalscript_raw, slot, geom, rss_size, "ndvi_gndvi_buffer3") for slot in slots]
list_of_requests = [request.download_list[0] for request in list_of_requests]
# download data with multiple threads
data = SentinelHubDownloadClient(config=config).download(list_of_requests, max_threads=5, show_progress=True)

# utils.print_file_names("ndvi_gndvi_buffer3")

# Create time-series for one pixel
series = []
x = []
x_dates = []
for idx, image in enumerate(data):
    series.append(image[2, 3, 1])
    x.append(idx)
    x_dates.append(slots[idx][0])

series = pd.Series(data=series)
print(series)
filtered = series.dropna()
print(len(filtered))
plt.scatter(filtered.index, filtered)

# SMOOTHING
gam_smoothing = False
loess_gam = True
method = ""

if loess_gam:
    frac = 0.1
    y = lowess(series, x, xvals=x, frac=frac, is_sorted=True, return_sorted=False)
    lams = np.logspace(-3, 5, 5)
    gam = LinearGAM(s(0, n_splines=40, basis="ps")).gridsearch(series.index[:,None], y, lam=lams)
    print(gam.summary())
    y = gam.predict(x)
    method = "LOESS"+str(frac)+" LinearGAM"
elif gam_smoothing:
    lams = np.logspace(-3, 5, 5)
    gam = LinearGAM(s(0, n_splines=40, basis="ps")).gridsearch(filtered.index[:,None], filtered, lam=lams)
    print(gam.summary())
    y = gam.predict(x)
    method = "LinearGAM"
else:
    frac = 0.1
    y = lowess(series, x, xvals=x, frac=frac, is_sorted=True, return_sorted=False)
    method = "LOESS "+str(frac)

# PLOTTING AND SAVING

plt.title("GNDVI - "+method+" smoothing")
plt.plot(y, color="g")
plt.ylabel("GNDVI")
plt.xlabel("BI-SETTIMANA")
plt.show()
to_save = pd.Series(y, x_dates)
to_save2 = pd.Series(y, x)
to_save.to_csv("series1_explicit_date.csv")
to_save2.to_csv("series1_seq_doy.csv")


# utils.show_images(data, slots, "2020-04-01",  rss_size)
