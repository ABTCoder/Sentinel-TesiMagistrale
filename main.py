import datetime
import os
import utils
import ts_pre_proc
from sentinelhub import SHConfig
import json
import pandas as pd
from evalscripts import evalscript_raw
import matplotlib.pyplot as plt

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
x_inc, x_dates, h_series, m_series = ts_pre_proc.multi_time_series(data, slots, "pixels.csv", 1)

'''series = pd.Series(data=series)
print(series)
filtered = series.dropna()
print(len(filtered))
plt.scatter(filtered.index, filtered)'''

# SMOOTHING
h_series, method = ts_pre_proc.smoothing_multi_ts(h_series, x_inc, "lowess")
m_series, method = ts_pre_proc.smoothing_multi_ts(m_series, x_inc, "lowess")

utils.plot_multi_series(h_series, "b")
utils.plot_multi_series(m_series, "r")

# PLOTTING AND SAVING

plt.title("GNDVI - "+method+" smoothing")
plt.ylabel("GNDVI")
plt.xlabel("BI-SETTIMANA")
plt.show()
'''to_save = pd.Series(y, x_dates)
to_save2 = pd.Series(y, x_inc)
to_save.to_csv("series1_explicit_date.csv")
to_save2.to_csv("series1_seq_doy.csv")'''

# utils.show_images(data, slots, "2020-04-01",  rss_size)
