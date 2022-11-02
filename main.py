import utils
import ts_pre_proc
from sentinelhub import SHConfig
import pandas as pd
from evalscripts import evalscript_raw, evalscript_true_color
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
geom, rss_size = utils.load_geometry("geoms/faggeta_1.geojson", resolution)

# Create list of days
slots = utils.all_days("2017-01-01", "2022-10-17", "7D", 7)
# create a list of requests
list_of_requests = [utils.get_request(config, evalscript_raw, slot, geom, rss_size, "ndvi_gndvi_buffer3") for slot in slots]
list_of_requests = [request.download_list[0] for request in list_of_requests]
# download data with multiple threads
data = SentinelHubDownloadClient(config=config).download(list_of_requests, max_threads=5, show_progress=True)

# utils.print_file_names("ndvi_gndvi_buffer3")


def main_multi():
    # Create time-series for one pixel
    request_true_color = utils.get_request(config, evalscript_true_color, ("2021-01-01", "2022-10-31"), geom, rss_size, "true_color", MimeType.PNG)
    true_color_imgs = request_true_color.get_data()
    image = true_color_imgs[0]
    plt.imshow(image)
    plt.show()
    utils.select_pixels(image, "pixels.csv", "a")
    utils.select_pixels(image, "pixels.csv", "b")
    x_inc, x_dates, h_series, m_series = ts_pre_proc.multi_time_series(data, slots, "pixels.csv", 0)
    h_series, method = ts_pre_proc.smoothing_multi_ts(h_series, x_inc, "lowess-gam2")
    m_series, method = ts_pre_proc.smoothing_multi_ts(m_series, x_inc, "lowess-gam2")

    utils.plot_multi_series(h_series, "b")
    utils.plot_multi_series(m_series, "r")
    plt.title("NDVI - " + method)
    plt.ylabel("NDVI")
    plt.xlabel("SETTIMANA")

    red_patch = mpatches.Patch(color='red', label='METANODOTTO')
    blue_patch = mpatches.Patch(color='blue', label='VEGETAZIONE')
    plt.legend(handles=[red_patch, blue_patch])
    plt.show()

    hpd = pd.concat(h_series, axis=1)
    hpd["date"] = x_dates
    hpd.to_csv("healthy_bw.csv")
    mpd = pd.concat(m_series, axis=1)
    mpd["date"] = x_dates
    mpd.to_csv("modified_bw.csv")


def main_all():
    x_inc, x_dates, image_series = ts_pre_proc.full_image_time_series(data, slots, 0)
    image_series, method = ts_pre_proc.smoothing_multi_ts(image_series, x_inc, "lowess-gam2")
    hpd = pd.concat(image_series, axis=1)
    hpd["date"] = x_dates
    hpd.to_csv("fullimage.csv")


def main():
    request_true_color = utils.get_request(config, evalscript_true_color, ("2021-01-01", "2022-10-31"), geom, rss_size,
                                           "true_color", MimeType.PNG)
    true_color_imgs = request_true_color.get_data()
    image = true_color_imgs[0]
    pixel = utils.select_single_pixel(image)
    x_inc, x_dates, series = ts_pre_proc.single_time_series(data, slots, pixel[0], pixel[1], 0)
    series = pd.Series(data=series)
    filtered = series.dropna()
    print(len(filtered))
    y, method = ts_pre_proc.smoothing(series, x_inc, "lowess-gam2", True)
    plt.scatter(filtered.index, filtered, alpha=0.5, color="lightblue")
    plt.title("NDVI - " + method)
    plt.plot(y, color="r")
    plt.ylabel("NDVI")
    plt.xlabel("SETTIMANA")
    plt.show()
# SMOOTHING


main_multi()
# PLOTTING AND SAVING
'''to_save = pd.Series(y, x_dates)
to_save2 = pd.Series(y, x_inc)
to_save.to_csv("series1_explicit_date.csv")
to_save2.to_csv("series1_seq_doy.csv")'''

# utils.show_images(data, slots, "2020-04-01",  rss_size)
