import utils
import ts_pre_proc
from sentinelhub import SHConfig
import pandas as pd
from evalscripts import evalscript_raw, evalscript_true_color, evalscript_raw_landsat8, evalscript_raw_landsat7, evalscript_true_color_ls7
import matplotlib.pyplot as plt
import matplotlib.image as plimg
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

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
'''config.instance_id = '<your instance id>'
config.sh_client_id = '<your client id>'
config.sh_client_secret = '<your client secret>'
config.save()'''

if not config.sh_client_id or not config.sh_client_secret:
    print("Warning! To use Process API, please provide the credentials (OAuth client ID and client secret).")

# MAIN PART

area = "cd6"

# CARICAMENTO GEOMETRIE
resolution = 10 #metri
geom, size = utils.load_geometry("geoms/"+area+".geojson", resolution)

# GEOMETRIE A COLORI
resolution_c = 10 #metri
geom_color, size_color = utils.load_geometry("geoms/"+area+".geojson", resolution_c)

# Crea lista di date : tupla (inizio, fine)
slots = utils.dates_list("2017-01-01", "2022-12-15", "7D", 7)
#slots = utils.fixed_weeks_per_year(2005, 18)
# crea lista di richieste all'api
list_of_requests = [utils.get_request(config, evalscript_raw, slot, geom, size, "ndvi_gndvi_buffer3",
                                      data_coll=DataCollection.SENTINEL2_L2A) for slot in slots]
request_true_color = utils.get_request(config, evalscript_true_color, ("2022-05-01", "2022-07-31"), geom_color, size_color,
                                       "true_color", MimeType.PNG, data_coll=DataCollection.SENTINEL2_L2A)
list_of_requests = [request.download_list[0] for request in list_of_requests]
# download data with multiple threads
data = SentinelHubDownloadClient(config=config).download(list_of_requests, max_threads=5, show_progress=True)

# utils.print_file_names("ndvi_gndvi_buffer3")


def main_multi():
    # Recupera immagine a colori per riferimento
    true_color_imgs = request_true_color.get_data()
    image = true_color_imgs[0]
    plt.imshow(image)
    plt.show()

    # Recupera time series per zona di controllo e zona di test
    utils.select_pixels(image, "pixels1.csv")
    utils.select_pixels(image, "pixels2.csv")
    x_dates, h_series = ts_pre_proc.multi_time_series(data, slots, "pixels1.csv", 0)
    _, m_series = ts_pre_proc.multi_time_series(data, slots, "pixels2.csv", 0)
    h_series, method = ts_pre_proc.smoothing_multi_ts(h_series, "lowess-gam2", params)
    m_series, method = ts_pre_proc.smoothing_multi_ts(m_series, "lowess-gam2", params)

    # Plotting
    utils.plot_multi_series(h_series, "b")
    utils.plot_multi_series(m_series, "r")
    plt.title("NDVI - " + method)
    plt.ylabel("NDVI")
    plt.xlabel("SETTIMANA")

    red_patch = mpatches.Patch(color='red', label='METANODOTTO')
    blue_patch = mpatches.Patch(color='blue', label='VEGETAZIONE')
    plt.legend(handles=[red_patch, blue_patch])
    plt.show()

    # Salva in csv
    hpd = pd.concat(h_series, axis=1)
    hpd["date"] = x_dates
    hpd.to_csv("ts/"+area+"_control.csv")
    mpd = pd.concat(m_series, axis=1)
    mpd["date"] = x_dates
    mpd.to_csv("ts/"+area+"_test.csv")


def main_full():
    true_color_imgs = request_true_color.get_data()
    image = true_color_imgs[0]
    plt.imshow(image)
    plimg.imsave("areas_img/" + area + ".png", image)
    plt.show()

    x_dates, image_series, height, width = ts_pre_proc.full_image_time_series(data, slots, 0)
    image_series, method = ts_pre_proc.smoothing_multi_ts(image_series, "lowess-gam2", params)
    hpd = pd.concat(image_series, axis=1)
    hpd["date"] = x_dates
    hpd.to_csv("ts/"+area+"_{0}_{1}.csv".format(height, width))


def main_single():
    true_color_imgs = request_true_color.get_data()
    image = true_color_imgs[0]
    pixel = utils.select_single_pixel(image)
    x_dates, series = ts_pre_proc.single_time_series(data, slots, pixel[0], pixel[1], 0)
    series = pd.Series(data=series)
    filtered = series.dropna()
    print(len(filtered))
    y, method = ts_pre_proc.smoothing(series, "lowess-gam2", params, True)
    plt.scatter(filtered.index, filtered, alpha=0.5, color="lightblue")
    plt.title("NDVI - " + method)
    plt.plot(y, color="r")
    plt.ylabel("NDVI")
    plt.xlabel("SETTIMANA")
    plt.show()
# SMOOTHING


def no_smoothing():
    true_color_imgs = request_true_color.get_data()
    image = true_color_imgs[0]
    plt.imshow(image)
    plimg.imsave("areas_img/" + area + ".png", image)
    plt.show()

    x_dates, image_series, height, width = ts_pre_proc.full_image_time_series(data, slots, 0)
    dataframe = ts_pre_proc.multi_remove_outliers(image_series, x_dates, params)
    dataframe.to_csv("ts/" + area + "_no_smoothing_{0}_{1}.csv".format(height, width))


def raw():
    true_color_imgs = request_true_color.get_data()
    image = true_color_imgs[0]
    plt.imshow(image)
    plimg.imsave("areas_img/" + area + ".png", image)
    plt.show()

    x_dates, image_series, height, width = ts_pre_proc.full_image_time_series(data, slots, 0)
    dataframe = pd.concat(image_series, axis=1)
    dataframe["date"] = x_dates
    dataframe.to_csv("ts/" + area + "_raw_{0}_{1}.csv".format(height, width))


def yearly_box_plot():
    fig, axs = plt.subplots(2, 1, sharey=True, figsize=(20, 10))
    true_color_imgs = request_true_color.get_data()
    image = true_color_imgs[0]
    pixel = utils.select_single_pixel(image)
    x_dates, series = ts_pre_proc.single_time_series(data, slots, pixel[0], pixel[1], 0)

    df = pd.DataFrame()
    df["date"] = pd.to_datetime(x_dates)
    outlier_removed = ts_pre_proc.remove_outliers(series, params)
    y, _ = ts_pre_proc.smoothing(series, "lowess", params, True)
    cols = [df, y, series]
    df = pd.concat(cols, axis=1)
    df.columns = ["date", 0, 1]
    df.loc[(df["date"].dt.month < 5) | (df["date"].dt.month >= 9), 0] = np.NaN
    df.loc[(df["date"].dt.month < 5) | (df["date"].dt.month >= 9), 1] = np.NaN
    #axs[0].plot(df["date"], df[0], color="r")
    df["year"] = df["date"].dt.year

    sns.boxplot(data=df, x="year", y=1, ax=axs[1])
    axs[0].scatter(df["date"], df[1])
    plt.show()


params = {
    "frac": 0.06,
    "n_splines": 120,
    "resid_fac": 0.2,
    "lam": 0.1
}

raw()
# tils.save_images(data, area, slots, "images")


# utils.show_images(data, slots, "2020-04-01",  rss_size)
