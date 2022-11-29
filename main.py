import utils
import ts_pre_proc
from sentinelhub import SHConfig
import pandas as pd
from evalscripts import evalscript_raw, evalscript_true_color, evalscript_raw_landsat
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
'''config.instance_id = '<your instance id>'
config.sh_client_id = '<your client id>'
config.sh_client_secret = '<your client secret>'
config.save()'''

if not config.sh_client_id or not config.sh_client_secret:
    print("Warning! To use Process API, please provide the credentials (OAuth client ID and client secret).")

# MAIN PART

area = "ced1"

# CARICAMENTO GEOMETRIE
resolution = 10 #metri
geom, rss_size = utils.load_geometry("geoms/"+area+".geojson", resolution)

# Crea lista di date : tupla (inizio, fine)
slots = utils.all_days("2013-01-01", "2022-11-23", "7D", 7)
# crea lista di richieste all'api
list_of_requests = [utils.get_request(config, evalscript_raw_landsat, slot, geom, rss_size, "ndvi_gndvi_buffer3",
                                      data_coll=DataCollection.LANDSAT_OT_L2) for slot in slots]
list_of_requests = [request.download_list[0] for request in list_of_requests]
# download data with multiple threads
data = SentinelHubDownloadClient(config=config).download(list_of_requests, max_threads=5, show_progress=True)

# utils.print_file_names("ndvi_gndvi_buffer3")


def main_multi():
    # Recupera immagine a colori per riferimento
    request_true_color = utils.get_request(config, evalscript_true_color, ("2020-05-01", "2020-05-31"), geom, rss_size, "true_color", MimeType.PNG)
    true_color_imgs = request_true_color.get_data()
    image = true_color_imgs[0]
    plt.imshow(image)
    plt.show()

    # Recupera time series per zona di controllo e zona di test
    utils.select_pixels(image, "pixels1.csv")
    utils.select_pixels(image, "pixels2.csv")
    x_dates, h_series = ts_pre_proc.multi_time_series(data, slots, "pixels1.csv", 0)
    _, m_series = ts_pre_proc.multi_time_series(data, slots, "pixels2.csv", 0)
    h_series, method = ts_pre_proc.smoothing_multi_ts(h_series, "lowess-gam2")
    m_series, method = ts_pre_proc.smoothing_multi_ts(m_series, "lowess-gam2")

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


def main_full_image():
    request_true_color = utils.get_request(config, evalscript_true_color, ("2020-01-01", "2020-01-31"), geom, rss_size,
                                           "true_color", MimeType.PNG)
    true_color_imgs = request_true_color.get_data()
    image = true_color_imgs[0]
    plt.imshow(image)
    plt.show()

    x_dates, image_series, height, width = ts_pre_proc.full_image_time_series(data, slots, 0)
    image_series, method = ts_pre_proc.smoothing_multi_ts(image_series, "lowess-gam2")
    hpd = pd.concat(image_series, axis=1)
    hpd["date"] = x_dates
    hpd.to_csv("ts/"+area+"_{0}_{1}.csv".format(height, width))


def main_single():
    request_true_color = utils.get_request(config, evalscript_true_color, ("2020-01-01", "2020-10-31"), geom, rss_size,
                                           "true_color", MimeType.PNG)
    true_color_imgs = request_true_color.get_data()
    image = true_color_imgs[0]
    pixel = utils.select_single_pixel(image)
    x_dates, series = ts_pre_proc.single_time_series(data, slots, pixel[0], pixel[1], 0)
    series = pd.Series(data=series)
    filtered = series.dropna()
    print(len(filtered))
    y, method = ts_pre_proc.smoothing(series, "lowess-gam2", True)
    plt.scatter(filtered.index, filtered, alpha=0.5, color="lightblue")
    plt.title("NDVI - " + method)
    plt.plot(y, color="r")
    plt.ylabel("NDVI")
    plt.xlabel("SETTIMANA")
    plt.show()
# SMOOTHING


#main_full_image()
utils.save_images(data, area, slots, "images")


# utils.show_images(data, slots, "2020-04-01",  rss_size)
