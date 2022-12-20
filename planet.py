import utils
import os
import matplotlib.image as plimg
import matplotlib.pyplot as plt
import rasterio as rs
import numpy as np
from bs4 import BeautifulSoup
import datetime
import pandas as pd
import utils
import ts_pre_proc
import matplotlib.patches as mpatches

slots = utils.dates_list("2016-01-01", "2022-12-31", "7D", 7)

tcd1 = datetime.datetime.strptime("2021-05-01", "%Y-%m-%d")
tcd2 = datetime.datetime.strptime("2021-05-30", "%Y-%m-%d")

dir_path = "ced_e"
img = None
ndvi_images = []
dates = []
height, width = 0, 0
true_color_img = None
for path in os.scandir(dir_path):
    if path.is_file():
        if path.name[-4:] == ".xml":
            date = path.name[:4] + "-" + path.name[4:6] + "-" + path.name[6:8]
            dates.append(date)
            dt = datetime.datetime.strptime(date, "%Y-%m-%d")

            file = open(path)
            doc = BeautifulSoup(file, "xml")
            factors = doc.find_all("ps:bandSpecificMetadata")
            for factor in factors:
                children = list(factor.children)
                band = int(children[1].text)
                fac = float(children[9].text)
                img[:, :, band-1] = img[:, :, band-1] * fac

            ndvi_img = (img[:, :, 3] - img[:, :, 0])/(img[:, :, 3] + img[:, :, 0])
            ndvi_images.append(ndvi_img)

            if tcd1 <= dt < tcd2:
                true_color_img = (img[:, :, :3]*3.5).astype(np.float32)

        if path.name[-4:] == ".tif":
            if "Analytic" in path.name:
                geo = rs.open(path)
                img = np.empty((geo.height, geo.width, 4))
                height = geo.height
                width = geo.width
                img[:, :, 0] = geo.read(1)
                img[:, :, 1] = geo.read(2)
                img[:, :, 2] = geo.read(3)
                img[:, :, 3] = geo.read(4)

k = 0
slots2 = []
data = []
empty = np.empty((height, width, 1))
empty[:, :] = np.nan
first_valid = None
for i, s in enumerate(slots):
    d1 = datetime.datetime.strptime(s[0], "%Y-%m-%d")
    d2 = datetime.datetime.strptime(s[1], "%Y-%m-%d")
    d = datetime.datetime.strptime(dates[k], "%Y-%m-%d")
    while d < d1:
        k += 1
        d = datetime.datetime.strptime(dates[k], "%Y-%m-%d")
    if d1 <= d < d2:
        if first_valid is None:
            first_valid = i
        slots2.append((dates[k], s[1]))
        data.append(ndvi_images[k])
        k += 1
        if k == len(dates)-1:
            break
    else:
        slots2.append(s)
        data.append(empty)


print(len(slots), len(slots2))
slots2 = slots2[first_valid:]
data = data[first_valid:]


def planet_single():
    pixel = utils.select_single_pixel(true_color_img)
    x_dates, series = ts_pre_proc.single_time_series(data, slots2, pixel[0], pixel[1], 0)
    series = pd.Series(data=series)
    filtered = series.dropna()
    print(len(filtered.index))
    y, method = ts_pre_proc.smoothing(series, "lowess-gam2", params, True)
    plt.scatter(filtered.index, filtered, alpha=0.5, color="lightblue")
    plt.title("NDVI - " + method)
    plt.plot(y, color="r")
    plt.ylabel("NDVI")
    plt.xlabel("SETTIMANA")
    plt.show()


def planet_multi():
    utils.select_pixels(true_color_img, "pixels1.csv")
    utils.select_pixels(true_color_img, "pixels2.csv")
    x_dates, h_series = ts_pre_proc.multi_time_series(data, slots2, "pixels1.csv", 0)
    _, m_series = ts_pre_proc.multi_time_series(data, slots2, "pixels2.csv", 0)
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
    hpd.to_csv("ts/" + area + "_control.csv")
    mpd = pd.concat(m_series, axis=1)
    mpd["date"] = x_dates
    mpd.to_csv("ts/" + area + "_test.csv")


def planet_full():
    plimg.imsave("areas_img/" + area + ".png", true_color_img)
    x_dates, image_series, _, _ = ts_pre_proc.full_image_time_series(data, slots2, 0)
    image_series, method = ts_pre_proc.smoothing_multi_ts(image_series, "lowess-gam2", params)
    hpd = pd.concat(image_series, axis=1)
    hpd["date"] = x_dates
    hpd.to_csv("ts/" + area + "_{0}_{1}.csv".format(height, width))


area = "planet_ced_e"

params = {
    "frac": 0.08,
    "n_splines": 120,
    "resid_fac": 0.2,
    "lam": 0.1
}

planet_multi()
