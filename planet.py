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
d1 = datetime.datetime.strptime(slots[k][0], "%Y-%m-%d")
d2 = datetime.datetime.strptime(slots[k][1], "%Y-%m-%d")
l = len(slots)
data = []
empty = np.empty((height, width, 1))
empty[:,:] = np.nan
for d, img in zip(dates, ndvi_images):
    d = datetime.datetime.strptime(d, "%Y-%m-%d")
    while d < d1 or d > d2:
        slots2.append(slots[k])
        data.append(empty)
        if k == l-1:
            break
        k += 1
        d1 = datetime.datetime.strptime(slots[k][0], "%Y-%m-%d")
        d2 = datetime.datetime.strptime(slots[k][1], "%Y-%m-%d")
    slots2.append((d, slots[k][1]))
    data.append(img)


pixel = utils.select_single_pixel(true_color_img)
x_dates, series = ts_pre_proc.single_time_series(data, slots2, pixel[0], pixel[1], 0)
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