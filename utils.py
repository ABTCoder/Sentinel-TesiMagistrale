import datetime
import random
import pandas as pd
import os
import json
import cv2

import matplotlib.pyplot as plt
from sentinelhub import (
    DataCollection,
    MimeType,
    MosaickingOrder,
    SentinelHubRequest,
    bbox_to_dimensions,
    Geometry
)


def intervals_by_chunks(start, end, n_chunks):
    start = datetime.datetime.strptime(start)
    end = datetime.datetime.strptime(end)
    tdelta = (end - start) / n_chunks
    edges = [(start + i * tdelta).date().isoformat() for i in range(n_chunks)]
    slots = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

    print("Monthly time windows:\n")
    for slot in slots:
        print(slot)

    return slots


def all_days(start, end, freq="D", interval_size=0):
    slots = []
    dates = pd.date_range(start=start, end=end, freq=freq)
    for d in dates:
        date2 = d + datetime.timedelta(days=interval_size)
        date = d.strftime("%Y-%m-%d")
        date2 = date2.strftime("%Y-%m-%d")
        slots.append((date, date2))
    return slots


def print_file_names(folder_name):
    for folder, _, filenames in os.walk(folder_name):
        for filename in filenames:
            print(os.path.join(folder, filename))


def get_request(config, evalscript, time_interval, geom=None, rss_size=None, data_folder=None, mimetype=MimeType.TIFF):
    return SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=time_interval,
                mosaicking_order=MosaickingOrder.LEAST_CC,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", mimetype)],
        bbox=geom.bbox,
        size=rss_size,
        # geometry=geom,
        config=config,
        data_folder= data_folder
    )
    # true_color_imgs = request_true_color.get_data()


# Loads a geojson geometry file, returns the geometry object and the image size at the specified resolution
def load_geometry(file, resolution):
    imported = json.load(open(file))
    geom = Geometry.from_geojson(imported["features"][0]["geometry"])
    image_size = bbox_to_dimensions(geom.bbox, resolution=resolution)
    print(f"Image shape at {resolution} m resolution: {image_size} pixels")
    return geom, image_size


def show_images(data, dates, start_date, image_size):
    ncols = 4
    nrows = 3
    aspect_ratio = image_size[0] / image_size[1]
    subplot_kw = {"xticks": [], "yticks": [], "frame_on": False}

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(5 * ncols * aspect_ratio, 5 * nrows),
                            subplot_kw=subplot_kw)
    i = 1
    for idx, image in enumerate(data):
        s = dates[idx][0]
        e = dates[idx][1]
        if s >= start_date:
            if i >= ncols*nrows:
                break
            ax = axs[i // ncols][i % ncols]
            ax.imshow(image[:, :, 0])
            ax.set_title(f"{s}  -  {e}", fontsize=10)
            i += 1
    plt.tight_layout()
    plt.show()


def plot_multi_series(series_list, color):
    for series in series_list:
        plt.plot(series, color=color, label=color)


def select_pixels(img, coord_file, type):
    """
    Opens an image, draws red dots, save coords
    """

    # mouse callback function
    def red_dot(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            with open(coord_file, "a") as f:
                f.write("{0}, {1}, {2}\n".format(x, y, type))
            cv2.circle(img, (x, y), 0, (0, 0, 255), -1)

    # interactive display
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    clone = img.copy()
    cv2.namedWindow('pixel selector '+type, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('pixel selector '+type, red_dot)

    # event handler
    while (1):
        cv2.imshow('pixel selector '+type, img)
        key = cv2.waitKey(1) & 0xFF
        # escape
        if key == 27 or key == ord('q'):
            cv2.destroyAllWindows()
            return
        # next
        if key == ord("n"):
            cv2.destroyAllWindows()
            return
        # refresh dots
        if key == ord('r'):
            with open(coord_file, "w") as f:
                f.write("")
            img = clone.copy()

    cv2.destroyAllWindows()
    return


def select_single_pixel(img):
    """
    Opens an image, draws red dots, save coords
    """
    current_pixel = ()

    # mouse callback function
    def red_dot(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            nonlocal img
            img = clone.copy()
            nonlocal current_pixel
            current_pixel = (x, y)
            cv2.circle(img, (x, y), 0, (0, 0, 255), -1)

    # interactive display
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    clone = img.copy()
    cv2.namedWindow('pixel selector', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('pixel selector', red_dot)

    # event handler
    while (1):
        cv2.imshow('pixel selector', img)
        key = cv2.waitKey(1) & 0xFF
        # escape
        if key == 27 or key == ord('q'):
            cv2.destroyAllWindows()
            return current_pixel
        # next
        if key == ord("n"):
            cv2.destroyAllWindows()
            return current_pixel
        # refresh dots
        if key == ord('r'):
            img = clone.copy()

    cv2.destroyAllWindows()
    return current_pixel
