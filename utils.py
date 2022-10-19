import datetime
import pandas as pd
import os
import json

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


def get_request(config, evalscript, time_interval, geom=None, rss_size=None, data_folder=None):
    return SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=time_interval,
                mosaicking_order=MosaickingOrder.LEAST_CC,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
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
