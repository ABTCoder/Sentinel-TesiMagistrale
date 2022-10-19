import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    Geometry,
    SentinelHubStatistical,
    SentinelHubStatisticalDownloadClient,
    SHConfig,
    parse_time,
    bbox_to_dimensions,
)


config = SHConfig()

if not config.sh_client_id or not config.sh_client_secret:
    print("Warning! To use Statistical API, please provide the credentials (OAuth client ID and client secret).")

resolution = 10

imported = json.load(open("Buffer3.geojson"))
print(imported)
geom = Geometry.from_geojson(imported["features"][0]["geometry"])
rss_size = bbox_to_dimensions(geom.bbox, resolution=resolution)


def stats_to_df(stats_data):
    """Transform Statistical API response into a pandas.DataFrame"""
    df_data = []

    for single_data in stats_data["data"]:
        df_entry = {}
        is_valid_entry = True

        df_entry["interval_from"] = parse_time(single_data["interval"]["from"]).date()
        df_entry["interval_to"] = parse_time(single_data["interval"]["to"]).date()

        for output_name, output_data in single_data["outputs"].items():
            for band_name, band_values in output_data["bands"].items():
                band_stats = band_values["stats"]
                if band_stats["sampleCount"] == band_stats["noDataCount"]:
                    is_valid_entry = False
                    break

                for stat_name, value in band_stats.items():
                    col_name = f"{output_name}_{band_name}_{stat_name}"
                    if stat_name == "percentiles":
                        for perc, perc_val in value.items():
                            perc_col_name = f"{col_name}_{perc}"
                            df_entry[perc_col_name] = perc_val
                    else:
                        df_entry[col_name] = value

        if is_valid_entry:
            df_data.append(df_entry)

    return pd.DataFrame(df_data)


yearly_time_interval = "2021-01-01", "2022-09-23"

ndvi_evalscript = """
//VERSION=3

function setup() {
  return {
    input: [
      {
        bands: [
          "B04",
          "B08",
          "dataMask"
        ]
      }
    ],
    output: [
      {
        id: "ndvi",
        bands: 1
      },
      {
        id: "dataMask",
        bands: 1
      }
    ]
  }
}

function evaluatePixel(samples) {
    return {
      ndvi: [index(samples.B08, samples.B04)],
      dataMask: [samples.dataMask]
    };
}
"""

ndvi_request = SentinelHubStatistical(
    aggregation=SentinelHubStatistical.aggregation(
        evalscript=ndvi_evalscript,
        time_interval=yearly_time_interval,
        aggregation_interval="P7D",
    ),
    input_data=[SentinelHubStatistical.input_data(DataCollection.SENTINEL2_L2A)], #, maxcc=0.8)],
    #bbox=geom.bbox,
    geometry=geom,
    config=config,
    #calculations={"ndvi": {"histograms": {"default": {"nBins": 20, "lowEdge": -1.0, "highEdge": 1.0}}}}
)

ndvi_stats = ndvi_request.get_data()[0]

pd_stats = stats_to_df(ndvi_stats)

print(ndvi_stats)


fig, ax = plt.subplots(figsize=(15, 8))

pd_stats.plot(ax=ax, x="interval_from", y="ndvi_B0_mean", color="red", label="NDVI")

ax.fill_between(
    pd_stats.interval_from.values,
    pd_stats["ndvi_B0_mean"] - pd_stats["ndvi_B0_stDev"],
    pd_stats["ndvi_B0_mean"] + pd_stats["ndvi_B0_stDev"],
    color="red",
    alpha=0.3,
)

plt.show()