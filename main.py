import utils
import ts_pre_proc
from sentinelhub import SHConfig
import pandas as pd
import evalscripts as es
import matplotlib.pyplot as plt
import matplotlib.image as plimg
import matplotlib.patches as mpatches
import seaborn as sns
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
# Specificare l'API Key di sentinel in questa sezione
'''config.instance_id = '<your instance id>'
config.sh_client_id = '<your client id>'
config.sh_client_secret = '<your client secret>'
config.save()'''

if not config.sh_client_id or not config.sh_client_secret:
    print("Warning! To use Process API, please provide the credentials (OAuth client ID and client secret).")

# MAIN PART

# Nome del geojson da utilizzare all'interno della cartella geoms
area = "cd5"

# Caricamento delle geometrie per le immagini NDVI/GNDVI
resolution = 10 #metri
geom, size = utils.load_geometry("geoms/"+area+".geojson", resolution)

# Caricamento delle geometrie per l'immagine a colori di riferimento
resolution_c = 10 #metri
geom_color, size_color = utils.load_geometry("geoms/"+area+".geojson", resolution_c)

# Crea lista di intervalli di date : lista di tuple (inizio, fine)
slots = utils.dates_list("2017-01-01", "2022-12-31", "7D", 7)

# crea lista di richieste all'api
# I parametri importanti sono l'evalscript e la data collection che specifica il satellite
list_of_requests = [utils.get_request(config, es.evalscript_raw, slot, geom, size, "cache",
                                      data_coll=DataCollection.SENTINEL2_L2A) for slot in slots]
request_true_color = utils.get_request(config, es.evalscript_true_color, ("2022-06-01", "2022-06-30"), geom_color, size_color,
                                       "true_color", MimeType.PNG, data_coll=DataCollection.SENTINEL2_L2A)
list_of_requests = [request.download_list[0] for request in list_of_requests]

# Scarica i dati con thread multipli
data = SentinelHubDownloadClient(config=config).download(list_of_requests, max_threads=5, show_progress=True)


def main_multi():
    """
    Main utilizzato per estrarre le time series di pixel scelti manualmente.
    Verranno creati due file: nomeArea_control.csv e nomeArea_test.csv (cartella ts)
    Durante l'esecuzione si potranno scegliere i pixel dei due file in modo separato.
    """
    # Recupera immagine a colori per riferimento
    true_color_imgs = request_true_color.get_data()
    image = true_color_imgs[0]
    plt.imshow(image)
    plt.show()

    # Recupera time series per zona di controllo e zona di test
    utils.select_pixels(image, "pixels_control.csv")
    utils.select_pixels(image, "pixels_test.csv")
    x_dates, control_series = ts_pre_proc.multi_time_series(data, slots, "pixels_control.csv", 0)
    _, test_series = ts_pre_proc.multi_time_series(data, slots, "pixels_test.csv", 0)
    control_series, method = ts_pre_proc.smoothing_multi_ts(control_series, params)
    test_series, method = ts_pre_proc.smoothing_multi_ts(test_series, params)

    # Plotting
    utils.plot_multi_series(control_series, "b")
    utils.plot_multi_series(test_series, "r")
    plt.title("NDVI - " + method)
    plt.ylabel("NDVI")
    plt.xlabel("SETTIMANA")

    red_patch = mpatches.Patch(color='red', label='METANODOTTO')
    blue_patch = mpatches.Patch(color='blue', label='VEGETAZIONE')
    plt.legend(handles=[red_patch, blue_patch])
    plt.show()

    # Salva in csv
    cdf = pd.concat(control_series, axis=1)
    cdf["date"] = x_dates
    cdf.to_csv("ts/"+area+"_control.csv")
    tdf = pd.concat(test_series, axis=1)
    tdf["date"] = x_dates
    tdf.to_csv("ts/"+area+"_test.csv")


def main_full():
    """
    Main utilizzato per estrarre le time series NDVI/GNDVI di tutti i pixel dell'area di studio (cartella ts)
    Nella cartella areas_img verrà salvata anche l'immagine a colori di riferimento
    """
    true_color_imgs = request_true_color.get_data()
    image = true_color_imgs[0]
    plt.imshow(image)
    plimg.imsave("areas_img/" + area + ".png", image)
    plt.show()

    x_dates, image_series, height, width = ts_pre_proc.full_image_time_series(data, slots, 0)
    image_series, method = ts_pre_proc.smoothing_multi_ts(image_series, params)
    hpd = pd.concat(image_series, axis=1)
    hpd["date"] = x_dates
    hpd.to_csv("ts/"+area+"_{0}_{1}.csv".format(height, width))
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.ylim([0, 1])
    plt.title("Faggeta - "+method)
    plt.ylabel("NDVI")
    for s in image_series:
        plt.plot(s)
    plt.xlabel("Settimana")
    plt.show()


def main_single():
    """
    Main utilizzato per estrarre la time series di un pixel a scelta
    Utile per testare i parametri di smoothing
    """
    true_color_imgs = request_true_color.get_data()
    image = true_color_imgs[0]
    pixel = utils.select_single_pixel(image)
    x_dates, series = ts_pre_proc.single_time_series(data, slots, pixel[0], pixel[1], 0)
    filtered = series.dropna()
    print(len(filtered))
    plt.rcParams["figure.figsize"] = (12, 6)
    #plt.ylim([0, 1])
    #plt.scatter(filtered.index, filtered, alpha=0.5, color="red")
    y, method = ts_pre_proc.smoothing(series, params)
    #plt.title("NDVI - " + method)
    plt.title("Ceduazioni, Landsat8 - "+method)
    plt.plot(y, color="g")
    plt.ylabel("NDVI")
    plt.xlabel("Settimana")
    plt.show()
# SMOOTHING


def no_smoothing():
    """
    Main utilizzato per ottenere le time series di tutti i pixel applicando solo la fase di rimozione degli outlier
    (cartella ts/no_smoothing
    """
    true_color_imgs = request_true_color.get_data()
    image = true_color_imgs[0]
    plt.imshow(image)
    plimg.imsave("areas_img/" + area + ".png", image)
    plt.show()

    x_dates, image_series, height, width = ts_pre_proc.full_image_time_series(data, slots, 0)
    dataframe = ts_pre_proc.multi_remove_outliers(image_series, x_dates, params)
    dataframe.to_csv("ts/no_smoothing/" + area + "_no_smoothing_{0}_{1}.csv".format(height, width))


def raw():
    """
    Main utilizzato per ottenere le time series grezze di tutti i pixel dell'area di studio
    (cartella ts/raw)
    """
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
    """
    Main di test
    """
    fig, axs = plt.subplots(3, 1, sharey=True, figsize=(20, 10))
    true_color_imgs = request_true_color.get_data()
    image = true_color_imgs[0]
    pixel = utils.select_single_pixel(image)
    x_dates, series = ts_pre_proc.single_time_series(data, slots, pixel[0], pixel[1], 0)

    df = pd.DataFrame()
    df["date"] = pd.to_datetime(x_dates)
    outlier_removed = ts_pre_proc.remove_outliers(series, params)
    cols = [df, outlier_removed]
    df = pd.concat(cols, axis=1)
    df.loc[(df["date"].dt.month < 5) | (df["date"].dt.month >= 9), 0] = np.NaN

    series = ts_pre_proc.group_by_year(df[0], df["date"])

    y = lowess(series[2].values, series.index, xvals=np.unique(series.index), frac=0.3, is_sorted=True, return_sorted=False)
    quantiles = []
    for y in np.unique(df["date"].dt.year):
        qt = series[series[1] == y][2].quantile(q=0.8)
        quantiles.append(qt)

    axs[1].plot(np.unique(df["date"].dt.year), quantiles, color="r")
    axs[1].scatter(series[1], series[2])

    df.columns = ["date", 0]

    df["year"] = df["date"].dt.year

    sns.boxplot(data=df, x="year", y=0, ax=axs[2])
    axs[0].scatter(df["date"], df[0])
    plt.show()


# PARAMETRI PRINCIPALI
params = {
    "frac": 0.06, # Utilizzato dal lowess
    "n_splines": 72, # Utilizzato dalla GAM
    "alpha": 1, # Utilizzato nella rimozione degli outlier
    "lam": 0.1, # Utilizzato dalla GAM
    "remove_outliers": True, # Effettuare o meno la rimozione degli outlier pre-smoothing
    "frac_outliers": 0.06, # Utilizzato dal lowess della rimozione degli outlier
    "method": "gam", # Metodo di smoothing da utilizzare
    "plot": True, # Attivare alcuni plot di grafici
}

main_full()



'''image = Image.open("images_faggeta/faggeta_1_2022-10-09_2022-10-16.tiff")
plt.imshow(np.asarray(image), vmin=0, vmax=1)
plt.colorbar()
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plt.show()'''

#utils.save_images(data, area, slots, "images_faggeta")
#utils.show_images(data, slots, "2020-04-01",  size)
