import datetime
import pandas as pd
import os
import json
import cv2
from PIL import Image

import matplotlib.pyplot as plt
from sentinelhub import (
    DataCollection,
    MimeType,
    MosaickingOrder,
    SentinelHubRequest,
    bbox_to_dimensions,
    Geometry
)

"""
Questo script contiene diverse funzioni di utilità per l'estrazione e la visualizzazione delle time series
"""

def intervals_by_chunks(start, end, n_chunks):
    """
    Crea n_chunks intervalli compresi tra le date di start ed end
    Esempio: start = 2017-01-01;  end = 2017-01-03;  n_chunks = 3
    [("2017-01-01, "2017-01-02"),  ("2017-01-02, "2017-01-03")]
    :param start: data di inizio
    :param end: data di fine
    :param n_chunks: numero di intervalli
    :return: lista  di lunghezza n_chunks-1 di tuple (ds, de)
    """
    start = datetime.datetime.strptime(start)
    end = datetime.datetime.strptime(end)
    tdelta = (end - start) / n_chunks
    edges = [(start + i * tdelta).date().isoformat() for i in range(n_chunks)]
    slots = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

    print("Monthly time windows:\n")
    for slot in slots:
        print(slot)

    return slots


def fixed_weeks_per_year(start_year, n_years):
    """
    Crea una lista di tuple (ds, de) di date a frequenza settimanale per il numero di anni specificato.
    Il numero di settimane e le date specifiche sono uguali per ogni anno.
    :param start_year: Anno di inizio
    :param n_years: numero di anni
    :return: lista di lunghezza n_years * 52 di tuple (ds, de)
    """
    slots = []
    for y in range(start_year, start_year+n_years):
        start = str(y)+"-01-01"
        end = str(y)+"-12-24"
        dates = pd.date_range(start=start, end=end, freq="7D")
        for d in dates:
            date2 = d + datetime.timedelta(days=7)
            date = d.strftime("%Y-%m-%d")
            date2 = date2.strftime("%Y-%m-%d")
            slots.append((date, date2))
    return slots


def dates_list(start, end, freq="D", interval_size=0):
    """
    Crea una lista di tuple (ds, de) di date a frequenza e lunghezza di intervallo variabile.
    :param start: data di inizio
    :param end: data di fine
    :param freq: frequenza degli intervalli (intervallo tra il primo elemento di una tupla e di quella successiva)
    :param interval_size: lunghezza dell'intervallo (intervallo tra gli elementi di una stessa tupla)
    :return: lista di tuple (ds, de)
    """
    slots = []
    dates = pd.date_range(start=start, end=end, freq=freq)
    for d in dates:
        date2 = d + datetime.timedelta(days=interval_size)
        date = d.strftime("%Y-%m-%d")
        date2 = date2.strftime("%Y-%m-%d")
        slots.append((date, date2))
    return slots


def print_file_names(folder_name):
    """
    Stampa i nomi dei file in una cartella
    :param folder_name: path della cartella
    """
    for folder, _, filenames in os.walk(folder_name):
        for filename in filenames:
            print(os.path.join(folder, filename))


def get_request(config, evalscript, time_interval, geom=None, image_size=None, data_folder=None, mimetype=MimeType.TIFF, data_coll=DataCollection.SENTINEL2_L2A):
    """
    Crea un oggetto di richiesta API di SentinelHub
    :param config: configurazioni API di SentinelHub
    :param evalscript: script di acquisizione delle bande
    :param time_interval: tupla (start, end) dell'intervallo temporale da cui estrarre l'immagine
    :param geom: oggetto della geometria geojson
    :param image_size: dimensioni dell'immagine (si ottiene con load_geometry)
    :param data_folder: cartella della cache delle immagini
    :param mimetype: formato di salvataggio delle immagini
    :param data_coll: collezioni di dati da cui cercare le immagini
    :return:
    """
    return SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=data_coll,
                time_interval=time_interval,
                mosaicking_order=MosaickingOrder.LEAST_CC, # Immagine con minor copertura nuvolosa nell'intervallo scelto
            )
        ],
        responses=[SentinelHubRequest.output_response("default", mimetype)],
        bbox=geom.bbox,
        size=image_size,
        # geometry=geom,
        config=config,
        data_folder= data_folder
    )


def load_geometry(file, resolution):
    """
    Carica un geojson e crea l'oggetto geometry e le dimensioni delle immagini da passare alla richiesta API alla risoluzione specificata
    :param file: path del file geojson
    :param resolution: risoluzione in metri delle immagini da ottenere
    :return: (geometry object, image size)
    """
    imported = json.load(open(file))
    geom = Geometry.from_geojson(imported["features"][0]["geometry"])
    image_size = bbox_to_dimensions(geom.bbox, resolution=resolution)
    print(f"Image shape at {resolution} m resolution: {image_size} pixels")
    return geom, image_size


def save_images(data, area, slots, folder, channel=0):
    """
    Salva le immagini scaricate dall'API in una cartella specificata
    :param data: lista delle immagini ottenute dall'API
    :param area: nome dell'area a fini di salvataggio
    :param slots: lista di date utilizzata in fase di richiesta API
    :param folder: path della cartella
    :param channel: canale delle immagini (in formato tiff) da salvare
    """
    script_dir = os.path.dirname(__file__)
    folder = os.path.join(script_dir, folder+"/")
    if not os.path.isdir(folder):
        os.makedirs(folder)
    for idx, image in enumerate(data):
        if len(image.shape) > 2:
            image = Image.fromarray(image[:,:,channel])
        else:
            image = Image.fromarray(image)
        image.save(folder+"/"+area+"_"+slots[idx][0]+"_"+slots[idx][1]+".tiff")


def show_images(data, slots, start_date, image_size, ncols=4, nrows=3):
    """
    Mostra ncols*nrows immagini dalla lista ottenuta dalla richiesta API partendo dalla data specificata
    :param data: lista delle immagini ottenute dall'API
    :param slots: lista di date utilizzata in fase di richiesta API
    :param start_date: data di inizio da cui mostrare le immagini
    :param image_size: tupla delle dimensioni delle immagini
    :param ncols: numero di colonne del plot
    :param nrows: numero di righe del plot
    """
    aspect_ratio = image_size[0] / image_size[1]
    subplot_kw = {"xticks": [], "yticks": [], "frame_on": False}

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(5 * ncols * aspect_ratio, 5 * nrows),
                            subplot_kw=subplot_kw)
    i = 1
    for idx, image in enumerate(data):
        s = slots[idx][0]
        e = slots[idx][1]
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
    """
    Grafica le time series della lista fornita
    :param series_list: lista di time series (lista python, numpy array, o pandas series)
    :param color: colore del plot
    """
    for series in series_list:
        plt.plot(series, color=color, label=color)


def select_pixels(img, coord_file):
    """
    Tool di selezione e salvataggio dei pixel di una data immagine.
    Premere R per resettare il file delle cordinate, ESC per confermare le selezioni.
    Evitare di chiudere la finestra tramite il tasto di chiusura.
    :param img: l'immagine da cui selezionare i pixel
    :param coord_file: file in cui salvare le coordinate
    """

    comms = " R: pulisci file, ESC conferma "
    # mouse callback function
    def red_dot(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            with open(coord_file, "a") as f:
                f.write("{0}, {1}\n".format(x, y))
            cv2.circle(img, (x, y), 0, (0, 0, 255), -1)

    # interactive display
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    clone = img.copy()
    cv2.namedWindow('Pixel selector '+coord_file+comms, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Pixel selector '+coord_file+comms, red_dot)

    # event handler
    while (1):
        cv2.imshow('Pixel selector '+coord_file+comms, img)
        key = cv2.waitKey(1) & 0xFF
        # escape
        if key == 27 or key == ord('q'):
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
    Tool per selezionare un solo pixel di una data immagine.
    Premere R per annullare, ESC per confermare la selezione.
    Evitare di chiudere la finestra tramite il tasto di chiusura.
    :param img: l'immagine da cui selezionare il pixel
    :return: (x, y) coordinate del pixel
    """
    current_pixel = ()
    comms = " R: reset, ESC conferma "

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
    cv2.namedWindow('Pixel selector'+comms, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Pixel selector'+comms, red_dot)

    # event handler
    while (1):
        cv2.imshow('Pixel selector'+comms, img)
        key = cv2.waitKey(1) & 0xFF
        # escape
        if key == 27 or key == ord('q'):
            cv2.destroyAllWindows()
            return current_pixel
        # refresh dots
        if key == ord('r'):
            img = clone.copy()

    cv2.destroyAllWindows()
    return current_pixel




