# Analisi Time Series NDVI
Questo repository contiene il codice sviluppato relativa alla tesi *Progettazione e Sviluppo di algoritmi basati su deep learning per l'analisi di traiettorie temporali spettrali tele-rilevate della vegetazione.*
![Change Detection](readme_imgs/change_detection.png?raw=true)
![Time Series Forecasting](readme_imgs/results1.png?raw=true)
![Valutazione del ripristino](readme_imgs/results2.gif?raw=true)

## Struttura
Il codice è suddiviso in alcuni script da eseguire localmente relativi all'estrazione delle time series dalle immagini Sentinel-2, Landsat8 e Planet; e da un notebook Jupyter (*Change_Detection_Forecasting_Restore_Point.ipynb*) per la fase di change detection, forecasting e valutazione dello stato di ripristino delle aree di studio.

### Estrazione Time Series
Installare i package in *requirements.txt* .
Lo script per le immagini Sentinel-2 e Landsat8 è *main.py*.
Prima di tutto impostare le credenziali API per SentinelHub

![](readme_imgs/config.png?raw=true)

Successivamente creare il geoJson del poligono che racchiude l'area di studio tramite QGIS. Inserire il geoJson nella cartella *geoms* (se assente, crearla nella stessa root dello script).
A questo punto inseririe i vari parametri quali nome del file geoJson, risoluzione in metri delle immagini NDVI e a colori, periodo temporale delle immagini, e i parametri della richiesta API. In particolare l'*evalscript* che specifica le bande da acquisire (vedere il file *evalscript.py* per quelli utilizzati), e *data_collection* che specifica la collezione di dati da cui scaricare le immagini (Sentinel-2 L2A, Landsat8, ecc...)

![](readme_imgs/area_settings.png?raw=true)

A questo punto, in fondo allo script è possibile impostare i parametri di smoothing e lanciare una delle diverse funzioni main disponibili. Per l'estrazione delle time series di tutti i pixel dell'area di studio si utilizza *main_full()*. 

![](readme_imgs/run_sentinel.png?raw=true)

L'esecuzione di *main_full()* genererà nella cartella *ts/* un csv con la nomenclatura *"nomeArea_h_w.csv"* dove *h* e *w* sono rispettivamente l'altezza e la larghezza delle immagini recuperate. Nella cartella *areas_img* invece verrà salvata l'immagine a colori di riferimento *"nomeArea.png"*.
Questi file vanno utilizzati successivamente nel notebook Jupyter.

Per le immagini Planet utilizzare invece lo script *planet.py*. Il procedimento è simile, tuttavia le immagini Planet vanno al momento scaricate manualmente dal portale web. Nello script va infine specificato il path della cartella contenente le immagini. Come nel caso di SentinelHub, vanno impostati i parametri di smoothing ed eseguita una delle funzioni *main* definite nello script.

### Change Detection, Forecasting, Stato di ripristino
Per queste fasi utilizzare il notebook Jupyter *Change_Detection_Forecasting_Restore_Point.ipynb* fornito nel repository. Per l'esecuzione seguire le istruzioni al suo interno.

