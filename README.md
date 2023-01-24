# Analisi Time Series NDVI
Questo repository contiene il codice sviluppato relativa alla tesi *Progettazione e Sviluppo di algoritmi basati su deep learning per l'analisi di traiettorie temporali spettrali tele-rilevate della vegetazione.*

## Struttura
Il codice è suddiviso in alcuni script da eseguire localmente relativi all'estrazione delle time series dalle immagini Sentinel-2, Landsat8 e Planet; e da un notebook Jupiter per la fase di change detection, forecasting e valutazione dello stato di ripristino delle aree di studio.

### Estrazione Time Series
Installare i package in *requirements.txt* .
Lo script per le immagini Sentinel-2 e Landsat8 è *main.py*.
Prima di tutto impostare le credenziali API per SentinelHub

![](readme_imgs/config.png?raw=true)




