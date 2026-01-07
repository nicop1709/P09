from datetime import timedelta
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from pandas import to_datetime
import numpy as np
from backtest import Backtest
from datetime import datetime
from utils import fetch_ohlcv_binance, prepare_data_advanced_features, plot_predictions
from sklearn.preprocessing import StandardScaler
import joblib

scaler = joblib.load("tabnet_scaler.pkl")
tabnet_model = joblib.load("tabnet_model.pkl")

st.set_page_config(page_icon=":moneybag:")

st.title("Predictions")

st.write("""Ce notebook prédit les signaux de trading pour la paire BTC-USDC 1h de 2015 à 2025.
Il utilise les modèles RandomForest et TabNet pour prédire les signaux de trading.
Il affiche les statistiques des performances des modèles.
Il affiche le backtest des modèles.
Il affiche la liste des trades des modèles.
""")


st.header("Predictions")

st.write(f"Nous sommes le {format(datetime.now(), '%d/%m/%Y')},  il est {format(datetime.now(), '%H:%M:%S')}")
progress_bar = st.progress(0.0)
progress_bar.progress(0.1, text="Récupération des données")
last_data = fetch_ohlcv_binance("BTCUSDC", "1h", format(datetime.now()-timedelta(days=7),"%Y-%m-%d"), format(datetime.now(),"%Y-%m-%d"))
last_data_model, features_cols_model = prepare_data_advanced_features(last_data,24,0.002)
progress_bar.progress(0.7, text="Préparation des données")
last_data_model_s  = scaler.transform(last_data_model[features_cols_model].values)
progress_bar.progress(0.8, text="Prédiction TabNet")
tabnet_pred_tab = tabnet_model.predict_proba(last_data_model_s)[:, 1]
progress_bar.progress(1.0, text="Fin de la prédiction")
tabnet_pred = (tabnet_pred_tab >= 0.5).astype(int)
st.info(f"Le cours du Bitcoin est actuellement de ***{last_data_model['Close'].iloc[-1]} USDC***")
if (to_datetime(datetime.utcnow()-timedelta(minutes=5),utc=True)  < last_data_model['Timestamp'].iloc[-1]) :
    st.success(f"Le signal TabNet pour les 5 premières minutes de l'heure est : ***{'Buy' if tabnet_pred[-1] == 1 else 'No-trade or Sell'}***")
    fig = plot_predictions(last_data, tabnet_pred[-1], plot=False)
    st.plotly_chart(fig)
else:
    st.warning(f"Le signal TabNet n'est pas disponible pour trader, il faut attendre **{round((last_data_model['Timestamp'].iloc[-1]+timedelta(minutes=60)-to_datetime(datetime.utcnow(),utc=True)).total_seconds()/60)} minutes**")
    st.warning(f"Si la tendance ne change pas le signal est actuellement : {'***Buy***' if tabnet_pred[-1] == 1 else 'No-trade or Sell'}")
    fig = plot_predictions(last_data, tabnet_pred[-1], plot=False)
    st.plotly_chart(fig)