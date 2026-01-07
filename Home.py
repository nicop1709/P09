from datetime import timedelta
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from pandas import to_datetime
import numpy as np
from backtest import Backtest

st.set_page_config(page_title="Projet 9 - RandomForest vs TabNet et TTM signaux trading", page_icon=":moneybag:")

st.title("Projet 9 - RandomForest vs TabNet et TTM signaux trading")

st.write("""Ce projet compare les performances de deux modèles de classification des signaux de trading pour Bitcoin (BTC) : RandomForest et TabNet.
RandomForest est un modèle de classification qui utilise un arbre de décision pour prédire les signaux de trading.
TabNet est un modèle de classification qui utilise un réseau de neurones pour prédire les signaux de trading.
TTM est un modèle de classification qui utilise un modèle de prédiction pour prédire les signaux de trading.
Les modèles sont entraînés sur un dataset de crypto currency prices et utilisent un modèle de classification pour prédire les signaux de trading.
Les modèles sont évalués sur un dataset de crypto currency prices et utilisent un modèle de classification pour prédire les signaux de trading.
""")

st.subheader("Pages")
st.page_link("pages/01_EDA.py", label="EDA")
st.page_link("pages/02_Predictions.py", label="Predictions")
# st.page_link("pages/03_Backtest.py", label="Backtest")
# st.page_link("pages/04_Conclusion.py", label="Conclusion")