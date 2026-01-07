from datetime import timedelta
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from pandas import to_datetime
import numpy as np
from backtest import Backtest
from utils import plot_backtest

st.set_page_config(page_icon=":moneybag:")

st.title("Exploratory Data Analysis")

st.write("""Ce notebook explore les données historiques de la paire BTC-USDC 1h de 2015 à 2025.
Il calcule les features et la target pour la prédiction du ROI futur.
Il affiche les statistiques des features et de la target.
Il affiche l'histogramme de la distribution des variations de prix Close.
Il affiche le backtest parfait avec réinjection de capital.
Il affiche la liste des trades du backtest parfait.
""")


st.header("EDA Dataset historique BTC-USDC 1h")

st.subheader("Données")
st.write("Données historiques BTC-USDC 1h de 2015 à 2025")
df = pd.read_csv("btc_usdc_1h_2015_2025.csv")


# Definir Sidebar
nb_days_displayed = st.sidebar.slider("Nombre de jours affichés", min_value=1, max_value=100, value=10)
horizon_steps = st.sidebar.slider("Nombre de bougies 1h pour la prédiction du ROI futur", min_value=1, max_value=24*5, value=24)
fee_roundtrip = st.sidebar.radio("Fee roundtrip", options=[0.0000, 0.0015, 0.0020], index=2)

# Affihcer les données brutes
df_display = df.tail(nb_days_displayed*24).copy() #24 pour timeframe 1h
df_display["Timestamp"] = pd.to_datetime(df_display["Timestamp"])
st.dataframe(df_display)


# Afficher les données brutes en candlestick chart et volume
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.1,
    subplot_titles=('Candlestick Chart', 'Volume'),
    row_heights=[0.6, 0.4]
)
fig.add_trace(
    go.Candlestick(x=df_display['Timestamp'],
                open=df_display['Open'],
                high=df_display['High'],
                low=df_display['Low'],
                close=df_display['Close']),
    row=1, col=1
)
fig.add_trace(  
    go.Bar(x=df_display['Timestamp'],
           y=df_display['Volume']),
    row=2, col=1
)
fig.update_layout(
    title=f"Candlestick Chart et Volume de {format(to_datetime(df_display['Timestamp'].iloc[0]),'%d %B %Y')} à {format(to_datetime(df_display['Timestamp'].iloc[-1]),'%d %B %Y')}",
    #xaxis=dict(title="Timestamp"),
    yaxis=dict(title="Price"),
    yaxis2=dict(title="Volume"),
)
# Ajouter le range slider en bas du graphique
fig.update_xaxes(rangeslider_visible=True, row=1, col=1,rangeslider_thickness=0.1)
st.plotly_chart(fig)

# Calculer les features et la target : 
df_display["Close_pct_change%"] = df_display["Close"].pct_change()*100
df_display["High_pct_change%"] = df_display["High"].pct_change()*100
df_display["Low_pct_change%"] = df_display["Low"].pct_change()*100
df_display["Volume_pct_change%"] = df_display["Volume"].pct_change()*100
df_display["Close_pct_change_last24%"] = df_display["Close"].pct_change(24)*100
df_display["target"] = ((df_display["Close"].shift(-horizon_steps)-df_display["Close"])/df_display["Close"])*100
features_cols = ["Close_pct_change%", "High_pct_change%", "Low_pct_change%", "Volume_pct_change%", "Close_pct_change_last24%"]
target_col = "target"
st.write("Target : Variation de prix Close (%) pour l'horizon de " + str(horizon_steps) + " bougies 1h")
st.dataframe(df_display[["Timestamp", "Close_pct_change%", "High_pct_change%", "Low_pct_change%", "Volume_pct_change%", "Close_pct_change_last24%", target_col]])

# AFficher les statistiques sur l'ensemble des données de 2015 à 2025
st.subheader("Statistiques des features et de la target")
st.dataframe(df_display[features_cols + [target_col]].describe())

# Définir le seuil en % => 0.002 => 0.2%    
seuil = 0.2
# Générer histogramme et barres de couleurs différentes
hist, bin_edges = np.histogram(df_display["target"].dropna(), bins=10*horizon_steps)

bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

colors = ['green' if c > seuil else 'red' for c in bin_centers]

fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=bin_centers,
        y=hist,
        marker_color=colors,
        name="target",
        width=(bin_edges[1] - bin_edges[0])
    )
)
fig.add_vline(
    x=seuil,
    line_dash="dash",
    line_color="blue",
    annotation_text=f"Seuil: {seuil}%",
    annotation_position="top right"
)

fig.update_layout(title="Distribution des variations de prix Close", xaxis_title="Variation de prix Close (%)", yaxis_title="Nombre de bougies 1h")
st.plotly_chart(fig)


st.latex(r"""
        ROI_{annualisé} = (1 + ROI)^{365/nb\_jours} - 1
        """)
col1, col2 = st.columns(2)
with col1:
    st.subheader("SANS réinjection de capital")
    st.latex(r"""
            ROI = \sum_{i=1}^{nb\_trades} ROI_i
            """)

with col2:
    st.subheader("AVEC réinjection de capital")
    st.latex(r"""
            ROI = \prod_{i=1}^{nb\_trades} (1 + ROI_i) - 1
            """)            

st.info(f"Choix d'une stratégie AVEC réinjection de capital :", icon="⚠️")

st.info(f"""
Backtest avec réinjection de capital :
* ACHAT **si** target > seuil, 
* VENTE **si** target < seuil **et si** au moins {horizon_steps} bougies 1h ont passé depuis l'achat

            """, icon="💡")
backtest_prefect = Backtest(df_display, pd.Series(df_display[target_col]>seuil), fee_roundtrip=fee_roundtrip)

st.subheader("Statistiques du backtest parfait")

st.metric("**ROI annualized** : ", value=f"{backtest_prefect.ROI_annualized_pct:.2f}%")
st.metric(f"**Max DrawDown** : ", value=f"{backtest_prefect.max_drawdown_pct:.2f}%")

stats_dict = {
    "Days": backtest_prefect.days,
    "Capital Final": f"{backtest_prefect.capital:.0f}",
    "PnL": f"{backtest_prefect.capital - backtest_prefect.capital_init:.0f}",
    "ROI": f"{backtest_prefect.ROI_pct:.2f}%",
    "ROI day": f"{backtest_prefect.ROI_day_pct:.2f}%",
    "Win rate": f"{backtest_prefect.win_rates:.0f}%",
    "Nb trades": f"{backtest_prefect.nb_trades:.0f}",
    "Nb trades par jour": f"{backtest_prefect.nb_trades_by_day:.2f}",
}
stats_df = pd.DataFrame(stats_dict.items(), columns=["Statistique", "Valeur"])
st.dataframe(stats_df)

st.subheader("Liste des trades du backtest parfait")
st.dataframe(pd.DataFrame(backtest_prefect.trade_list))

fig = plot_backtest(backtest_prefect, plot=False)
st.plotly_chart(fig)