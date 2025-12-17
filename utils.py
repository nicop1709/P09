
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta

def plot_backtest(backtester):
    # On suppose que trades_df == backtester.df_trades déjà généré avec l'algo ci-dessus
    trades_df = backtester.df_trades

    # Pour le graphique, récupérer le temps et close price
    df_curves = backtester.df_bt.reset_index(drop=True)
    df_curves["Timestamp_entry"] = df_curves["Timestamp"]
    df_curves = pd.merge(df_curves, trades_df[["Timestamp", "exit_price","Capital"]], on="Timestamp", how="left")
    df_curves = pd.merge(df_curves, trades_df[["Timestamp_entry", "entry_price"]], on="Timestamp_entry", how="left")
    df_curves["Capital"] = df_curves["Capital"].fillna(method="ffill").fillna(backtester.capital_init)

    timestamps = df_curves["Timestamp"]
    close_prices = df_curves["Close"]
    capital_curve = df_curves["Capital"]
    buy_time = df_curves["Timestamp_entry"]
    buy_price = df_curves["entry_price"]
    sell_time = df_curves["Timestamp"]
    sell_price = df_curves["exit_price"]

    # Créer un subplot avec 2 graphiques (prix en haut, capital en bas)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Cours Close avec signaux Buy/Sell', 'Évolution du Capital'),
        row_heights=[0.6, 0.4]
    )

    # Graphique 1 : Prix avec signaux
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=close_prices,
            mode='lines',
            name='Close',
            line=dict(color='blue')
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=buy_time,
            y=buy_price,
            mode='markers',
            marker=dict(color='green', symbol='triangle-up', size=10),
            name='Buy'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=sell_time,
            y=sell_price,
            mode='markers',
            marker=dict(color='red', symbol='triangle-down', size=10),
            name='Sell'
        ),
        row=1, col=1
    )

    # Graphique 2 : Capital
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=capital_curve,
            mode='lines',
            name='Capital',
            line=dict(color='purple', width=2)
        ),
        row=2, col=1
    )

    # Ligne de référence pour le capital initial
    fig.add_hline(
        y=backtester.capital_init,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Capital initial: {backtester.capital_init:.2f}",
        row=2, col=1
    )

    fig.update_layout(
        title='Cours Close avec signaux Buy/Sell et Évolution du Capital',
        height=800,
        width=1200,
        showlegend=True,
        legend=dict(x=0, y=1)
    )

    fig.update_xaxes(title_text="Timestamp", row=2, col=1)
    fig.update_yaxes(title_text="Prix", row=1, col=1)
    fig.update_yaxes(title_text="Capital", row=2, col=1)

    fig.show()

class Trader:
    def __init__(self, row: pd.Series, idx: int, idx_entry: int, signal: np.ndarray, capital: float, portfolio: float, position: float, qty: float, entry_price: float, exit_price: float, fee_roundtrip=0.002, pct_capital=1, debug=False, trade_list=[], horizon_steps=24,capital_before_buy=0):
        self.row = row
        self.idx = idx
        self.signal = signal
        self.fee_roundtrip = fee_roundtrip
        self.pct_capital = pct_capital
        self.capital = capital    
        self.portfolio = portfolio
        self.position = position
        self.qty = qty
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.debug = debug
        self.idx_entry = idx_entry
        self.trade_list = trade_list
        self.timestamp_entry = None
        self.max_drawdown_pct = 0
        self.horizon_steps = horizon_steps
        self.capital_before_buy = capital_before_buy

    def _buy(self):
        self.qty = self.pct_capital * self.capital / self.row["Close"]
        position_value = self.qty * self.row["Close"]
        self.position = position_value  # Montant investi dans la position
        self.entry_price = self.row["Close"]
        buy_fees = self.fee_roundtrip * position_value / 2
        self.capital_before_buy = self.capital
        self.capital -= (position_value + buy_fees)
        self.portfolio = position_value  # Portfolio = valeur de la position
        self.idx_entry = self.idx
        self.timestamp_entry = self.row["Timestamp"]
        if self.debug:
            print(f"Idx: {self.idx} / Buy: {self.qty:.8f} @ {self.entry_price:.2f}")
        return True

    def _sell(self):
        sell_value = self.qty * self.row["Close"]
        sell_fees = self.fee_roundtrip * sell_value / 2
        PnL = self.qty * (self.row["Close"] - self.entry_price)
        PnL_net = PnL - sell_fees
        self.capital += sell_value - sell_fees
        self.position = 0  # Plus de position ouverte
        self.exit_price = self.row["Close"]
        self.portfolio = 0  # Portfolio vide après vente
        self.max_drawdown_pct = (PnL_net/self.capital_before_buy)*100 if PnL_net<0 else 0
        self.trade_list.append({
            "idx": self.idx,
            "idx_entry": self.idx_entry,
            "Timestamp": self.row["Timestamp"],
            "Timestamp_entry": self.timestamp_entry,
            "qty": self.qty,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "PnL": PnL,
            "PnL_net": PnL_net,
            "Capital": self.capital,
            "MaxDrawDown": self.max_drawdown_pct,
        })

        if self.debug:
            print(f"Idx: {self.idx} / Sell: {self.qty:.8f} @ {self.exit_price:.2f}")
            print(f"PnL: {PnL:.2f}")
            print(f"PnL net (après frais): {PnL_net:.2f}")
            print(f"Portfolio: {self.portfolio:.2f}")
            print(f"Capital: {self.capital:.2f}")
        return True

    def run(self):
        # Conversion du signal en int (gère les cas numpy array et scalar)
        sig = int(self.signal) if isinstance(self.signal, (np.ndarray, np.generic)) else int(self.signal)
        
        # Mise à jour du portfolio si position ouverte (valeur actuelle de la position)
        if self.position > 0:
            self.portfolio = self.qty * self.row["Close"]
        
        if self.debug:
            print(f"Idx: {self.idx} / Signal: {sig} / Position: {self.position:.2f} / Portfolio: {self.portfolio:.2f}")
        
        # Achat : signal=1 et pas de position ouverte
        if sig == 1 and self.position == 0:
            self._buy()
        # Vente : signal=0 et position ouverte (on vend dès que le signal passe à 0)
        elif sig == 0 and self.position > 0 and self.idx >= self.idx_entry + self.horizon_steps:
            self._sell()
        
        return self.portfolio, self.capital, self.position, self.qty, self.entry_price, self.exit_price, self.trade_list     


class Backtest:
    def __init__(self, df_bt: pd.DataFrame, signal: np.ndarray, fee_roundtrip=0.002, pct_capital=1, capital_init=1000, debug=False, horizon_steps=24):
        self.df_bt = df_bt
        self.signal = signal
        self.fee_roundtrip = fee_roundtrip
        self.pct_capital = pct_capital
        self.capital_init = capital_init  # Sauvegarder le capital initial
        self.capital = capital_init
        self.position = 0
        self.qty = 0
        self.entry_price = 0
        self.exit_price = 0
        self.portfolio = 0
        self.debug = debug
        self.idx_entry = 0
        self.trade_list = []
        self.max_drawdown_pct = 0
        self.horizon_steps = horizon_steps
        self.run()
        self.print_stats()

    def run(self):
        trader = Trader([], 0, 0, 0, self.capital, self.portfolio, self.position, self.qty, self.entry_price, self.exit_price, self.fee_roundtrip, self.pct_capital, debug=self.debug, trade_list=self.trade_list, horizon_steps=self.horizon_steps, capital_before_buy=self.capital_init)
        last_idx = None
        for i, row in self.df_bt.iterrows():
            trader.row = row
            trader.idx = i
            trader.signal = self.signal[i]
            trader.run()
            self.portfolio = trader.portfolio
            self.capital = trader.capital   
            self.position = trader.position
            self.qty = trader.qty
            self.entry_price = trader.entry_price
            self.exit_price = trader.exit_price
            self.idx_entry = trader.idx_entry
            self.timestamp_entry = trader.timestamp_entry
            self.capital_before_buy = trader.capital_before_buy
            last_idx = i

        # Clôture forcée si position ouverte en fin de backtest
        if self.position > 0 and last_idx is not None:
            last_row = self.df_bt.iloc[last_idx]
            sell_value = self.qty * last_row["Close"]
            sell_fees = self.fee_roundtrip * sell_value / 2
            PnL = self.qty * (last_row["Close"] - self.entry_price)
            PnL_net = PnL - sell_fees
            self.capital += sell_value - sell_fees
            self.portfolio = 0
            self.position = 0
            self.trade_list.append({
                "idx": last_idx,
                "idx_entry": self.idx_entry,
                "Timestamp": last_row["Timestamp"],
                "Timestamp_entry": self.timestamp_entry,
                "qty": self.qty,
                "entry_price": self.entry_price,
                "exit_price": last_row["Close"],
                "PnL": PnL,
                "PnL_net": PnL_net,
                "Capital": self.capital,
                "MaxDrawDown": self.max_drawdown_pct,
            })
            self.qty = 0
            self.entry_price = 0
            self.exit_price = 0
        
        days = (self.df_bt.iloc[-1]["Timestamp"] - self.df_bt.iloc[0]["Timestamp"]).days
        if days <= 0:
            days = 1  # Avoid division by zero
        self.days = days
        self.PnL = self.capital - self.capital_init
        self.ROI_pct = self.PnL / self.capital_init *100
        self.ROI_day_pct = self.PnL / self.capital_init / days * 100
        # Calculate annualized ROI: convert ROI_pct from percentage to decimal first
        roi_decimal = self.ROI_pct / 100
        if roi_decimal <= -1:
            # If we lost more than 100%, return -100%
            self.ROI_annualized_pct = -100.0
        else:
            self.ROI_annualized_pct = ((1 + roi_decimal) ** (365.0 / days) - 1) * 100
        self.df_trades = pd.DataFrame(self.trade_list)
        self.win_rates = self.df_trades["PnL"].apply(lambda x: x > 0).mean()*100
        self.nb_trades = len(self.df_trades)
        self.nb_trades_by_day = self.nb_trades / days
        self.max_drawdown_pct = self.df_trades["MaxDrawDown"].max()
        return self.portfolio, self.capital, self.position, self.qty, self.entry_price, self.exit_price, self.trade_list
    
    def print_stats(self):
        print(f"Days: {self.days}")
        print(f"Portfolio: {self.portfolio}")
        print(f"Capital: {self.capital}")
        print(f"PnL: {self.capital - self.capital_init}")
        print(f"Position: {self.position}")
        print(f"ROI: {self.ROI_pct:.2f}%")
        print(f"ROI annualized: {self.ROI_annualized_pct:.2f}%")
        print(f"ROI day: {self.ROI_day_pct:.2f}%")
        print(f"Win rate: {self.win_rates:.2f}%")
        print(f"Nb trades: {self.nb_trades}")
        print(f"Nb trades par jour: {self.nb_trades_by_day:.2f}")
        print(f"Max DrawDown: {self.max_drawdown_pct:.2f}%")
        return self.df_trades


def clean_data(df):
    df = df.copy()
    df = df.dropna().reset_index(drop=True)
    return df

def calculate_features_pct_change(df):
    df = df.copy()
    df["Close_pct_change"] = df["Close"].pct_change()
    df["High_pct_change"] = df["High"].pct_change()
    df["Low_pct_change"] = df["Low"].pct_change()          
    df["Volume_pct_change"] = df["Volume"].pct_change()
    features_cols = ["Close_pct_change", "High_pct_change", "Low_pct_change", "Volume_pct_change"]
    return df, features_cols

def calculate_features_technical(df):
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low  = df["Low"]
    vol  = df["Volume"]
    open = df["Open"]

    df["logret_1"] = np.log(close).diff()
    df["logret_5"] = np.log(close).diff(5)
    df["logret_20"] = np.log(close).diff(20)

    df["vol_20"] = df["logret_1"].rolling(20).std()
    df["vol_50"] = df["logret_1"].rolling(50).std()

    df["ma20"] = close.rolling(20).mean()
    df["ma50"] = close.rolling(50).mean()
    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()

    df["ma_diff"] = df["ma20"] - df["ma50"]
    df["ema_diff"] = df["ema20"] - df["ema50"]

    df["rsi14"] = ta.momentum.rsi(close, window=14)
    macd = ta.trend.MACD(close)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    df["atr14"] = ta.volatility.average_true_range(high, low, close, window=14) / close
    df["adx14"] = ta.trend.adx(high, low, close, window=14)

    df["hl_range"] = (high - low) / close
    df["np_range"] = (high - low) / (close - open+1e-6)
    features_cols = []

    # On retire les lignes avec NaNs (features + futur)
    features_cols = ["logret_1", "logret_5", "logret_20", "vol_20", 
    "vol_50", "ma20", "ma50", "ema20", "ema50", "ma_diff", "ema_diff", 
    "rsi14", "macd", "macd_signal", "atr14", "adx14", "hl_range", "np_range"]
    
    return df, features_cols

def calculate_label(df, horizon_steps, threshold):
    df = df.copy()
    # --- Label : ROI futur à horizon_steps ---
    df["future_close"] = df["Close"].shift(-horizon_steps)
    df["roi_H"] = (df["future_close"] - df["Close"]) / df["Close"]
    df["y"] = (df["roi_H"] > threshold).astype(int)
    return df

def prepare_data_min_features(df):
    # Nettoyage basique
    df = clean_data(df)
    df, features_cols = calculate_features_pct_change(df)
    df = calculate_label(df)
    df_model = df.dropna(subset=features_cols + ["y"]).reset_index(drop=True)
    df_model["Volume_pct_change"] = df_model["Volume_pct_change"].replace([np.inf, -np.inf], 0)

    return df_model, features_cols


def prepare_data_advanced_features(df):
    # Nettoyage basique
    df = clean_data(df)
    df,features_cols = calculate_features_technical(df)
    df = calculate_label(df)
    df_model = df.dropna(subset=features_cols + ["y"]).reset_index(drop=True)
    return df_model, features_cols
    