import pandas as pd
import numpy as np

class Trader:
    def __init__(self, row: pd.Series, idx: int, idx_entry: int, signal: np.ndarray, capital: float, portfolio: float, position: float, qty: float, entry_price: float, exit_price: float, fee_roundtrip=0.002, pct_capital=1, debug=False, trade_list=[], horizon_steps=24,capital_before_buy=0,allow_short=False):
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
        self.allow_short = allow_short

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
