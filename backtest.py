from trader import Trader
import pandas as pd
import numpy as np

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
        last_pos_idx = None
        last_df_idx = None
        print(self.df_bt.head())
        for pos_idx, (i, row) in enumerate(self.df_bt.iterrows()):
            trader.row = row
            trader.idx = i
            # Use positional index for signal to ensure alignment with DataFrame rows
            if isinstance(self.signal, pd.Series):
                trader.signal = self.signal.iloc[pos_idx] if pos_idx < len(self.signal) else False
            else:
                trader.signal = self.signal[pos_idx] if pos_idx < len(self.signal) else False
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
            last_pos_idx = pos_idx
            last_df_idx = i

        # Clôture forcée si position ouverte en fin de backtest
        if last_pos_idx is not None and last_pos_idx < len(self.df_bt):
            last_row = self.df_bt.iloc[last_pos_idx]
            if self.position > 0:
                sell_value = self.qty * last_row["Close"]
                sell_fees = self.fee_roundtrip * sell_value / 2
                PnL = self.qty * (last_row["Close"] - self.entry_price)
                PnL_net = PnL - sell_fees
                self.capital += sell_value - sell_fees
                self.portfolio = 0
                self.position = 0
                self.trade_list.append({
                    "idx": last_df_idx,
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
                    "position_type": "long",
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
        self.ROI_day_pct = ((1 + self.ROI_pct / 100) ** (1  / days) - 1) * 100
        
        # Calculate annualized ROI: convert ROI_pct from percentage to decimal first
        roi_decimal = self.ROI_pct / 100
        if roi_decimal <= -1:
            # If we lost more than 100%, return -100%
            self.ROI_annualized_pct = -100.0
        else:
            self.ROI_annualized_pct = ((1 + roi_decimal) ** (365.0 / days) - 1) * 100
        self.df_trades = pd.DataFrame(self.trade_list)
        if len(self.df_trades) > 0 and "PnL" in self.df_trades.columns:
            self.win_rates = self.df_trades["PnL"].apply(lambda x: x > 0).mean()*100
            self.max_drawdown_pct = self.df_trades["MaxDrawDown"].max()
        else:
            self.win_rates = 0.0
            self.max_drawdown_pct = 0.0
        self.nb_trades = len(self.df_trades)
        self.nb_trades_by_day = self.nb_trades / days if days > 0 else 0
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