from trader_v2 import Trader
import pandas as pd
import numpy as np

class Backtest:
    """
    Backtest pour Isolated Margin Binance avec support Long et Short.
    
    Signaux attendus:
        - signal = 1  : Ouvrir Long
        - signal = -1 : Ouvrir Short
        - signal = 0  : Fermer la position / Ne rien faire
    
    Paramètres:
        - leverage: Levier utilisé (ex: 10 pour 10x)
        - enable_liquidation: Active/désactive la liquidation automatique
    """
    
    def __init__(self, df_bt: pd.DataFrame, signal: np.ndarray, fee_roundtrip=0.002, 
                 pct_capital=1, capital_init=1000, debug=False, horizon_steps=24,
                 leverage=1, enable_liquidation=True):
        self.df_bt = df_bt
        self.signal = signal
        self.fee_roundtrip = fee_roundtrip
        self.pct_capital = pct_capital
        self.capital_init = capital_init
        self.capital = capital_init
        self.position = 0  # 1 = Long, -1 = Short, 0 = Pas de position
        self.qty = 0
        self.entry_price = 0
        self.exit_price = 0
        self.portfolio = 0
        self.debug = debug
        self.idx_entry = 0
        self.trade_list = []
        self.max_drawdown_pct = 0
        self.horizon_steps = horizon_steps
        self.leverage = leverage
        self.enable_liquidation = enable_liquidation
        self.margin = 0
        self.liquidation_price = 0
        self.run()
        self.print_stats()

    def run(self):
        trader = Trader(
            row=[], 
            idx=0, 
            idx_entry=0, 
            signal=0, 
            capital=self.capital, 
            portfolio=self.portfolio, 
            position=self.position, 
            qty=self.qty, 
            entry_price=self.entry_price, 
            exit_price=self.exit_price, 
            fee_roundtrip=self.fee_roundtrip, 
            pct_capital=self.pct_capital, 
            debug=self.debug, 
            trade_list=self.trade_list, 
            horizon_steps=self.horizon_steps, 
            capital_before_buy=self.capital_init,
            leverage=self.leverage,
            enable_liquidation=self.enable_liquidation
        )
        
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
            self.margin = trader.margin
            self.liquidation_price = trader.liquidation_price
            last_idx = i

        # Clôture forcée si position ouverte en fin de backtest
        if last_idx is not None and self.position != 0:
            last_row = self.df_bt.iloc[last_idx]
            current_price = last_row["Close"]
            
            if self.position == 1:  # Long
                # Calcul PnL Long
                position_value = self.qty * current_price
                close_fees = self.fee_roundtrip * position_value / 2
                PnL = self.qty * (current_price - self.entry_price)
                PnL_net = PnL - close_fees
                self.capital += self.margin + PnL - close_fees
                position_type = "long"
                
            elif self.position == -1:  # Short
                # Calcul PnL Short
                position_value = self.qty * current_price
                close_fees = self.fee_roundtrip * position_value / 2
                PnL = self.qty * (self.entry_price - current_price)
                PnL_net = PnL - close_fees
                self.capital += self.margin + PnL - close_fees
                position_type = "short"
            
            self.max_drawdown_pct = (PnL_net / self.capital_before_buy) * 100 if PnL_net < 0 else 0
            
            self.trade_list.append({
                "idx": last_idx,
                "idx_entry": self.idx_entry,
                "Timestamp": last_row["Timestamp"],
                "Timestamp_entry": self.timestamp_entry,
                "qty": self.qty,
                "entry_price": self.entry_price,
                "exit_price": current_price,
                "PnL": PnL,
                "PnL_net": PnL_net,
                "Capital": self.capital,
                "MaxDrawDown": self.max_drawdown_pct,
                "position_type": position_type,
                "leverage": self.leverage,
                "margin": self.margin,
                "reason": "end_of_backtest",
            })
            
            # Reset
            self.portfolio = 0
            self.position = 0
            self.qty = 0
            self.entry_price = 0
            self.exit_price = 0
            self.margin = 0
        
        # Calcul des statistiques
        days = (self.df_bt.iloc[-1]["Timestamp"] - self.df_bt.iloc[0]["Timestamp"]).days
        if days <= 0:
            days = 1
        self.days = days
        self.PnL = self.capital - self.capital_init
        self.ROI_pct = self.PnL / self.capital_init * 100
        self.ROI_day_pct = ((1 + self.ROI_pct / 100) ** (365.0 / days) - 1) * 100
        
        roi_decimal = self.ROI_pct / 100
        if roi_decimal <= -1:
            self.ROI_annualized_pct = -100.0
        else:
            self.ROI_annualized_pct = ((1 + roi_decimal) ** (365.0 / days) - 1) * 100
        
        self.df_trades = pd.DataFrame(self.trade_list)
        
        if len(self.df_trades) > 0 and "PnL" in self.df_trades.columns:
            self.win_rates = self.df_trades["PnL"].apply(lambda x: x > 0).mean() * 100
            self.max_drawdown_pct = self.df_trades["MaxDrawDown"].min()  # Min car drawdown est négatif
            
            # Stats par type de position
            if "position_type" in self.df_trades.columns:
                long_trades = self.df_trades[self.df_trades["position_type"] == "long"]
                short_trades = self.df_trades[self.df_trades["position_type"] == "short"]
                
                self.nb_long_trades = len(long_trades)
                self.nb_short_trades = len(short_trades)
                self.win_rate_long = long_trades["PnL"].apply(lambda x: x > 0).mean() * 100 if len(long_trades) > 0 else 0
                self.win_rate_short = short_trades["PnL"].apply(lambda x: x > 0).mean() * 100 if len(short_trades) > 0 else 0
                self.pnl_long = long_trades["PnL_net"].sum() if len(long_trades) > 0 else 0
                self.pnl_short = short_trades["PnL_net"].sum() if len(short_trades) > 0 else 0
                
                # Nombre de liquidations
                self.nb_liquidations = len(self.df_trades[self.df_trades["reason"] == "liquidation"])
            else:
                self.nb_long_trades = len(self.df_trades)
                self.nb_short_trades = 0
                self.win_rate_long = self.win_rates
                self.win_rate_short = 0
                self.pnl_long = self.df_trades["PnL_net"].sum()
                self.pnl_short = 0
                self.nb_liquidations = 0
        else:
            self.win_rates = 0.0
            self.max_drawdown_pct = 0.0
            self.nb_long_trades = 0
            self.nb_short_trades = 0
            self.win_rate_long = 0
            self.win_rate_short = 0
            self.pnl_long = 0
            self.pnl_short = 0
            self.nb_liquidations = 0
        
        self.nb_trades = len(self.df_trades)
        self.nb_trades_by_day = self.nb_trades / days if days > 0 else 0
        
        return self.portfolio, self.capital, self.position, self.qty, self.entry_price, self.exit_price, self.trade_list
    
    def print_stats(self):
        print("=" * 50)
        print("BACKTEST RESULTS - Isolated Margin")
        print("=" * 50)
        print(f"Leverage: {self.leverage}x")
        print(f"Days: {self.days}")
        print("-" * 50)
        print(f"Capital Initial: {self.capital_init:.2f}")
        print(f"Capital Final: {self.capital:.2f}")
        print(f"PnL Total: {self.capital - self.capital_init:.2f}")
        print("-" * 50)
        print(f"ROI: {self.ROI_pct:.2f}%")
        print(f"ROI Annualized: {self.ROI_annualized_pct:.2f}%")
        print(f"ROI/Day: {self.ROI_day_pct:.2f}%")
        print("-" * 50)
        print(f"Nb Trades: {self.nb_trades}")
        print(f"  - Long: {self.nb_long_trades} (Win Rate: {self.win_rate_long:.2f}%, PnL: {self.pnl_long:.2f})")
        print(f"  - Short: {self.nb_short_trades} (Win Rate: {self.win_rate_short:.2f}%, PnL: {self.pnl_short:.2f})")
        print(f"Nb Trades/Day: {self.nb_trades_by_day:.2f}")
        print("-" * 50)
        print(f"Win Rate Global: {self.win_rates:.2f}%")
        print(f"Max DrawDown: {self.max_drawdown_pct:.2f}%")
        print(f"Nb Liquidations: {self.nb_liquidations}")
        print("=" * 50)
        return self.df_trades
