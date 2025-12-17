"""
Classe Backtest centralisée pour tous les scripts de trading.

Cette classe implémente un backtest simple long-only basé sur les changements
de signaux. Elle entre en position longue quand signal=1 et sort quand signal=0.
"""

import numpy as np
import pandas as pd
from typing import Union


class Backtest:
    """Simple long‑only backtest based on signal changes.

    The backtest enters a long position when signal=1 and no position is open.
    It exits when signal=0 and a position is open.

    The position size is a fraction ``pct_capital`` of current capital.
    Half of the roundtrip fee is paid at entry and half at exit.
    """

    def __init__(
        self,
        df_bt: pd.DataFrame,
        signals: Union[np.ndarray, pd.Series],
        fee_roundtrip: float = 0.002,
        pct_capital: float = 0.1,
        capital_init: float = 1000.0,
    ) -> None:
        """
        Parameters
        ----------
        df_bt : DataFrame
            DataFrame with OHLCV data, must include 'Close' column.
        signals : array-like
            Binary signals (0 or 1) indicating when to buy/sell.
        fee_roundtrip : float, default=0.002
            Roundtrip fee (e.g. 0.002 = 0.2%).
        pct_capital : float, default=0.1
            Fraction of capital to invest per trade.
        capital_init : float, default=1000.0
            Initial capital.
        """
        self.df_bt = df_bt.reset_index(drop=True)
        self.signals = np.asarray(signals).astype(int)
        self.fee_roundtrip = fee_roundtrip
        self.pct_capital = pct_capital
        self.capital_init = capital_init
        self.capital = capital_init
        self.position = 0  # Track if position is open
        self.qty = 0
        self.entry_price = 0.0
        self.num_trades = 0  # Count number of trades (entries)

    def run(self) -> float:
        """Run the backtest. Returns final capital."""
        self.num_trades = 0  # Reset trade count
        for i, row in self.df_bt.iterrows():
            if i >= len(self.signals):
                break
            sig = int(self.signals[i])
            
            # Buy signal: signal=1 AND no position open
            if sig == 1 and self.position == 0:
                self.qty = self.pct_capital * self.capital / row["Close"]
                position = self.qty * row["Close"]
                buy_fees = self.fee_roundtrip * position / 2
                self.capital -= (position + buy_fees)
                self.position = position
                self.entry_price = row["Close"]
                self.num_trades += 1  # Count trade entry
            
            # Sell signal: signal=0 AND position is open
            elif sig == 0 and self.position != 0:
                sell_value = self.qty * row["Close"]
                sell_fees = self.fee_roundtrip * sell_value / 2
                self.capital += sell_value - sell_fees
                self.position = 0
                self.qty = 0
                self.entry_price = 0.0
        
        # Force close if position still open at end
        if self.position != 0:
            i = len(self.df_bt) - 1
            sell_value = self.qty * self.df_bt["Close"].iloc[i]
            sell_fees = self.fee_roundtrip * sell_value / 2
            self.capital += sell_value - sell_fees
            self.position = 0
        
        return self.capital

    def get_roi_annualized(self) -> float:
        """Calculate annualized ROI percentage.
        
        Returns
        -------
        float
            Annualized ROI as a percentage (e.g., 10.5 for 10.5%).
        """
        if len(self.df_bt) == 0:
            return 0.0
        
        # Get start and end dates
        if "Timestamp" in self.df_bt.columns:
            start_date = pd.to_datetime(self.df_bt["Timestamp"].iloc[0])
            end_date = pd.to_datetime(self.df_bt["Timestamp"].iloc[-1])
        else:
            # If no Timestamp, use index as proxy (assume daily data)
            start_date = pd.Timestamp.now() - pd.Timedelta(days=len(self.df_bt))
            end_date = pd.Timestamp.now()
        
        # Calculate number of days
        days = (end_date - start_date).days
        if days <= 0:
            days = 1  # Avoid division by zero
        
        # Calculate ROI
        roi = (self.capital - self.capital_init) / self.capital_init
        
        # Annualize: (1 + roi)^(365/days) - 1
        if roi <= -1:
            # If we lost more than 100%, return -100%
            return -100.0
        
        roi_annualized = ((1 + roi) ** (365.0 / days) - 1) * 100
        
        return roi_annualized

    def get_avg_trades_per_day(self) -> float:
        """Calculate average number of trades per day.
        
        Returns
        -------
        float
            Average number of trades per day.
        """
        if len(self.df_bt) == 0:
            return 0.0
        
        # Get start and end dates
        if "Timestamp" in self.df_bt.columns:
            start_date = pd.to_datetime(self.df_bt["Timestamp"].iloc[0])
            end_date = pd.to_datetime(self.df_bt["Timestamp"].iloc[-1])
        else:
            # If no Timestamp, use index as proxy (assume daily data)
            start_date = pd.Timestamp.now() - pd.Timedelta(days=len(self.df_bt))
            end_date = pd.Timestamp.now()
        
        # Calculate number of days
        days = (end_date - start_date).days
        if days <= 0:
            days = 1  # Avoid division by zero
        
        # Calculate average trades per day
        # num_trades is set during run()
        avg_trades = self.num_trades / days if days > 0 else 0.0
        
        return avg_trades

