import pandas as pd
import numpy as np

class Trader:
    """
    Trader pour Isolated Margin Binance avec support Long et Short.
    
    Signaux:
        - signal = 1  : Ouvrir une position Long
        - signal = -1 : Ouvrir une position Short
        - signal = 0  : Fermer la position actuelle (ou ne rien faire)
    
    Isolated Margin:
        - Chaque position a son propre margin isolé
        - Le leverage multiplie la taille de position
        - Liquidation si les pertes >= margin (optionnel)
    """
    
    def __init__(self, row: pd.Series, idx: int, idx_entry: int, signal: np.ndarray, 
                 capital: float, portfolio: float, position: int, qty: float, 
                 entry_price: float, exit_price: float, fee_roundtrip=0.002, 
                 pct_capital=1, debug=False, trade_list=[], horizon_steps=24,
                 capital_before_buy=0, leverage=1, enable_liquidation=True):
        self.row = row
        self.idx = idx
        self.signal = signal
        self.fee_roundtrip = fee_roundtrip
        self.pct_capital = pct_capital
        self.capital = capital    
        self.portfolio = portfolio
        self.position = position  # 1 = Long, -1 = Short, 0 = Pas de position
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
        self.leverage = leverage
        self.enable_liquidation = enable_liquidation
        self.margin = 0  # Margin alloué à la position actuelle
        self.liquidation_price = 0  # Prix de liquidation

    def _calculate_liquidation_price(self, entry_price, position_type):
        """
        Calcule le prix de liquidation pour une position Isolated Margin.
        Liquidation quand PnL = -margin (perte = 100% du margin)
        """
        if position_type == "long":
            # Long: liquidé si le prix baisse trop
            return entry_price * (1 - 1/self.leverage) if self.leverage > 0 else 0
        else:
            # Short: liquidé si le prix monte trop
            return entry_price * (1 + 1/self.leverage) if self.leverage > 0 else float('inf')

    def _open_long(self):
        """Ouvre une position Long avec Isolated Margin."""
        # Margin = capital alloué à cette position
        self.margin = self.pct_capital * self.capital
        # Taille de position = margin * leverage
        position_value = self.margin * self.leverage
        self.qty = position_value / self.row["Close"]
        self.position = 1  # Long
        self.entry_price = self.row["Close"]
        
        # Frais d'ouverture (sur la valeur notionnelle)
        open_fees = self.fee_roundtrip * position_value / 2
        self.capital_before_buy = self.capital
        self.capital -= (self.margin + open_fees)  # On retire le margin + frais
        
        self.portfolio = position_value
        self.idx_entry = self.idx
        self.timestamp_entry = self.row["Timestamp"]
        self.liquidation_price = self._calculate_liquidation_price(self.entry_price, "long")
        
        if self.debug:
            print(f"[LONG OPEN] Idx: {self.idx} | Qty: {self.qty:.8f} @ {self.entry_price:.2f}")
            print(f"  Margin: {self.margin:.2f} | Leverage: {self.leverage}x | Position Value: {position_value:.2f}")
            print(f"  Liquidation Price: {self.liquidation_price:.2f}")
        return True

    def _close_long(self, forced=False, reason="signal"):
        """Ferme une position Long."""
        current_price = self.row["Close"]
        position_value = self.qty * current_price
        
        # Frais de fermeture
        close_fees = self.fee_roundtrip * position_value / 2
        
        # PnL = (prix_sortie - prix_entrée) * qty
        PnL = self.qty * (current_price - self.entry_price)
        PnL_net = PnL - close_fees
        
        # Capital récupéré = margin + PnL - frais
        self.capital += self.margin + PnL - close_fees
        
        self.exit_price = current_price
        self.max_drawdown_pct = (PnL_net / self.capital_before_buy) * 100 if PnL_net < 0 else 0
        
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
            "position_type": "long",
            "leverage": self.leverage,
            "margin": self.margin,
            "reason": reason,
        })
        
        if self.debug:
            print(f"[LONG CLOSE] Idx: {self.idx} | Qty: {self.qty:.8f} @ {self.exit_price:.2f}")
            print(f"  PnL: {PnL:.2f} | PnL Net: {PnL_net:.2f} | Capital: {self.capital:.2f}")
            print(f"  Reason: {reason}")
        
        # Reset position
        self.position = 0
        self.qty = 0
        self.entry_price = 0
        self.portfolio = 0
        self.margin = 0
        self.liquidation_price = 0
        return True

    def _open_short(self):
        """Ouvre une position Short avec Isolated Margin."""
        # Margin = capital alloué à cette position
        self.margin = self.pct_capital * self.capital
        # Taille de position = margin * leverage
        position_value = self.margin * self.leverage
        self.qty = position_value / self.row["Close"]
        self.position = -1  # Short
        self.entry_price = self.row["Close"]
        
        # Frais d'ouverture
        open_fees = self.fee_roundtrip * position_value / 2
        self.capital_before_buy = self.capital
        self.capital -= (self.margin + open_fees)  # On retire le margin + frais
        
        self.portfolio = position_value
        self.idx_entry = self.idx
        self.timestamp_entry = self.row["Timestamp"]
        self.liquidation_price = self._calculate_liquidation_price(self.entry_price, "short")
        
        if self.debug:
            print(f"[SHORT OPEN] Idx: {self.idx} | Qty: {self.qty:.8f} @ {self.entry_price:.2f}")
            print(f"  Margin: {self.margin:.2f} | Leverage: {self.leverage}x | Position Value: {position_value:.2f}")
            print(f"  Liquidation Price: {self.liquidation_price:.2f}")
        return True

    def _close_short(self, forced=False, reason="signal"):
        """Ferme une position Short."""
        current_price = self.row["Close"]
        position_value = self.qty * current_price
        
        # Frais de fermeture
        close_fees = self.fee_roundtrip * position_value / 2
        
        # PnL Short = (prix_entrée - prix_sortie) * qty (profit si le prix baisse)
        PnL = self.qty * (self.entry_price - current_price)
        PnL_net = PnL - close_fees
        
        # Capital récupéré = margin + PnL - frais
        self.capital += self.margin + PnL - close_fees
        
        self.exit_price = current_price
        self.max_drawdown_pct = (PnL_net / self.capital_before_buy) * 100 if PnL_net < 0 else 0
        
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
            "position_type": "short",
            "leverage": self.leverage,
            "margin": self.margin,
            "reason": reason,
        })
        
        if self.debug:
            print(f"[SHORT CLOSE] Idx: {self.idx} | Qty: {self.qty:.8f} @ {self.exit_price:.2f}")
            print(f"  PnL: {PnL:.2f} | PnL Net: {PnL_net:.2f} | Capital: {self.capital:.2f}")
            print(f"  Reason: {reason}")
        
        # Reset position
        self.position = 0
        self.qty = 0
        self.entry_price = 0
        self.portfolio = 0
        self.margin = 0
        self.liquidation_price = 0
        return True

    def _check_liquidation(self):
        """Vérifie si la position doit être liquidée."""
        if not self.enable_liquidation or self.position == 0:
            return False
        
        current_price = self.row["Close"]
        
        if self.position == 1:  # Long
            if current_price <= self.liquidation_price:
                if self.debug:
                    print(f"[LIQUIDATION LONG] Prix {current_price:.2f} <= Liq Price {self.liquidation_price:.2f}")
                self._close_long(forced=True, reason="liquidation")
                return True
        elif self.position == -1:  # Short
            if current_price >= self.liquidation_price:
                if self.debug:
                    print(f"[LIQUIDATION SHORT] Prix {current_price:.2f} >= Liq Price {self.liquidation_price:.2f}")
                self._close_short(forced=True, reason="liquidation")
                return True
        
        return False

    def _update_portfolio_value(self):
        """Met à jour la valeur du portfolio basée sur le prix actuel."""
        if self.position == 0:
            self.portfolio = 0
            return
        
        current_price = self.row["Close"]
        
        if self.position == 1:  # Long
            # Valeur = qty * prix actuel
            PnL = self.qty * (current_price - self.entry_price)
            self.portfolio = self.margin + PnL
        elif self.position == -1:  # Short
            # Valeur = margin + PnL (PnL positif si prix baisse)
            PnL = self.qty * (self.entry_price - current_price)
            self.portfolio = self.margin + PnL

    def run(self):
        """Exécute la logique de trading pour cette étape."""
        # Conversion du signal en int
        sig = int(self.signal) if isinstance(self.signal, (np.ndarray, np.generic)) else int(self.signal)
        
        # Mise à jour de la valeur du portfolio
        self._update_portfolio_value()
        
        # Vérification de liquidation
        if self._check_liquidation():
            return self.portfolio, self.capital, self.position, self.qty, self.entry_price, self.exit_price, self.trade_list
        
        if self.debug:
            print(f"Idx: {self.idx} | Signal: {sig} | Position: {self.position} | Portfolio: {self.portfolio:.2f}")
        
        # Logique de trading
        can_close = self.idx >= self.idx_entry + self.horizon_steps
        
        # === Position Long (position = 1) ===
        if self.position == 1:
            # Fermer Long si signal = 0 ou -1 (après horizon)
            if sig <= 0 and can_close:
                self._close_long(reason="signal")
                # Si signal = -1, ouvrir Short immédiatement
                if sig == -1:
                    self._open_short()
        
        # === Position Short (position = -1) ===
        elif self.position == -1:
            # Fermer Short si signal = 0 ou 1 (après horizon)
            if sig >= 0 and can_close:
                self._close_short(reason="signal")
                # Si signal = 1, ouvrir Long immédiatement
                if sig == 1:
                    self._open_long()
        
        # === Pas de position (position = 0) ===
        elif self.position == 0:
            if sig == 1:
                self._open_long()
            elif sig == -1:
                self._open_short()
        
        return self.portfolio, self.capital, self.position, self.qty, self.entry_price, self.exit_price, self.trade_list
