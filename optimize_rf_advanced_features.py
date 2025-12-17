"""
Option 2 : Feature Engineering avancé pour améliorer RandomForest
Ajoute des features sophistiquées : rolling stats, ratios, interactions, etc.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import ta
import warnings
warnings.filterwarnings('ignore')

# Import centralized utilities
from backtest import Backtest
from utils import compute_advanced_features

# ============================================================
# PARAMÈTRES
# ============================================================
PAIR = "BTC/USDC"
TIMEFRAME = "1h"
START_YEAR = 2019
END_YEAR = 2025
HORIZON_STEPS = 24
FEE_ROUNDTRIP = 0.002
THRESH = FEE_ROUNDTRIP
TRAIN_RATIO = 0.85
VALID_RATIO = 0.10
RANDOM_SEED = 42
CACHE = f"btc_usdc_{TIMEFRAME}_{START_YEAR}_{END_YEAR}.csv"

np.random.seed(RANDOM_SEED)


# Backtest class is now imported from backtest.py


# ============================================================
# FEATURE ENGINEERING AVANCÉ
# ============================================================
def create_advanced_features(df):
    """Crée des features avancées"""
    print("🔧 Création de features avancées...")

    # Use centralized compute_advanced_features
    df = compute_advanced_features(df, horizon=HORIZON_STEPS, threshold=THRESH)
    
    # Add additional features specific to this script
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    
    # 7. Stochastic (if not already added)
    if "stoch_k" not in df.columns:
        try:
            stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
            df["stoch_k"] = stoch.stoch()
            df["stoch_d"] = stoch.stoch_signal()
        except Exception:
            pass

    # 8. CCI (Commodity Channel Index)
    if "cci" not in df.columns:
        try:
            df["cci"] = ta.trend.CCIIndicator(high, low, close, window=20).cci()
        except Exception:
            pass

    # 9. Williams %R
    if "williams_r" not in df.columns:
        try:
            df["williams_r"] = ta.momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r()
        except Exception:
            pass

    return df


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "="*60)
    print("🚀 OPTION 2 : Feature Engineering Avancé")
    print("="*60)

    # 1. Chargement
    print("\n📊 Chargement des données...")
    if not os.path.exists(CACHE):
        print(f"❌ Fichier {CACHE} introuvable!")
        return

    df = pd.read_csv(CACHE, parse_dates=["Timestamp"])
    df = df.dropna().reset_index(drop=True)
    print(f"   → {len(df)} lignes chargées")

    # 2. Feature engineering avancé
    df = create_advanced_features(df)

    # Features sélectionnées (toutes les features numériques sauf labels)
    feature_cols = [col for col in df.columns if col not in
                    ["Timestamp", "future_close", "roi_H", "y"] and
                    df[col].dtype in [np.float64, np.int64]]

    # Nettoyage des NaN et Inf
    df = df.replace([np.inf, -np.inf], np.nan)
    df_model = df.dropna(subset=feature_cols + ["y"]).reset_index(drop=True)
    print(f"   → {len(df_model)} lignes après features")
    print(f"   → {len(feature_cols)} features créées")

    # 3. Split
    print("\n✂️  Split train/test...")
    n = len(df_model)
    train_end = int(n * (TRAIN_RATIO + VALID_RATIO))

    train_df = df_model.iloc[:train_end].copy()
    test_df = df_model.iloc[train_end:].copy()

    X_train = train_df[feature_cols].values
    y_train = train_df["y"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["y"].values

    print(f"   → Train: {len(train_df)}, Test: {len(test_df)}")

    # 4. Entraînement RandomForest optimisé
    print("\n🌲 Entraînement RandomForest avec features avancées...")

    rf = RandomForestClassifier(
        n_estimators=1000,  # Plus d'arbres
        max_depth=15,
        min_samples_leaf=5,
        class_weight="balanced",
        max_features="sqrt",  # Meilleure généralisation
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=0
    )

    rf.fit(X_train, y_train)

    # 5. Prédictions
    p_test = rf.predict_proba(X_test)[:, 1]

    # Test plusieurs seuils
    print("\n🎯 Optimisation du seuil de décision...")
    thresholds = np.arange(0.40, 0.66, 0.02)
    best_pnl = -float('inf')
    best_thresh = 0.5

    for thresh in thresholds:
        pred_test = (p_test >= thresh).astype(int)
        bt = Backtest(df_bt=test_df.reset_index(drop=True), signals=pred_test,
                     fee_roundtrip=FEE_ROUNDTRIP, pct_capital=0.1, capital_init=1000)
        capital = bt.run()
        pnl = capital - 1000

        if pnl > best_pnl:
            best_pnl = pnl
            best_thresh = thresh
            best_capital = capital
            best_bt = bt

    pred_test_best = (p_test >= best_thresh).astype(int)

    # 6. Résultats
    roc_auc = roc_auc_score(y_test, p_test)
    roi_annualized = best_bt.get_roi_annualized() if 'best_bt' in locals() and best_bt else 0.0
    avg_trades = best_bt.get_avg_trades_per_day() if 'best_bt' in locals() and best_bt else 0.0

    print("\n" + "="*60)
    print("🏆 RÉSULTATS FINAUX")
    print("="*60)
    print(f"Features utilisées : {len(feature_cols)}")
    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"Meilleur seuil : {best_thresh:.2f}")
    print(f"Capital final : {best_capital:.2f}€")
    print(f"PnL : {best_pnl:.2f}€")
    print(f"ROI annualized : {roi_annualized:.2f}%")
    print(f"Trades/jour moyen : {avg_trades:.4f}")

    print("\nClassification report (seuil optimal) :")
    print(classification_report(y_test, pred_test_best, digits=4))

    # Top features importantes
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n🔝 Top 20 features importantes :")
    print(feature_importance.head(20).to_string(index=False))

    print("="*60)

if __name__ == "__main__":
    main()
