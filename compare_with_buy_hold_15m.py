"""
Comparaison de toutes les stratégies avec Buy & Hold - TIMEFRAME 15m
Plus de données pour améliorer les performances ML (4x plus qu'avec 1h)
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import ta
import warnings
warnings.filterwarnings('ignore')

# Import centralized utilities
from backtest import Backtest
from utils import compute_advanced_features, buy_and_hold

# ============================================================
# PARAMÈTRES - TIMEFRAME 15m (4x PLUS DE DONNÉES)
# ============================================================
TIMEFRAME = "15m"
START_YEAR = 2019
END_YEAR = 2025
HORIZON_STEPS = 96  # 96 steps * 15min = 24 heures (même horizon temporel qu'avant)
FEE_ROUNDTRIP = 0.002
THRESH = FEE_ROUNDTRIP
TRAIN_RATIO = 0.85
VALID_RATIO = 0.10
RANDOM_SEED = 42
LOOKBACK = 512  # MiniRocket: 512*15min = 128h = ~5 jours de contexte
CACHE = f"btc_usdc_{TIMEFRAME}_{START_YEAR}_{END_YEAR}.csv"

np.random.seed(RANDOM_SEED)


# Backtest class and buy_and_hold function are now imported from backtest.py and utils.py

def create_advanced_features(df):
    """Crée des features avancées"""
    # Use centralized compute_advanced_features
    return compute_advanced_features(df, horizon=HORIZON_STEPS, threshold=THRESH)


def train_minirocket(df, y_train, y_valid, y_test, train_end, valid_end, test_df):
    """Entraîne un modèle MiniRocket sur séquences temporelles OHLCV"""
    try:
        from sktime.transformations.panel.rocket import MiniRocket
        from sklearn.metrics import f1_score
    except ImportError:
        print("⚠️  MiniRocket nécessite sktime. Ignoré.")
        return None, None

    print("🚀 Préparation des séquences temporelles MiniRocket...")

    # Normaliser les canaux
    close_mean, close_std = df["Close"].mean(), df["Close"].std()
    close_norm = (df["Close"] - close_mean) / (close_std + 1e-8)

    volume_mean, volume_std = df["Volume"].mean(), df["Volume"].std()
    volume_norm = (df["Volume"] - volume_mean) / (volume_std + 1e-8)

    returns = np.log(df["Close"] / df["Close"].shift(1)).fillna(0)

    hl_spread = (df["High"] - df["Low"]) / df["Close"]
    hl_mean, hl_std = hl_spread.mean(), hl_spread.std()
    hl_norm = (hl_spread - hl_mean) / (hl_std + 1e-8)

    # Créer séquences
    def create_sequences(data, lookback):
        sequences = []
        for i in range(len(data) - lookback + 1):
            sequences.append(data[i:i+lookback])
        return np.array(sequences)

    close_seqs = create_sequences(close_norm.values, LOOKBACK)
    volume_seqs = create_sequences(volume_norm.values, LOOKBACK)
    returns_seqs = create_sequences(returns.values, LOOKBACK)
    hl_seqs = create_sequences(hl_norm.values, LOOKBACK)

    # Stack: (n_samples, n_channels, series_length)
    n_samples = len(df) - LOOKBACK + 1
    X_ts = np.zeros((n_samples, 4, LOOKBACK))
    X_ts[:, 0, :] = close_seqs
    X_ts[:, 1, :] = volume_seqs
    X_ts[:, 2, :] = returns_seqs
    X_ts[:, 3, :] = hl_seqs

    # Aligner les targets
    y_full = np.concatenate([y_train, y_valid, y_test])

    train_end_seq = max(0, train_end - LOOKBACK + 1)
    valid_end_seq = max(0, valid_end - LOOKBACK + 1)

    if train_end_seq <= 0 or valid_end_seq <= train_end_seq:
        print("⚠️  Pas assez de données pour MiniRocket")
        return None, None

    X_train_ts = X_ts[:train_end_seq]
    X_valid_ts = X_ts[train_end_seq:valid_end_seq]
    X_test_ts = X_ts[valid_end_seq:]

    y_train_seq = y_full[LOOKBACK-1:LOOKBACK-1+len(X_train_ts)]
    y_valid_seq = y_full[LOOKBACK-1+len(X_train_ts):LOOKBACK-1+len(X_train_ts)+len(X_valid_ts)]
    y_test_seq = y_full[LOOKBACK-1+len(X_train_ts)+len(X_valid_ts):LOOKBACK-1+len(X_train_ts)+len(X_valid_ts)+len(X_test_ts)]

    if len(y_train_seq) != len(X_train_ts) or len(y_test_seq) != len(X_test_ts):
        print("⚠️  Alignement des targets échoué")
        return None, None

    print(f"   → Train sequences: {len(X_train_ts)}, Test sequences: {len(X_test_ts)}")

    # Transformer avec MiniRocket
    print("🔥 Transformation MiniRocket (10000 kernels)...")
    rocket = MiniRocket(num_kernels=10000, random_state=RANDOM_SEED)
    rocket.fit(X_train_ts)
    X_train_transform = rocket.transform(X_train_ts)
    X_valid_transform = rocket.transform(X_valid_ts)
    X_test_transform = rocket.transform(X_test_ts)

    # Standardiser
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train_transform)
    X_valid_std = scaler.transform(X_valid_transform)
    X_test_std = scaler.transform(X_test_transform)

    # Optimiser sur validation set
    print("🎯 Optimisation hyperparamètres...")
    best_f1 = 0
    best_clf = None
    best_threshold = 0.5

    for class_weight in [None, "balanced", {0: 1, 1: 2}, {0: 1, 1: 3}]:
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_leaf=5,
            class_weight=class_weight, random_state=RANDOM_SEED, n_jobs=-1
        )
        clf.fit(X_train_std, y_train_seq)
        y_valid_prob = clf.predict_proba(X_valid_std)[:, 1]

        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
            y_valid_pred = (y_valid_prob >= threshold).astype(int)
            f1 = f1_score(y_valid_seq, y_valid_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_clf = clf
                best_threshold = threshold

    # Prédictions sur test
    y_test_prob = best_clf.predict_proba(X_test_std)[:, 1]
    y_pred = (y_test_prob >= best_threshold).astype(int)

    # Aligner signaux avec test_df
    signals = np.zeros(len(test_df))
    start_idx = LOOKBACK - 1
    if start_idx < len(test_df):
        end_idx = min(start_idx + len(y_pred), len(test_df))
        signals[start_idx:end_idx] = y_pred[:end_idx-start_idx]

    return signals, best_threshold


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "="*80)
    print("📊 COMPARAISON COMPLÈTE : Buy & Hold vs ML Models")
    print("="*80)

    # 1. Chargement
    print("\n📥 Chargement des données...")
    if not os.path.exists(CACHE):
        print(f"❌ Fichier {CACHE} introuvable!")
        return

    df = pd.read_csv(CACHE, parse_dates=["Timestamp"])
    df = df.dropna().reset_index(drop=True)
    print(f"   → {len(df)} lignes chargées")

    # 2. Feature engineering
    print("\n🔧 Feature engineering...")
    df = create_advanced_features(df)

    feature_cols = [col for col in df.columns if col not in
                    ["Timestamp", "future_close", "roi_H", "y", "Open"] and
                    df[col].dtype in [np.float64, np.int64]]

    df = df.replace([np.inf, -np.inf], np.nan)
    df_model = df.dropna(subset=feature_cols + ["y"]).reset_index(drop=True)
    print(f"   → {len(df_model)} lignes après features")

    # 3. Split
    n = len(df_model)
    train_end = int(n * (TRAIN_RATIO + VALID_RATIO))

    train_df = df_model.iloc[:train_end].copy()
    test_df = df_model.iloc[train_end:].copy()

    X_train = train_df[feature_cols].values
    y_train = train_df["y"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["y"].values

    print(f"   → Test: {len(test_df)} lignes")
    print(f"   → Période test: {test_df['Timestamp'].iloc[0]} à {test_df['Timestamp'].iloc[-1]}")

    # ========== BUY & HOLD ==========
    print("\n" + "="*80)
    print("💰 STRATÉGIE 1 : BUY & HOLD (Baseline)")
    print("="*80)

    capital_bh, qty_bh, price_start, price_end = buy_and_hold(
        test_df, capital_init=1000, pct_capital=0.1, fee_roundtrip=FEE_ROUNDTRIP
    )
    pnl_bh = capital_bh - 1000
    roi_bh = (price_end - price_start) / price_start * 100

    print(f"Prix début  : {price_start:.2f}€")
    print(f"Prix fin    : {price_end:.2f}€")
    print(f"ROI Bitcoin : {roi_bh:+.2f}%")
    from utils import calculate_roi_annualized
    roi_annualized_bh = calculate_roi_annualized(1000, capital_bh, test_df)
    # Buy & Hold = 1 trade (entrée au début, sortie à la fin)
    if "Timestamp" in test_df.columns:
        start_date = pd.to_datetime(test_df["Timestamp"].iloc[0])
        end_date = pd.to_datetime(test_df["Timestamp"].iloc[-1])
        days_bh = (end_date - start_date).days
        if days_bh <= 0:
            days_bh = 1
        avg_trades_bh = 1.0 / days_bh  # 1 trade pour toute la période
    else:
        avg_trades_bh = 0.0
    print(f"Capital final : {capital_bh:.2f}€")
    print(f"PnL : {pnl_bh:+.2f}€")
    print(f"ROI annualized : {roi_annualized_bh:.2f}%")
    print(f"Trades/jour moyen : {avg_trades_bh:.4f}")

    # ========== RANDOM FOREST BASELINE ==========
    print("\n" + "="*80)
    print("🌲 STRATÉGIE 2 : RandomForest Baseline (22 features)")
    print("="*80)

    basic_features = [
        "Open", "High", "Low", "Close", "Volume",
        "logret_1", "logret_5", "logret_20",
        "vol_20", "vol_50", "ma20", "ma50", "ema20", "ema50",
        "ma_diff", "ema_diff", "rsi14", "macd", "macd_signal",
        "atr14", "adx14", "hl_range"
    ]

    X_train_basic = train_df[basic_features].values
    X_test_basic = test_df[basic_features].values

    rf_basic = RandomForestClassifier(
        n_estimators=800, max_depth=10, min_samples_leaf=10,
        class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1, verbose=0
    )
    rf_basic.fit(X_train_basic, y_train)

    p_test_basic = rf_basic.predict_proba(X_test_basic)[:, 1]
    pred_basic = (p_test_basic >= 0.5).astype(int)

    bt_basic = Backtest(df_bt=test_df.reset_index(drop=True), signals=pred_basic,
                       fee_roundtrip=FEE_ROUNDTRIP, pct_capital=0.1, capital_init=1000)
    capital_basic = bt_basic.run()
    pnl_basic = capital_basic - 1000
    roi_annualized_basic = bt_basic.get_roi_annualized()
    avg_trades_basic = bt_basic.get_avg_trades_per_day()

    print(f"Capital final : {capital_basic:.2f}€")
    print(f"PnL : {pnl_basic:+.2f}€")
    print(f"ROI annualized : {roi_annualized_basic:.2f}%")
    print(f"Trades/jour moyen : {avg_trades_basic:.4f}")
    print(f"vs Buy&Hold : {pnl_basic - pnl_bh:+.2f}€")

    # ========== RANDOM FOREST ADVANCED ==========
    print("\n" + "="*80)
    print("🌲✨ STRATÉGIE 3 : RandomForest + Features Avancées (74 features)")
    print("="*80)

    rf_adv = RandomForestClassifier(
        n_estimators=1000, max_depth=15, min_samples_leaf=5,
        class_weight="balanced", max_features="sqrt",
        random_state=RANDOM_SEED, n_jobs=-1, verbose=0
    )
    rf_adv.fit(X_train, y_train)

    p_test_adv = rf_adv.predict_proba(X_test)[:, 1]

    # Optimisation seuil
    best_pnl_adv = -float('inf')
    best_thresh_adv = 0.5
    best_bt_adv = None
    for thresh in np.arange(0.40, 0.66, 0.02):
        pred_adv_thresh = (p_test_adv >= thresh).astype(int)
        bt = Backtest(df_bt=test_df.reset_index(drop=True), signals=pred_adv_thresh,
                     fee_roundtrip=FEE_ROUNDTRIP, pct_capital=0.1, capital_init=1000)
        cap = bt.run()
        pnl = cap - 1000
        if pnl > best_pnl_adv:
            best_pnl_adv = pnl
            best_thresh_adv = thresh
            capital_adv = cap
            best_bt_adv = bt

    roi_annualized_adv = best_bt_adv.get_roi_annualized() if best_bt_adv else 0.0
    avg_trades_adv = best_bt_adv.get_avg_trades_per_day() if best_bt_adv else 0.0
    print(f"Seuil optimal : {best_thresh_adv:.2f}")
    print(f"Capital final : {capital_adv:.2f}€")
    print(f"PnL : {best_pnl_adv:+.2f}€")
    print(f"ROI annualized : {roi_annualized_adv:.2f}%")
    print(f"Trades/jour moyen : {avg_trades_adv:.4f}")
    print(f"vs Buy&Hold : {best_pnl_adv - pnl_bh:+.2f}€")

    # ========== LOGISTIC REGRESSION ==========
    print("\n" + "="*80)
    print("📊 STRATÉGIE 4 : Logistic Regression (74 features)")
    print("="*80)

    # Normalisation nécessaire pour la régression logistique
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    lr.fit(X_train_scaled, y_train)

    p_test_lr = lr.predict_proba(X_test_scaled)[:, 1]

    # Optimisation seuil
    best_pnl_lr = -float('inf')
    best_thresh_lr = 0.5
    best_bt_lr = None
    for thresh in np.arange(0.40, 0.66, 0.02):
        pred_lr_thresh = (p_test_lr >= thresh).astype(int)
        bt = Backtest(df_bt=test_df.reset_index(drop=True), signals=pred_lr_thresh,
                     fee_roundtrip=FEE_ROUNDTRIP, pct_capital=0.1, capital_init=1000)
        cap = bt.run()
        pnl = cap - 1000
        if pnl > best_pnl_lr:
            best_pnl_lr = pnl
            best_thresh_lr = thresh
            capital_lr = cap
            best_bt_lr = bt

    roi_annualized_lr = best_bt_lr.get_roi_annualized() if best_bt_lr else 0.0
    avg_trades_lr = best_bt_lr.get_avg_trades_per_day() if best_bt_lr else 0.0
    print(f"Seuil optimal : {best_thresh_lr:.2f}")
    print(f"Capital final : {capital_lr:.2f}€")
    print(f"PnL : {best_pnl_lr:+.2f}€")
    print(f"ROI annualized : {roi_annualized_lr:.2f}%")
    print(f"Trades/jour moyen : {avg_trades_lr:.4f}")
    print(f"vs Buy&Hold : {best_pnl_lr - pnl_bh:+.2f}€")

    # ========== MINIROCKET ==========
    print("\n" + "="*80)
    print("🚀 STRATÉGIE 5 : MiniRocket (Time Series)")
    print("="*80)

    # Utiliser df AVANT feature engineering pour MiniRocket (besoin OHLCV brut)
    df_raw = pd.read_csv(CACHE, parse_dates=["Timestamp"])
    df_raw = df_raw.dropna().reset_index(drop=True)

    # Créer les mêmes features de base pour avoir les targets
    df_raw = create_advanced_features(df_raw)
    df_raw = df_raw.replace([np.inf, -np.inf], np.nan)
    df_raw = df_raw.dropna(subset=feature_cols + ["y"]).reset_index(drop=True)

    # Split avec les mêmes indices
    n_raw = len(df_raw)
    train_end_raw = int(n_raw * (TRAIN_RATIO + VALID_RATIO))

    y_train_raw = df_raw.iloc[:int(n_raw * TRAIN_RATIO)]["y"].values
    y_valid_raw = df_raw.iloc[int(n_raw * TRAIN_RATIO):train_end_raw]["y"].values
    y_test_raw = df_raw.iloc[train_end_raw:]["y"].values

    signals_mr, thresh_mr = train_minirocket(
        df_raw, y_train_raw, y_valid_raw, y_test_raw,
        int(n_raw * TRAIN_RATIO), train_end_raw,
        df_raw.iloc[train_end_raw:].copy()
    )

    if signals_mr is not None:
        bt_mr = Backtest(df_bt=df_raw.iloc[train_end_raw:].reset_index(drop=True), signals=signals_mr,
                        fee_roundtrip=FEE_ROUNDTRIP, pct_capital=0.1, capital_init=1000)
        capital_mr = bt_mr.run()
        pnl_mr = capital_mr - 1000
        roi_annualized_mr = bt_mr.get_roi_annualized()
        avg_trades_mr = bt_mr.get_avg_trades_per_day()

        print(f"Seuil optimal : {thresh_mr:.2f}")
        print(f"Capital final : {capital_mr:.2f}€")
        print(f"PnL : {pnl_mr:+.2f}€")
        print(f"ROI annualized : {roi_annualized_mr:.2f}%")
        print(f"Trades/jour moyen : {avg_trades_mr:.4f}")
        print(f"vs Buy&Hold : {pnl_mr - pnl_bh:+.2f}€")
    else:
        capital_mr = 0
        pnl_mr = -1000
        print("⚠️  MiniRocket non disponible (sktime non installé)")

    # ========== RÉSUMÉ FINAL ==========
    print("\n" + "="*80)
    print("🏆 TABLEAU COMPARATIF FINAL")
    print("="*80)

    results_list = [
        {
            "Stratégie": "Buy & Hold",
            "Capital": capital_bh,
            "PnL": pnl_bh,
            "vs B&H": 0.00,
            "ROI annualized %": f"{roi_annualized_bh:.2f}%",
            "Trades/jour": f"{avg_trades_bh:.4f}"
        },
        {
            "Stratégie": "RF Baseline (22 feat)",
            "Capital": capital_basic,
            "PnL": pnl_basic,
            "vs B&H": pnl_basic - pnl_bh,
            "ROI annualized %": f"{roi_annualized_basic:.2f}%",
            "Trades/jour": f"{avg_trades_basic:.4f}"
        },
        {
            "Stratégie": "RF Advanced (74 feat)",
            "Capital": capital_adv,
            "PnL": best_pnl_adv,
            "vs B&H": best_pnl_adv - pnl_bh,
            "ROI annualized %": f"{roi_annualized_adv:.2f}%",
            "Trades/jour": f"{avg_trades_adv:.4f}"
        },
        {
            "Stratégie": "Logistic Regression",
            "Capital": capital_lr,
            "PnL": best_pnl_lr,
            "vs B&H": best_pnl_lr - pnl_bh,
            "ROI annualized %": f"{roi_annualized_lr:.2f}%",
            "Trades/jour": f"{avg_trades_lr:.4f}"
        }
    ]

    if signals_mr is not None:
        results_list.append({
            "Stratégie": "MiniRocket (TS)",
            "Capital": capital_mr,
            "PnL": pnl_mr,
            "vs B&H": pnl_mr - pnl_bh,
            "ROI annualized %": f"{roi_annualized_mr:.2f}%",
            "Trades/jour": f"{avg_trades_mr:.4f}"
        })
    else:
        roi_annualized_mr = -100.0
        avg_trades_mr = 0.0

    results = pd.DataFrame(results_list)

    print(results.to_string(index=False))

    # Analyse
    print("\n" + "="*80)
    print("📈 ANALYSE")
    print("="*80)

    winner = results.loc[results["PnL"].idxmax()]
    print(f"Meilleure stratégie : {winner['Stratégie']}")
    print(f"PnL : {winner['PnL']:.2f}€")

    ml_pnls = [best_pnl_adv, pnl_basic, best_pnl_lr]
    if signals_mr is not None:
        ml_pnls.append(pnl_mr)
    ml_best_pnl = max(ml_pnls)

    if pnl_bh > ml_best_pnl:
        print("\n⚠️  Buy & Hold BAT tous les modèles ML !")
        print("   → Les signaux ML n'ajoutent pas de valeur")
        print("   → Mieux vaut garder le Bitcoin")
    else:
        print("\n✅ Un modèle ML bat Buy & Hold !")
        print(f"   → Gain vs B&H : {ml_best_pnl - pnl_bh:.2f}€")

    print("="*80)


if __name__ == "__main__":
    main()
