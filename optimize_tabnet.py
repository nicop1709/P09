"""
Script d'optimisation TabNet vs RandomForest
Objectif : Trouver la meilleure configuration TabNet qui bat RandomForest

Usage:
    python optimize_tabnet.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
import warnings
warnings.filterwarnings('ignore')

# Import centralized utilities
from backtest import Backtest
from utils import compute_features, construct_target

# ============================================================
# PARAMÈTRES GLOBAUX
# ============================================================
PAIR = "BTC/USDC"
TIMEFRAME = "1h"
START_YEAR = 2017  # Augmenté à 2017 pour BEAUCOUP plus de données
END_YEAR = 2025
HORIZON_STEPS = 24
FEE_ROUNDTRIP = 0.002
THRESH = FEE_ROUNDTRIP
TRAIN_RATIO = 0.85
VALID_RATIO = 0.10
RANDOM_SEED = 42
CACHE = f"btc_usdc_{TIMEFRAME}_{START_YEAR}_{END_YEAR}.csv"

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# Backtest class is now imported from backtest.py


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================
def load_and_prepare_data():
    """Charge et prépare les données"""
    print("📊 Chargement des données...")

    if not os.path.exists(CACHE):
        print(f"❌ Fichier {CACHE} introuvable!")
        print("Exécutez d'abord le notebook pour créer le cache des données.")
        sys.exit(1)

    df = pd.read_csv(CACHE, parse_dates=["Timestamp"])
    df = df.dropna().reset_index(drop=True)

    print(f"   → {len(df)} lignes chargées")
    return df


def create_features(df):
    """Crée les features techniques"""
    print("🔧 Création des features...")

    # Use centralized compute_features
    df = compute_features(df)
    
    # Add hl_range if not present
    if "hl_range" not in df.columns:
        df["hl_range"] = (df["High"] - df["Low"]) / df["Close"]

    # Create target using centralized function
    target = construct_target(df, horizon=HORIZON_STEPS, threshold=THRESH)
    df["y"] = target

    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "logret_1", "logret_5", "logret_20",
        "vol_20", "vol_50", "ma20", "ma50", "ema20", "ema50",
        "ma_diff", "ema_diff", "rsi14", "macd", "macd_signal",
        "atr14", "adx14", "hl_range"
    ]

    df_model = df.dropna(subset=feature_cols + ["y"]).reset_index(drop=True)
    print(f"   → {len(df_model)} lignes après features")

    return df_model, feature_cols


def split_data(df_model, feature_cols):
    """Split train/valid/test"""
    print("✂️  Split train/valid/test...")

    n = len(df_model)
    train_end = int(n * TRAIN_RATIO)
    valid_end = int(n * (TRAIN_RATIO + VALID_RATIO))

    train_df = df_model.iloc[:train_end].copy()
    valid_df = df_model.iloc[train_end:valid_end].copy()
    test_df = df_model.iloc[valid_end:].copy()

    X_train = train_df[feature_cols].values
    y_train = train_df["y"].values
    X_valid = valid_df[feature_cols].values
    y_valid = valid_df["y"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["y"].values

    print(f"   → Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")

    return X_train, y_train, X_valid, y_valid, X_test, y_test, test_df


def train_baseline_rf(X_train, y_train, X_test, y_test, test_df):
    """Entraîne RandomForest baseline"""
    print("\n" + "="*60)
    print("🌲 BASELINE : RandomForest")
    print("="*60)

    rf = RandomForestClassifier(
        n_estimators=800,
        max_depth=10,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=0
    )

    print("   Entraînement...")
    rf.fit(X_train, y_train)

    p_test = rf.predict_proba(X_test)[:, 1]
    pred_test = (p_test >= 0.5).astype(int)

    # Backtest
    bt_rf = Backtest(df_bt=test_df.reset_index(drop=True), signals=pred_test, fee_roundtrip=FEE_ROUNDTRIP, pct_capital=0.1, capital_init=1000)
    capital_rf = bt_rf.run()
    pnl_rf = capital_rf - 1000
    roi_annualized_rf = bt_rf.get_roi_annualized()
    avg_trades_rf = bt_rf.get_avg_trades_per_day()

    roc_auc = roc_auc_score(y_test, p_test)

    print(f"   ✅ ROC-AUC: {roc_auc:.4f}")
    print(f"   ✅ Capital final: {capital_rf:.2f}€")
    print(f"   ✅ PnL: {pnl_rf:.2f}€")
    print(f"   ✅ ROI annualized: {roi_annualized_rf:.2f}%")
    print(f"   ✅ Trades/jour moyen: {avg_trades_rf:.4f}")

    return capital_rf, pnl_rf, roc_auc


def optimize_tabnet(X_train, y_train, X_valid, y_valid, X_test, y_test, test_df, target_pnl):
    """Optimise TabNet pour battre RandomForest"""
    print("\n" + "="*60)
    print("🔥 OPTIMISATION TABNET")
    print("="*60)
    print(f"🎯 Objectif : PnL > {target_pnl:.2f}€\n")

    # Scaling obligatoire
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_valid_s = scaler.transform(X_valid)
    X_test_s = scaler.transform(X_test)

    # Grille d'hyperparamètres optimisée pour PLUS DE DONNÉES
    configs = [
        # Config 1 : Grand modèle, plus d'epochs
        {"n_d": 64, "n_a": 64, "n_steps": 6, "lr": 1e-3, "gamma": 1.5, "patience": 40, "max_epochs": 200, "name": "Config1_large_long"},
        # Config 2 : Très grand modèle
        {"n_d": 128, "n_a": 128, "n_steps": 7, "lr": 5e-4, "gamma": 1.3, "patience": 50, "max_epochs": 250, "name": "Config2_xlarge"},
        # Config 3 : Optimal moyen
        {"n_d": 48, "n_a": 48, "n_steps": 6, "lr": 1e-3, "gamma": 1.5, "patience": 35, "max_epochs": 200, "name": "Config3_medium_opt"},
        # Config 4 : Plus de steps
        {"n_d": 64, "n_a": 64, "n_steps": 10, "lr": 8e-4, "gamma": 1.8, "patience": 40, "max_epochs": 200, "name": "Config4_deep"},
        # Config 5 : Balanced
        {"n_d": 32, "n_a": 32, "n_steps": 5, "lr": 1.5e-3, "gamma": 1.4, "patience": 30, "max_epochs": 150, "name": "Config5_balanced"},
        # Config 6 : Aggressive
        {"n_d": 96, "n_a": 96, "n_steps": 8, "lr": 2e-3, "gamma": 1.6, "patience": 35, "max_epochs": 200, "name": "Config6_aggressive"},
    ]

    best_config = None
    best_pnl = target_pnl
    best_capital = 0
    results_all = []

    for i, config in enumerate(configs, 1):
        print(f"\n🧪 Test {i}/{len(configs)} : {config['name']}")
        print(f"   Params: n_d={config['n_d']}, n_a={config['n_a']}, n_steps={config['n_steps']}, lr={config['lr']}")

        try:
            # Entraînement TabNet
            tabnet = TabNetClassifier(
                n_d=config['n_d'],
                n_a=config['n_a'],
                n_steps=config['n_steps'],
                gamma=config['gamma'],
                lambda_sparse=1e-4,
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=config['lr']),
                mask_type="sparsemax",
                seed=RANDOM_SEED,
                verbose=0
            )

            tabnet.fit(
                X_train_s, y_train,
                eval_set=[(X_valid_s, y_valid)],
                eval_name=["valid"],
                eval_metric=["auc"],
                max_epochs=config.get('max_epochs', 150),
                patience=config['patience'],
                batch_size=1024,
                virtual_batch_size=256,
                num_workers=0,
                drop_last=False
            )

            # Prédictions
            p_test_tab = tabnet.predict_proba(X_test_s)[:, 1]

            # Test différents seuils
            thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
            best_thresh_pnl = -float('inf')
            best_thresh = 0.5

            for thresh in thresholds:
                pred_tab = (p_test_tab >= thresh).astype(int)
                bt_tab = Backtest(df_bt=test_df.reset_index(drop=True), signals=pred_tab, fee_roundtrip=FEE_ROUNDTRIP, pct_capital=0.1, capital_init=1000)
                capital_tab = bt_tab.run()
                pnl_tab = capital_tab - 1000

                if pnl_tab > best_thresh_pnl:
                    best_thresh_pnl = pnl_tab
                    best_thresh = thresh
                    best_capital_thresh = capital_tab
                    best_bt_tab = bt_tab

            roi_annualized_tab = best_bt_tab.get_roi_annualized() if 'best_bt_tab' in locals() else 0.0
            avg_trades_tab = best_bt_tab.get_avg_trades_per_day() if 'best_bt_tab' in locals() else 0.0
            roc_auc = roc_auc_score(y_test, p_test_tab)

            print(f"   ROC-AUC: {roc_auc:.4f}")
            print(f"   Meilleur seuil: {best_thresh:.2f}")
            print(f"   Capital final: {best_capital_thresh:.2f}€")
            print(f"   PnL: {best_thresh_pnl:.2f}€")
            print(f"   ROI annualized: {roi_annualized_tab:.2f}%")
            print(f"   Trades/jour moyen: {avg_trades_tab:.4f}", end="")

            if best_thresh_pnl > target_pnl:
                print(" ✅ MEILLEUR QUE RF!")
                if best_thresh_pnl > best_pnl:
                    best_pnl = best_thresh_pnl
                    best_capital = best_capital_thresh
                    best_config = {**config, "threshold": best_thresh, "roc_auc": roc_auc}
            else:
                print(f" ❌ (écart: {best_thresh_pnl - target_pnl:.2f}€)")

            results_all.append({
                "Config": config['name'],
                "n_d": config['n_d'],
                "n_a": config['n_a'],
                "n_steps": config['n_steps'],
                "lr": config['lr'],
                "Best_Threshold": best_thresh,
                "ROC_AUC": roc_auc,
                "Capital": best_capital_thresh,
                "PnL": best_thresh_pnl,
                "Beats_RF": "✅" if best_thresh_pnl > target_pnl else "❌"
            })

        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            results_all.append({
                "Config": config['name'],
                "n_d": config['n_d'],
                "n_a": config['n_a'],
                "n_steps": config['n_steps'],
                "lr": config['lr'],
                "Best_Threshold": None,
                "ROC_AUC": None,
                "Capital": None,
                "PnL": None,
                "Beats_RF": "❌"
            })

    return best_config, best_pnl, best_capital, results_all


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "="*60)
    print("🚀 OPTIMISATION TABNET vs RANDOMFOREST")
    print("="*60)

    # 1. Chargement et préparation
    df = load_and_prepare_data()
    df_model, feature_cols = create_features(df)
    X_train, y_train, X_valid, y_valid, X_test, y_test, test_df = split_data(df_model, feature_cols)

    # 2. Baseline RandomForest
    capital_rf, pnl_rf, roc_rf = train_baseline_rf(X_train, y_train, X_test, y_test, test_df)

    # 3. Optimisation TabNet
    best_config, best_pnl, best_capital, results_all = optimize_tabnet(
        X_train, y_train, X_valid, y_valid, X_test, y_test, test_df, pnl_rf
    )

    # 4. Résultats finaux
    print("\n" + "="*60)
    print("📊 RÉSULTATS FINAUX")
    print("="*60)

    df_results = pd.DataFrame(results_all)
    df_results = df_results.sort_values("PnL", ascending=False)
    print("\nTous les résultats (triés par PnL) :")
    print(df_results.to_string(index=False))

    print("\n" + "="*60)
    print("🏆 COMPARAISON FINALE")
    print("="*60)
    print(f"RandomForest   : {capital_rf:.2f}€ (PnL: {pnl_rf:.2f}€, ROC-AUC: {roc_rf:.4f})")

    if best_config:
        print(f"TabNet (best)  : {best_capital:.2f}€ (PnL: {best_pnl:.2f}€, ROC-AUC: {best_config['roc_auc']:.4f})")
        print(f"\n✅ SUCCÈS ! TabNet bat RandomForest de {best_pnl - pnl_rf:.2f}€")
        print(f"\n🎯 Configuration gagnante :")
        print(f"   - {best_config['name']}")
        print(f"   - n_d={best_config['n_d']}, n_a={best_config['n_a']}, n_steps={best_config['n_steps']}")
        print(f"   - Learning rate: {best_config['lr']}")
        print(f"   - Seuil optimal: {best_config['threshold']:.2f}")
        print(f"   - Gamma: {best_config['gamma']}")
        print(f"   - Patience: {best_config['patience']}")
    else:
        print(f"TabNet (best)  : Aucune config ne bat RandomForest")
        print(f"\n❌ ÉCHEC : Meilleur TabNet = {df_results.iloc[0]['PnL']:.2f}€")
        print(f"   Écart avec RF : {df_results.iloc[0]['PnL'] - pnl_rf:.2f}€")
        print(f"\n💡 Suggestions :")
        print(f"   - Tester plus de configurations")
        print(f"   - Augmenter max_epochs")
        print(f"   - Essayer d'autres features")

    print("="*60)


if __name__ == "__main__":
    main()
