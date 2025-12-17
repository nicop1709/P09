#!/usr/bin/env python3
"""
Script pour charger un modèle TFT depuis un checkpoint et faire les prédictions
sans avoir à réentraîner le modèle.

Usage:
    python load_tft_checkpoint.py
    python load_tft_checkpoint.py --checkpoint checkpoints/epoch=29-step=27060.ckpt
"""

import argparse
import sys
import os

# Ajouter le répertoire parent au path pour importer compare_with_buy_hold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compare_with_buy_hold import load_tft_from_checkpoint_and_predict
import pandas as pd
import numpy as np
from utils import compute_advanced_features

def main():
    parser = argparse.ArgumentParser(description='Charger TFT depuis checkpoint et faire prédictions')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Chemin vers le checkpoint (défaut: utilise le plus récent)')
    parser.add_argument('--csv-file', type=str, required=True,
                       help='Fichier CSV avec les données')
    parser.add_argument('--horizon', type=int, default=24,
                       help='Horizon de prédiction (défaut: 24)')
    parser.add_argument('--threshold', type=float, default=0.02,
                       help='Seuil pour la classification (défaut: 0.02)')
    parser.add_argument('--fee', type=float, default=0.002,
                       help='Frais de transaction roundtrip (défaut: 0.002)')
    
    args = parser.parse_args()
    
    # Charger les données
    print(f"→ Chargement des données depuis {args.csv_file}...")
    df = pd.read_csv(args.csv_file)
    
    # Préparer les features (même logique que dans compare_with_buy_hold.py)
    print("→ Préparation des features...")
    df = compute_advanced_features(df, horizon=args.horizon, threshold=args.threshold)
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Extraire les colonnes de features
    feature_cols = [col for col in df.columns 
                   if col not in ['timestamp', 'datetime', 'y', 'target', 'close', 'open', 'high', 'low', 'volume']]
    
    # Split train/valid/test (même ratio que par défaut)
    train_ratio = 0.6
    valid_ratio = 0.2
    n = len(df)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))
    
    train_df = df.iloc[:train_end].copy()
    valid_df = df.iloc[train_end:valid_end].copy()
    test_df = df.iloc[valid_end:].copy()
    
    # Extraire X et y
    X_train = train_df[feature_cols].values
    y_train = train_df['y'].values
    X_valid = valid_df[feature_cols].values
    y_valid = valid_df['y'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['y'].values
    
    print(f"→ Données: {len(train_df)} train, {len(valid_df)} valid, {len(test_df)} test")
    
    # Charger le modèle et faire les prédictions
    print("\n" + "="*80)
    print("🕐 Chargement du modèle TFT depuis le checkpoint")
    print("="*80)
    
    result = load_tft_from_checkpoint_and_predict(
        X_train, y_train, X_valid, y_valid, X_test, y_test, test_df,
        train_df, valid_df, feature_cols,
        fee_roundtrip=args.fee,
        random_seed=42,
        checkpoint_path=args.checkpoint
    )
    
    if result[0] is not None:
        capital, pnl, roi_annualized, thresh, bt = result
        avg_trades = bt.get_avg_trades_per_day()
        
        print("\n" + "="*80)
        print("✅ RÉSULTATS")
        print("="*80)
        print(f"Seuil optimal : {thresh:.2f}")
        print(f"Capital final : {capital:.2f}€")
        print(f"PnL : {pnl:+.2f}€")
        print(f"ROI annualized : {roi_annualized:.2f}%")
        print(f"Trades/jour moyen : {avg_trades:.4f}")
    else:
        print("\n❌ Échec du chargement ou des prédictions")
        sys.exit(1)

if __name__ == "__main__":
    main()


