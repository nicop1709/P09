"""
Script pour extraire et visualiser la feature importance depuis un modèle TFT entraîné.

Ce script peut être utilisé de deux manières :
1. Avec un modèle TFT déjà entraîné (chargé depuis un checkpoint)
2. En intégrant l'extraction dans le processus d'entraînement

Usage:
    python tft_feature_importance.py --model-path checkpoint.ckpt --dataset-path data.csv
    python tft_feature_importance.py --integrate  # Intègre dans compare_with_buy_hold.py
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from compare_with_buy_hold import extract_tft_feature_importance

try:
    import torch
    import pytorch_lightning as pl
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
except ImportError as e:
    print(f"⚠️  Erreur d'import: {e}")
    print("   → Assurez-vous d'avoir installé pytorch-forecasting et pytorch-lightning")
    exit(1)


def load_tft_model(model_path, training_dataset):
    """
    Charge un modèle TFT depuis un checkpoint.
    
    Parameters
    ----------
    model_path : str
        Chemin vers le fichier checkpoint (.ckpt)
    training_dataset : TimeSeriesDataSet
        Dataset utilisé pour l'entraînement (nécessaire pour initialiser le modèle)
    
    Returns
    -------
    TemporalFusionTransformer
        Modèle TFT chargé
    """
    try:
        # Charger le modèle depuis le checkpoint
        tft = TemporalFusionTransformer.load_from_checkpoint(model_path)
        return tft
    except Exception as e:
        print(f"⚠️  Erreur lors du chargement du modèle: {e}")
        return None


def visualize_feature_importance(importance_results, output_path=None, top_k=20):
    """
    Visualise les résultats de feature importance.
    
    Parameters
    ----------
    importance_results : dict
        Résultats retournés par extract_tft_feature_importance
    output_path : str, optional
        Chemin pour sauvegarder les graphiques
    top_k : int
        Nombre de top features à afficher
    """
    if 'vsn_importance' not in importance_results or len(importance_results['vsn_importance']) == 0:
        print("⚠️  Aucune donnée d'importance disponible pour la visualisation")
        return
    
    df_importance = importance_results['vsn_importance']
    
    # Créer la figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Bar plot des top features
    top_features = df_importance.head(top_k)
    ax1 = axes[0]
    ax1.barh(range(len(top_features)), top_features['importance_normalized'].values)
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features['feature'].values, fontsize=9)
    ax1.set_xlabel('Importance normalisée', fontsize=11)
    ax1.set_title(f'Top {top_k} Features - Variable Selection Network (VSN)', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. Distribution des importances
    ax2 = axes[1]
    ax2.hist(df_importance['importance_normalized'].values, bins=20, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Importance normalisée', fontsize=11)
    ax2.set_ylabel('Nombre de features', fontsize=11)
    ax2.set_title('Distribution des Importances', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   → Graphique sauvegardé: {output_path}")
    else:
        plt.show()
    
    plt.close()


def print_importance_summary(importance_results, top_k=20):
    """
    Affiche un résumé textuel de la feature importance.
    
    Parameters
    ----------
    importance_results : dict
        Résultats retournés par extract_tft_feature_importance
    top_k : int
        Nombre de top features à afficher
    """
    print("\n" + "="*80)
    print("📊 FEATURE IMPORTANCE - Temporal Fusion Transformer")
    print("="*80)
    
    # VSN Importance
    if 'vsn_importance' in importance_results and len(importance_results['vsn_importance']) > 0:
        df_importance = importance_results['vsn_importance']
        print(f"\n🔝 Top {top_k} Features (Variable Selection Network):")
        print("-" * 80)
        print(f"{'Rang':<6} {'Feature':<40} {'Importance':<15} {'Normalisée':<15}")
        print("-" * 80)
        
        for idx, row in df_importance.head(top_k).iterrows():
            print(f"{idx+1:<6} {row['feature']:<40} {row['importance']:<15.6f} {row['importance_normalized']:<15.4f}")
        
        print(f"\n📈 Statistiques:")
        print(f"   → Total features: {len(df_importance)}")
        print(f"   → Importance moyenne: {df_importance['importance'].mean():.6f}")
        print(f"   → Importance médiane: {df_importance['importance'].median():.6f}")
        print(f"   → Importance max: {df_importance['importance'].max():.6f}")
        print(f"   → Importance min: {df_importance['importance'].min():.6f}")
    else:
        print("\n⚠️  Aucune donnée VSN disponible")
    
    # Attention Stats
    if 'attention_stats' in importance_results:
        att_stats = importance_results['attention_stats']
        if 'error' not in att_stats and 'note' not in att_stats:
            print(f"\n🎯 Statistiques des Attention Weights:")
            print(f"   → Moyenne: {att_stats.get('mean', 'N/A')}")
            print(f"   → Écart-type: {att_stats.get('std', 'N/A')}")
            print(f"   → Min: {att_stats.get('min', 'N/A')}")
            print(f"   → Max: {att_stats.get('max', 'N/A')}")
            if 'shape' in att_stats:
                print(f"   → Shape: {att_stats['shape']}")
        else:
            print(f"\n⚠️  Attention weights: {att_stats.get('note', att_stats.get('error', 'Non disponible'))}")
    
    # Summary
    if 'summary' in importance_results and 'error' not in importance_results['summary']:
        summary = importance_results['summary']
        if 'top_features' in summary:
            print(f"\n📋 Résumé:")
            print(f"   → Top {summary.get('top_k', top_k)} features identifiées")
            print(f"   → Total features analysées: {summary.get('total_features', 'N/A')}")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Extract TFT Feature Importance')
    parser.add_argument('--model-path', type=str, help='Path to TFT checkpoint file')
    parser.add_argument('--dataset-path', type=str, help='Path to training dataset CSV')
    parser.add_argument('--feature-cols', type=str, help='JSON file with feature columns list')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory for results')
    parser.add_argument('--top-k', type=int, default=20, help='Number of top features to display')
    parser.add_argument('--save-csv', action='store_true', help='Save importance results to CSV')
    parser.add_argument('--plot', action='store_true', help='Generate visualization plots')
    
    args = parser.parse_args()
    
    if not args.model_path:
        print("⚠️  --model-path est requis")
        print("\n💡 Pour intégrer la feature importance dans compare_with_buy_hold.py,")
        print("   modifiez le script pour appeler extract_tft_feature_importance()")
        print("   après l'entraînement du modèle TFT.")
        return
    
    # Charger le modèle et le dataset
    # Note: Cette partie nécessite de recréer le dataset avec les mêmes paramètres
    # que lors de l'entraînement. Pour une utilisation complète, il faudrait
    # sauvegarder aussi les paramètres du dataset.
    
    print("⚠️  Pour utiliser ce script, vous devez:")
    print("   1. Sauvegarder le modèle TFT après l'entraînement")
    print("   2. Recréer le TimeSeriesDataSet avec les mêmes paramètres")
    print("   3. Charger le modèle et extraire l'importance")
    print("\n💡 Alternative: Intégrez extract_tft_feature_importance() directement")
    print("   dans compare_with_buy_hold.py après l'entraînement du modèle.")


if __name__ == "__main__":
    main()


