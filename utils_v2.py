
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize


# =============================================================================
# MÉTRIQUES MULTICLASS (Long/Short/Hold)
# =============================================================================

def compute_multiclass_metrics(y_true, y_pred, y_proba=None, class_labels=[-1, 0, 1], 
                                class_names=["Short", "Hold", "Long"]):
    """
    Calcule les métriques pour classification multiclass (Long/Short/Hold).
    
    Args:
        y_true: Labels réels (-1, 0, 1)
        y_pred: Prédictions (-1, 0, 1)
        y_proba: Probabilités prédites (shape: n_samples x n_classes), optionnel
                 Les colonnes doivent être dans l'ordre de class_labels
        class_labels: Liste des labels de classe [-1, 0, 1]
        class_names: Noms des classes ["Short", "Hold", "Long"]
    
    Returns:
        dict: Dictionnaire de métriques
    """
    metrics = {}
    
    # Classification report
    report = classification_report(y_true, y_pred, labels=class_labels, 
                                   target_names=class_names, output_dict=True, 
                                   zero_division=0)
    
    # Accuracy globale
    metrics["accuracy"] = report["accuracy"]
    
    # Métriques par classe
    for label, name in zip(class_labels, class_names):
        if name in report:
            metrics[f"precision_{name.lower()}"] = report[name]["precision"]
            metrics[f"recall_{name.lower()}"] = report[name]["recall"]
            metrics[f"f1_{name.lower()}"] = report[name]["f1-score"]
            metrics[f"support_{name.lower()}"] = report[name]["support"]
    
    # Métriques macro/weighted
    metrics["precision_macro"] = report["macro avg"]["precision"]
    metrics["recall_macro"] = report["macro avg"]["recall"]
    metrics["f1_macro"] = report["macro avg"]["f1-score"]
    metrics["precision_weighted"] = report["weighted avg"]["precision"]
    metrics["recall_weighted"] = report["weighted avg"]["recall"]
    metrics["f1_weighted"] = report["weighted avg"]["f1-score"]
    
    # ROC-AUC multiclass (si probas disponibles)
    if y_proba is not None:
        roc_auc = _compute_roc_auc_multiclass(y_true, y_proba, class_labels)
        metrics["roc_auc_ovr"] = roc_auc.get("macro")
        metrics["roc_auc_ovr_weighted"] = roc_auc.get("weighted")
        # ROC-AUC par classe
        for i, name in enumerate(class_names):
            metrics[f"roc_auc_{name.lower()}"] = roc_auc.get(f"class_{i}")
    
    return metrics


def _compute_roc_auc_multiclass(y_true, y_proba, class_labels=[-1, 0, 1]):
    """
    Calcule ROC-AUC pour multiclass de manière robuste.
    Gère les cas où certaines classes n'ont pas d'échantillons.
    """
    from sklearn.metrics import roc_auc_score
    
    result = {"macro": None, "weighted": None}
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)
    n_classes = len(class_labels)
    
    # Vérifier que toutes les classes sont présentes dans y_true
    unique_classes = set(np.unique(y_true))
    expected_classes = set(class_labels)
    
    if unique_classes != expected_classes:
        missing = expected_classes - unique_classes
        print(f"⚠️ Classes manquantes dans y_true: {missing}")
        print(f"  ROC-AUC calculé seulement pour les classes présentes")
    
    # Calcul One-vs-Rest pour chaque classe
    roc_scores = []
    weights = []
    
    for i, cls in enumerate(class_labels):
        # Binary: cette classe vs les autres
        y_binary = (y_true == cls).astype(int)
        n_pos = y_binary.sum()
        n_neg = len(y_binary) - n_pos
        
        if n_pos == 0 or n_neg == 0:
            # Pas possible de calculer ROC-AUC
            result[f"class_{i}"] = None
            continue
        
        try:
            score = roc_auc_score(y_binary, y_proba[:, i])
            roc_scores.append(score)
            weights.append(n_pos)
            result[f"class_{i}"] = score
        except Exception as e:
            result[f"class_{i}"] = None
    
    # Calcul macro et weighted
    if len(roc_scores) > 0:
        result["macro"] = np.mean(roc_scores)
        result["weighted"] = np.average(roc_scores, weights=weights)
    
    return result


def print_multiclass_report(y_true, y_pred, y_proba=None, class_labels=[-1, 0, 1],
                            class_names=["Short", "Hold", "Long"]):
    """
    Affiche un rapport de classification multiclass complet.
    
    Args:
        y_true: Labels réels (-1, 0, 1)
        y_pred: Prédictions (-1, 0, 1)
        y_proba: Probabilités prédites (optionnel)
        class_labels: Liste des labels de classe
        class_names: Noms des classes
    """
    print("=" * 60)
    print("CLASSIFICATION REPORT - 3 Classes (Long/Short/Hold)")
    print("=" * 60)
    
    # Classification report sklearn
    print(classification_report(y_true, y_pred, labels=class_labels, 
                               target_names=class_names, digits=4, zero_division=0))
    
    # Matrice de confusion
    print("Matrice de confusion:")
    print("-" * 40)
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)
    print()
    
    # ROC-AUC si probas disponibles
    if y_proba is not None:
        roc_auc = _compute_roc_auc_multiclass(y_true, y_proba, class_labels)
        if roc_auc["macro"] is not None:
            print(f"ROC-AUC (OvR macro):    {roc_auc['macro']:.4f}")
            print(f"ROC-AUC (OvR weighted): {roc_auc['weighted']:.4f}")
            for i, name in enumerate(class_names):
                score = roc_auc.get(f"class_{i}")
                if score is not None:
                    print(f"  - {name}: {score:.4f}")
        else:
            print("ROC-AUC non calculable (classes manquantes)")
    
    print("=" * 60)


def plot_confusion_matrix_3class(y_true, y_pred, class_labels=[-1, 0, 1],
                                  class_names=["Short", "Hold", "Long"], 
                                  title="Matrice de confusion", figsize=(8, 6)):
    """
    Affiche une matrice de confusion pour 3 classes.
    
    Returns:
        fig: Figure matplotlib
    """
    from sklearn.metrics import ConfusionMatrixDisplay
    
    fig, ax = plt.subplots(figsize=figsize)
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(title)
    plt.tight_layout()
    
    return fig

def plot_backtest(backtester):
    """
    Affiche le backtest avec les positions Long et Short.
    - Long Entry: triangle vert vers le haut
    - Long Exit: triangle rouge vers le bas
    - Short Entry: triangle orange vers le bas
    - Short Exit: triangle bleu vers le haut
    """
    trades_df = backtester.df_trades

    df_curves = backtester.df_bt.reset_index(drop=True)
    df_curves["Timestamp_entry"] = df_curves["Timestamp"]
    
    # Merge des données de trades
    if len(trades_df) > 0 and "Timestamp" in trades_df.columns:
        df_curves = pd.merge(
            df_curves, 
            trades_df[["Timestamp", "exit_price", "Capital", "position_type"]], 
            on="Timestamp", 
            how="left"
        )
        df_curves = pd.merge(
            df_curves, 
            trades_df[["Timestamp_entry", "entry_price", "position_type"]], 
            on="Timestamp_entry", 
            how="left",
            suffixes=('', '_entry')
        )
    else:
        df_curves["exit_price"] = None
        df_curves["Capital"] = backtester.capital_init
        df_curves["entry_price"] = None
        df_curves["position_type"] = None
        df_curves["position_type_entry"] = None
    
    df_curves["Capital"] = df_curves["Capital"].ffill().fillna(backtester.capital_init)

    timestamps = df_curves["Timestamp"]
    close_prices = df_curves["Close"]
    capital_curve = df_curves["Capital"]

    # Créer un subplot avec 2 graphiques
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Prix avec signaux Long/Short', 'Évolution du Capital'),
        row_heights=[0.6, 0.4]
    )

    # Graphique 1 : Prix
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=close_prices,
            mode='lines',
            name='Close',
            line=dict(color='gray', width=1)
        ),
        row=1, col=1
    )

    # Séparation des trades Long et Short
    if len(trades_df) > 0 and "position_type" in trades_df.columns:
        long_trades = trades_df[trades_df["position_type"] == "long"]
        short_trades = trades_df[trades_df["position_type"] == "short"]
        
        # Long Entry (vert, triangle up)
        if len(long_trades) > 0:
            fig.add_trace(
                go.Scatter(
                    x=long_trades["Timestamp_entry"],
                    y=long_trades["entry_price"],
                    mode='markers',
                    marker=dict(color='#00C853', symbol='triangle-up', size=12, line=dict(width=1, color='darkgreen')),
                    name='Long Entry'
                ),
                row=1, col=1
            )
            # Long Exit (rouge, triangle down)
            fig.add_trace(
                go.Scatter(
                    x=long_trades["Timestamp"],
                    y=long_trades["exit_price"],
                    mode='markers',
                    marker=dict(color='#FF1744', symbol='triangle-down', size=12, line=dict(width=1, color='darkred')),
                    name='Long Exit'
                ),
                row=1, col=1
            )
        
        # Short Entry (orange, triangle down)
        if len(short_trades) > 0:
            fig.add_trace(
                go.Scatter(
                    x=short_trades["Timestamp_entry"],
                    y=short_trades["entry_price"],
                    mode='markers',
                    marker=dict(color='#FF9100', symbol='triangle-down', size=12, line=dict(width=1, color='darkorange')),
                    name='Short Entry'
                ),
                row=1, col=1
            )
            # Short Exit (bleu, triangle up)
            fig.add_trace(
                go.Scatter(
                    x=short_trades["Timestamp"],
                    y=short_trades["exit_price"],
                    mode='markers',
                    marker=dict(color='#2979FF', symbol='triangle-up', size=12, line=dict(width=1, color='darkblue')),
                    name='Short Exit'
                ),
                row=1, col=1
            )
    else:
        # Fallback pour ancien format sans position_type
        if len(trades_df) > 0:
            fig.add_trace(
                go.Scatter(
                    x=trades_df["Timestamp_entry"],
                    y=trades_df["entry_price"],
                    mode='markers',
                    marker=dict(color='green', symbol='triangle-up', size=10),
                    name='Buy'
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=trades_df["Timestamp"],
                    y=trades_df["exit_price"],
                    mode='markers',
                    marker=dict(color='red', symbol='triangle-down', size=10),
                    name='Sell'
                ),
                row=1, col=1
            )

    # Graphique 2 : Capital
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=capital_curve,
            mode='lines',
            name='Capital',
            line=dict(color='purple', width=2)
        ),
        row=2, col=1
    )

    # Ligne de référence pour le capital initial
    fig.add_hline(
        y=backtester.capital_init,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Capital initial: {backtester.capital_init:.2f}",
        row=2, col=1
    )

    # Titre avec stats
    title = f'Backtest - Leverage: {backtester.leverage}x | ROI: {backtester.ROI_pct:.2f}% | Win Rate: {backtester.win_rates:.2f}%'
    
    fig.update_layout(
        title=title,
        height=800,
        width=1200,
        showlegend=True,
        legend=dict(x=0, y=1)
    )

    fig.update_xaxes(title_text="Timestamp", row=2, col=1)
    fig.update_yaxes(title_text="Prix", row=1, col=1)
    fig.update_yaxes(title_text="Capital", row=2, col=1)

    fig.show()


def clean_data(df):
    df = df.copy()
    df = df.dropna().reset_index(drop=True)
    return df

def calculate_features_pct_change(df):
    df = df.copy()
    df["Close_pct_change"] = df["Close"].pct_change()
    df["High_pct_change"] = df["High"].pct_change()
    df["Low_pct_change"] = df["Low"].pct_change()          
    df["Volume_pct_change"] = df["Volume"].pct_change()
    features_cols = ["Close_pct_change", "High_pct_change", "Low_pct_change", "Volume_pct_change"]
    return df, features_cols

def calculate_features_technical(df):
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low  = df["Low"]
    vol  = df["Volume"]
    open = df["Open"]

    df["logret_1"] = np.log(close).diff()
    df["logret_5"] = np.log(close).diff(5)
    df["logret_20"] = np.log(close).diff(20)

    df["vol_20"] = df["logret_1"].rolling(20).std()
    df["vol_50"] = df["logret_1"].rolling(50).std()

    df["ma20"] = close.rolling(20).mean()
    df["ma50"] = close.rolling(50).mean()
    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()

    df["ma_diff"] = df["ma20"] - df["ma50"]
    df["ema_diff"] = df["ema20"] - df["ema50"]

    df["rsi14"] = ta.momentum.rsi(close, window=14)
    macd = ta.trend.MACD(close)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    df["atr14"] = ta.volatility.average_true_range(high, low, close, window=14) / close
    df["adx14"] = ta.trend.adx(high, low, close, window=14)

    df["hl_range"] = (high - low) / close
    df["np_range"] = (high - low) / (close - open+1e-6)
    features_cols = []

    # On retire les lignes avec NaNs (features + futur)
    features_cols = ["logret_1", "logret_5", "logret_20", "vol_20", 
    "vol_50", "ma20", "ma50", "ema20", "ema50", "ma_diff", "ema_diff", 
    "rsi14", "macd", "macd_signal", "atr14", "adx14", "hl_range", "np_range"]
    
    return df, features_cols

def calculate_label(df, horizon_steps, threshold):
    """
    Calcule le label pour classification binaire (Long only).
    y = 1 si ROI futur > threshold, sinon 0
    """
    df = df.copy()
    df["future_close"] = df["Close"].shift(-horizon_steps)
    df["roi_H"] = (df["future_close"] - df["Close"]) / df["Close"]
    df["y"] = (df["roi_H"] > threshold).astype(int)
    return df

def calculate_label_3class(df, horizon_steps, threshold_long, threshold_short=None):
    """
    Calcule le label pour classification 3 classes (Long/Short/Hold).
    
    Labels:
        - y = 1  : Long (ROI futur > threshold_long)
        - y = -1 : Short (ROI futur < -threshold_short)
        - y = 0  : Hold (entre les deux)
    
    Args:
        df: DataFrame avec colonne "Close"
        horizon_steps: Nombre de pas pour calculer le ROI futur
        threshold_long: Seuil pour signal Long (ex: 0.01 = 1%)
        threshold_short: Seuil pour signal Short (par défaut = threshold_long)
    """
    df = df.copy()
    if threshold_short is None:
        threshold_short = threshold_long
    
    df["future_close"] = df["Close"].shift(-horizon_steps)
    df["roi_H"] = (df["future_close"] - df["Close"]) / df["Close"]
    
    # Labels: 1 = Long, -1 = Short, 0 = Hold
    conditions = [
        df["roi_H"] > threshold_long,   # Long
        df["roi_H"] < -threshold_short,  # Short
    ]
    choices = [1, -1]
    df["y"] = np.select(conditions, choices, default=0)
    
    return df

def prepare_data_min_features(df, horizon_steps, threshold):
    """Préparation des données avec features minimales (classification binaire)."""
    df = clean_data(df)
    df, features_cols = calculate_features_pct_change(df)
    df = calculate_label(df, horizon_steps, threshold)
    df_model = df.dropna(subset=features_cols + ["y"]).reset_index(drop=True)
    df_model["Volume_pct_change"] = df_model["Volume_pct_change"].replace([np.inf, -np.inf], 0)
    return df_model, features_cols

def prepare_data_min_features_3class(df, horizon_steps, threshold_long, threshold_short=None):
    """Préparation des données avec features minimales (classification 3 classes)."""
    df = clean_data(df)
    df, features_cols = calculate_features_pct_change(df)
    df = calculate_label_3class(df, horizon_steps, threshold_long, threshold_short)
    df_model = df.dropna(subset=features_cols + ["y"]).reset_index(drop=True)
    df_model["Volume_pct_change"] = df_model["Volume_pct_change"].replace([np.inf, -np.inf], 0)
    return df_model, features_cols

def prepare_data_advanced_features(df, horizon_steps, threshold):
    """Préparation des données avec features avancées (classification binaire)."""
    df = clean_data(df)
    df, features_cols = calculate_features_technical(df)
    df = calculate_label(df, horizon_steps, threshold)
    df_model = df.dropna(subset=features_cols + ["y"]).reset_index(drop=True)
    return df_model, features_cols

def prepare_data_advanced_features_3class(df, horizon_steps, threshold_long, threshold_short=None):
    """Préparation des données avec features avancées (classification 3 classes)."""
    df = clean_data(df)
    df, features_cols = calculate_features_technical(df)
    df = calculate_label_3class(df, horizon_steps, threshold_long, threshold_short)
    df_model = df.dropna(subset=features_cols + ["y"]).reset_index(drop=True)
    return df_model, features_cols
