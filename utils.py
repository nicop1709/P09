"""
Fonctions utilitaires centralisées pour tous les scripts de trading.

Ce module contient les fonctions communes utilisées par différents scripts :
- fetch_ohlcv_binance : récupération de données depuis Binance
- compute_features : création de features techniques de base
- compute_advanced_features : création de features avancées
- construct_target : construction de la cible binaire
- split_data : division chronologique des données
- evaluate_metrics : évaluation des métriques de classification
- buy_and_hold : stratégie Buy & Hold
"""

import datetime as _dt
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    import ta  # type: ignore
except ImportError:
    ta = None

try:
    from sklearn.metrics import (
        classification_report,
        precision_recall_curve,
        roc_auc_score,
        auc,
    )
except ImportError:
    classification_report = None
    precision_recall_curve = None
    roc_auc_score = None
    auc = None


def fetch_ohlcv_binance(
    pair: str,
    timeframe: str,
    start: _dt.datetime,
    end: _dt.datetime,
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch OHLCV data from Binance using ccxt.

    Parameters
    ----------
    pair : str
        Trading pair, e.g. 'BTC/USDC'.
    timeframe : str
        Timeframe for OHLCV data, e.g. '1h', '15m'.
    start : datetime
        Start datetime (UTC).  Data will be retrieved from this point.
    end : datetime
        End datetime (UTC).  Data will be retrieved up to this point.
    limit : int, optional
        Maximum number of candles per API call.

    Returns
    -------
    DataFrame
        DataFrame with columns ``Timestamp``, ``Open``, ``High``,
        ``Low``, ``Close``, ``Volume``.
    """
    try:
        import ccxt  # type: ignore
    except ImportError:
        raise ImportError(
            "ccxt is required to fetch data from Binance. "
            "Install with `pip install ccxt` or provide a CSV file via ``--csv-file``."
        )

    exchange = ccxt.binance({"enableRateLimit": True})
    since = int(start.replace(tzinfo=_dt.timezone.utc).timestamp() * 1000)
    end_ms = int(end.replace(tzinfo=_dt.timezone.utc).timestamp() * 1000)

    data: list[list] = []
    while since < end_ms:
        batch = exchange.fetch_ohlcv(pair, timeframe=timeframe, since=since, limit=limit)
        if not batch:
            break
        data.extend(batch)
        since = batch[-1][0] + 1  # move past last candle
        # break early to avoid extremely long runs without rate limiting
        if len(batch) < limit // 2:
            break
    df = pd.DataFrame(
        data, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"]
    )
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute basic technical features for the given OHLCV DataFrame.

    This function creates standard technical indicators. New columns are
    appended to a copy of the input DataFrame.

    Parameters
    ----------
    df : DataFrame
        DataFrame with OHLCV data (columns: Open, High, Low, Close, Volume).

    Returns
    -------
    DataFrame
        DataFrame with added technical features.
    """
    if ta is None:
        raise ImportError("The 'ta' library is required for technical indicators. Install with `pip install ta`.")

    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

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

    # Drop rows with NaNs introduced by rolling calculations
    df = df.dropna().reset_index(drop=True)
    return df


def compute_advanced_features(df: pd.DataFrame, horizon: Optional[int] = None, threshold: Optional[float] = None) -> pd.DataFrame:
    """Compute advanced technical features for the given OHLCV DataFrame.

    This function creates extended technical indicators including rolling
    statistics, momentum features, Bollinger Bands, OBV, etc.

    Parameters
    ----------
    df : DataFrame
        DataFrame with OHLCV data (columns: Open, High, Low, Close, Volume).
    horizon : int, optional
        If provided, creates target column 'y' based on future ROI.
    threshold : float, optional
        Threshold for ROI to label a Buy (used if horizon is provided).

    Returns
    -------
    DataFrame
        DataFrame with added advanced technical features.
    """
    if ta is None:
        raise ImportError("The 'ta' library is required for technical indicators. Install with `pip install ta`.")

    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # Basic features (same as compute_features)
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

    # Advanced features
    # 1. Rolling statistics multi-périodes
    for window in [5, 10, 20, 50]:
        df[f"close_mean_{window}"] = close.rolling(window).mean() / close - 1
        df[f"close_std_{window}"] = close.rolling(window).std() / close
        df[f"close_max_{window}"] = close.rolling(window).max() / close - 1
        df[f"close_min_{window}"] = close.rolling(window).min() / close - 1
        df[f"volume_mean_{window}"] = volume.rolling(window).mean()
        df[f"volume_std_{window}"] = volume.rolling(window).std()

    # 2. Momentum features
    for lag in [1, 3, 5, 10, 20]:
        df[f"price_momentum_{lag}"] = close.pct_change(lag)
        df[f"volume_momentum_{lag}"] = volume.pct_change(lag)

    # 3. Ratios et interactions
    df["close_to_ma20"] = close / df["ma20"] - 1
    df["close_to_ma50"] = close / df["ma50"] - 1
    df["ma20_to_ma50"] = df["ma20"] / df["ma50"] - 1
    df["volume_to_ma"] = volume / volume.rolling(20).mean() - 1
    df["rsi_ema"] = df["rsi14"].ewm(span=14).mean()

    # 4. Volatility features
    df["parkinson_vol"] = np.sqrt(1 / (4 * np.log(2)) * (np.log(high / low) ** 2))
    df["garman_klass_vol"] = np.sqrt(
        0.5 * (np.log(high / low) ** 2) -
        (2 * np.log(2) - 1) * (np.log(close / df["Open"]) ** 2)
    )

    # 5. Support/Resistance indicators
    df["dist_to_high_20"] = (close - high.rolling(20).max()) / close
    df["dist_to_low_20"] = (close - low.rolling(20).min()) / close

    # 6. Bollinger Bands
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_position"] = (close - df["bb_low"]) / (df["bb_high"] - df["bb_low"])

    # 7. OBV
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    df["obv_ema"] = df["obv"].ewm(span=20).mean()

    # 8. Stochastic (if available)
    try:
        stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()
    except Exception:
        pass

    # Create target if horizon and threshold are provided
    if horizon is not None and threshold is not None:
        df["future_close"] = df["Close"].shift(-horizon)
        df["roi_H"] = (df["future_close"] - df["Close"]) / df["Close"]
        df["y"] = (df["roi_H"] > threshold).astype(int)

    # Drop rows with NaNs
    df = df.dropna().reset_index(drop=True)
    return df


def construct_target(df: pd.DataFrame, horizon: int, threshold: float = 0, fee_roundtrip: float = 0.002) -> pd.Series:
    """Construct a binary target vector indicating when to buy.

    The target is 1 (Buy) if the future return over ``horizon`` steps
    exceeds ``threshold``.  Otherwise 0 (No trade).

    Parameters
    ----------
    df : DataFrame
        DataFrame with OHLCV data, must include 'Close' column.
    horizon : int
        Number of steps ahead to look for ROI.
    threshold : float
        Minimum ROI threshold to label as Buy (1).

    Returns
    -------
    Series
        Binary target vector (0 or 1).
    """
    close = df["Close"].values
    # shift forward by horizon steps
    future = np.roll(close, -horizon)
    # last ``horizon`` future returns will be NaN since we can't look that far
    roi = (future*(1-fee_roundtrip/2) - close*(1+fee_roundtrip/2)) 
    df = df.copy()
    df["roi_future"] = roi
    df.loc[df.index[-horizon:], "roi_future"] = np.nan
    target = (df["roi_future"] > threshold).astype(int)
    target.iloc[-horizon:] = 0  # can't trade near the end
    return target


def split_data(
    df: pd.DataFrame,
    features: list[str],
    target: pd.Series,
    train_ratio: float,
    valid_ratio: float,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Split data into train/valid/test sets chronologically.

    Parameters
    ----------
    df : DataFrame
        Data with engineered features.
    features : list[str]
        Column names to use as features.
    target : Series
        Binary target vector.
    train_ratio : float
        Fraction of data for training.
    valid_ratio : float
        Fraction of data for validation.  The remainder is test.

    Returns
    -------
    X_train, X_valid, X_test, y_train, y_valid, y_test
        NumPy arrays for each split.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    valid_end = train_end + int(n * valid_ratio)

    X = df[features].values
    y = target.values
    X_train, X_valid, X_test = X[:train_end], X[train_end:valid_end], X[valid_end:]
    y_train, y_valid, y_test = y[:train_end], y[train_end:valid_end], y[valid_end:]
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> dict:
    """Compute common classification metrics.

    Returns a dictionary with keys ``accuracy``, ``precision``, ``recall``,
    ``f1`` and optionally ``roc_auc`` and ``pr_auc`` if probabilities are
    provided.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted binary labels.
    y_prob : array-like, optional
        Predicted probabilities for positive class.

    Returns
    -------
    dict
        Dictionary with classification metrics.
    """
    if classification_report is None:
        raise ImportError("scikit-learn is required. Install with `pip install scikit-learn`.")

    report = classification_report(y_true, y_pred, output_dict=True)
    metrics = {
        "accuracy": report["accuracy"],
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"],
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics["roc_auc"] = None
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            metrics["pr_auc"] = auc(recall, precision)
        except Exception:
            metrics["pr_auc"] = None
    return metrics


def buy_and_hold(df_test: pd.DataFrame, capital_init: float = 1000.0, pct_capital: float = 0.1, fee_roundtrip: float = 0.002) -> Tuple[float, float, float, float]:
    """Stratégie Buy & Hold : acheter au début et vendre à la fin.

    Parameters
    ----------
    df_test : DataFrame
        DataFrame with OHLCV data, must include 'Close' column.
    capital_init : float, default=1000.0
        Initial capital.
    pct_capital : float, default=0.1
        Fraction of capital to invest.
    fee_roundtrip : float, default=0.002
        Roundtrip fee (e.g. 0.002 = 0.2%).

    Returns
    -------
    final_capital : float
        Final capital after buy and hold strategy.
    qty : float
        Quantity purchased.
    first_price : float
        First price (entry).
    last_price : float
        Last price (exit).
    """
    first_price = df_test["Close"].iloc[0]
    last_price = df_test["Close"].iloc[-1]

    # Achat au début
    invest_amount = pct_capital * capital_init
    buy_fees = fee_roundtrip * invest_amount / 2
    qty = (invest_amount - buy_fees) / first_price

    # Vente à la fin
    sell_value = qty * last_price
    sell_fees = fee_roundtrip * sell_value / 2
    final_capital = capital_init - invest_amount + sell_value - sell_fees

    return final_capital, qty, first_price, last_price


def calculate_roi_annualized(
    capital_init: float,
    capital_final: float,
    df: pd.DataFrame,
    date_col: str = "Timestamp",
) -> float:
    """Calculate annualized ROI percentage from capital and date range.

    Parameters
    ----------
    capital_init : float
        Initial capital.
    capital_final : float
        Final capital.
    df : DataFrame
        DataFrame with date information (must include Timestamp column or index).
    date_col : str, default="Timestamp"
        Column name containing dates.

    Returns
    -------
    float
        Annualized ROI as a percentage (e.g., 10.5 for 10.5%).
    """
    if len(df) == 0:
        return 0.0
    
    # Get start and end dates
    if date_col in df.columns:
        start_date = pd.to_datetime(df[date_col].iloc[0])
        end_date = pd.to_datetime(df[date_col].iloc[-1])
    elif hasattr(df.index, 'dtype') and pd.api.types.is_datetime64_any_dtype(df.index):
        start_date = pd.to_datetime(df.index[0])
        end_date = pd.to_datetime(df.index[-1])
    else:
        # If no Timestamp, use index as proxy (assume daily data)
        start_date = pd.Timestamp.now() - pd.Timedelta(days=len(df))
        end_date = pd.Timestamp.now()
    
    # Calculate number of days
    days = (end_date - start_date).days
    if days <= 0:
        days = 1  # Avoid division by zero
    
    # Calculate ROI
    roi = (capital_final - capital_init) / capital_init
    
    # Annualize: (1 + roi)^(365/days) - 1
    if roi <= -1:
        # If we lost more than 100%, return -100%
        return -100.0
    
    roi_annualized = ((1 + roi) ** (365.0 / days) - 1) * 100
    
    return roi_annualized

