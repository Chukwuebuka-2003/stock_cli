"""
Technical Indicators Module

Provides calculation of various technical indicators for stock analysis:
- Moving Averages (SMA, EMA)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Stochastic Oscillator
- ATR (Average True Range)
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Calculate technical indicators for stock price data."""

    @staticmethod
    def calculate_sma(df: pd.DataFrame, period: int = 20, price_col: str = "Close") -> pd.Series:
        """
        Calculate Simple Moving Average.

        Args:
            df: DataFrame with price data
            period: Number of periods for SMA
            price_col: Column name for price data

        Returns:
            Series with SMA values
        """
        try:
            return df[price_col].rolling(window=period).mean()
        except Exception as e:
            logger.error(f"Error calculating SMA: {e}")
            return pd.Series([None] * len(df))

    @staticmethod
    def calculate_ema(df: pd.DataFrame, period: int = 20, price_col: str = "Close") -> pd.Series:
        """
        Calculate Exponential Moving Average.

        Args:
            df: DataFrame with price data
            period: Number of periods for EMA
            price_col: Column name for price data

        Returns:
            Series with EMA values
        """
        try:
            return df[price_col].ewm(span=period, adjust=False).mean()
        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            return pd.Series([None] * len(df))

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14, price_col: str = "Close") -> pd.Series:
        """
        Calculate Relative Strength Index.

        Args:
            df: DataFrame with price data
            period: Number of periods for RSI (default: 14)
            price_col: Column name for price data

        Returns:
            Series with RSI values (0-100)
        """
        try:
            delta = df[price_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series([None] * len(df))

    @staticmethod
    def calculate_macd(
        df: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        price_col: str = "Close"
    ) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            df: DataFrame with price data
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line period (default: 9)
            price_col: Column name for price data

        Returns:
            Dictionary with 'macd', 'signal', and 'histogram' Series
        """
        try:
            ema_fast = df[price_col].ewm(span=fast_period, adjust=False).mean()
            ema_slow = df[price_col].ewm(span=slow_period, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            histogram = macd_line - signal_line

            return {
                "macd": macd_line,
                "signal": signal_line,
                "histogram": histogram
            }
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return {
                "macd": pd.Series([None] * len(df)),
                "signal": pd.Series([None] * len(df)),
                "histogram": pd.Series([None] * len(df))
            }

    @staticmethod
    def calculate_bollinger_bands(
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0,
        price_col: str = "Close"
    ) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands.

        Args:
            df: DataFrame with price data
            period: Number of periods for moving average (default: 20)
            std_dev: Number of standard deviations (default: 2.0)
            price_col: Column name for price data

        Returns:
            Dictionary with 'upper', 'middle', and 'lower' bands
        """
        try:
            middle_band = df[price_col].rolling(window=period).mean()
            std = df[price_col].rolling(window=period).std()
            upper_band = middle_band + (std * std_dev)
            lower_band = middle_band - (std * std_dev)

            return {
                "upper": upper_band,
                "middle": middle_band,
                "lower": lower_band
            }
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return {
                "upper": pd.Series([None] * len(df)),
                "middle": pd.Series([None] * len(df)),
                "lower": pd.Series([None] * len(df))
            }

    @staticmethod
    def calculate_stochastic(
        df: pd.DataFrame,
        period: int = 14,
        smooth_period: int = 3
    ) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator.

        Args:
            df: DataFrame with High, Low, Close columns
            period: Number of periods (default: 14)
            smooth_period: Smoothing period (default: 3)

        Returns:
            Dictionary with '%K' and '%D' lines
        """
        try:
            low_min = df["Low"].rolling(window=period).min()
            high_max = df["High"].rolling(window=period).max()
            k = 100 * ((df["Close"] - low_min) / (high_max - low_min))
            d = k.rolling(window=smooth_period).mean()

            return {
                "K": k,
                "D": d
            }
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {e}")
            return {
                "K": pd.Series([None] * len(df)),
                "D": pd.Series([None] * len(df))
            }

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range.

        Args:
            df: DataFrame with High, Low, Close columns
            period: Number of periods (default: 14)

        Returns:
            Series with ATR values
        """
        try:
            high_low = df["High"] - df["Low"]
            high_close = np.abs(df["High"] - df["Close"].shift())
            low_close = np.abs(df["Low"] - df["Close"].shift())

            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(period).mean()

            return atr
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return pd.Series([None] * len(df))

    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to a DataFrame.

        Args:
            df: DataFrame with OHLCV data (Open, High, Low, Close, Volume)

        Returns:
            DataFrame with all indicators added as new columns
        """
        df = df.copy()

        # Moving Averages
        df["SMA_20"] = TechnicalIndicators.calculate_sma(df, 20)
        df["SMA_50"] = TechnicalIndicators.calculate_sma(df, 50)
        df["SMA_200"] = TechnicalIndicators.calculate_sma(df, 200)
        df["EMA_12"] = TechnicalIndicators.calculate_ema(df, 12)
        df["EMA_26"] = TechnicalIndicators.calculate_ema(df, 26)

        # RSI
        df["RSI"] = TechnicalIndicators.calculate_rsi(df)

        # MACD
        macd = TechnicalIndicators.calculate_macd(df)
        df["MACD"] = macd["macd"]
        df["MACD_Signal"] = macd["signal"]
        df["MACD_Histogram"] = macd["histogram"]

        # Bollinger Bands
        bb = TechnicalIndicators.calculate_bollinger_bands(df)
        df["BB_Upper"] = bb["upper"]
        df["BB_Middle"] = bb["middle"]
        df["BB_Lower"] = bb["lower"]

        # Stochastic (if High, Low columns exist)
        if "High" in df.columns and "Low" in df.columns:
            stoch = TechnicalIndicators.calculate_stochastic(df)
            df["Stoch_K"] = stoch["K"]
            df["Stoch_D"] = stoch["D"]

            # ATR
            df["ATR"] = TechnicalIndicators.calculate_atr(df)

        logger.info("All technical indicators calculated successfully")
        return df

    @staticmethod
    def get_indicator_signals(df: pd.DataFrame) -> Dict[str, str]:
        """
        Generate trading signals based on technical indicators.

        Args:
            df: DataFrame with calculated indicators

        Returns:
            Dictionary with indicator signals (bullish, bearish, neutral)
        """
        if len(df) == 0:
            return {}

        signals = {}
        latest = df.iloc[-1]

        # RSI Signal
        if "RSI" in df.columns and pd.notna(latest["RSI"]):
            if latest["RSI"] > 70:
                signals["RSI"] = "Overbought (>70)"
            elif latest["RSI"] < 30:
                signals["RSI"] = "Oversold (<30)"
            else:
                signals["RSI"] = f"Neutral ({latest['RSI']:.1f})"

        # MACD Signal
        if "MACD" in df.columns and "MACD_Signal" in df.columns:
            if pd.notna(latest["MACD"]) and pd.notna(latest["MACD_Signal"]):
                if latest["MACD"] > latest["MACD_Signal"]:
                    signals["MACD"] = "Bullish (MACD > Signal)"
                else:
                    signals["MACD"] = "Bearish (MACD < Signal)"

        # Moving Average Signal (Price vs SMA_50)
        if "SMA_50" in df.columns and "Close" in df.columns:
            if pd.notna(latest["SMA_50"]):
                if latest["Close"] > latest["SMA_50"]:
                    signals["MA_Trend"] = "Bullish (Price > SMA50)"
                else:
                    signals["MA_Trend"] = "Bearish (Price < SMA50)"

        # Bollinger Bands Signal
        if all(col in df.columns for col in ["BB_Upper", "BB_Lower", "Close"]):
            if pd.notna(latest["BB_Upper"]) and pd.notna(latest["BB_Lower"]):
                if latest["Close"] >= latest["BB_Upper"]:
                    signals["BB"] = "Overbought (At Upper Band)"
                elif latest["Close"] <= latest["BB_Lower"]:
                    signals["BB"] = "Oversold (At Lower Band)"
                else:
                    signals["BB"] = "Neutral (Within Bands)"

        return signals
