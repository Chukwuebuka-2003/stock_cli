"""
Feature Engineering Module

Creates features for machine learning models from stock price data and technical indicators.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from src.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Engineer features for ML models from stock price data."""

    @staticmethod
    def create_price_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create price-based features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added price features
        """
        df = df.copy()

        try:
            # Price changes
            df["Price_Change"] = df["Close"].pct_change()
            df["Price_Change_5d"] = df["Close"].pct_change(periods=5)
            df["Price_Change_10d"] = df["Close"].pct_change(periods=10)
            df["Price_Change_20d"] = df["Close"].pct_change(periods=20)

            # Price momentum
            df["Momentum_5d"] = df["Close"] - df["Close"].shift(5)
            df["Momentum_10d"] = df["Close"] - df["Close"].shift(10)
            df["Momentum_20d"] = df["Close"] - df["Close"].shift(20)

            # Price volatility
            df["Volatility_5d"] = df["Close"].rolling(window=5).std()
            df["Volatility_10d"] = df["Close"].rolling(window=10).std()
            df["Volatility_20d"] = df["Close"].rolling(window=20).std()

            # High-Low spread
            df["HL_Spread"] = df["High"] - df["Low"]
            df["HL_Spread_Pct"] = (df["High"] - df["Low"]) / df["Close"]

            # Close position in daily range
            df["Close_Position"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"])

            # Volume features
            df["Volume_Change"] = df["Volume"].pct_change()
            df["Volume_MA_5"] = df["Volume"].rolling(window=5).mean()
            df["Volume_MA_20"] = df["Volume"].rolling(window=20).mean()
            df["Volume_Ratio"] = df["Volume"] / df["Volume_MA_20"]

            logger.info("Price features created successfully")
            return df

        except Exception as e:
            logger.error(f"Error creating price features: {e}")
            return df

    @staticmethod
    def create_technical_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from technical indicators.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with technical indicator features
        """
        df = df.copy()

        try:
            # Add all technical indicators
            df = TechnicalIndicators.add_all_indicators(df)

            # RSI-based features
            if "RSI" in df.columns:
                df["RSI_Overbought"] = (df["RSI"] > 70).astype(int)
                df["RSI_Oversold"] = (df["RSI"] < 30).astype(int)
                df["RSI_Change"] = df["RSI"].diff()

            # MACD-based features
            if "MACD" in df.columns and "MACD_Signal" in df.columns:
                df["MACD_Diff"] = df["MACD"] - df["MACD_Signal"]
                df["MACD_Cross_Up"] = ((df["MACD"] > df["MACD_Signal"]) &
                                       (df["MACD"].shift(1) <= df["MACD_Signal"].shift(1))).astype(int)
                df["MACD_Cross_Down"] = ((df["MACD"] < df["MACD_Signal"]) &
                                         (df["MACD"].shift(1) >= df["MACD_Signal"].shift(1))).astype(int)

            # Bollinger Bands features
            if all(col in df.columns for col in ["BB_Upper", "BB_Lower", "Close"]):
                df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
                df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])
                df["BB_Upper_Break"] = (df["Close"] > df["BB_Upper"]).astype(int)
                df["BB_Lower_Break"] = (df["Close"] < df["BB_Lower"]).astype(int)

            # Moving average crossovers
            if "SMA_20" in df.columns and "SMA_50" in df.columns:
                df["MA_20_50_Diff"] = df["SMA_20"] - df["SMA_50"]
                df["MA_Cross_Up"] = ((df["SMA_20"] > df["SMA_50"]) &
                                     (df["SMA_20"].shift(1) <= df["SMA_50"].shift(1))).astype(int)
                df["MA_Cross_Down"] = ((df["SMA_20"] < df["SMA_50"]) &
                                       (df["SMA_20"].shift(1) >= df["SMA_50"].shift(1))).astype(int)

            logger.info("Technical features created successfully")
            return df

        except Exception as e:
            logger.error(f"Error creating technical features: {e}")
            return df

    @staticmethod
    def create_lag_features(df: pd.DataFrame, lags: List[int] = None) -> pd.DataFrame:
        """
        Create lagged features for time series.

        Args:
            df: DataFrame with features
            lags: List of lag periods (default: [1, 2, 3, 5, 10])

        Returns:
            DataFrame with lagged features
        """
        if lags is None:
            lags = [1, 2, 3, 5, 10]

        df = df.copy()

        try:
            # Create lags for Close price
            for lag in lags:
                df[f"Close_Lag_{lag}"] = df["Close"].shift(lag)
                df[f"Volume_Lag_{lag}"] = df["Volume"].shift(lag)
                df[f"Returns_Lag_{lag}"] = df["Price_Change"].shift(lag)

            logger.info(f"Lag features created for lags: {lags}")
            return df

        except Exception as e:
            logger.error(f"Error creating lag features: {e}")
            return df

    @staticmethod
    def create_target_variable(df: pd.DataFrame, horizon: int = 1,
                             target_type: str = "price") -> pd.DataFrame:
        """
        Create target variable for prediction.

        Args:
            df: DataFrame with price data
            horizon: Number of periods ahead to predict
            target_type: Type of target - "price", "returns", or "direction"

        Returns:
            DataFrame with target variable
        """
        df = df.copy()

        try:
            if target_type == "price":
                # Predict future price
                df["Target"] = df["Close"].shift(-horizon)

            elif target_type == "returns":
                # Predict future returns
                df["Target"] = df["Close"].pct_change(periods=horizon).shift(-horizon)

            elif target_type == "direction":
                # Predict price direction (1 = up, 0 = down)
                future_returns = df["Close"].pct_change(periods=horizon).shift(-horizon)
                df["Target"] = (future_returns > 0).astype(int)

            else:
                raise ValueError(f"Unknown target_type: {target_type}")

            logger.info(f"Target variable created: {target_type}, horizon: {horizon}")
            return df

        except Exception as e:
            logger.error(f"Error creating target variable: {e}")
            return df

    @staticmethod
    def prepare_data_for_ml(df: pd.DataFrame, horizon: int = 5,
                           target_type: str = "direction") -> tuple:
        """
        Prepare complete feature set for machine learning.

        Args:
            df: DataFrame with OHLCV data
            horizon: Prediction horizon in days
            target_type: Type of prediction target

        Returns:
            Tuple of (features_df, target_series, feature_names)
        """
        try:
            # Create all features
            df = FeatureEngineer.create_price_features(df)
            df = FeatureEngineer.create_technical_features(df)
            df = FeatureEngineer.create_lag_features(df)

            # Create target variable
            df = FeatureEngineer.create_target_variable(df, horizon, target_type)

            # Define feature columns (exclude non-feature columns)
            exclude_cols = ["Date", "Open", "High", "Low", "Close", "Volume",
                          "Target", "Dividends", "Stock Splits"]
            feature_cols = [col for col in df.columns if col not in exclude_cols]

            # Remove rows with NaN values
            df_clean = df[feature_cols + ["Target"]].dropna()

            X = df_clean[feature_cols]
            y = df_clean["Target"]

            logger.info(f"Prepared {len(X)} samples with {len(feature_cols)} features")
            logger.info(f"Feature names: {feature_cols[:10]}... (showing first 10)")

            return X, y, feature_cols

        except Exception as e:
            logger.error(f"Error preparing data for ML: {e}")
            return None, None, None

    @staticmethod
    def get_feature_importance_summary(feature_names: List[str],
                                      importances: np.ndarray,
                                      top_n: int = 20) -> pd.DataFrame:
        """
        Get summary of feature importances.

        Args:
            feature_names: List of feature names
            importances: Array of feature importance values
            top_n: Number of top features to return

        Returns:
            DataFrame with feature names and importances
        """
        try:
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            })

            importance_df = importance_df.sort_values("Importance", ascending=False)
            importance_df = importance_df.head(top_n)

            logger.info(f"Top {top_n} features extracted")
            return importance_df

        except Exception as e:
            logger.error(f"Error getting feature importance summary: {e}")
            return pd.DataFrame()

    @staticmethod
    def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> tuple:
        """
        Scale features using StandardScaler.

        Args:
            X_train: Training features
            X_test: Test features (optional)

        Returns:
            Tuple of (X_train_scaled, X_test_scaled, scaler)
        """
        try:
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )

            X_test_scaled = None
            if X_test is not None:
                X_test_scaled = pd.DataFrame(
                    scaler.transform(X_test),
                    columns=X_test.columns,
                    index=X_test.index
                )

            logger.info("Features scaled successfully")
            return X_train_scaled, X_test_scaled, scaler

        except Exception as e:
            logger.error(f"Error scaling features: {e}")
            return X_train, X_test, None
