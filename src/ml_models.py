"""
Machine Learning Models Module

Implements various ML models for stock price prediction:
- Prophet for time series forecasting
- Random Forest for classification/regression
- XGBoost for gradient boosting
"""

import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                            mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from src.feature_engineering import FeatureEngineer

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ProphetPredictor:
    """Time series forecasting using Facebook Prophet."""

    def __init__(self):
        self.model = None
        self.symbol = None

    def train(self, df: pd.DataFrame, symbol: str) -> None:
        """
        Train Prophet model on historical data.

        Args:
            df: DataFrame with Date and Close columns
            symbol: Stock symbol
        """
        try:
            self.symbol = symbol

            # Prepare data for Prophet (requires 'ds' and 'y' columns)
            prophet_df = pd.DataFrame({
                'ds': pd.to_datetime(df['Date']),
                'y': df['Close']
            })

            # Initialize and fit model
            self.model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )

            logger.info(f"Training Prophet model for {symbol}...")
            self.model.fit(prophet_df)
            logger.info(f"Prophet model trained successfully for {symbol}")

        except Exception as e:
            logger.error(f"Error training Prophet model: {e}")
            raise

    def predict(self, periods: int = 30) -> pd.DataFrame:
        """
        Make predictions for future periods.

        Args:
            periods: Number of days to forecast

        Returns:
            DataFrame with predictions
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained. Call train() first.")

            # Create future dataframe
            future = self.model.make_future_dataframe(periods=periods)

            # Make predictions
            forecast = self.model.predict(future)

            # Extract relevant columns
            result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
            result.columns = ['Date', 'Predicted_Price', 'Lower_Bound', 'Upper_Bound']

            logger.info(f"Generated {periods} day forecast for {self.symbol}")
            return result

        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return pd.DataFrame()

    def get_forecast_plot_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get data for plotting forecast.

        Returns:
            Tuple of (historical_df, forecast_df)
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained")

            future = self.model.make_future_dataframe(periods=30)
            forecast = self.model.predict(future)

            return forecast, self.model

        except Exception as e:
            logger.error(f"Error getting forecast plot data: {e}")
            return pd.DataFrame(), None


class MLPredictor:
    """Machine learning predictor using Random Forest and XGBoost."""

    def __init__(self, model_type: str = "random_forest", task: str = "classification"):
        """
        Initialize ML predictor.

        Args:
            model_type: "random_forest" or "xgboost"
            task: "classification" or "regression"
        """
        self.model_type = model_type
        self.task = task
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.symbol = None

        # Initialize model
        if model_type == "random_forest":
            if task == "classification":
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                self.model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
        elif model_type == "xgboost":
            if task == "classification":
                self.model = XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                self.model = XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1
                )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def train(self, df: pd.DataFrame, symbol: str, horizon: int = 5,
             test_size: float = 0.2) -> Dict:
        """
        Train ML model.

        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol
            horizon: Prediction horizon in days
            test_size: Fraction of data for testing

        Returns:
            Dictionary with training metrics
        """
        try:
            self.symbol = symbol
            logger.info(f"Training {self.model_type} {self.task} model for {symbol}...")

            # Prepare features and target
            target_type = "direction" if self.task == "classification" else "returns"
            X, y, feature_names = FeatureEngineer.prepare_data_for_ml(
                df, horizon=horizon, target_type=target_type
            )

            if X is None or len(X) < 50:
                raise ValueError("Insufficient data for training")

            self.feature_names = feature_names

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=False
            )

            # Scale features
            X_train_scaled, X_test_scaled, scaler = FeatureEngineer.scale_features(
                X_train, X_test
            )
            self.scaler = scaler

            # Train model
            self.model.fit(X_train_scaled, y_train)

            # Evaluate
            y_pred = self.model.predict(X_test_scaled)

            metrics = self._calculate_metrics(y_test, y_pred)
            metrics['train_size'] = len(X_train)
            metrics['test_size'] = len(X_test)
            metrics['num_features'] = len(feature_names)

            logger.info(f"Model trained successfully. Test accuracy/R2: {metrics.get('accuracy', metrics.get('r2', 0)):.4f}")

            return metrics

        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise

    def predict(self, df: pd.DataFrame, horizon: int = 5) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            df: DataFrame with OHLCV data
            horizon: Prediction horizon

        Returns:
            Array of predictions
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained. Call train() first.")

            # Prepare features
            target_type = "direction" if self.task == "classification" else "returns"
            X, _, _ = FeatureEngineer.prepare_data_for_ml(
                df, horizon=horizon, target_type=target_type
            )

            if X is None:
                raise ValueError("Failed to prepare features")

            # Scale features
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )

            # Make predictions
            predictions = self.model.predict(X_scaled)

            return predictions

        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return np.array([])

    def predict_proba(self, df: pd.DataFrame, horizon: int = 5) -> np.ndarray:
        """
        Get prediction probabilities (for classification only).

        Args:
            df: DataFrame with OHLCV data
            horizon: Prediction horizon

        Returns:
            Array of prediction probabilities
        """
        try:
            if self.task != "classification":
                raise ValueError("predict_proba only available for classification")

            if self.model is None:
                raise ValueError("Model not trained")

            # Prepare features
            X, _, _ = FeatureEngineer.prepare_data_for_ml(
                df, horizon=horizon, target_type="direction"
            )

            if X is None:
                raise ValueError("Failed to prepare features")

            # Scale features
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )

            # Get probabilities
            probas = self.model.predict_proba(X_scaled)

            return probas

        except Exception as e:
            logger.error(f"Error getting prediction probabilities: {e}")
            return np.array([])

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from trained model.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importances
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained")

            importances = self.model.feature_importances_

            return FeatureEngineer.get_feature_importance_summary(
                self.feature_names, importances, top_n
            )

        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return pd.DataFrame()

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate evaluation metrics based on task type."""
        metrics = {}

        try:
            if self.task == "classification":
                metrics['accuracy'] = accuracy_score(y_true, y_pred)
                metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

                # Classification report
                report = classification_report(y_true, y_pred, output_dict=True)
                metrics['precision'] = report['weighted avg']['precision']
                metrics['recall'] = report['weighted avg']['recall']
                metrics['f1_score'] = report['weighted avg']['f1-score']

            else:  # regression
                metrics['mae'] = mean_absolute_error(y_true, y_pred)
                metrics['mse'] = mean_squared_error(y_true, y_pred)
                metrics['rmse'] = np.sqrt(metrics['mse'])
                metrics['r2'] = r2_score(y_true, y_pred)

        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")

        return metrics

    def save_model(self, filepath: str) -> None:
        """Save trained model to disk."""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'model_type': self.model_type,
                'task': self.task,
                'symbol': self.symbol
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load_model(self, filepath: str) -> None:
        """Load trained model from disk."""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.model_type = model_data['model_type']
            self.task = model_data['task']
            self.symbol = model_data['symbol']
            logger.info(f"Model loaded from {filepath}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise


def train_ensemble_models(df: pd.DataFrame, symbol: str, horizon: int = 5) -> Dict:
    """
    Train multiple models and return their predictions.

    Args:
        df: DataFrame with OHLCV data
        symbol: Stock symbol
        horizon: Prediction horizon

    Returns:
        Dictionary with trained models and metrics
    """
    try:
        logger.info(f"Training ensemble models for {symbol}...")

        results = {}

        # Train Random Forest Classifier
        rf_clf = MLPredictor("random_forest", "classification")
        rf_clf_metrics = rf_clf.train(df, symbol, horizon)
        results['rf_classifier'] = {
            'model': rf_clf,
            'metrics': rf_clf_metrics
        }

        # Train XGBoost Classifier
        xgb_clf = MLPredictor("xgboost", "classification")
        xgb_clf_metrics = xgb_clf.train(df, symbol, horizon)
        results['xgb_classifier'] = {
            'model': xgb_clf,
            'metrics': xgb_clf_metrics
        }

        # Train Prophet
        prophet = ProphetPredictor()
        prophet.train(df, symbol)
        results['prophet'] = {
            'model': prophet,
            'metrics': {'type': 'time_series'}
        }

        logger.info(f"Ensemble models trained successfully for {symbol}")

        return results

    except Exception as e:
        logger.error(f"Error training ensemble models: {e}")
        return {}
