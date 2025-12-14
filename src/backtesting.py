"""
Backtesting Module

Provides backtesting capabilities for trading strategies and ML predictions.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Backtester:
    """Backtest trading strategies on historical data."""

    def __init__(self, initial_capital: float = 100000.0):
        """
        Initialize backtester.

        Args:
            initial_capital: Starting capital for backtesting
        """
        self.initial_capital = initial_capital
        self.trades = []
        self.equity_curve = []

    def run_strategy(self, df: pd.DataFrame, signals: pd.Series,
                    commission: float = 0.001) -> Dict:
        """
        Run backtest on a trading strategy.

        Args:
            df: DataFrame with OHLCV data and Date column
            signals: Series with trading signals (1=buy, 0=hold, -1=sell)
            commission: Commission per trade (default: 0.1%)

        Returns:
            Dictionary with backtest results
        """
        try:
            logger.info("Running backtest...")

            # Initialize tracking variables
            cash = self.initial_capital
            position = 0  # Number of shares held
            equity = self.initial_capital
            trades = []
            equity_curve = []

            # Iterate through data
            for idx in range(len(df)):
                date = df.iloc[idx]['Date'] if 'Date' in df.columns else idx
                close = df.iloc[idx]['Close']
                signal = signals.iloc[idx] if idx < len(signals) else 0

                # Calculate current equity
                equity = cash + (position * close)

                # Execute trades based on signals
                if signal == 1 and position == 0:  # Buy signal
                    # Buy as many shares as possible
                    shares_to_buy = int(cash / (close * (1 + commission)))
                    if shares_to_buy > 0:
                        cost = shares_to_buy * close * (1 + commission)
                        cash -= cost
                        position += shares_to_buy

                        trades.append({
                            'date': date,
                            'action': 'BUY',
                            'shares': shares_to_buy,
                            'price': close,
                            'cost': cost,
                            'equity': equity
                        })

                elif signal == -1 and position > 0:  # Sell signal
                    # Sell all shares
                    proceeds = position * close * (1 - commission)
                    cash += proceeds
                    position = 0

                    trades.append({
                        'date': date,
                        'action': 'SELL',
                        'shares': position,
                        'price': close,
                        'proceeds': proceeds,
                        'equity': equity
                    })

                # Record equity
                equity_curve.append({
                    'date': date,
                    'equity': equity,
                    'cash': cash,
                    'position_value': position * close
                })

            # Close any open positions at the end
            if position > 0:
                final_price = df.iloc[-1]['Close']
                proceeds = position * final_price * (1 - commission)
                cash += proceeds
                position = 0

            final_equity = cash

            # Calculate metrics
            metrics = self._calculate_metrics(
                pd.DataFrame(equity_curve),
                initial_capital=self.initial_capital,
                final_equity=final_equity,
                trades=trades
            )

            self.trades = trades
            self.equity_curve = equity_curve

            logger.info(f"Backtest completed. Final equity: ${final_equity:,.2f}")

            return {
                'metrics': metrics,
                'trades': trades,
                'equity_curve': equity_curve
            }

        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {}

    def run_prediction_backtest(self, df: pd.DataFrame,
                               predictions: np.ndarray,
                               threshold: float = 0.5) -> Dict:
        """
        Backtest ML model predictions.

        Args:
            df: DataFrame with OHLCV data
            predictions: Array of predictions (probabilities or binary)
            threshold: Threshold for classification (default: 0.5)

        Returns:
            Dictionary with backtest results
        """
        try:
            # Convert predictions to signals
            signals = pd.Series(0, index=range(len(predictions)))

            if len(predictions.shape) > 1:  # Probabilities
                # Use probability of upward movement
                up_prob = predictions[:, 1] if predictions.shape[1] == 2 else predictions[:, 0]
                signals = pd.Series(np.where(up_prob > threshold, 1, -1))
            else:  # Binary predictions
                signals = pd.Series(np.where(predictions > 0.5, 1, -1))

            return self.run_strategy(df, signals)

        except Exception as e:
            logger.error(f"Error running prediction backtest: {e}")
            return {}

    def _calculate_metrics(self, equity_df: pd.DataFrame,
                          initial_capital: float,
                          final_equity: float,
                          trades: List[Dict]) -> Dict:
        """Calculate performance metrics."""
        metrics = {}

        try:
            # Basic metrics
            total_return = ((final_equity - initial_capital) / initial_capital) * 100
            metrics['initial_capital'] = initial_capital
            metrics['final_equity'] = final_equity
            metrics['total_return_pct'] = total_return
            metrics['total_return'] = final_equity - initial_capital

            # Number of trades
            metrics['num_trades'] = len(trades)
            buy_trades = [t for t in trades if t['action'] == 'BUY']
            sell_trades = [t for t in trades if t['action'] == 'SELL']
            metrics['num_buy_trades'] = len(buy_trades)
            metrics['num_sell_trades'] = len(sell_trades)

            # Equity curve statistics
            if not equity_df.empty and 'equity' in equity_df.columns:
                equity_series = equity_df['equity']

                # Max drawdown
                running_max = equity_series.expanding().max()
                drawdown = (equity_series - running_max) / running_max * 100
                metrics['max_drawdown_pct'] = drawdown.min()

                # Calculate daily returns
                returns = equity_series.pct_change().dropna()

                if len(returns) > 0:
                    # Sharpe ratio (assuming 252 trading days per year, 0% risk-free rate)
                    metrics['sharpe_ratio'] = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

                    # Sortino ratio (downside deviation)
                    downside_returns = returns[returns < 0]
                    if len(downside_returns) > 0:
                        downside_std = downside_returns.std()
                        metrics['sortino_ratio'] = (returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0
                    else:
                        metrics['sortino_ratio'] = 0

                    # Win rate
                    win_count = (returns > 0).sum()
                    total_count = len(returns)
                    metrics['win_rate_pct'] = (win_count / total_count) * 100 if total_count > 0 else 0

                    # Average win/loss
                    wins = returns[returns > 0]
                    losses = returns[returns < 0]
                    metrics['avg_win_pct'] = wins.mean() * 100 if len(wins) > 0 else 0
                    metrics['avg_loss_pct'] = losses.mean() * 100 if len(losses) > 0 else 0

                    # Profit factor
                    total_wins = wins.sum()
                    total_losses = abs(losses.sum())
                    metrics['profit_factor'] = total_wins / total_losses if total_losses > 0 else 0

        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")

        return metrics

    def compare_with_buy_and_hold(self, df: pd.DataFrame,
                                  strategy_results: Dict) -> Dict:
        """
        Compare strategy performance with buy-and-hold.

        Args:
            df: DataFrame with price data
            strategy_results: Results from run_strategy()

        Returns:
            Dictionary with comparison metrics
        """
        try:
            # Calculate buy-and-hold returns
            initial_price = df.iloc[0]['Close']
            final_price = df.iloc[-1]['Close']
            bh_return_pct = ((final_price - initial_price) / initial_price) * 100

            # Calculate equity for buy-and-hold
            shares = self.initial_capital / initial_price
            bh_final_equity = shares * final_price

            strategy_return_pct = strategy_results['metrics']['total_return_pct']

            comparison = {
                'buy_and_hold': {
                    'return_pct': bh_return_pct,
                    'final_equity': bh_final_equity
                },
                'strategy': {
                    'return_pct': strategy_return_pct,
                    'final_equity': strategy_results['metrics']['final_equity']
                },
                'outperformance_pct': strategy_return_pct - bh_return_pct
            }

            logger.info(f"Strategy vs Buy-and-Hold: {comparison['outperformance_pct']:+.2f}%")

            return comparison

        except Exception as e:
            logger.error(f"Error comparing with buy-and-hold: {e}")
            return {}


class WalkForwardAnalysis:
    """Perform walk-forward analysis for model validation."""

    def __init__(self, train_window: int = 252, test_window: int = 63):
        """
        Initialize walk-forward analysis.

        Args:
            train_window: Number of periods for training (default: 1 year)
            test_window: Number of periods for testing (default: 3 months)
        """
        self.train_window = train_window
        self.test_window = test_window

    def run_analysis(self, df: pd.DataFrame, model_func,
                    horizon: int = 5) -> List[Dict]:
        """
        Run walk-forward analysis.

        Args:
            df: DataFrame with OHLCV data
            model_func: Function that trains and returns a model
            horizon: Prediction horizon

        Returns:
            List of dictionaries with results for each window
        """
        results = []

        try:
            total_len = len(df)
            current_idx = self.train_window

            while current_idx + self.test_window <= total_len:
                # Split data
                train_df = df.iloc[current_idx - self.train_window:current_idx]
                test_df = df.iloc[current_idx:current_idx + self.test_window]

                # Train model
                logger.info(f"Walk-forward window {len(results) + 1}: Train on {len(train_df)} samples, test on {len(test_df)} samples")

                try:
                    model = model_func(train_df)

                    # Make predictions
                    predictions = model.predict(test_df, horizon)

                    # Backtest
                    backtester = Backtester()
                    backtest_results = backtester.run_prediction_backtest(
                        test_df, predictions
                    )

                    results.append({
                        'train_start': train_df.iloc[0]['Date'] if 'Date' in train_df.columns else 0,
                        'train_end': train_df.iloc[-1]['Date'] if 'Date' in train_df.columns else current_idx,
                        'test_start': test_df.iloc[0]['Date'] if 'Date' in test_df.columns else current_idx,
                        'test_end': test_df.iloc[-1]['Date'] if 'Date' in test_df.columns else current_idx + self.test_window,
                        'metrics': backtest_results.get('metrics', {}),
                        'num_predictions': len(predictions)
                    })

                except Exception as e:
                    logger.warning(f"Error in walk-forward window: {e}")

                # Move to next window
                current_idx += self.test_window

            logger.info(f"Walk-forward analysis completed with {len(results)} windows")

        except Exception as e:
            logger.error(f"Error running walk-forward analysis: {e}")

        return results

    def aggregate_results(self, results: List[Dict]) -> Dict:
        """
        Aggregate results from multiple walk-forward windows.

        Args:
            results: List of results from run_analysis()

        Returns:
            Dictionary with aggregated metrics
        """
        try:
            if not results:
                return {}

            # Extract metrics from each window
            returns = [r['metrics'].get('total_return_pct', 0) for r in results]
            sharpe_ratios = [r['metrics'].get('sharpe_ratio', 0) for r in results if 'sharpe_ratio' in r['metrics']]
            max_drawdowns = [r['metrics'].get('max_drawdown_pct', 0) for r in results if 'max_drawdown_pct' in r['metrics']]

            aggregated = {
                'num_windows': len(results),
                'avg_return_pct': np.mean(returns) if returns else 0,
                'std_return_pct': np.std(returns) if returns else 0,
                'min_return_pct': np.min(returns) if returns else 0,
                'max_return_pct': np.max(returns) if returns else 0,
                'avg_sharpe_ratio': np.mean(sharpe_ratios) if sharpe_ratios else 0,
                'avg_max_drawdown_pct': np.mean(max_drawdowns) if max_drawdowns else 0,
                'win_rate_pct': (sum(1 for r in returns if r > 0) / len(returns)) * 100 if returns else 0
            }

            logger.info(f"Aggregated results: Avg return {aggregated['avg_return_pct']:.2f}%, Win rate {aggregated['win_rate_pct']:.1f}%")

            return aggregated

        except Exception as e:
            logger.error(f"Error aggregating results: {e}")
            return {}
