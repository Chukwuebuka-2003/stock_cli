import json
import logging
from datetime import datetime, timedelta

from stock_cli.file_paths import HISTORY_PATH

logger = logging.getLogger(__name__)


class PortfolioHistory:
    def __init__(self, history_path=HISTORY_PATH):
        """
        Initializes the Portfolio History tracker.
        Args:
            history_path (str): The path to the history file.
        """
        self.history_path = history_path
        self.history = self.load_history()

    def load_history(self):
        """Load history from JSON file."""
        try:
            with open(self.history_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.info("History file not found. Starting with empty history.")
            return []
        except json.JSONDecodeError:
            logger.error("Error decoding history.json. Starting with empty history.")
            return []

    def save_history(self):
        """Save history to JSON file."""
        try:
            with open(self.history_path, "w") as f:
                json.dump(self.history, f, indent=4)
            logger.info(f"History saved successfully to {self.history_path}")
        except IOError as e:
            logger.error(f"Error saving history file: {e}")

    def add_snapshot(self, portfolio, data_fetcher):
        """
        Add a portfolio snapshot for today.
        Args:
            portfolio: Portfolio instance
            data_fetcher: DataFetcher instance
        """
        today = datetime.now().strftime("%Y-%m-%d")

        # Check if we already have a snapshot for today
        if self.history and self.history[-1].get("date") == today:
            logger.info(f"Snapshot for {today} already exists. Updating it.")
            self.history.pop()  # Remove old snapshot for today

        positions = portfolio.get_positions()
        total_value = 0
        total_cost = 0
        position_details = []

        for position in positions:
            symbol = position["symbol"]
            quantity = position["quantity"]
            purchase_price = position["purchase_price"]

            stock_data = data_fetcher.get_stock_data(symbol)
            if stock_data:
                current_price = stock_data["currentPrice"]
                value = current_price * quantity
                cost = purchase_price * quantity

                total_value += value
                total_cost += cost

                position_details.append({
                    "symbol": symbol,
                    "quantity": quantity,
                    "purchase_price": purchase_price,
                    "current_price": current_price,
                    "value": value,
                    "cost": cost
                })

        gain_loss = total_value - total_cost
        gain_loss_percent = (gain_loss / total_cost * 100) if total_cost > 0 else 0

        snapshot = {
            "date": today,
            "total_value": total_value,
            "total_cost": total_cost,
            "gain_loss": gain_loss,
            "gain_loss_percent": gain_loss_percent,
            "positions": position_details
        }

        self.history.append(snapshot)
        self.save_history()
        logger.info(f"Added snapshot for {today}: ${total_value:.2f}")

        return snapshot

    def get_performance(self, days=None):
        """
        Get performance metrics for a specific time period.
        Args:
            days (int): Number of days to look back. None for all-time.
        Returns:
            dict: Performance metrics including start value, end value, change, etc.
        """
        if not self.history:
            return None

        # Sort history by date to ensure chronological order
        sorted_history = sorted(self.history, key=lambda x: x["date"])

        if days is None:
            # All-time performance
            start_snapshot = sorted_history[0]
            end_snapshot = sorted_history[-1]
        else:
            # Performance for last N days
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

            # Filter snapshots >= cutoff_date
            filtered_history = [s for s in sorted_history if s["date"] >= cutoff_date]

            if not filtered_history:
                return None

            start_snapshot = filtered_history[0]
            end_snapshot = sorted_history[-1]

        start_value = start_snapshot["total_value"]
        end_value = end_snapshot["total_value"]
        value_change = end_value - start_value
        percent_change = (value_change / start_value * 100) if start_value > 0 else 0

        return {
            "start_date": start_snapshot["date"],
            "end_date": end_snapshot["date"],
            "start_value": start_value,
            "end_value": end_value,
            "value_change": value_change,
            "percent_change": percent_change,
            "days": days if days else len(sorted_history)
        }

    def get_latest_snapshot(self):
        """Get the most recent snapshot."""
        if not self.history:
            return None

        sorted_history = sorted(self.history, key=lambda x: x["date"])
        return sorted_history[-1]

    def get_snapshots_for_period(self, days):
        """
        Get all snapshots for the last N days.
        Args:
            days (int): Number of days to look back
        Returns:
            list: List of snapshots
        """
        if not self.history:
            return []

        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        sorted_history = sorted(self.history, key=lambda x: x["date"])

        return [s for s in sorted_history if s["date"] >= cutoff_date]

    def format_performance_report(self, period_name, performance):
        """
        Format a performance report as a string.
        Args:
            period_name (str): Name of the period (e.g., "7 Days", "All Time")
            performance (dict): Performance metrics
        Returns:
            str: Formatted report
        """
        if not performance:
            return f"{period_name}: No data available"

        report = f"\n{period_name} Performance:\n"
        report += f"  Period: {performance['start_date']} to {performance['end_date']}\n"
        report += f"  Start Value: ${performance['start_value']:,.2f}\n"
        report += f"  End Value: ${performance['end_value']:,.2f}\n"

        change_symbol = "+" if performance['value_change'] >= 0 else ""
        report += f"  Change: {change_symbol}${performance['value_change']:,.2f} "
        report += f"({change_symbol}{performance['percent_change']:.2f}%)\n"

        return report
