import json
import logging
from datetime import datetime

from stock_cli.file_paths import ALERTS_PATH

logger = logging.getLogger(__name__)


class PriceAlerts:
    def __init__(self, alerts_path=ALERTS_PATH):
        """
        Initializes the Price Alerts manager.
        Args:
            alerts_path (str): The path to the alerts file.
        """
        self.alerts_path = alerts_path
        self.alerts = self.load_alerts()

    def load_alerts(self):
        """Load alerts from JSON file."""
        try:
            with open(self.alerts_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.info("Alerts file not found. Starting with empty alerts.")
            return []
        except json.JSONDecodeError:
            logger.error("Error decoding alerts.json. Starting with empty alerts.")
            return []

    def save_alerts(self):
        """Save alerts to JSON file."""
        try:
            with open(self.alerts_path, "w") as f:
                json.dump(self.alerts, f, indent=4)
            logger.info(f"Alerts saved successfully to {self.alerts_path}")
        except IOError as e:
            logger.error(f"Error saving alerts file: {e}")

    def add_alert(self, symbol, above=None, below=None):
        """
        Add a price alert for a stock.
        Args:
            symbol (str): Stock symbol
            above (float): Trigger alert when price goes above this value
            below (float): Trigger alert when price goes below this value
        Returns:
            dict: The created alert
        """
        symbol = symbol.upper()

        if above is None and below is None:
            raise ValueError("At least one of 'above' or 'below' must be specified")

        alert = {
            "id": self._generate_alert_id(),
            "symbol": symbol,
            "above": above,
            "below": below,
            "created_at": datetime.now().isoformat(),
            "last_checked": None,
            "triggered": False
        }

        self.alerts.append(alert)
        self.save_alerts()
        logger.info(f"Added alert for {symbol}: above={above}, below={below}")

        return alert

    def remove_alert(self, alert_id):
        """
        Remove an alert by ID.
        Args:
            alert_id (str): Alert ID
        Returns:
            bool: True if removed, False if not found
        """
        original_count = len(self.alerts)
        self.alerts = [a for a in self.alerts if a["id"] != alert_id]

        if len(self.alerts) < original_count:
            self.save_alerts()
            logger.info(f"Removed alert with ID: {alert_id}")
            return True
        else:
            logger.warning(f"Alert not found with ID: {alert_id}")
            return False

    def remove_alerts_for_symbol(self, symbol):
        """
        Remove all alerts for a specific symbol.
        Args:
            symbol (str): Stock symbol
        Returns:
            int: Number of alerts removed
        """
        symbol = symbol.upper()
        original_count = len(self.alerts)
        self.alerts = [a for a in self.alerts if a["symbol"] != symbol]

        removed_count = original_count - len(self.alerts)
        if removed_count > 0:
            self.save_alerts()
            logger.info(f"Removed {removed_count} alert(s) for {symbol}")

        return removed_count

    def get_alerts(self, symbol=None, active_only=False):
        """
        Get alerts, optionally filtered by symbol and/or active status.
        Args:
            symbol (str): Optional stock symbol to filter by
            active_only (bool): Only return non-triggered alerts
        Returns:
            list: List of alerts
        """
        alerts = self.alerts

        if symbol:
            alerts = [a for a in alerts if a["symbol"] == symbol.upper()]

        if active_only:
            alerts = [a for a in alerts if not a["triggered"]]

        return alerts

    def check_alerts(self, data_fetcher):
        """
        Check all active alerts against current prices.
        Args:
            data_fetcher: DataFetcher instance
        Returns:
            list: List of triggered alerts
        """
        triggered_alerts = []
        active_alerts = self.get_alerts(active_only=True)

        for alert in active_alerts:
            symbol = alert["symbol"]
            stock_data = data_fetcher.get_stock_data(symbol)

            if not stock_data:
                logger.warning(f"Could not fetch data for {symbol}, skipping alert check")
                continue

            current_price = stock_data["currentPrice"]
            alert["last_checked"] = datetime.now().isoformat()

            triggered = False
            trigger_reason = None

            if alert["above"] is not None and current_price > alert["above"]:
                triggered = True
                trigger_reason = f"Price ${current_price:.2f} is above ${alert['above']:.2f}"

            if alert["below"] is not None and current_price < alert["below"]:
                triggered = True
                trigger_reason = f"Price ${current_price:.2f} is below ${alert['below']:.2f}"

            if triggered:
                alert["triggered"] = True
                alert["triggered_at"] = datetime.now().isoformat()
                alert["triggered_price"] = current_price
                alert["trigger_reason"] = trigger_reason

                triggered_alerts.append(alert)
                logger.info(f"Alert triggered for {symbol}: {trigger_reason}")

        if triggered_alerts:
            self.save_alerts()

        return triggered_alerts

    def _generate_alert_id(self):
        """Generate a unique alert ID."""
        if not self.alerts:
            return "alert_1"

        # Find the highest existing ID number
        max_id = 0
        for alert in self.alerts:
            try:
                alert_num = int(alert["id"].split("_")[1])
                max_id = max(max_id, alert_num)
            except (IndexError, ValueError):
                pass

        return f"alert_{max_id + 1}"

    def format_alert(self, alert):
        """
        Format an alert as a readable string.
        Args:
            alert (dict): Alert dictionary
        Returns:
            str: Formatted alert string
        """
        status = "TRIGGERED" if alert["triggered"] else "ACTIVE"
        conditions = []

        if alert["above"]:
            conditions.append(f"above ${alert['above']:.2f}")
        if alert["below"]:
            conditions.append(f"below ${alert['below']:.2f}")

        condition_str = " and ".join(conditions)

        output = f"[{status}] {alert['symbol']}: {condition_str} (ID: {alert['id']})"

        if alert["triggered"]:
            output += f"\n  Triggered at: {alert.get('triggered_at', 'N/A')}"
            output += f"\n  Trigger price: ${alert.get('triggered_price', 0):.2f}"
            output += f"\n  Reason: {alert.get('trigger_reason', 'N/A')}"

        return output
