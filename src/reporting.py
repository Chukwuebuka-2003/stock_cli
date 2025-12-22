import html
import logging
import smtplib
import socket
import ssl
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def is_safe_url(url: str) -> bool:
    """
    Validate that a URL is safe (http/https scheme only).
    Prevents javascript:, data:, vbscript: and other dangerous URL schemes.
    """
    if not url:
        return False
    try:
        parsed = urlparse(url)
        return parsed.scheme in ('http', 'https')
    except Exception:
        return False


def sanitize_url_for_html(url: str) -> str:
    """
    Return the HTML-escaped URL if it's safe, otherwise return '#' as a safe fallback.
    """
    if is_safe_url(url):
        return html.escape(url)
    return '#'


class Reporting:
    def __init__(self, config):
        self.config = config

    def _get_report_data(self, portfolio, data_fetcher):
        """Helper function to gather and calculate all report data."""
        positions_data = []
        total_value = 0
        total_cost = 0

        for position in portfolio.get_positions():
            data = data_fetcher.get_stock_data(position["symbol"])
            if data:
                current_price = data.get("currentPrice", 0)
                market_value = current_price * position["quantity"]
                purchase_cost = position["purchase_price"] * position["quantity"]
                gain_loss = market_value - purchase_cost
                total_value += market_value
                total_cost += purchase_cost

                positions_data.append(
                    {
                        "symbol": position["symbol"],
                        "quantity": position["quantity"],
                        "purchase_price": position["purchase_price"],
                        "current_price": current_price,
                        "market_value": market_value,
                        "gain_loss": gain_loss,
                    }
                )

        total_gain_loss = total_value - total_cost
        return {
            "positions": positions_data,
            "total_value": total_value,
            "total_gain_loss": total_gain_loss,
        }

    def generate_text_report(self, portfolio, data_fetcher):
        """Generate a clean, table-formatted plain text report."""
        report_data = self._get_report_data(portfolio, data_fetcher)

        # Header
        header = f"{'Symbol':<10} {'Quantity':>12} {'Purchase Price':>18} {'Current Price':>17} {'Market Value':>16} {'Gain/Loss':>15}"
        separator = "-" * len(header)

        # Rows
        rows = [header, separator]
        for pos in report_data["positions"]:
            row = (
                f"{pos['symbol']:<10} "
                f"{pos['quantity']:>12.4f} "
                f"${pos['purchase_price']:>16,.2f} "
                f"${pos['current_price']:>15,.2f} "
                f"${pos['market_value']:>14,.2f} "
                f"${pos['gain_loss']:>13,.2f}"
            )
            rows.append(row)

        # Summary
        summary = (
            f"\n{separator}\n"
            f"Total Portfolio Value: ${report_data['total_value']:,.2f}\n"
            f"Total Gain/Loss: ${report_data['total_gain_loss']:,.2f}\n"
        )

        return "\n".join(rows) + summary

    def generate_html_report(self, portfolio, data_fetcher, ai_analysis="", market_events=None):
        """Generate a styled HTML report, perfect for emails."""
        report_data = self._get_report_data(portfolio, data_fetcher)

        # Market Events HTML Formatting (shown first if present)
        events_html = ""
        if market_events and market_events.get("symbol_events"):
            events_html = '<div class="events-card"><h2>📰 Market Events Detected</h2>'
            events_html += '<p style="color: #666; font-size: 14px;">This report was triggered by significant market events affecting your portfolio:</p>'

            for symbol, events in market_events["symbol_events"].items():
                # Escape symbol to prevent XSS
                safe_symbol = html.escape(str(symbol))
                events_html += f'<div style="margin: 15px 0;"><h3 style="color: #1a73e8; margin-bottom: 10px;">{safe_symbol}</h3>'
                for event in events[:3]:  # Limit to top 3 events per symbol
                    # Escape all event data to prevent XSS
                    safe_title = html.escape(str(event.get("title", "")))
                    safe_content = html.escape(str(event.get("content", ""))[:200])
                    safe_url = sanitize_url_for_html(event.get("url", ""))
                    events_html += f'''
                    <div style="border-left: 3px solid #1a73e8; padding-left: 15px; margin-bottom: 15px;">
                        <p style="font-weight: bold; margin: 5px 0;">{safe_title}</p>
                        <p style="margin: 5px 0; color: #666;">{safe_content}...</p>
                        <p style="margin: 5px 0;"><a href="{safe_url}" style="color: #1a73e8;">Read more →</a></p>
                    </div>
                    '''
                events_html += '</div>'
            events_html += '</div>'

        # AI Analysis HTML Formatting
        ai_html = ""
        if ai_analysis:
            # First escape HTML to prevent XSS, then apply safe formatting
            safe_analysis = html.escape(str(ai_analysis))
            # Convert markdown-like bold markers to HTML bold tags
            # Since content is escaped, we need to handle the escaped ** as well
            formatted_analysis = safe_analysis.replace("**", "<b>")
            # Replace newlines with <br> and double spaces with non-breaking spaces
            formatted_analysis = formatted_analysis.replace("\n", "<br>").replace("  ", "&nbsp;&nbsp;")
            ai_html = f"""
            <h2>AI Analysis</h2>
            <div class="analysis-card">
                <p>{formatted_analysis}</p>
            </div>
            """

        # Table rows
        rows_html = ""
        for pos in report_data["positions"]:
            gain_loss_color = "green" if pos["gain_loss"] >= 0 else "red"
            # Escape symbol to prevent XSS (user-provided input)
            safe_symbol = html.escape(str(pos["symbol"]))
            rows_html += f"""
            <tr>
                <td>{safe_symbol}</td>
                <td>{pos["quantity"]:.4f}</td>
                <td>${pos["purchase_price"]:,.2f}</td>
                <td>${pos["current_price"]:,.2f}</td>
                <td>${pos["market_value"]:,.2f}</td>
                <td style="color:{gain_loss_color};">${pos["gain_loss"]:,.2f}</td>
            </tr>
            """

        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; color: #333; }}
            .container {{ border: 1px solid #ddd; border-radius: 8px; padding: 20px; max-width: 800px; margin: auto; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            h1, h2 {{ color: #1a73e8; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .summary-card, .analysis-card {{ background-color: #f9f9f9; border: 1px solid #eee; border-radius: 8px; padding: 15px; margin-top: 20px; }}
            .events-card {{ background-color: #fff3cd; border: 1px solid #ffc107; border-radius: 8px; padding: 15px; margin-bottom: 20px; }}
            b {{ color: #1a73e8; }}
        </style>
        </head>
        <body>
            <div class="container">
                <h1>Stock Portfolio Report</h1>
                {events_html}
                <h2>Portfolio Overview</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Quantity</th>
                            <th>Purchase Price</th>
                            <th>Current Price</th>
                            <th>Market Value</th>
                            <th>Gain/Loss</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
                <div class="summary-card">
                    <p><b>Total Portfolio Value:</b> ${report_data["total_value"]:,.2f}</p>
                    <p><b>Total Gain/Loss:</b> <span style="color:{"green" if report_data["total_gain_loss"] >= 0 else "red"};">${report_data["total_gain_loss"]:,.2f}</span></p>
                </div>
                {ai_html}
            </div>
        </body>
        </html>
        """
        return html_template

    def send_email_report(self, html_content, report_type="daily"):
        """Send an HTML report via email."""
        email_settings = self.config.get("email_settings", {})
        if not all(
            [
                email_settings.get("smtp_server"),
                email_settings.get("email"),
                email_settings.get("password"),
                email_settings.get("recipient"),
            ]
        ):
            logger.warning("Email settings not configured. Skipping email send.")
            return False

        logger.info(f"Sending {report_type} report to {email_settings['recipient']}...")
        msg = MIMEMultipart()
        msg["From"] = email_settings["email"]
        msg["To"] = email_settings["recipient"]
        msg["Subject"] = (
            f"📈 {report_type.title()} Stock Portfolio Report - {datetime.now().strftime('%Y-%m-%d')}"
        )

        # Attach the HTML content
        msg.attach(MIMEText(html_content, "html"))

        try:
            context = ssl.create_default_context()
            with smtplib.SMTP(
                email_settings["smtp_server"],
                int(email_settings["smtp_port"]),
                timeout=30,
            ) as server:
                server.starttls(context=context)
                server.login(email_settings["email"], email_settings["password"])
                server.send_message(msg)
            logger.info(f"✅ Report sent successfully to {email_settings['recipient']}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to send email: {e}", exc_info=True)
            return False
