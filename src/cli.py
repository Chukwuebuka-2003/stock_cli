import json
import logging
import os
from datetime import datetime

import click

from .ai import AIAnalyzer
from .alerts import PriceAlerts
from .config import Config
from .data_fetcher import DataFetcher
from .history import PortfolioHistory
from .portfolio import Portfolio
from .reporting import Reporting
from .watchlist import Watchlist

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """Stock Tracker CLI - Track your investments and get AI-powered reports"""
    pass


@cli.command()
@click.option("--symbol", prompt="Stock Symbol", help="The stock symbol to add.")
@click.option("--quantity", prompt="Quantity", type=float, help="The number of shares.")
@click.option(
    "--price", prompt="Purchase Price", type=float, help="The purchase price per share."
)
def add(symbol, quantity, price):
    """Add a new stock position to your portfolio"""
    portfolio = Portfolio()
    portfolio.add_position(symbol, quantity, price)
    click.echo(f"Added {quantity} shares of {symbol.upper()} at ${price}")


@cli.command()
@click.option("--symbol", prompt="Stock Symbol", help="The stock symbol to remove.")
def remove(symbol):
    """Remove a stock position from your portfolio"""
    portfolio = Portfolio()
    if portfolio.remove_position(symbol):
        click.echo(f"Removed {symbol.upper()} from your portfolio.")
    else:
        click.echo(f"{symbol.upper()} not found in your portfolio.")


@cli.command()
def report():
    """Generate and display a report of your portfolio"""
    config = Config()
    api_key = config.get("alpha_vantage_api_key")
    if not api_key:
        click.echo(
            "Alpha Vantage API key not set. Please run 'setup-alpha-vantage' first."
        )
        return

    portfolio = Portfolio()
    data_fetcher = DataFetcher(api_key=api_key)
    reporting = Reporting(config)

    # Generate and print the plain text report for the console
    text_report = reporting.generate_text_report(portfolio, data_fetcher)
    click.echo(text_report)


@cli.command()
@click.option(
    "--email", is_flag=True, help="Send the report to the configured email address."
)
@click.option(
    "--events",
    default=None,
    help="JSON string of market events that triggered this report (for event-based reporting)."
)
def ai_report(email, events):
    """Generate an AI-powered analysis of your portfolio"""
    config = Config()
    alpha_vantage_key = config.get("alpha_vantage_api_key")
    if not alpha_vantage_key:
        click.echo(
            "Alpha Vantage API key not set. Please run 'setup-alpha-vantage' first."
        )
        return

    groq_key = config.get("groq_api_key")
    if not groq_key:
        click.echo("Groq API key not configured. Please run 'setup-ai' first.")
        return

    portfolio = Portfolio()
    data_fetcher = DataFetcher(api_key=alpha_vantage_key)
    reporting = Reporting(config)

    # Parse events from command-line argument or environment variable
    market_events = None

    # First try command-line argument
    if events:
        try:
            market_events = json.loads(events)
            logger.info("Loaded market events from --events argument")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing events JSON from argument: {e}")
            click.echo("⚠️  Warning: Could not parse market events from --events argument")

    # If not provided via argument, try environment variable
    if not market_events:
        events_env = os.getenv("MARKET_EVENTS_JSON")
        if events_env:
            try:
                market_events = json.loads(events_env)
                logger.info("Loaded market events from MARKET_EVENTS_JSON environment variable")
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing MARKET_EVENTS_JSON environment variable: {e}")
                click.echo("⚠️  Warning: Could not parse market events from environment variable")

    # Generate the text report for console display
    text_report = reporting.generate_text_report(portfolio, data_fetcher)
    ai_analyzer = AIAnalyzer(api_key=groq_key)
    analysis = ai_analyzer.get_analysis(text_report)

    # Print clean report to console
    click.echo(text_report)
    click.echo("\nAI Analysis:\n")
    click.echo(analysis)

    # If email flag is set, generate and send the HTML report
    if email:
        click.echo("\nSending email with HTML report...")
        html_report = reporting.generate_html_report(
            portfolio, data_fetcher, ai_analysis=analysis, market_events=market_events
        )

        # Determine report type based on whether events were provided
        report_type = "Event-Triggered" if market_events else "AI-Powered"

        success = reporting.send_email_report(html_report, report_type)
        if success:
            click.echo("✅ Email sent successfully!")
        else:
            click.echo("❌ Failed to send email. Please check your settings and logs.")


@cli.command()
def setup_ai():
    """Set up your Groq API key"""
    config = Config()
    api_key = click.prompt("Enter your Groq API key", hide_input=True)
    config.set("groq_api_key", api_key)
    click.echo("Groq API key saved.")


@cli.command()
def setup_alpha_vantage():
    """Set up your Alpha Vantage API key"""
    config = Config()
    api_key = click.prompt("Enter your Alpha Vantage API key", hide_input=True)
    config.set("alpha_vantage_api_key", api_key)
    click.echo("Alpha Vantage API key saved.")


@cli.command()
@click.option(
    "--smtp-server",
    prompt="SMTP Server",
    default=None,
    help="SMTP server (e.g., smtp.gmail.com)",
)
@click.option("--smtp-port", default=None, help="SMTP port (default: 587 for Gmail)")
@click.option("--email", prompt="Your Email", help="Your email address")
@click.option(
    "--password",
    prompt="App Password",
    hide_input=True,
    help="Your App Password (16-digit for Gmail)",
)
@click.option("--recipient", prompt="Recipient Email", help="Report recipient email")
def setup_email(smtp_server, smtp_port, email, password, recipient):
    """Setup email settings for report delivery (Gmail App Password compatible)"""
    config = Config()

    is_gmail = "gmail.com" in email.lower()

    if smtp_server is None:
        if is_gmail:
            smtp_server = "smtp.gmail.com"
            click.echo(f"✅ Auto-detected Gmail server: {smtp_server}")
        else:
            smtp_server = click.prompt("SMTP Server", default="smtp.gmail.com")

    if smtp_port is None:
        if is_gmail:
            smtp_port = 587
            click.echo(f"✅ Auto-detected Gmail port: {smtp_port}")
        else:
            smtp_port = click.prompt("SMTP Port", default=587, type=int)

    if is_gmail:
        if len(password.replace(" ", "")) != 16:
            click.echo("⚠️  Gmail App Password should be 16 digits")
            click.echo("💡 Generate one at: https://myaccount.google.com/apppasswords")
            confirm = click.confirm("Continue anyway?", default=False)
            if not confirm:
                click.echo("❌ Setup cancelled")
                return

    email_settings = {
        "smtp_server": smtp_server,
        "smtp_port": int(smtp_port),
        "email": email,
        "password": password,
        "recipient": recipient,
    }
    config.set("email_settings", email_settings)

    click.echo("📧 Testing email configuration...")
    reporting = Reporting(config)
    test_html = reporting.generate_html_report(
        Portfolio(), DataFetcher(api_key="DEMO")
    )  # Dummy data for test
    success = reporting.send_email_report(test_html, "test")

    if success:
        click.echo("✅ Email settings configured successfully!")
    else:
        click.echo("❌ Test email failed. Please check your settings.")


# Portfolio History Commands
@cli.group()
def history():
    """Manage portfolio history and view performance over time"""
    pass


@history.command(name="snapshot")
def history_snapshot():
    """Take a snapshot of your current portfolio for historical tracking"""
    config = Config()
    api_key = config.get("alpha_vantage_api_key")
    if not api_key:
        click.echo(
            "Alpha Vantage API key not set. Please run 'setup-alpha-vantage' first."
        )
        return

    portfolio = Portfolio()
    if not portfolio.get_positions():
        click.echo("Your portfolio is empty. Add some positions first.")
        return

    data_fetcher = DataFetcher(api_key=api_key)
    portfolio_history = PortfolioHistory()

    click.echo("Taking portfolio snapshot...")
    snapshot = portfolio_history.add_snapshot(portfolio, data_fetcher)

    click.echo(f"\n✅ Snapshot saved for {snapshot['date']}")
    click.echo(f"Total Value: ${snapshot['total_value']:,.2f}")
    click.echo(f"Total Cost: ${snapshot['total_cost']:,.2f}")
    click.echo(f"Gain/Loss: ${snapshot['gain_loss']:,.2f} ({snapshot['gain_loss_percent']:+.2f}%)")


@history.command(name="show")
@click.option(
    "--period",
    type=click.Choice(["7d", "30d", "90d", "1y", "all"]),
    default="all",
    help="Time period to show performance for",
)
def history_show(period):
    """Show portfolio performance over time"""
    portfolio_history = PortfolioHistory()

    if not portfolio_history.history:
        click.echo("No historical data available.")
        click.echo("Run 'stock-tracker history snapshot' to start tracking your portfolio.")
        return

    # Map period to days
    period_days_map = {
        "7d": 7,
        "30d": 30,
        "90d": 90,
        "1y": 365,
        "all": None,
    }

    period_name_map = {
        "7d": "7 Days",
        "30d": "30 Days",
        "90d": "90 Days",
        "1y": "1 Year",
        "all": "All Time",
    }

    days = period_days_map[period]
    period_name = period_name_map[period]

    performance = portfolio_history.get_performance(days)

    if not performance:
        click.echo(f"No data available for {period_name}.")
        return

    report = portfolio_history.format_performance_report(period_name, performance)
    click.echo(report)

    # Show all available periods if showing all
    if period == "all":
        click.echo("\nPerformance by Period:")
        for p in ["7d", "30d", "90d", "1y"]:
            if perf := portfolio_history.get_performance(period_days_map[p]):
                click.echo(
                    f"  {period_name_map[p]:10} {perf['value_change']:+,.2f} ({perf['percent_change']:+.2f}%)"
                )


# Alert Commands
@cli.group()
def alert():
    """Manage price alerts for stocks"""
    pass


@alert.command(name="add")
@click.argument("symbol")
@click.option("--above", type=float, help="Alert when price goes above this value")
@click.option("--below", type=float, help="Alert when price goes below this value")
def alert_add(symbol, above, below):
    """Add a price alert for a stock"""
    if above is None and below is None:
        click.echo("❌ Error: You must specify at least one of --above or --below")
        return

    alerts = PriceAlerts()

    try:
        alert = alerts.add_alert(symbol, above=above, below=below)
        click.echo(f"✅ Alert added for {alert['symbol']}")
        click.echo(f"ID: {alert['id']}")
        if above:
            click.echo(f"Trigger above: ${above:.2f}")
        if below:
            click.echo(f"Trigger below: ${below:.2f}")
    except ValueError as e:
        click.echo(f"❌ Error: {e}")


@alert.command(name="list")
@click.option("--symbol", help="Filter alerts by symbol")
@click.option("--active-only", is_flag=True, help="Show only active (non-triggered) alerts")
def alert_list(symbol, active_only):
    """List all price alerts"""
    alerts = PriceAlerts()
    alert_list = alerts.get_alerts(symbol=symbol, active_only=active_only)

    if not alert_list:
        if symbol:
            click.echo(f"No alerts found for {symbol}")
        else:
            click.echo("No alerts configured.")
            click.echo("Add an alert with: stock-tracker alert add SYMBOL --above PRICE")
        return

    click.echo(f"\n{'Active' if active_only else 'All'} Alerts:")
    click.echo("=" * 60)
    for alert_item in alert_list:
        click.echo(f"\n{alerts.format_alert(alert_item)}")


@alert.command(name="remove")
@click.argument("alert_id")
def alert_remove(alert_id):
    """Remove a price alert by ID"""
    alerts = PriceAlerts()

    if alerts.remove_alert(alert_id):
        click.echo(f"✅ Alert {alert_id} removed")
    else:
        click.echo(f"❌ Alert {alert_id} not found")


@alert.command(name="check")
def alert_check():
    """Check all active alerts against current prices"""
    config = Config()
    api_key = config.get("alpha_vantage_api_key")
    if not api_key:
        click.echo(
            "Alpha Vantage API key not set. Please run 'setup-alpha-vantage' first."
        )
        return

    alerts = PriceAlerts()
    active_alerts = alerts.get_alerts(active_only=True)

    if not active_alerts:
        click.echo("No active alerts to check.")
        return

    click.echo(f"Checking {len(active_alerts)} active alert(s)...")

    data_fetcher = DataFetcher(api_key=api_key)

    if triggered := alerts.check_alerts(data_fetcher):
        click.echo(f"\n🚨 {len(triggered)} alert(s) triggered!")
        click.echo("=" * 60)
        for alert_item in triggered:
            click.echo(f"\n{alerts.format_alert(alert_item)}")
    else:
        click.echo("\n✅ No alerts triggered")


# Watchlist Commands
@cli.group()
def watchlist():
    """Manage your stock watchlist"""
    pass


@watchlist.command(name="add")
@click.argument("symbol")
@click.option("--note", help="Optional note about the stock")
def watchlist_add(symbol, note):
    """Add a stock to your watchlist"""
    wl = Watchlist()

    # Validate symbol format before attempting to add
    symbol_normalized = symbol.strip().upper()
    if not symbol_normalized or not symbol_normalized.replace('.', '').replace('-', '').isalnum():
        click.echo(f"❌ Invalid stock symbol: '{symbol}'")
        return

    if wl.add_stock(symbol, note=note):
        click.echo(f"✅ Added {symbol_normalized} to watchlist")
        if note:
            click.echo(f"Note: {note}")
    else:
        click.echo(f"⚠️  {symbol_normalized} is already in your watchlist")


@watchlist.command(name="remove")
@click.argument("symbol")
def watchlist_remove(symbol):
    """Remove a stock from your watchlist"""
    wl = Watchlist()

    if wl.remove_stock(symbol):
        click.echo(f"✅ Removed {symbol.upper()} from watchlist")
    else:
        click.echo(f"❌ {symbol.upper()} not found in watchlist")


@watchlist.command(name="list")
def watchlist_list():
    """List all stocks in your watchlist"""
    wl = Watchlist()
    stocks = wl.get_stocks()

    if not stocks:
        click.echo("Your watchlist is empty.")
        click.echo("Add stocks with: stock-tracker watchlist add SYMBOL")
        return

    click.echo(f"\nWatchlist ({len(stocks)} stock{'s' if len(stocks) != 1 else ''}):")
    click.echo("=" * 60)

    for stock in stocks:
        click.echo(f"\n{stock['symbol']}")
        click.echo(f"  Added: {stock['added_at'][:10]}")
        if stock.get("note"):
            click.echo(f"  Note: {stock['note']}")


@watchlist.command(name="report")
def watchlist_report():
    """Generate a detailed report for your watchlist"""
    config = Config()
    api_key = config.get("alpha_vantage_api_key")
    if not api_key:
        click.echo(
            "Alpha Vantage API key not set. Please run 'setup-alpha-vantage' first."
        )
        return

    wl = Watchlist()
    stocks = wl.get_stocks()

    if not stocks:
        click.echo("Your watchlist is empty.")
        return

    data_fetcher = DataFetcher(api_key=api_key)

    click.echo(f"\nWatchlist Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    click.echo("=" * 80)

    for stock in stocks:
        symbol = stock["symbol"]
        click.echo(f"\n{symbol}")

        if stock_data := data_fetcher.get_stock_data(symbol):
            click.echo(f"  Current Price: ${stock_data['currentPrice']:,.2f}")
            click.echo(f"  Change: {stock_data['change']:+.2f} ({stock_data['changePercent']})")
            click.echo(f"  Previous Close: ${stock_data['previousClose']:,.2f}")
        else:
            click.echo("  ⚠️  Could not fetch price data")

        if stock.get("note"):
            click.echo(f"  Note: {stock['note']}")

    click.echo("\n" + "=" * 80)
