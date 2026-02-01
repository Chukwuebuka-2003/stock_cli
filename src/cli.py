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
from .agents.orchestrator import AgentOrchestrator
from .rag.vector_store import VectorStore
from .rag.embeddings import EmbeddingService
from .backtesting import Backtester
from .technical_indicators import TechnicalIndicators
from .ml_models import ProphetPredictor
from .sec_filings import SECFilingsClient
from groq import Groq
import appdirs
import yfinance as yf
import pandas as pd
import numpy as np


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
@click.argument("query", nargs=-1)
def chat(query):
    """Chat with the AI about your portfolio and the market"""
    query_text = " ".join(query)
    if not query_text:
        query_text = click.prompt("What would you like to know?")

    config = Config()
    groq_key = config.get("groq_api_key")
    if not groq_key:
        click.echo("Groq API key not configured. Please run 'setup-ai' first.")
        return

    tavily_key = config.get("tavily_api_key") or os.getenv("TAVILY_API_KEY")
    if tavily_key:
        os.environ.setdefault("TAVILY_API_KEY", tavily_key)

    twelvedata_key = config.get("twelvedata_api_key") or os.getenv("TWELVE_DATA_API_KEY")
    data_fetcher = None
    if twelvedata_key:
        data_fetcher = DataFetcher(twelvedata_api_key=twelvedata_key)
    else:
        click.echo("⚠️  Twelve Data API key not configured. Live prices unavailable in chat responses.")

    # Initialize components
    click.echo("Initializing agents...")
    groq_client = Groq(api_key=groq_key)
    
    # Initialize RAG
    user_data_dir = appdirs.user_data_dir("StockTrackerCLI", "Chukwuebuka")
    rag_dir = os.path.join(user_data_dir, "rag_storage")
    embedding_service = EmbeddingService()
    vector_store = VectorStore(persist_directory=rag_dir, embedding_service=embedding_service)
    
    orchestrator = AgentOrchestrator(
        model_client=groq_client,
        vector_store=vector_store,
        tavily_api_key=tavily_key,
        data_fetcher=data_fetcher,
    )
    
    click.echo(f"\nProcessing query: {query_text}\n")
    response = orchestrator.run(query_text)
    
    click.echo("\n" + "="*60)
    click.echo("AI Response:")
    click.echo("="*60 + "\n")
    click.echo(response)

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


# Backtesting Commands
@cli.command()
@click.argument("symbol")
@click.option("--strategy", type=click.Choice(["sma_crossover"]), default="sma_crossover", help="Trading strategy to backtest")
@click.option("--period", default="2y", help="Historical data period (e.g., 1y, 2y, 5y)")
@click.option("--capital", type=float, default=100000.0, help="Initial capital for backtesting")
@click.option("--fast", type=int, default=50, help="Fast moving average period")
@click.option("--slow", type=int, default=200, help="Slow moving average period")
def backtest(symbol, strategy, period, capital, fast, slow):
    """Backtest a trading strategy on historical data"""
    click.echo(f"\n{'='*80}")
    click.echo(f"Backtesting {symbol.upper()} - {strategy.upper()} Strategy")
    click.echo(f"{'='*80}\n")
    
    try:
        # Fetch historical data
        click.echo(f"Fetching {period} of historical data for {symbol.upper()}...")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        
        if df.empty:
            click.echo(f"❌ No data available for {symbol.upper()}")
            return
        
        click.echo(f"✅ Loaded {len(df)} days of data ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")
        
        # Reset index to make Date a column
        df = df.reset_index()
        df.rename(columns={'index': 'Date'}, inplace=True)
        
        # Generate trading signals based on strategy
        if strategy == "sma_crossover":
            click.echo(f"\nGenerating signals using SMA crossover (fast={fast}, slow={slow})...")
            
            # Calculate moving averages
            ti = TechnicalIndicators()
            df['SMA_Fast'] = ti.calculate_sma(df, period=fast)
            df['SMA_Slow'] = ti.calculate_sma(df, period=slow)
            
            # Generate signals: 1 = buy, -1 = sell, 0 = hold
            signals = pd.Series(0, index=range(len(df)))
            
            for i in range(1, len(df)):
                # Buy signal: fast crosses above slow
                if df.iloc[i-1]['SMA_Fast'] <= df.iloc[i-1]['SMA_Slow'] and df.iloc[i]['SMA_Fast'] > df.iloc[i]['SMA_Slow']:
                    signals.iloc[i] = 1
                # Sell signal: fast crosses below slow
                elif df.iloc[i-1]['SMA_Fast'] >= df.iloc[i-1]['SMA_Slow'] and df.iloc[i]['SMA_Fast'] < df.iloc[i]['SMA_Slow']:
                    signals.iloc[i] = -1
        
        # Run backtest
        click.echo(f"\nRunning backtest with ${capital:,.2f} initial capital...")
        backtester = Backtester(initial_capital=capital)
        results = backtester.run_strategy(df, signals)
        
        if not results:
            click.echo("❌ Backtest failed")
            return
        
        # Display results
        metrics = results['metrics']
        trades = results['trades']
        
        click.echo(f"\n{'='*80}")
        click.echo("BACKTEST RESULTS")
        click.echo(f"{'='*80}\n")
        
        click.echo(f"Initial Capital:     ${metrics['initial_capital']:,.2f}")
        click.echo(f"Final Equity:        ${metrics['final_equity']:,.2f}")
        click.echo(f"Total Return:        ${metrics['total_return']:,.2f} ({metrics['total_return_pct']:+.2f}%)")
        click.echo(f"\nNumber of Trades:    {metrics['num_trades']}")
        click.echo(f"  Buy Trades:        {metrics['num_buy_trades']}")
        click.echo(f"  Sell Trades:       {metrics['num_sell_trades']}")
        
        if 'sharpe_ratio' in metrics:
            click.echo(f"\nSharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
        if 'sortino_ratio' in metrics:
            click.echo(f"Sortino Ratio:       {metrics['sortino_ratio']:.2f}")
        if 'max_drawdown_pct' in metrics:
            click.echo(f"Max Drawdown:        {metrics['max_drawdown_pct']:.2f}%")
        if 'win_rate_pct' in metrics:
            click.echo(f"Win Rate:            {metrics['win_rate_pct']:.2f}%")
        if 'profit_factor' in metrics:
            click.echo(f"Profit Factor:       {metrics['profit_factor']:.2f}")
        
        # Compare with buy and hold
        comparison = backtester.compare_with_buy_and_hold(df, results)
        if comparison:
            click.echo(f"\n{'='*80}")
            click.echo("STRATEGY vs BUY & HOLD")
            click.echo(f"{'='*80}\n")
            click.echo(f"Buy & Hold Return:   {comparison['buy_and_hold']['return_pct']:+.2f}%")
            click.echo(f"Strategy Return:     {comparison['strategy']['return_pct']:+.2f}%")
            click.echo(f"Outperformance:      {comparison['outperformance_pct']:+.2f}%")
        
        # Show recent trades
        if trades:
            click.echo(f"\n{'='*80}")
            click.echo(f"RECENT TRADES (Last 10)")
            click.echo(f"{'='*80}\n")
            
            recent_trades = trades[-10:]
            for trade in recent_trades:
                date_str = trade['date'].strftime('%Y-%m-%d') if hasattr(trade['date'], 'strftime') else str(trade['date'])
                click.echo(f"{date_str} - {trade['action']:4s} {trade['shares']:6.0f} shares @ ${trade['price']:8.2f}")
        
        click.echo(f"\n{'='*80}\n")
        
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        click.echo(f"❌ Error running backtest: {e}")


# ML Prediction Commands
@cli.command()
@click.argument("symbol")
@click.option("--days", type=int, default=30, help="Number of days to predict")
@click.option("--model", type=click.Choice(["prophet"]), default="prophet", help="ML model to use")
@click.option("--period", default="2y", help="Historical data period for training (e.g., 1y, 2y, 5y)")
def predict(symbol, days, model, period):
    """Predict future stock prices using machine learning"""
    click.echo(f"\n{'='*80}")
    click.echo(f"Stock Price Prediction - {symbol.upper()}")
    click.echo(f"{'='*80}\n")
    
    try:
        # Fetch historical data
        click.echo(f"Fetching {period} of historical data for training...")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        
        if df.empty:
            click.echo(f"❌ No data available for {symbol.upper()}")
            return
        
        # Reset index to make Date a column
        df = df.reset_index()
        df.rename(columns={'index': 'Date'}, inplace=True)
        
        click.echo(f"✅ Loaded {len(df)} days of historical data")
        click.echo(f"   Period: {df['Date'].iloc[0].strftime('%Y-%m-%d')} to {df['Date'].iloc[-1].strftime('%Y-%m-%d')}")
        click.echo(f"   Latest Close: ${df['Close'].iloc[-1]:.2f}")
        
        # Train model and make predictions
        if model == "prophet":
            click.echo(f"\nTraining Prophet model (this may take a moment)...")
            predictor = ProphetPredictor()
            predictor.train(df, symbol)
            
            click.echo(f"Generating {days}-day forecast...")
            predictions = predictor.predict(periods=days)
            
            if predictions.empty:
                click.echo("❌ Prediction failed")
                return
            
            # Display predictions
            click.echo(f"\n{'='*80}")
            click.echo(f"PRICE PREDICTIONS - Next {days} Days")
            click.echo(f"{'='*80}\n")
            
            click.echo(f"{'Date':<12} {'Predicted':<12} {'Lower Bound':<12} {'Upper Bound':<12}")
            click.echo("-" * 80)
            
            for _, row in predictions.iterrows():
                date_str = row['Date'].strftime('%Y-%m-%d')
                click.echo(
                    f"{date_str:<12} "
                    f"${row['Predicted_Price']:<11.2f} "
                    f"${row['Lower_Bound']:<11.2f} "
                    f"${row['Upper_Bound']:<11.2f}"
                )
            
            # Calculate predicted change
            current_price = df['Close'].iloc[-1]
            final_prediction = predictions['Predicted_Price'].iloc[-1]
            price_change = final_prediction - current_price
            pct_change = (price_change / current_price) * 100
            
            click.echo(f"\n{'='*80}")
            click.echo("FORECAST SUMMARY")
            click.echo(f"{'='*80}\n")
            click.echo(f"Current Price:       ${current_price:.2f}")
            click.echo(f"Predicted Price:     ${final_prediction:.2f} (in {days} days)")
            click.echo(f"Expected Change:     ${price_change:+.2f} ({pct_change:+.2f}%)")
            
            # Determine trend
            if pct_change > 2:
                trend = "📈 BULLISH"
            elif pct_change < -2:
                trend = "📉 BEARISH"
            else:
                trend = "➡️  NEUTRAL"
            
            click.echo(f"Trend:               {trend}")
            click.echo(f"\n{'='*80}\n")
            
            click.echo("⚠️  Disclaimer: Predictions are for informational purposes only.")
            click.echo("    Past performance does not guarantee future results.\n")
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        click.echo(f"❌ Error making predictions: {e}")


# SEC Filings Command
@cli.command()
@click.argument("symbol")
@click.option("--filing-type", type=click.Choice(["10-K", "10-Q", "8-K", "all"]), default="all", help="Type of SEC filing")
@click.option("--limit", type=int, default=5, help="Number of filings to fetch")
def sec(symbol, filing_type, limit):
    """View recent SEC filings for a stock"""
    click.echo(f"\n{'='*80}")
    click.echo(f"SEC Filings - {symbol.upper()}")
    click.echo(f"{'='*80}\n")
    
    try:
        # Get user data directory for caching
        user_data_dir = appdirs.user_data_dir("StockTrackerCLI", "Chukwuebuka")
        cache_dir = os.path.join(user_data_dir, "sec_cache")
        
        # Initialize SEC client
        user_agent = "StockTrackerCLI/0.3.0 (ebulamicheal@gmail.com)"
        sec_client = SECFilingsClient(
            cache_dir=cache_dir,
            user_agent=user_agent,
            cache_ttl_days=7
        )
        
        # Fetch filings
        form_type = None if filing_type == "all" else filing_type
        click.echo(f"Fetching {filing_type} filings for {symbol.upper()}...")
        
        filings = sec_client.fetch_filings(symbol, form_type=form_type, limit=limit)
        
        if not filings:
            click.echo(f"❌ No {filing_type} filings found for {symbol.upper()}")
            click.echo("\nNote: Make sure the symbol is correct and the company files with the SEC.")
            return
        
        click.echo(f"✅ Found {len(filings)} filing(s)\n")
        
        # Display filings
        click.echo(f"{'='*80}")
        click.echo(f"RECENT FILINGS")
        click.echo(f"{'='*80}\n")
        
        for i, filing in enumerate(filings, 1):
            click.echo(f"Filing #{i}")
            click.echo(f"  Type:         {filing.get('form_type', 'N/A')}")
            click.echo(f"  Filing Date:  {filing.get('filing_date', 'N/A')}")
            if filing.get('report_date'):
                click.echo(f"  Report Date:  {filing['report_date']}")
            click.echo(f"  URL:          {filing.get('document_url', 'N/A')}")
            
            # Show content preview if available
            if filing.get('content'):
                # Get first 500 chars of text content
                content = filing['content']
                if isinstance(content, str):
                    # Remove HTML tags for preview
                    import re
                    text_content = re.sub(r'<[^>]+>', ' ', content)
                    text_content = ' '.join(text_content.split())[:500]
                    click.echo(f"  Preview:      {text_content}...")
            
            click.echo("")
        
        click.echo(f"{'='*80}\n")
        click.echo("💡 Tip: Visit the URLs above to view full filings on SEC.gov")
        
    except ValueError as e:
        click.echo(f"❌ Error: {e}")
    except Exception as e:
        logger.error(f"SEC filings error: {e}")
        click.echo(f"❌ Error fetching SEC filings: {e}")


# Portfolio Comparison Command
@cli.command()
@click.option("--benchmark", default="^GSPC", help="Benchmark symbol (default: ^GSPC for S&P 500)")
@click.option("--period", default="1y", help="Comparison period (e.g., 1m, 3m, 6m, 1y, 2y)")
def compare(benchmark, period):
    """Compare portfolio performance against a benchmark"""
    config = Config()
    api_key = config.get("alpha_vantage_api_key")
    if not api_key:
        click.echo("Alpha Vantage API key not set. Please run 'setup-alpha-vantage' first.")
        return
    
    portfolio = Portfolio()
    positions = portfolio.get_positions()
    
    if not positions:
        click.echo("Your portfolio is empty. Add some positions first.")
        return
    
    click.echo(f"\n{'='*80}")
    click.echo(f"Portfolio Comparison vs {benchmark}")
    click.echo(f"{'='*80}\n")
    
    try:
        # Fetch benchmark data
        click.echo(f"Fetching {period} benchmark data...")
        benchmark_ticker = yf.Ticker(benchmark)
        benchmark_df = benchmark_ticker.history(period=period)
        
        if benchmark_df.empty:
            click.echo(f"❌ Could not fetch data for benchmark {benchmark}")
            return
        
        benchmark_start = benchmark_df['Close'].iloc[0]
        benchmark_end = benchmark_df['Close'].iloc[-1]
        benchmark_return = ((benchmark_end - benchmark_start) / benchmark_start) * 100
        
        # Calculate portfolio returns
        data_fetcher = DataFetcher(api_key=api_key)
        portfolio_history = PortfolioHistory()
        
        # Try to get historical snapshots
        snapshots = portfolio_history.history
        
        if snapshots and len(snapshots) >= 2:
            # Use historical data if available
            # Find snapshots within the period
            from dateutil.parser import parse
            from datetime import datetime, timedelta
            
            period_days = {
                "1m": 30, "3m": 90, "6m": 180, "1y": 365, "2y": 730, "5y": 1825
            }
            days = period_days.get(period, 365)
            cutoff_date = datetime.now() - timedelta(days=days)
            
            recent_snapshots = [s for s in snapshots if parse(s['date']) >= cutoff_date]
            
            if len(recent_snapshots) >= 2:
                recent_snapshots.sort(key=lambda x: x['date'])
                start_snapshot = recent_snapshots[0]
                end_snapshot = recent_snapshots[-1]
                
                portfolio_start = start_snapshot['total_value']
                portfolio_end = end_snapshot['total_value']
                portfolio_return = ((portfolio_end - portfolio_start) / portfolio_start) * 100
            else:
                click.echo("⚠️  Not enough historical data for accurate comparison.")
                click.echo("    Using current portfolio value estimate.\n")
                
                # Estimate using current prices
                total_value = 0
                total_cost = 0
                for position in positions:
                    stock_data = data_fetcher.get_stock_data(position['symbol'])
                    if stock_data:
                        value = position['quantity'] * stock_data['currentPrice']
                        cost = position['quantity'] * position['purchase_price']
                        total_value += value
                        total_cost += cost
                
                portfolio_return = ((total_value - total_cost) / total_cost) * 100 if total_cost > 0 else 0
        else:
            click.echo("⚠️  No historical snapshots found. Using current portfolio value.\n")
            
            # Calculate based on current prices
            total_value = 0
            total_cost = 0
            for position in positions:
                stock_data = data_fetcher.get_stock_data(position['symbol'])
                if stock_data:
                    value = position['quantity'] * stock_data['currentPrice']
                    cost = position['quantity'] * position['purchase_price']
                    total_value += value
                    total_cost += cost
            
            portfolio_return = ((total_value - total_cost) / total_cost) * 100 if total_cost > 0 else 0
        
        # Display comparison
        click.echo(f"{'='*80}")
        click.echo(f"PERFORMANCE COMPARISON - {period.upper()}")
        click.echo(f"{'='*80}\n")
        
        click.echo(f"Portfolio Return:    {portfolio_return:+.2f}%")
        click.echo(f"Benchmark Return:    {benchmark_return:+.2f}% ({benchmark})")
        
        alpha = portfolio_return - benchmark_return
        click.echo(f"\nAlpha:               {alpha:+.2f}%")
        
        if alpha > 0:
            click.echo(f"Performance:         📈 Outperforming benchmark")
        elif alpha < 0:
            click.echo(f"Performance:         📉 Underperforming benchmark")
        else:
            click.echo(f"Performance:         ➡️  Matching benchmark")
        
        # Calculate additional metrics if historical data available
        if snapshots and len(snapshots) >= 10:
            # Calculate volatility
            returns = []
            for i in range(1, len(snapshots)):
                prev_val = snapshots[i-1]['total_value']
                curr_val = snapshots[i]['total_value']
                ret = (curr_val - prev_val) / prev_val
                returns.append(ret)
            
            if returns:
                volatility = np.std(returns) * np.sqrt(252) * 100  # Annualized
                sharpe = (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
                
                click.echo(f"\nAnnualized Volatility: {volatility:.2f}%")
                click.echo(f"Sharpe Ratio:         {sharpe:.2f}")
        
        click.echo(f"\n{'='*80}\n")
        click.echo("💡 Tip: Run 'stock-tracker history snapshot' regularly for better tracking")
        
    except Exception as e:
        logger.error(f"Comparison error: {e}")
        click.echo(f"❌ Error comparing portfolio: {e}")


# Export/Import Commands
@cli.command()
@click.option("--format", type=click.Choice(["json", "csv"]), default="json", help="Export format")
@click.option("--output", help="Output file path (default: portfolio_export.json/csv)")
def export(format, output):
    """Export portfolio data to a file"""
    portfolio = Portfolio()
    positions = portfolio.get_positions()
    
    if not positions:
        click.echo("Your portfolio is empty. Nothing to export.")
        return
    
    # Set default output filename
    if not output:
        output = f"portfolio_export.{format}"
    
    try:
        if format == "json":
            # Export as JSON
            export_data = {
                "export_date": datetime.now().isoformat(),
                "positions": positions
            }
            
            with open(output, 'w') as f:
                json.dump(export_data, f, indent=2)
            
        elif format == "csv":
            # Export as CSV
            df = pd.DataFrame(positions)
            df.to_csv(output, index=False)
        
        click.echo(f"✅ Portfolio exported to {output}")
        click.echo(f"   {len(positions)} position(s) exported")
        
    except Exception as e:
        logger.error(f"Export error: {e}")
        click.echo(f"❌ Error exporting portfolio: {e}")


@cli.command()
@click.argument("file")
@click.option("--replace", is_flag=True, help="Replace existing portfolio (default: merge)")
def import_portfolio(file, replace):
    """Import portfolio data from a file"""
    if not os.path.exists(file):
        click.echo(f"❌ File not found: {file}")
        return
    
    try:
        portfolio = Portfolio()
        
        # Determine format from file extension
        file_ext = os.path.splitext(file)[1].lower()
        
        if file_ext == ".json":
            with open(file, 'r') as f:
                data = json.load(f)
                
            # Handle both old and new export formats
            if isinstance(data, dict) and 'positions' in data:
                positions_to_import = data['positions']
            elif isinstance(data, list):
                positions_to_import = data
            else:
                click.echo("❌ Invalid JSON format")
                return
                
        elif file_ext == ".csv":
            df = pd.read_csv(file)
            positions_to_import = df.to_dict('records')
        else:
            click.echo(f"❌ Unsupported file format: {file_ext}")
            click.echo("    Supported formats: .json, .csv")
            return
        
        # Clear portfolio if replace flag is set
        if replace:
            current_positions = portfolio.get_positions()
            for pos in current_positions:
                portfolio.remove_position(pos['symbol'])
            click.echo(f"Cleared {len(current_positions)} existing position(s)")
        
        # Import positions
        imported = 0
        for pos in positions_to_import:
            try:
                portfolio.add_position(
                    symbol=pos['symbol'],
                    quantity=float(pos['quantity']),
                    purchase_price=float(pos['purchase_price'])
                )
                imported += 1
            except Exception as e:
                click.echo(f"⚠️  Failed to import {pos.get('symbol', 'unknown')}: {e}")
        
        mode = "replaced" if replace else "merged"
        click.echo(f"✅ Portfolio {mode} successfully")
        click.echo(f"   {imported} position(s) imported")
        
    except json.JSONDecodeError:
        click.echo("❌ Invalid JSON file")
    except Exception as e:
        logger.error(f"Import error: {e}")
        click.echo(f"❌ Error importing portfolio: {e}")
