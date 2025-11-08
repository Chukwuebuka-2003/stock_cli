import logging
from datetime import datetime

import click

from .ai import AIAnalyzer
from .config import Config
from .data_fetcher import DataFetcher
from .portfolio import Portfolio
from .reporting import Reporting

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
def ai_report(email):
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
            portfolio, data_fetcher, ai_analysis=analysis
        )
        success = reporting.send_email_report(html_report, "AI-Powered")
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
