# Stock Tracker CLI

[![PyPI version](https://badge.fury.io/py/stock-tracker-cli.svg)](https://badge.fury.io/py/stock-tracker-cli)
[![PyPI Downloads](https://img.shields.io/pypi/dm/stock-tracker-cli.svg)](https://pypi.org/project/stock-tracker-cli/)
[![Total Downloads](https://static.pepy.tech/badge/stock-tracker-cli)](https://pepy.tech/project/stock-tracker-cli)
[![Python Versions](https://img.shields.io/pypi/pyversions/stock-tracker-cli.svg)](https://pypi.org/project/stock-tracker-cli/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/Chukwuebuka-2003/stock_cli.svg)](https://github.com/Chukwuebuka-2003/stock_cli/stargazers)
[![GitHub Issues](https://img.shields.io/github/issues/Chukwuebuka-2003/stock_cli.svg)](https://github.com/Chukwuebuka-2003/stock_cli/issues)

A command-line tool to track your stock portfolio and get AI-powered investment analysis.

## Features

### Core Features
- 📊 **Portfolio Management**: Add and remove stock positions with purchase prices
- 💹 **Real-time Data**: Fetch current stock prices using Alpha Vantage API
- 🤖 **AI Analysis**: Get intelligent insights using Groq's LLM (llama-3.3-70b-versatile)
- 📧 **Email Reports**: Send HTML-formatted reports with AI analysis via email
- ⚡ **Smart Caching**: 15-minute cache to reduce API calls
- 🐳 **Docker Support**: Containerized deployment with automated scheduled reports
- 🚨 **GitHub Actions Automation**: Scheduled and event-based reporting with Tavily API integration
- 📰 **Market Event Detection**: Automatically trigger reports when significant market events affect your portfolio
- 📈 **Portfolio History**: Track portfolio performance over time with daily snapshots and time-period analysis
- 🔔 **Price Alerts**: Set price alerts for stocks with customizable above/below thresholds
- 👀 **Watchlist**: Track stocks you're interested in without adding them to your portfolio

### 🆕 Advanced Trading Features
- 📉 **Backtesting**: Test trading strategies on historical data with detailed performance metrics
- 🔮 **ML Price Predictions**: Forecast future stock prices using Prophet machine learning model
- 📄 **SEC Filings**: Access and view recent SEC filings (10-K, 10-Q, 8-K) for any stock
- 📊 **Portfolio Comparison**: Compare your portfolio performance against market benchmarks (S&P 500, NASDAQ, etc.)
- 💾 **Export/Import**: Backup and restore your portfolio in JSON or CSV format

### 🆕 Interactive Dashboard (Streamlit UI)
- 📊 **Real-time Portfolio Dashboard**: Visual overview of your entire portfolio with interactive charts
- 📈 **Advanced Price Charts**: Candlestick charts with technical indicators (RSI, MACD, Bollinger Bands, Moving Averages)
- 🥧 **Portfolio Composition**: Pie charts showing asset allocation
- 📉 **Performance Analytics**: Visual gain/loss tracking for each position
- 🎯 **Technical Signals**: Automated buy/sell signals based on technical indicators
- 🔄 **Multi-timeframe Analysis**: Analyze stocks across different time periods (1M, 3M, 6M, 1Y, 2Y, 5Y, Max)
- 📊 **Interactive Indicators**: Toggle between different technical indicators in real-time

<img width="1366" height="768" alt="Screenshot From 2025-11-29 16-48-13" src="https://github.com/user-attachments/assets/897a48b8-8fa7-4c54-8269-c5e72cf6c123" />

<img width="1366" height="768" alt="Screenshot From 2025-11-29 16-48-24" src="https://github.com/user-attachments/assets/134f5edb-6605-4d7f-b400-56f5a055cddf" />


<img width="1366" height="768" alt="Screenshot From 2025-11-29 20-32-11" src="https://github.com/user-attachments/assets/f15a1b42-d270-4ec1-a917-1f063d807d1a" />


## Installation

### From PyPI

```bash
pip install stock-tracker-cli
```

### From Source

```bash
git clone https://github.com/Chukwuebuka-2003/stock_cli.git
cd stock_cli
pip install -e .
```

## Quick Start

### 1. Configure API Keys

```bash
# Setup Groq API key for AI analysis
stock-tracker setup-ai

# Setup Alpha Vantage API key for stock data
stock-tracker setup-alpha-vantage

# Setup email settings (optional, for email reports)
stock-tracker setup-email
```

**Get Free API Keys:**
- Groq: https://console.groq.com/
- Alpha Vantage: https://www.alphavantage.co/support/#api-key

### 2. Add Stock Positions

```bash
# Add a stock position
stock-tracker add AAPL 10 150.00

# Arguments: SYMBOL QUANTITY PURCHASE_PRICE
stock-tracker add MSFT 5 300.00
stock-tracker add GOOGL 3 2800.00
```

### 3. Generate Reports

```bash
# View portfolio report in terminal
stock-tracker report

# Get AI-powered analysis
stock-tracker ai-report

# Send AI report via email
stock-tracker ai-report --email recipient@example.com
```

## Commands

### Portfolio Management

```bash
# Add a stock position
stock-tracker add <SYMBOL> <QUANTITY> <PURCHASE_PRICE>

# Remove a stock position
stock-tracker remove <SYMBOL>
```

### Reports

```bash
# Generate plain text report
stock-tracker report

# Generate AI-powered analysis report
stock-tracker ai-report [--email EMAIL]
```

### Configuration

```bash
# Configure Groq API key
stock-tracker setup-ai

# Configure Alpha Vantage API key
stock-tracker setup-alpha-vantage

# Configure email settings
stock-tracker setup-email
```

### Portfolio History

```bash
# Take a snapshot of your current portfolio
stock-tracker history snapshot

# View performance over time
stock-tracker history show --period 7d    # Last 7 days
stock-tracker history show --period 30d   # Last 30 days
stock-tracker history show --period 90d   # Last 90 days
stock-tracker history show --period 1y    # Last year
stock-tracker history show --period all   # All time (default)
```

### Price Alerts

```bash
# Add a price alert
stock-tracker alert add AAPL --above 200           # Alert when AAPL goes above $200
stock-tracker alert add TSLA --below 180           # Alert when TSLA goes below $180
stock-tracker alert add MSFT --above 400 --below 350  # Dual threshold alert

# List all alerts
stock-tracker alert list
stock-tracker alert list --active-only    # Show only non-triggered alerts
stock-tracker alert list --symbol AAPL    # Filter by symbol

# Check alerts against current prices
stock-tracker alert check

# Remove an alert
stock-tracker alert remove alert_1
```

### Watchlist

```bash
# Add stocks to watchlist
stock-tracker watchlist add NVDA
stock-tracker watchlist add GOOGL --note "Considering for tech exposure"

# List all watchlist stocks
stock-tracker watchlist list

# Generate detailed watchlist report with current prices
stock-tracker watchlist report

# Remove a stock from watchlist
stock-tracker watchlist remove NVDA
```

### 🆕 Backtesting

```bash
# Backtest a trading strategy on historical data
stock-tracker backtest AAPL --period 2y           # Default SMA crossover strategy
stock-tracker backtest MSFT --capital 50000       # Custom initial capital
stock-tracker backtest GOOGL --fast 20 --slow 50  # Custom MA periods

# Strategy options:
# - sma_crossover: Simple Moving Average crossover (fast MA crosses slow MA)

# Results include:
# - Total return and Sharpe ratio
# - Maximum drawdown and win rate
# - Comparison with buy-and-hold strategy
# - Trade history
```

### 🆕 Price Predictions

```bash
# Predict future stock prices using machine learning
stock-tracker predict AAPL               # 30-day forecast (default)
stock-tracker predict TSLA --days 60     # 60-day forecast
stock-tracker predict NVDA --period 5y   # Train on 5 years of data

# Models available:
# - prophet: Facebook Prophet time series forecasting (default)

# Shows:
# - Predicted prices with confidence intervals (upper/lower bounds)
# - Expected price change and percentage
# - Trend analysis (bullish/bearish/neutral)
```

### 🆕 SEC Filings

```bash
# View recent SEC filings for a stock
stock-tracker sec AAPL                    # All recent filings
stock-tracker sec MSFT --filing-type 10-K # Annual reports only
stock-tracker sec TSLA --filing-type 10-Q # Quarterly reports
stock-tracker sec GOOGL --filing-type 8-K # Current event reports
stock-tracker sec NVDA --limit 10         # Get more filings

# Shows:
# - Filing type, date, and report date
# - Direct links to SEC.gov documents
# - Content preview
```

### 🆕 Portfolio Analysis

```bash
# Compare portfolio performance against benchmarks
stock-tracker compare                         # Compare vs S&P 500 (default)
stock-tracker compare --benchmark ^DJI        # Compare vs Dow Jones
stock-tracker compare --benchmark ^IXIC       # Compare vs NASDAQ
stock-tracker compare --period 6m             # 6-month comparison

# Shows:
# - Portfolio return vs benchmark return
# - Alpha (outperformance/underperformance)
# - Volatility and Sharpe ratio (if historical data available)
```

### 🆕 Portfolio Export/Import

```bash
# Export portfolio to file
stock-tracker export                          # Export to portfolio_export.json
stock-tracker export --format csv             # Export to CSV
stock-tracker export --output backup.json     # Custom output path

# Import portfolio from file
stock-tracker import-portfolio backup.json    # Merge with existing portfolio
stock-tracker import-portfolio data.csv --replace  # Replace entire portfolio

# Use cases:
# - Backup your portfolio
# - Share portfolio with others
# - Migrate between devices
# - Bulk import positions
```

## 🆕 Interactive Dashboard (Streamlit UI)

Launch the interactive web-based dashboard for a visual portfolio experience:

```bash
# Launch the Streamlit dashboard
streamlit run src/streamlit_app.py

# Or use the command after installation
stock-tracker-ui
```

The dashboard will open in your default web browser at `http://localhost:8501`

### Dashboard Features

#### 📊 Portfolio Overview Tab
- Real-time portfolio metrics (total value, cost basis, gain/loss)
- Interactive pie chart showing portfolio composition by stock
- Bar chart displaying individual stock performance
- Detailed positions table with color-coded gains/losses

#### 📈 Stock Analysis Tab
- Select any stock from your portfolio
- Choose time periods: 1 month, 3 months, 6 months, 1 year, 2 years, 5 years, or maximum
- Interactive candlestick charts with zoom and pan capabilities
- Technical indicators overlay:
  - **Moving Averages**: SMA 20, SMA 50, SMA 200
  - **Bollinger Bands**: Upper, middle, and lower bands
  - **RSI (Relative Strength Index)**: Overbought/oversold levels
  - **MACD**: Signal line and histogram
- Automated technical signals (bullish/bearish/neutral)
- Volume bars with price correlation

#### 👀 Watchlist Tab
- Real-time prices for all watchlist stocks
- Quick view of price changes and percentages
- Notes for each watched stock

### Dashboard Screenshots

The dashboard provides:
- ✨ Modern, responsive design
- 🎨 Color-coded metrics (green for gains, red for losses)
- 📱 Works on desktop and tablet
- ⚡ Auto-refreshing data with cache management
- 🔄 Manual refresh button for instant updates

### Keyboard Shortcuts & Tips

- Use the sidebar for quick CLI command reference
- Click "Clear Cache" in the sidebar to force fresh data fetch
- All charts are interactive - hover for details, zoom, and pan
- Charts can be downloaded as PNG images using the toolbar

## Example Output

```
Stock Portfolio Report
Generated on: 2025-11-09 10:30:00

╔════════╦══════════╦════════════════╦═══════════════╦════════════╦═════════════════╗
║ Symbol ║ Quantity ║ Purchase Price ║ Current Price ║    Value   ║ Gain/Loss (%)   ║
╠════════╬══════════╬════════════════╬═══════════════╬════════════╬═════════════════╣
║  AAPL  ║   10.0   ║    $150.00     ║    $178.50    ║  $1,785.00 ║ +$285.00 (+19%) ║
║  MSFT  ║    5.0   ║    $300.00     ║    $385.20    ║  $1,926.00 ║ +$426.00 (+28%) ║
║ GOOGL  ║    3.0   ║   $2,800.00    ║   $2,950.00   ║  $8,850.00 ║ +$450.00 (+5%)  ║
╚════════╩══════════╩════════════════╩═══════════════╩════════════╩═════════════════╝

Portfolio Summary:
Total Value: $12,561.00
Total Gain/Loss: +$1,161.00 (+10.2%)
```

## GitHub Actions Automation ⚡

**NEW!** Automate your stock reports with GitHub Actions - no server required!

### Features

- 📅 **Scheduled Reports**: Daily, weekly, and monthly automated reports
- 🚨 **Event-Based Triggers**: Automatically generate reports when market events affect your portfolio
- 🔒 **Secure**: All credentials stored in GitHub Secrets
- 📧 **Email Delivery**: Reports sent directly to your inbox
- 🐳 **Docker Execution**: Containerized workflows for consistency

### Quick Setup

1. **Configure GitHub Secrets** (Settings → Secrets → Actions):
   ```
   GROQ_API_KEY
   ALPHA_VANTAGE_API_KEY
   TAVILY_API_KEY
   EMAIL_SMTP_SERVER
   EMAIL_SMTP_PORT
   EMAIL_ADDRESS
   EMAIL_PASSWORD
   EMAIL_RECIPIENT
   PORTFOLIO_POSITIONS
   ```

2. **Set Portfolio Positions** (JSON format):
   ```json
   [{"symbol":"AAPL","quantity":10,"purchase_price":150.0},{"symbol":"GOOGL","quantity":5,"purchase_price":2800.0}]
   ```

3. **Enable GitHub Actions** in your repository settings

4. **Done!** Reports will be automatically generated and emailed on schedule

### Workflows

- **Scheduled Reports**: Daily (8 AM UTC), Weekly (Mon 6 AM), Monthly (1st at 7 AM)
- **Event-Based**: Checks every 2 hours for market events affecting your portfolio
- **Docker-Based**: Containerized execution (8:30 AM UTC daily)

### Get Started

📖 **[Complete Setup Guide](GITHUB_ACTIONS_SETUP.md)** - Detailed instructions with troubleshooting

**Get Tavily API Key**: https://tavily.com (Free tier: 1,000 credits/month)

---

## Docker Deployment

### Build and Run

```bash
# Build the image
docker build -t stock-tracker:latest .

# Run one-time report
docker run --rm stock-tracker:latest report

# Run AI report with email
docker run --rm \
  -e GROQ_API_KEY="your_key" \
  -e ALPHA_VANTAGE_API_KEY="your_key" \
  -e EMAIL_ADDRESS="your@gmail.com" \
  -e EMAIL_PASSWORD="app_password" \
  -e EMAIL_RECIPIENT="recipient@example.com" \
  -e PORTFOLIO_POSITIONS='[{"symbol":"AAPL","quantity":10,"purchase_price":150}]' \
  stock-tracker:latest ai-report --email
```

### Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
TAVILY_API_KEY=your_tavily_api_key
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_ADDRESS=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
EMAIL_RECIPIENT=recipient@example.com
PORTFOLIO_POSITIONS=[{"symbol":"AAPL","quantity":10,"purchase_price":150.0}]
```

## Configuration Files

The CLI stores data in platform-specific directories:

- **Linux**: `~/.config/StockTrackerCLI/` and `~/.local/share/StockTrackerCLI/`
- **macOS**: `~/Library/Application Support/StockTrackerCLI/`
- **Windows**: `%LOCALAPPDATA%\StockTrackerCLI\`

### Files:
- `config.json` - API keys and email settings
- `positions.json` - Your stock positions
- `cache.json` - Cached stock data (15-minute expiry)
- `history.json` - Portfolio snapshots and historical performance data
- `alerts.json` - Price alerts configuration
- `watchlist.json` - Watchlist stocks
- `logs/stock_tracker.log` - Application logs

## Requirements

- Python 3.8 or higher
- Internet connection for API access
- API keys:
  - Groq API (for AI analysis)
  - Alpha Vantage API (for stock data)
- SMTP server access (optional, for email reports)

## Dependencies

### Core Dependencies
- click - CLI framework
- pandas - Data manipulation
- groq - AI inference
- alpha-vantage - Stock data API
- yfinance - Historical stock data
- tavily-python - Market event detection and news search
- python-dateutil - Date utilities
- appdirs - Cross-platform directories
- schedule - Task scheduling
- python-dotenv - Environment variable management

### UI & Visualization (New!)
- streamlit - Interactive web dashboard
- plotly - Interactive charts and graphs
- matplotlib - Additional plotting capabilities
- seaborn - Statistical visualizations

### Technical Analysis (New!)
- ta - Technical indicators library
- numpy - Numerical computations
- scipy - Scientific computing

## Gmail Setup

For Gmail users, you'll need to use an App Password:

1. Enable 2-factor authentication on your Google account
2. Go to https://myaccount.google.com/apppasswords
3. Generate an app password for "Mail"
4. Use this app password when running `stock-tracker setup-email`

## License

MIT License - see LICENSE file for details

## Author

Chukwuebuka Ezeokeke - [GitHub](https://github.com/Chukwuebuka-2003)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Issues

Found a bug? Please report it on the [GitHub Issues](https://github.com/Chukwuebuka-2003/stock_cli/issues) page.
