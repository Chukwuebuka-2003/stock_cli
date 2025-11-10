# GitHub Actions Automated Reports Setup Guide

This guide explains how to set up automated stock portfolio reports using GitHub Actions with event-based triggers powered by Tavily API.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Workflow Descriptions](#workflow-descriptions)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

---

## 🎯 Overview

This repository includes three GitHub Actions workflows for automated stock portfolio reporting:

1. **Scheduled Reports** - Daily, weekly, and monthly reports on a fixed schedule
2. **Event-Based Reports** - Triggered by significant market events detected via Tavily API
3. **Docker-Based Reports** - Containerized execution for consistent environments

---

## ✨ Features

- **🕐 Scheduled Reporting**: Daily (weekdays), weekly (Mondays), and monthly (1st of month)
- **🚨 Event-Driven**: Automatically generates reports when market events affect your portfolio
- **🐳 Docker Support**: Run reports in isolated containers
- **🔒 Secure**: All sensitive data stored in GitHub Secrets
- **📧 Email Delivery**: Reports sent directly to your email
- **🤖 AI-Powered**: Uses Groq LLM for intelligent portfolio analysis
- **📰 Market Intelligence**: Integrates Tavily API for real-time market event detection

---

## 📦 Prerequisites

### Required API Keys

1. **Groq API Key** - For AI-powered analysis
   - Sign up at: https://console.groq.com
   - Free tier available

2. **Alpha Vantage API Key** - For real-time stock price data
   - Get free key at: https://www.alphavantage.co/support/#api-key
   - Free tier: 25 requests/day

3. **Tavily API Key** - For market event detection
   - Sign up at: https://tavily.com
   - Free tier: 1,000 credits/month

4. **Email Credentials** (Gmail recommended)
   - Gmail App Password (16 digits)
   - Generate at: https://myaccount.google.com/apppasswords
   - Required for 2FA-enabled accounts

### Repository Access

- Admin access to your GitHub repository
- Ability to add GitHub Secrets

---

## 🚀 Setup Instructions

### Step 1: Configure GitHub Secrets

Navigate to your repository: **Settings → Secrets and variables → Actions → New repository secret**

Add the following secrets:

#### API Keys
```
GROQ_API_KEY=your_groq_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
TAVILY_API_KEY=your_tavily_api_key_here
```

#### Email Configuration
```
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_ADDRESS=your.email@gmail.com
EMAIL_PASSWORD=your_16_digit_app_password
EMAIL_RECIPIENT=recipient@example.com
```

#### Portfolio Positions

This is the most important secret for automation. Format your portfolio as a JSON array:

```json
PORTFOLIO_POSITIONS=[{"symbol":"AAPL","quantity":10,"purchase_price":150.0},{"symbol":"GOOGL","quantity":5,"purchase_price":2800.0},{"symbol":"MSFT","quantity":15,"purchase_price":300.0}]
```

**Important**:
- Remove all whitespace/newlines from the JSON
- Each position must have: `symbol`, `quantity`, `purchase_price`
- All on one line

**Example with more stocks:**
```json
[{"symbol":"AAPL","quantity":10,"purchase_price":150.0},{"symbol":"GOOGL","quantity":5,"purchase_price":2800.0},{"symbol":"MSFT","quantity":15,"purchase_price":300.0},{"symbol":"TSLA","quantity":8,"purchase_price":700.0},{"symbol":"NVDA","quantity":12,"purchase_price":400.0}]
```

### Step 2: Enable GitHub Actions

1. Go to **Settings → Actions → General**
2. Under "Actions permissions", select:
   - ✅ Allow all actions and reusable workflows
3. Under "Workflow permissions", select:
   - ✅ Read and write permissions
4. Click **Save**

### Step 3: Verify Workflows

1. Navigate to **Actions** tab in your repository
2. You should see three workflows:
   - 📊 Scheduled Stock Reports
   - 🚨 Event-Based Stock Reports
   - 🐳 Docker-Based Stock Reports

### Step 4: Test Manual Trigger

1. Go to **Actions → Scheduled Stock Reports**
2. Click **Run workflow** dropdown
3. Select report type: `daily`
4. Click **Run workflow**
5. Wait for completion and check your email!

---

## 📅 Workflow Descriptions

### 1. Scheduled Reports (`scheduled-reports.yml`)

**Triggers:**
- **Daily**: Monday-Friday at 8:00 AM UTC (before US market open)
- **Weekly**: Every Monday at 6:00 AM UTC
- **Monthly**: 1st day of month at 7:00 AM UTC
- **Manual**: On-demand via workflow_dispatch

**What it does:**
- Generates AI-powered portfolio report
- Includes current positions, values, and gains/losses
- Sends HTML-formatted email with analysis
- Creates GitHub Actions summary

**Customization:**
Edit the cron schedules in `.github/workflows/scheduled-reports.yml`:
```yaml
schedule:
  - cron: '0 8 * * 1-5'  # Daily at 8 AM UTC, weekdays
  - cron: '0 6 * * 1'    # Weekly on Monday at 6 AM UTC
  - cron: '0 7 1 * *'    # Monthly on 1st at 7 AM UTC
```

### 2. Event-Based Reports (`event-based-reports.yml`)

**Triggers:**
- **Automatic**: Every 2 hours during trading days (Monday-Friday)
- **Manual**: On-demand with option to force report

**How it works:**
1. **Event Detection**: Uses Tavily API to search for:
   - News about stocks in your portfolio
   - General market-moving events
   - Earnings announcements, regulatory changes, etc.

2. **Intelligence Filter**: Only generates reports if:
   - High-relevance events detected (score ≥ 0.7)
   - Events specifically mention your portfolio stocks
   - OR manual force_report flag is set

3. **Smart Reporting**:
   - Skips report if no significant events
   - Includes event context in the email
   - Saves API costs by filtering noise

**Customization:**
Adjust detection frequency in `.github/workflows/event-based-reports.yml`:
```yaml
schedule:
  - cron: '0 */2 * * 1-5'  # Every 2 hours, weekdays
  # Change */2 to */4 for every 4 hours
  # Change */2 to */1 for every 1 hour
```

Adjust relevance threshold in `src/market_events.py`:
```python
should_trigger = detector.should_trigger_report(
    events_by_symbol,
    threshold=0.7  # Lower for more reports (0.5), higher for fewer (0.8)
)
```

### 3. Docker-Based Reports (`docker-reports.yml`)

**Triggers:**
- **Scheduled**: Daily at 8:30 AM UTC (weekdays)
- **Manual**: Run any stock-tracker command

**Benefits:**
- Consistent execution environment
- Isolated dependencies
- Reproducible builds
- Easy local testing

**Manual Execution Examples:**
```bash
# Generate AI report
gh workflow run docker-reports.yml -f report_command="ai-report --email"

# Generate plain text report
gh workflow run docker-reports.yml -f report_command="report"

# Custom command
gh workflow run docker-reports.yml -f report_command="ai-report"
```

---

## 🔧 Troubleshooting

### No Email Received

**Check:**
1. Email secrets are correctly set (no extra spaces)
2. Gmail App Password is 16 digits (no spaces)
3. SMTP server is `smtp.gmail.com` and port is `587`
4. Check workflow logs for email errors
5. Verify EMAIL_RECIPIENT is correct

**Gmail App Password Setup:**
1. Enable 2-Factor Authentication on your Google account
2. Go to: https://myaccount.google.com/apppasswords
3. Select "Mail" and your device
4. Copy the 16-digit password (no spaces)
5. Use this as EMAIL_PASSWORD secret

### Portfolio Positions Not Loading

**Common Issues:**
- JSON format incorrect (use validator: https://jsonlint.com)
- Whitespace/newlines in the secret
- Missing required fields: `symbol`, `quantity`, `purchase_price`

**Test your JSON locally:**
```bash
export PORTFOLIO_POSITIONS='[{"symbol":"AAPL","quantity":10,"purchase_price":150.0}]'
stock-tracker report
```

### Tavily API Not Working

**Check:**
1. API key is valid and has credits
2. Run manual test:
```bash
export TAVILY_API_KEY="tvly-your-key"
export PORTFOLIO_POSITIONS='[{"symbol":"AAPL","quantity":10,"purchase_price":150}]'
python -c "
import sys
sys.path.insert(0, 'src')
from market_events import check_portfolio_events
import json, os
result = check_portfolio_events(json.loads(os.getenv('PORTFOLIO_POSITIONS')))
print(result)
"
```

### Workflow Not Running

**Check:**
1. Actions are enabled in Settings → Actions
2. Cron schedule is correct (uses UTC timezone)
3. Repository has activity (scheduled workflows require recent commits)
4. Check Actions tab for any error messages

**Force a workflow run:**
- Use "Run workflow" button for manual trigger
- Make a commit to activate scheduled workflows

### API Rate Limits

**Alpha Vantage (25 requests/day on free tier):**
- Scheduled reports use ~1 request per stock per day
- Cache is 15 minutes (900 seconds)
- Consider upgrading for larger portfolios

**Tavily (1,000 credits/month on free tier):**
- Event checks use ~2-10 credits per run
- Running every 2 hours ≈ 300-600 credits/month
- Adjust check frequency if needed

---

## ⚙️ Advanced Configuration

### Adjust Report Schedules

Edit cron expressions in workflow files. Use https://crontab.guru for help.

**Examples:**
```yaml
# Before market open (9:30 AM ET = 1:30 PM UTC)
- cron: '30 13 * * 1-5'

# After market close (4:00 PM ET = 8:00 PM UTC)
- cron: '0 20 * * 1-5'

# Every 4 hours during trading
- cron: '0 */4 * * 1-5'
```

### Customize Email Recipients

**Single recipient:**
Set `EMAIL_RECIPIENT` to one email address.

**Multiple recipients:**
Modify `src/reporting.py` to support comma-separated recipients:
```python
recipients = self.email_settings.get("recipient", "").split(",")
msg["To"] = ", ".join([r.strip() for r in recipients])
```

Then set secret:
```
EMAIL_RECIPIENT=user1@example.com,user2@example.com,user3@example.com
```

### Local Testing with Docker

Build and run locally:
```bash
# Build image
docker build -t stock-tracker:latest .

# Run with your environment variables
docker run --rm \
  -e GROQ_API_KEY="your_key" \
  -e ALPHA_VANTAGE_API_KEY="your_key" \
  -e PORTFOLIO_POSITIONS='[{"symbol":"AAPL","quantity":10,"purchase_price":150}]' \
  stock-tracker:latest ai-report

# Test with email
docker run --rm \
  -e GROQ_API_KEY="your_key" \
  -e ALPHA_VANTAGE_API_KEY="your_key" \
  -e EMAIL_SMTP_SERVER="smtp.gmail.com" \
  -e EMAIL_SMTP_PORT="587" \
  -e EMAIL_ADDRESS="your@gmail.com" \
  -e EMAIL_PASSWORD="your_app_password" \
  -e EMAIL_RECIPIENT="recipient@example.com" \
  -e PORTFOLIO_POSITIONS='[{"symbol":"AAPL","quantity":10,"purchase_price":150}]' \
  stock-tracker:latest ai-report --email
```

### Environment-Specific Configurations

Create different workflows for different environments:

**Production** (`.github/workflows/prod-reports.yml`):
- Uses main branch
- Sends to production email list
- Higher event detection threshold

**Staging** (`.github/workflows/staging-reports.yml`):
- Uses develop branch
- Sends to test email
- Lower event threshold for testing

### Monitoring and Alerts

**Setup notifications:**
1. Go to repository **Settings → Notifications**
2. Enable "Send notifications for failed workflows"
3. Configure email/Slack/Discord webhooks

**View workflow history:**
- Actions tab shows all runs with logs
- Filter by workflow, status, branch
- Download logs for debugging

---

## 📊 Usage Statistics

Track your automation usage:

### API Calls (Approximate)

**With 5 stocks in portfolio:**
- Daily reports: ~5 Alpha Vantage calls/day
- Event checks (every 2 hours): ~20 Tavily credits/day
- Monthly totals:
  - Alpha Vantage: ~100 calls (within free tier)
  - Tavily: ~400-600 credits (within free tier)

**Optimization tips:**
- Reduce event check frequency to every 4 hours
- Use cache effectively (already implemented)
- Monitor usage in respective dashboards

---

## 🎉 Success Checklist

- [ ] All GitHub Secrets configured
- [ ] GitHub Actions enabled
- [ ] Manual workflow test successful
- [ ] Email received successfully
- [ ] Portfolio positions loading correctly
- [ ] Tavily event detection working
- [ ] Docker build successful
- [ ] Scheduled workflows enabled
- [ ] Notifications configured

---

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Cron Schedule Expressions](https://crontab.guru)
- [Groq API Documentation](https://console.groq.com/docs)
- [Alpha Vantage API Docs](https://www.alphavantage.co/documentation/)
- [Tavily API Documentation](https://docs.tavily.com)
- [Gmail App Passwords Guide](https://support.google.com/accounts/answer/185833)

---

## 🆘 Getting Help

If you encounter issues:

1. Check workflow logs in Actions tab
2. Review this troubleshooting guide
3. Verify all secrets are correctly set
4. Test commands locally first
5. Open an issue with:
   - Workflow logs (remove sensitive data)
   - Environment details
   - Steps to reproduce

---

## 🔄 Updating Portfolio Positions

To update your portfolio in GitHub Actions:

1. Go to **Settings → Secrets → Actions**
2. Find `PORTFOLIO_POSITIONS` secret
3. Click **Update**
4. Paste new JSON (format: same as setup)
5. Click **Update secret**

**Quick Update Script:**
```python
# Generate JSON from your current positions
import json

positions = [
    {"symbol": "AAPL", "quantity": 10, "purchase_price": 150.0},
    {"symbol": "GOOGL", "quantity": 5, "purchase_price": 2800.0},
    # Add more...
]

print(json.dumps(positions, separators=(',', ':')))
```

Copy the output and update the secret.

---

**🎉 Congratulations! Your automated stock reporting system is ready!** 📈
