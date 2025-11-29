"""
Streamlit UI for Stock Tracker CLI

Interactive dashboard for portfolio analysis, stock charts, and technical indicators.
"""

import html
import logging
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

try:
    from markdown import markdown as md_to_html
except ImportError:  # pragma: no cover
    md_to_html = None

from src.backtesting import Backtester
from src.config import ConfigManager
from src.data_fetcher import DataFetcher
from src.feature_engineering import FeatureEngineer
from src.ml_models import MLPredictor, ProphetPredictor, train_ensemble_models
from src.portfolio import PortfolioManager
from src.technical_indicators import TechnicalIndicators
from src.watchlist import WatchlistManager
from src.agents.orchestrator import AgentOrchestrator
from src.rag.vector_store import VectorStore
from src.rag.embeddings import EmbeddingService
from src.rag.processor import DocumentProcessor
from src.sec_filings import SECFilingsClient
from groq import Groq
import appdirs
from stock_cli.file_paths import CONFIG_PATH, POSITIONS_PATH, WATCHLIST_PATH

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Stock Tracker Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .positive {
        color: #00c853;
    }
    .negative {
        color: #ff1744;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
    .chat-history {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .chat-bubble {
        max-width: 75%;
        padding: 0.9rem 1rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background-color: rgba(240, 242, 246, 0.05);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
    }
    .chat-bubble.user {
        margin-left: auto;
        background: linear-gradient(135deg, #1f8ef1, #5ac8fa);
        color: #fff;
    }
    .chat-bubble.assistant {
        margin-right: auto;
        background: rgba(255,255,255,0.04);
    }
    .chat-role {
        font-size: 0.8rem;
        opacity: 0.8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.2rem;
    }
    .chat-text {
        font-size: 0.95rem;
        line-height: 1.4;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .chat-input-wrapper {
        margin-top: 1.5rem;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        background-color: rgba(255,255,255,0.03);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_config_manager():
    """Get cached ConfigManager instance."""
    return ConfigManager(CONFIG_PATH)


@st.cache_resource
def get_data_fetcher():
    """Get cached DataFetcher instance with Twelve Data."""
    config = get_config_manager()
    
    # Get Twelve Data API key
    twelvedata_key = config.get("twelvedata_api_key")
    
    if not twelvedata_key:
        st.error("⚠️ Twelve Data API key not configured. Please add TWELVE_DATA_API_KEY to your .env file.")
        st.stop()
    
    st.success("✅ Using Twelve Data API (800 calls/day)")
    
    return DataFetcher(twelvedata_api_key=twelvedata_key)



@st.cache_data(ttl=900)  # Cache for 15 minutes
def get_historical_data(symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
    """Fetch and cache historical data for a symbol."""
    fetcher = get_data_fetcher()
    return fetcher.get_historical_data(symbol, period=period)


@st.cache_data(ttl=900)
def get_stock_quote(symbol: str) -> Optional[dict]:
    """Fetch and cache current stock quote."""
    fetcher = get_data_fetcher()
    return fetcher.get_stock_data(symbol)


@st.cache_resource
def get_embedding_service() -> EmbeddingService:
    """Load embedding model once per session."""
    return EmbeddingService()


@st.cache_resource
def get_vector_store() -> VectorStore:
    """Provide a persistent vector store for RAG features."""
    try:
        rag_dir = resolve_rag_directory()
    except RuntimeError as exc:
        st.error(f"⚠️ {exc}")
        st.stop()

    embedding_service = get_embedding_service()
    return VectorStore(persist_directory=rag_dir, embedding_service=embedding_service)


@st.cache_resource
def get_sec_client(user_agent: str, sec_api_key: Optional[str]) -> SECFilingsClient:
    """Cache the SEC filings client per user-agent string."""
    cache_dir = os.path.join(appdirs.user_cache_dir("StockTrackerCLI", "Chukwuebuka"), "sec_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return SECFilingsClient(cache_dir=cache_dir, user_agent=user_agent, sec_api_key=sec_api_key)


def _ensure_writable_directory(path: Path) -> Optional[Path]:
    """Return the path if it is writable, otherwise None."""
    resolved = path.expanduser()
    try:
        resolved.mkdir(parents=True, exist_ok=True)
        test_file = resolved / ".write_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
        return resolved
    except OSError as exc:
        logger.warning("Unable to use vector store directory '%s': %s", resolved, exc)
        return None


def resolve_rag_directory() -> str:
    """Pick a writable directory for the vector store."""
    candidates: List[Path] = []

    custom_dir = os.getenv("STOCK_TRACKER_RAG_DIR")
    if custom_dir:
        candidates.append(Path(custom_dir))

    appdir_path = Path(appdirs.user_data_dir("StockTrackerCLI", "Chukwuebuka")) / "rag_storage"
    candidates.append(appdir_path)

    project_root = Path(__file__).resolve().parent.parent
    candidates.append(project_root / ".rag_storage")

    candidates.append(Path.cwd() / ".rag_storage")
    candidates.append(Path(tempfile.gettempdir()) / "stock_tracker_cli_rag")

    for candidate in candidates:
        writable = _ensure_writable_directory(candidate)
        if writable:
            return str(writable)

    raise RuntimeError("Unable to find a writable directory for the vector store.")


def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure metadata only contains types supported by Chroma."""
    sanitized: Dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            sanitized[key] = ""
        elif isinstance(value, (bool, int, float, str)):
            sanitized[key] = value
        else:
            sanitized[key] = str(value)
    return sanitized


def aggregate_positions_by_symbol(positions_list: List[dict]) -> dict:
    """
    Aggregate multiple positions of the same symbol.
    
    When a user buys the same stock multiple times at different prices,
    this function combines them into a single position with:
    - Total quantity (sum of all positions)
    - Weighted average purchase price (based on cost basis)
    
    Args:
        positions_list: List of position dictionaries
        
    Returns:
        Dictionary with symbol as key and aggregated data as value
    """
    aggregated = {}
    
    for pos in positions_list:
        symbol = pos["symbol"]
        
        if symbol not in aggregated:
            aggregated[symbol] = {
                "symbol": symbol,
                "quantity": 0,
                "purchase_price": 0,  # Will be weighted average
                "total_cost": 0,
                "positions": []  # Keep track of individual positions
            }
        
        # Calculate cost basis for this position
        cost = pos["quantity"] * pos["purchase_price"]
        
        # Aggregate
        aggregated[symbol]["quantity"] += pos["quantity"]
        aggregated[symbol]["total_cost"] += cost
        aggregated[symbol]["positions"].append(pos)
    
    # Calculate weighted average purchase price for each symbol
    for symbol, data in aggregated.items():
        if data["quantity"] > 0:
            data["purchase_price"] = data["total_cost"] / data["quantity"]
    
    return aggregated


def run_portfolio_optimization(
    price_history: pd.DataFrame,
    risk_free_rate: float = 0.02,
    n_simulations: int = 4000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run a simple Monte Carlo portfolio optimization to approximate the efficient frontier.

    Args:
        price_history: DataFrame of price history indexed by date with one column per symbol.
        risk_free_rate: Annual risk-free rate (decimal).
        n_simulations: Number of random portfolios to simulate.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary containing frontier data, best Sharpe portfolio, and minimum-volatility portfolio.
    """
    if price_history is None or price_history.empty:
        raise ValueError("Price history is empty. Cannot optimize portfolio.")

    returns = price_history.pct_change().dropna()
    if returns.empty:
        raise ValueError("Insufficient historical data to compute returns.")

    mean_returns = returns.mean() * 252  # annualize
    cov_matrix = returns.cov() * 252     # annualize
    symbols = list(price_history.columns)

    if len(symbols) < 2:
        raise ValueError("At least two symbols are required for portfolio optimization.")

    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(len(symbols)), size=n_simulations)

    exp_returns = weights @ mean_returns.values
    portfolio_vars = np.einsum('ij,jk,ik->i', weights, cov_matrix.values, weights)
    volatilities = np.sqrt(np.maximum(portfolio_vars, 0))

    sharpe_ratios = np.where(
        volatilities > 0,
        (exp_returns - risk_free_rate) / volatilities,
        0
    )

    frontier = pd.DataFrame({
        "expected_return": exp_returns,
        "volatility": volatilities,
        "sharpe": sharpe_ratios
    })

    best_idx = int(np.argmax(sharpe_ratios))
    min_vol_idx = int(np.argmin(volatilities))

    def build_portfolio(idx: int) -> Dict[str, Any]:
        weight_map = {symbol: float(weights[idx][i]) for i, symbol in enumerate(symbols)}
        return {
            "weights": weight_map,
            "expected_return": float(exp_returns[idx]),
            "volatility": float(volatilities[idx]),
            "sharpe": float(sharpe_ratios[idx])
        }

    return {
        "frontier": frontier,
        "best": build_portfolio(best_idx),
        "min_vol": build_portfolio(min_vol_idx),
        "mean_returns": mean_returns,
        "cov_matrix": cov_matrix,
        "symbols": symbols
    }

def format_currency(value: float) -> str:
    """Format value as currency."""
    return f"${value:,.2f}"


def format_percent(value: float) -> str:
    """Format value as percentage."""
    return f"{value:+.2f}%"


def create_candlestick_chart(df: pd.DataFrame, symbol: str, indicators: List[str] = None) -> go.Figure:
    """
    Create an interactive candlestick chart with technical indicators.

    Args:
        df: DataFrame with OHLCV data
        symbol: Stock symbol
        indicators: List of indicators to display

    Returns:
        Plotly figure
    """
    if indicators is None:
        indicators = []

    # Add technical indicators to dataframe
    df_with_indicators = TechnicalIndicators.add_all_indicators(df)

    # Create subplots
    rows = 1
    row_heights = [0.7]
    subplot_titles = [f"{symbol} Price Chart"]

    if "RSI" in indicators:
        rows += 1
        row_heights.append(0.15)
        subplot_titles.append("RSI")

    if "MACD" in indicators:
        rows += 1
        row_heights.append(0.15)
        subplot_titles.append("MACD")

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=row_heights,
        subplot_titles=subplot_titles
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df_with_indicators["Date"],
            open=df_with_indicators["Open"],
            high=df_with_indicators["High"],
            low=df_with_indicators["Low"],
            close=df_with_indicators["Close"],
            name="Price"
        ),
        row=1, col=1
    )

    # Add moving averages
    if "MA" in indicators:
        for ma in ["SMA_20", "SMA_50", "SMA_200"]:
            if ma in df_with_indicators.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_with_indicators["Date"],
                        y=df_with_indicators[ma],
                        name=ma,
                        line=dict(width=1.5)
                    ),
                    row=1, col=1
                )

    # Add Bollinger Bands
    if "BB" in indicators:
        fig.add_trace(
            go.Scatter(
                x=df_with_indicators["Date"],
                y=df_with_indicators["BB_Upper"],
                name="BB Upper",
                line=dict(width=1, dash="dash", color="gray"),
                showlegend=False
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df_with_indicators["Date"],
                y=df_with_indicators["BB_Lower"],
                name="BB Lower",
                line=dict(width=1, dash="dash", color="gray"),
                fill="tonexty",
                fillcolor="rgba(128, 128, 128, 0.1)",
                showlegend=False
            ),
            row=1, col=1
        )

    # Add volume bars
    colors = ["red" if close < open_ else "green"
              for close, open_ in zip(df_with_indicators["Close"], df_with_indicators["Open"])]

    fig.add_trace(
        go.Bar(
            x=df_with_indicators["Date"],
            y=df_with_indicators["Volume"],
            name="Volume",
            marker_color=colors,
            opacity=0.3,
            yaxis="y2"
        ),
        row=1, col=1
    )

    # RSI subplot
    current_row = 2
    if "RSI" in indicators and "RSI" in df_with_indicators.columns:
        fig.add_trace(
            go.Scatter(
                x=df_with_indicators["Date"],
                y=df_with_indicators["RSI"],
                name="RSI",
                line=dict(color="purple", width=1.5)
            ),
            row=current_row, col=1
        )
        # Add overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=current_row, col=1)
        fig.update_yaxes(range=[0, 100], row=current_row, col=1)
        current_row += 1

    # MACD subplot
    if "MACD" in indicators and "MACD" in df_with_indicators.columns:
        fig.add_trace(
            go.Scatter(
                x=df_with_indicators["Date"],
                y=df_with_indicators["MACD"],
                name="MACD",
                line=dict(color="blue", width=1.5)
            ),
            row=current_row, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df_with_indicators["Date"],
                y=df_with_indicators["MACD_Signal"],
                name="Signal",
                line=dict(color="red", width=1.5)
            ),
            row=current_row, col=1
        )
        fig.add_trace(
            go.Bar(
                x=df_with_indicators["Date"],
                y=df_with_indicators["MACD_Histogram"],
                name="Histogram",
                marker_color="gray",
                opacity=0.3
            ),
            row=current_row, col=1
        )

    # Update layout
    fig.update_layout(
        height=700 if rows == 1 else 900,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Add volume axis
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", secondary_y=True, row=1, col=1, showgrid=False)

    return fig


def create_portfolio_pie_chart(portfolio_data: List[dict]) -> go.Figure:
    """Create portfolio composition pie chart."""
    # Filter out positions with None current value (failed quote fetches)
    valid_data = [p for p in portfolio_data if p.get("currentValue") is not None]
    
    if not valid_data:
        # No valid data - return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No current price data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title="Portfolio Composition")
        return fig
    
    symbols = [p["symbol"] for p in valid_data]
    values = [p["currentValue"] for p in valid_data]

    fig = go.Figure(data=[go.Pie(
        labels=symbols,
        values=values,
        hole=0.3,
        textinfo="label+percent",
        textposition="auto",
        marker=dict(line=dict(color="white", width=2))
    )])

    fig.update_layout(
        title="Portfolio Composition",
        height=400,
        showlegend=True
    )

    return fig


def create_performance_chart(portfolio_data: List[dict]) -> go.Figure:
    """Create performance bar chart."""
    # Filter out positions with None gain/loss (failed quote fetches)
    valid_data = [p for p in portfolio_data if p.get("gainLoss") is not None]
    
    if not valid_data:
        # No valid data - return empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No current price data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title="Performance by Stock")
        return fig
    
    symbols = [p["symbol"] for p in valid_data]
    gains = [p["gainLoss"] for p in valid_data]
    colors = ["green" if g > 0 else "red" for g in gains]

    fig = go.Figure(data=[go.Bar(
        x=symbols,
        y=gains,
        marker_color=colors,
        text=[f"${g:,.2f}" for g in gains],
        textposition="auto"
    )])

    fig.update_layout(
        title="Gain/Loss by Stock",
        xaxis_title="Symbol",
        yaxis_title="Gain/Loss ($)",
        height=400
    )

    return fig


def display_portfolio_overview():
    """Display portfolio overview tab."""
    st.header("📊 Portfolio Overview")

    try:
        portfolio = PortfolioManager(POSITIONS_PATH)
        positions_list = portfolio.get_positions()
        
        # Aggregate positions by symbol (handles multiple buys of same stock)
        positions = aggregate_positions_by_symbol(positions_list)

        if not positions:
            st.info("📝 No positions in portfolio. Add stocks using the CLI: `stock-tracker add SYMBOL QUANTITY PRICE`")
            return

        # Fetch current data for all positions
        fetcher = get_data_fetcher()
        portfolio_data = []
        total_value = 0
        total_cost = 0

        for symbol, data in positions.items():
            quote = get_stock_quote(symbol)
            quantity = data["quantity"]
            purchase_price = data["purchase_price"]
            
            if quote:
                # Successfully fetched current price
                current_price = quote["currentPrice"]
                current_value = current_price * quantity
                cost_basis = purchase_price * quantity
                gain_loss = current_value - cost_basis
                gain_loss_pct = (gain_loss / cost_basis) * 100 if cost_basis > 0 else 0

                portfolio_data.append({
                    "symbol": symbol,
                    "quantity": quantity,
                    "purchasePrice": purchase_price,
                    "currentPrice": current_price,
                    "currentValue": current_value,
                    "costBasis": cost_basis,
                    "gainLoss": gain_loss,
                    "gainLossPct": gain_loss_pct
                })

                total_value += current_value
                total_cost += cost_basis
            else:
                # Quote fetch failed - show position with N/A for current data
                cost_basis = purchase_price * quantity
                
                portfolio_data.append({
                    "symbol": symbol,
                    "quantity": quantity,
                    "purchasePrice": purchase_price,
                    "currentPrice": None,  # Will display as N/A
                    "currentValue": None,
                    "costBasis": cost_basis,
                    "gainLoss": None,
                    "gainLossPct": None
                })
                
                total_cost += cost_basis
                # Note: Don't add to total_value since we don't have current price

        # Display key metrics
        total_gain_loss = total_value - total_cost
        total_gain_loss_pct = (total_gain_loss / total_cost) * 100 if total_cost > 0 else 0

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Total Portfolio Value",
                value=format_currency(total_value),
                delta=format_currency(total_gain_loss)
            )

        with col2:
            st.metric(
                label="Total Cost Basis",
                value=format_currency(total_cost)
            )

        with col3:
            st.metric(
                label="Total Gain/Loss",
                value=format_currency(total_gain_loss),
                delta=format_percent(total_gain_loss_pct)
            )

        with col4:
            st.metric(
                label="Number of Holdings",
                value=len(portfolio_data)
            )

        st.markdown("---")

        # Portfolio composition and performance charts
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                create_portfolio_pie_chart(portfolio_data),
                width='stretch'
            )

        with col2:
            st.plotly_chart(
                create_performance_chart(portfolio_data),
                width='stretch'
            )

        st.markdown("---")

        # Detailed positions table
        st.subheader("📋 Detailed Positions")

        df = pd.DataFrame(portfolio_data)
        df = df[[
            "symbol", "quantity", "purchasePrice", "currentPrice",
            "currentValue", "gainLoss", "gainLossPct"
        ]]
        df.columns = [
            "Symbol", "Quantity", "Purchase Price", "Current Price",
            "Current Value", "Gain/Loss ($)", "Gain/Loss (%)"
        ]

        # Format the dataframe
        styled_df = df.style.format({
            "Purchase Price": "${:.2f}",
            "Current Price": "${:.2f}",
            "Current Value": "${:,.2f}",
            "Gain/Loss ($)": "${:,.2f}",
            "Gain/Loss (%)": "{:+.2f}%"
        }).applymap(
            lambda x: "color: green" if isinstance(x, (int, float)) and x > 0 else "color: red" if isinstance(x, (int, float)) and x < 0 else "",
            subset=["Gain/Loss ($)", "Gain/Loss (%)"]
        )

        st.dataframe(styled_df, width='stretch', hide_index=True)

    except Exception as e:
        st.error(f"Error loading portfolio: {e}")
        logger.error(f"Portfolio overview error: {e}", exc_info=True)


def display_stock_analysis():
    """Display individual stock analysis tab."""
    st.header("📈 Stock Analysis")

    # Stock selector
    portfolio = PortfolioManager(POSITIONS_PATH)
    positions_list = portfolio.get_positions()
    
    # Aggregate positions by symbol (handles multiple buys of same stock)
    positions = aggregate_positions_by_symbol(positions_list)

    if not positions:
        st.info("📝 No positions in portfolio. Add stocks using the CLI.")
        return

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        selected_symbol = st.selectbox(
            "Select Stock",
            options=list(positions.keys()),
            index=0
        )

    with col2:
        period = st.selectbox(
            "Time Period",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
            index=3
        )

    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        refresh = st.button("🔄 Refresh")

    if refresh:
        st.cache_data.clear()

    # Fetch historical data
    with st.spinner(f"Loading {selected_symbol} data..."):
        df = get_historical_data(selected_symbol, period)

    if df is None or df.empty:
        st.error(f"Failed to fetch data for {selected_symbol}")
        return

    # Current quote
    quote = get_stock_quote(selected_symbol)
    if quote:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Current Price",
                value=format_currency(quote["currentPrice"]),
                delta=format_currency(quote["change"])
            )

        with col2:
            st.metric(
                label="Change %",
                value=quote["changePercent"]
            )

        with col3:
            st.metric(
                label="Previous Close",
                value=format_currency(quote["previousClose"])
            )

        with col4:
            position = positions[selected_symbol]
            gain_loss = (quote["currentPrice"] - position["purchase_price"]) * position["quantity"]
            st.metric(
                label="Your Position P/L",
                value=format_currency(gain_loss)
            )

    st.markdown("---")

    # Technical indicators selection
    st.subheader("Technical Indicators")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        show_ma = st.checkbox("Moving Averages", value=True)
    with col2:
        show_bb = st.checkbox("Bollinger Bands", value=False)
    with col3:
        show_rsi = st.checkbox("RSI", value=True)
    with col4:
        show_macd = st.checkbox("MACD", value=True)

    indicators = []
    if show_ma:
        indicators.append("MA")
    if show_bb:
        indicators.append("BB")
    if show_rsi:
        indicators.append("RSI")
    if show_macd:
        indicators.append("MACD")

    # Display chart
    fig = create_candlestick_chart(df, selected_symbol, indicators)
    st.plotly_chart(fig, width='stretch')

    # Technical signals
    st.subheader("📊 Technical Signals")
    df_with_indicators = TechnicalIndicators.add_all_indicators(df)
    signals = TechnicalIndicators.get_indicator_signals(df_with_indicators)

    if signals:
        cols = st.columns(len(signals))
        for idx, (indicator, signal) in enumerate(signals.items()):
            with cols[idx]:
                signal_color = "🟢" if "Bullish" in signal or "Oversold" in signal else "🔴" if "Bearish" in signal or "Overbought" in signal else "🟡"
                st.info(f"{signal_color} **{indicator}**\n\n{signal}")
    else:
        st.info("Not enough data to generate signals")


def display_watchlist():
    """Display watchlist tab."""
    st.header("👀 Watchlist")

    try:
        watchlist = WatchlistManager(WATCHLIST_PATH)
        stocks = watchlist.get_stocks()

        if not stocks:
            st.info("📝 No stocks in watchlist. Add stocks using: `stock-tracker watchlist add SYMBOL`")
            return

        # Fetch current data for watchlist
        watchlist_data = []

        for stock in stocks:
            symbol = stock["symbol"]
            quote = get_stock_quote(symbol)
            if quote:
                watchlist_data.append({
                    "Symbol": symbol,
                    "Current Price": quote["currentPrice"],
                    "Change": quote["change"],
                    "Change %": quote["changePercent"],
                    "Previous Close": quote["previousClose"],
                    "Note": stock.get("note", "")
                })

        if watchlist_data:
            df = pd.DataFrame(watchlist_data)

            styled_df = df.style.format({
                "Current Price": "${:.2f}",
                "Change": "${:+.2f}",
                "Previous Close": "${:.2f}"
            }).applymap(
                lambda x: "color: green" if isinstance(x, (int, float)) and x > 0 else "color: red" if isinstance(x, (int, float)) and x < 0 else "",
                subset=["Change"]
            )

            st.dataframe(styled_df, width='stretch', hide_index=True)

        st.markdown("---")
        st.info("💡 Use the CLI to add or remove watchlist items: `stock-tracker watchlist add/remove SYMBOL`")

    except Exception as e:
        st.error(f"Error loading watchlist: {e}")
        logger.error(f"Watchlist error: {e}", exc_info=True)


def display_ml_predictions():
    """Display ML predictions tab."""
    st.header("🤖 ML Price Predictions")

    portfolio = PortfolioManager(POSITIONS_PATH)
    positions_list = portfolio.get_positions()
    
    # Aggregate positions by symbol (handles multiple buys of same stock)
    positions = aggregate_positions_by_symbol(positions_list)

    if not positions:
        st.info("📝 No positions in portfolio. Add stocks using the CLI.")
        return

    # Warning about data limitations
    # Warning about data limitations
    # Twelve Data free tier supports historical data, but we should still be mindful of limits
    st.info(
        "💡 **Tip**: ML models require significant historical data (500+ days) for reliable predictions. "
        "Ensure your API plan supports the requested data range."
    )
    
    st.info(
        "💡 **Tip**: With limited data, the technical analysis in the 'Stock Analysis' tab "
        "provides more reliable insights than ML predictions."
    )
    
    st.markdown("---")

    # Model selection
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        selected_symbol = st.selectbox(
            "Select Stock for Prediction",
            options=list(positions.keys()),
            index=0,
            key="ml_symbol"
        )

    with col2:
        period = st.selectbox(
            "Training Data Period",
            options=["6mo", "1y", "2y", "5y"],
            index=2,
            key="ml_period"
        )

    with col3:
        horizon = st.number_input(
            "Predict Days Ahead",
            min_value=1,
            max_value=30,
            value=5
        )

    st.markdown("---")

    # Train models button
    if st.button("🚀 Train Models & Generate Predictions", type="primary"):
        with st.spinner(f"Training models for {selected_symbol}..."):
            try:
                # Fetch historical data
                df = get_historical_data(selected_symbol, period)

                if df is None or len(df) < 100:
                    st.error(
                        "❌ **Insufficient Data**: Unable to fetch enough historical data for training. "
                        f"Received only {len(df) if df is not None else 0} days of data."
                    )
                    st.info(
                        "💡 Ensure your API key has access to historical data. "
                        "ML models need 500+ days for reliable predictions. "
                        "Consider upgrading to Premium for full historical data access."
                    )
                    return

                st.info(f"📊 Training on {len(df)} days of historical data...")

                # Train ensemble models
                results = train_ensemble_models(df, selected_symbol, horizon)

                if not results:
                    st.error(
                        "❌ **Training Failed**: Unable to train models with available data. "
                        "This typically happens when there's insufficient data after feature engineering."
                    )
                    st.info(
                        "🔍 **Why this happens**: Creating technical indicators (50-day moving averages, etc.) "
                        "requires sufficient historical data. "
                        "After removing incomplete rows, there's not enough data left for training."
                    )
                    return

                # Display results in tabs
                pred_tabs = st.tabs(["📈 Prophet Forecast", "🌲 Random Forest", "⚡ XGBoost", "📊 Model Comparison"])

                # Prophet tab
                with pred_tabs[0]:
                    st.subheader("Prophet Time Series Forecast")

                    prophet_model = results['prophet']['model']
                    forecast_df = prophet_model.predict(periods=30)

                    # Create forecast plot
                    fig = go.Figure()

                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=df['Date'],
                        y=df['Close'],
                        name="Historical",
                        line=dict(color="blue", width=2)
                    ))

                    # Forecast
                    fig.add_trace(go.Scatter(
                        x=forecast_df['Date'],
                        y=forecast_df['Predicted_Price'],
                        name="Forecast",
                        line=dict(color="green", width=2, dash="dash")
                    ))

                    # Confidence intervals
                    fig.add_trace(go.Scatter(
                        x=forecast_df['Date'],
                        y=forecast_df['Upper_Bound'],
                        name="Upper Bound",
                        line=dict(width=0),
                        showlegend=False
                    ))

                    fig.add_trace(go.Scatter(
                        x=forecast_df['Date'],
                        y=forecast_df['Lower_Bound'],
                        name="Lower Bound",
                        line=dict(width=0),
                        fillcolor="rgba(0,255,0,0.2)",
                        fill="tonexty",
                        showlegend=False
                    ))

                    fig.update_layout(
                        title=f"{selected_symbol} - 30 Day Forecast",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=500,
                        hovermode="x unified"
                    )

                    st.plotly_chart(fig, width='stretch')

                    # Show prediction table
                    st.subheader("Forecast Values")
                    st.dataframe(
                        forecast_df.head(10).style.format({
                            "Predicted_Price": "${:.2f}",
                            "Lower_Bound": "${:.2f}",
                            "Upper_Bound": "${:.2f}"
                        }),
                        width='stretch',
                        hide_index=True
                    )

                # Random Forest tab
                with pred_tabs[1]:
                    st.subheader("Random Forest Classification")

                    rf_model = results['rf_classifier']['model']
                    rf_metrics = results['rf_classifier']['metrics']

                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Accuracy", f"{rf_metrics['accuracy']:.2%}")
                    with col2:
                        st.metric("Precision", f"{rf_metrics['precision']:.2%}")
                    with col3:
                        st.metric("Recall", f"{rf_metrics['recall']:.2%}")
                    with col4:
                        st.metric("F1 Score", f"{rf_metrics['f1_score']:.2%}")

                    # Feature importance
                    st.subheader("Top 20 Important Features")
                    importance_df = rf_model.get_feature_importance(top_n=20)

                    fig = go.Figure(go.Bar(
                        x=importance_df['Importance'],
                        y=importance_df['Feature'],
                        orientation='h',
                        marker_color='lightblue'
                    ))

                    fig.update_layout(
                        title="Feature Importance",
                        xaxis_title="Importance",
                        yaxis_title="Feature",
                        height=600,
                        yaxis=dict(autorange="reversed")
                    )

                    st.plotly_chart(fig, width='stretch')

                    # Confusion matrix
                    if 'confusion_matrix' in rf_metrics:
                        st.subheader("Confusion Matrix")
                        cm = np.array(rf_metrics['confusion_matrix'])

                        fig = go.Figure(data=go.Heatmap(
                            z=cm,
                            x=['Predicted Down', 'Predicted Up'],
                            y=['Actual Down', 'Actual Up'],
                            colorscale='Blues',
                            text=cm,
                            texttemplate='%{text}',
                            textfont={"size": 16}
                        ))

                        fig.update_layout(
                            title="Confusion Matrix",
                            height=400
                        )

                        st.plotly_chart(fig, width='stretch')

                # XGBoost tab
                with pred_tabs[2]:
                    st.subheader("XGBoost Classification")

                    xgb_model = results['xgb_classifier']['model']
                    xgb_metrics = results['xgb_classifier']['metrics']

                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Accuracy", f"{xgb_metrics['accuracy']:.2%}")
                    with col2:
                        st.metric("Precision", f"{xgb_metrics['precision']:.2%}")
                    with col3:
                        st.metric("Recall", f"{xgb_metrics['recall']:.2%}")
                    with col4:
                        st.metric("F1 Score", f"{xgb_metrics['f1_score']:.2%}")

                    # Feature importance
                    st.subheader("Top 20 Important Features")
                    importance_df = xgb_model.get_feature_importance(top_n=20)

                    fig = go.Figure(go.Bar(
                        x=importance_df['Importance'],
                        y=importance_df['Feature'],
                        orientation='h',
                        marker_color='lightgreen'
                    ))

                    fig.update_layout(
                        title="Feature Importance",
                        xaxis_title="Importance",
                        yaxis_title="Feature",
                        height=600,
                        yaxis=dict(autorange="reversed")
                    )

                    st.plotly_chart(fig, width='stretch')
                    
                    # Confusion matrix
                    if 'confusion_matrix' in xgb_metrics:
                        st.subheader("Confusion Matrix")
                        cm = np.array(xgb_metrics['confusion_matrix'])

                        fig = go.Figure(data=go.Heatmap(
                            z=cm,
                            x=['Predicted Down', 'Predicted Up'],
                            y=['Actual Down', 'Actual Up'],
                            colorscale='Greens',
                            text=cm,
                            texttemplate='%{text}',
                            textfont={"size": 16}
                        ))

                        fig.update_layout(
                            title="Confusion Matrix",
                            height=400
                        )

                        st.plotly_chart(fig, width='stretch')

                # Model comparison tab
                with pred_tabs[3]:
                    st.subheader("Model Performance Comparison")

                    # Create comparison dataframe
                    comparison_data = {
                        'Model': ['Random Forest', 'XGBoost'],
                        'Accuracy': [rf_metrics['accuracy'], xgb_metrics['accuracy']],
                        'Precision': [rf_metrics['precision'], xgb_metrics['precision']],
                        'Recall': [rf_metrics['recall'], xgb_metrics['recall']],
                        'F1 Score': [rf_metrics['f1_score'], xgb_metrics['f1_score']]
                    }

                    comparison_df = pd.DataFrame(comparison_data)

                    # Display as table
                    st.dataframe(
                        comparison_df.style.format({
                            'Accuracy': '{:.2%}',
                            'Precision': '{:.2%}',
                            'Recall': '{:.2%}',
                            'F1 Score': '{:.2%}'
                        }),
                        width='stretch',
                        hide_index=True
                    )

                    # Radar chart comparison
                    fig = go.Figure()

                    metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
                    rf_values = [rf_metrics['accuracy'], rf_metrics['precision'],
                               rf_metrics['recall'], rf_metrics['f1_score']]
                    xgb_values = [xgb_metrics['accuracy'], xgb_metrics['precision'],
                                xgb_metrics['recall'], xgb_metrics['f1_score']]

                    fig.add_trace(go.Scatterpolar(
                        r=rf_values,
                        theta=metrics_list,
                        fill='toself',
                        name='Random Forest'
                    ))

                    fig.add_trace(go.Scatterpolar(
                        r=xgb_values,
                        theta=metrics_list,
                        fill='toself',
                        name='XGBoost'
                    ))

                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        title="Model Performance Radar Chart",
                        height=500
                    )

                    st.plotly_chart(fig, width='stretch')

                    # Recommendations
                    st.subheader("💡 Model Recommendations")

                    best_model = "Random Forest" if rf_metrics['accuracy'] > xgb_metrics['accuracy'] else "XGBoost"
                    st.success(f"**Best Performing Model:** {best_model}")

                    st.info("""
                    **Interpretation Guide:**
                    - **Accuracy**: Overall correctness of predictions
                    - **Precision**: Of all positive predictions, how many were correct
                    - **Recall**: Of all actual positives, how many were predicted
                    - **F1 Score**: Harmonic mean of precision and recall
                    """)

            except Exception as e:
                st.error(f"Error during model training: {e}")
                logger.error(f"ML prediction error: {e}", exc_info=True)

    else:
        st.info("👆 Click the button above to train models and generate predictions")


def display_backtesting():
    """Display backtesting results tab."""
    st.header("📊 Strategy Backtesting")

    portfolio = PortfolioManager(POSITIONS_PATH)
    positions_list = portfolio.get_positions()
    
    # Aggregate positions by symbol (handles multiple buys of same stock)
    positions = aggregate_positions_by_symbol(positions_list)

    if not positions:
        st.info("📝 No positions in portfolio. Add stocks using the CLI.")
        return

    # Warning about data limitations
    # Warning about data limitations
    st.info(
        "💡 **Tip**: Backtesting requires significant historical data. "
        "Ensure your API plan supports the requested data range."
    )
    
    st.info(
        "💡 **Tip**: With limited data, the technical analysis and current portfolio metrics "
        "provide more reliable insights than backtesting."
    )
    
    st.markdown("---")

    # Configuration
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_symbol = st.selectbox(
            "Select Stock",
            options=list(positions.keys()),
            index=0,
            key="backtest_symbol"
        )

    with col2:
        period = st.selectbox(
            "Backtest Period",
            options=["1y", "2y", "3y", "5y"],
            index=1,
            key="backtest_period"
        )

    with col3:
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=1000,
            max_value=1000000,
            value=100000,
            step=1000
        )

    st.markdown("---")

    # Run backtest button
    if st.button("🔄 Run Backtest", type="primary"):
        with st.spinner(f"Running backtest for {selected_symbol}..."):
            try:
                # Fetch historical data
                df = get_historical_data(selected_symbol, period)

                if df is None or len(df) < 100:
                    st.error("Insufficient data for backtesting.")
                    return

                # Train a simple ML model for signals
                st.info("Training model for trading signals...")
                model = MLPredictor("random_forest", "classification")
                model.train(df, selected_symbol, horizon=5, test_size=0.3)

                # Generate predictions
                predictions = model.predict(df)

                # Create signals (1 for buy, -1 for sell, 0 for hold)
                signals = pd.Series(0, index=range(len(df)))
                if len(predictions) > 0:
                    # Use predictions to generate signals
                    for i in range(len(predictions) - 1):
                        if i + 1 < len(predictions):
                            if predictions[i] == 1 and predictions[i - 1] == 0:  # Buy signal
                                signals.iloc[i] = 1
                            elif predictions[i] == 0 and predictions[i - 1] == 1:  # Sell signal
                                signals.iloc[i] = -1

                # Run backtest
                backtester = Backtester(initial_capital=initial_capital)
                results = backtester.run_strategy(df, signals, commission=0.001)

                if not results:
                    st.error("Backtest failed.")
                    return

                metrics = results['metrics']
                equity_curve = pd.DataFrame(results['equity_curve'])

                # Display key metrics
                st.subheader("📈 Performance Metrics")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Total Return",
                        f"${metrics['total_return']:,.2f}",
                        f"{metrics['total_return_pct']:.2f}%"
                    )

                with col2:
                    st.metric(
                        "Sharpe Ratio",
                        f"{metrics.get('sharpe_ratio', 0):.2f}"
                    )

                with col3:
                    st.metric(
                        "Max Drawdown",
                        f"{metrics.get('max_drawdown_pct', 0):.2f}%"
                    )

                with col4:
                    st.metric(
                        "Win Rate",
                        f"{metrics.get('win_rate_pct', 0):.1f}%"
                    )

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Number of Trades", metrics['num_trades'])

                with col2:
                    st.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.2f}")

                with col3:
                    st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")

                with col4:
                    final_equity = metrics['final_equity']
                    st.metric("Final Equity", f"${final_equity:,.2f}")

                st.markdown("---")

                # Equity curve
                st.subheader("💰 Equity Curve")

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=equity_curve['date'],
                    y=equity_curve['equity'],
                    name="Portfolio Value",
                    line=dict(color="blue", width=2),
                    fill='tonexty'
                ))

                # Add buy-and-hold comparison
                bh_comparison = backtester.compare_with_buy_and_hold(df, results)
                initial_price = df.iloc[0]['Close']
                shares = initial_capital / initial_price
                bh_equity = [shares * df.iloc[i]['Close'] for i in range(len(df))]

                fig.add_trace(go.Scatter(
                    x=df['Date'],
                    y=bh_equity,
                    name="Buy & Hold",
                    line=dict(color="green", width=2, dash="dash")
                ))

                fig.update_layout(
                    title=f"{selected_symbol} Backtest - Equity Curve",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    height=500,
                    hovermode="x unified"
                )

                st.plotly_chart(fig, width='stretch')

                # Comparison with buy-and-hold
                st.subheader("📊 Strategy vs Buy-and-Hold")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Strategy Return",
                        f"{bh_comparison['strategy']['return_pct']:.2f}%"
                    )

                with col2:
                    st.metric(
                        "Buy-and-Hold Return",
                        f"{bh_comparison['buy_and_hold']['return_pct']:.2f}%"
                    )

                with col3:
                    outperformance = bh_comparison['outperformance_pct']
                    st.metric(
                        "Outperformance",
                        f"{outperformance:+.2f}%",
                        delta=f"{outperformance:+.2f}%"
                    )

                # Trade history
                if results['trades']:
                    st.subheader("📜 Trade History")

                    trades_df = pd.DataFrame(results['trades'][:50])  # Show last 50 trades

                    if not trades_df.empty:
                        st.dataframe(
                            trades_df.style.format({
                                'price': '${:.2f}',
                                'cost': '${:.2f}',
                                'proceeds': '${:.2f}',
                                'equity': '${:,.2f}'
                            }),
                            width='stretch',
                            hide_index=True
                        )

            except Exception as e:
                if "Insufficient data" in str(e):
                    st.error(
                        "❌ **Backtesting Failed**: Unable to train model with available data. "
                        "This typically happens when there's insufficient data after feature engineering."
                    )
                    st.info(
                        "🔍 **Why this happens**: Creating technical indicators (50-day moving averages, etc.) "
                        "requires sufficient historical data. "
                        "After removing incomplete rows, there's not enough data left for training."
                    )
                else:
                    st.error(f"Error during backtesting: {e}")
                logger.error(f"Backtesting error: {e}", exc_info=True)

    else:
        st.info("👆 Click the button above to run backtest")

    # Portfolio optimization + risk analytics
    st.markdown("---")
    st.subheader("⚖️ Portfolio Optimization & Risk Analytics")

    available_symbols = list(positions.keys())
    if len(available_symbols) < 2:
        st.info("Add at least two positions to unlock portfolio optimization analytics.")
        return

    opt_col1, opt_col2, opt_col3 = st.columns([2, 1, 1])

    with opt_col1:
        selected_assets = st.multiselect(
            "Select Assets",
            options=available_symbols,
            default=available_symbols,
            key="opt_assets"
        )

    with opt_col2:
        optimization_period = st.selectbox(
            "Historical Window",
            options=["6mo", "1y", "3y", "5y"],
            index=1,
            key="opt_period"
        )

    with opt_col3:
        risk_free_rate = st.number_input(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.25,
            key="opt_risk_free"
        )

    simulations = st.slider(
        "Simulated Portfolios",
        min_value=1000,
        max_value=10000,
        value=4000,
        step=500,
        key="opt_simulations"
    )

    if st.button("🚀 Optimize Portfolio", key="run_portfolio_optimization"):
        if len(selected_assets) < 2:
            st.warning("Select at least two assets to optimize.")
        else:
            with st.spinner("Running mean-variance optimization..."):
                price_series = {}
                unavailable_assets = []

                for symbol in selected_assets:
                    hist = get_historical_data(symbol, optimization_period)
                    if hist is None or hist.empty or 'Close' not in hist.columns:
                        unavailable_assets.append(symbol)
                        continue

                    series = hist[['Date', 'Close']].copy()
                    series['Date'] = pd.to_datetime(series['Date'])
                    series = series.set_index('Date')['Close']
                    price_series[symbol] = series

                if len(price_series) < 2:
                    st.error("Unable to gather overlapping price history for the selected assets.")
                    if unavailable_assets:
                        st.info(f"No historical data for: {', '.join(unavailable_assets)}")
                    return

                price_history = pd.DataFrame(price_series).dropna()

                if price_history.empty or len(price_history) < 30:
                    st.error("Not enough overlapping historical data to run optimization.")
                    return

                try:
                    optimization_results = run_portfolio_optimization(
                        price_history=price_history,
                        risk_free_rate=risk_free_rate / 100,
                        n_simulations=simulations
                    )

                    best_portfolio = optimization_results['best']
                    min_vol_portfolio = optimization_results['min_vol']
                    frontier = optimization_results['frontier']
                    symbols = optimization_results['symbols']

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Max Sharpe Return",
                            f"{best_portfolio['expected_return'] * 100:.2f}%",
                            help="Annualized expected return of the recommended allocation"
                        )

                    with col2:
                        st.metric(
                            "Max Sharpe Volatility",
                            f"{best_portfolio['volatility'] * 100:.2f}%",
                            help="Annualized standard deviation of returns"
                        )

                    with col3:
                        st.metric(
                            "Sharpe Ratio",
                            f"{best_portfolio['sharpe']:.2f}",
                            help="(Return - Risk Free) / Volatility"
                        )

                    st.subheader("Recommended Allocation (Max Sharpe)")
                    weights_df = pd.DataFrame([
                        {
                            "Symbol": symbol,
                            "Allocation (%)": best_portfolio['weights'].get(symbol, 0) * 100
                        }
                        for symbol in symbols
                    ])

                    st.dataframe(
                        weights_df.style.format({"Allocation (%)": "{:.2f}"}),
                        hide_index=True,
                        width='stretch'
                    )

                    cov_matrix = optimization_results['cov_matrix']
                    weight_vector = np.array([best_portfolio['weights'].get(symbol, 0) for symbol in symbols])
                    portfolio_variance = float(weight_vector.T @ cov_matrix.values @ weight_vector)

                    if portfolio_variance > 0:
                        marginal_contrib = cov_matrix.values @ weight_vector
                        risk_contrib = weight_vector * marginal_contrib / portfolio_variance
                        risk_df = pd.DataFrame({
                            "Symbol": symbols,
                            "Risk Contribution (%)": risk_contrib * 100
                        })

                        st.subheader("Risk Contribution by Asset")
                        st.dataframe(
                            risk_df.style.format({"Risk Contribution (%)": "{:.2f}"}),
                            hide_index=True,
                            width='stretch'
                        )

                    st.subheader("Efficient Frontier (Simulated)")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=frontier['volatility'] * 100,
                        y=frontier['expected_return'] * 100,
                        mode='markers',
                        marker=dict(
                            size=6,
                            color=frontier['sharpe'],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Sharpe")
                        ),
                        name="Simulated Portfolios",
                        hovertemplate="Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>"
                    ))

                    fig.add_trace(go.Scatter(
                        x=[best_portfolio['volatility'] * 100],
                        y=[best_portfolio['expected_return'] * 100],
                        mode='markers',
                        marker=dict(size=12, color='red'),
                        name="Max Sharpe"
                    ))

                    fig.add_trace(go.Scatter(
                        x=[min_vol_portfolio['volatility'] * 100],
                        y=[min_vol_portfolio['expected_return'] * 100],
                        mode='markers',
                        marker=dict(size=12, color='orange'),
                        name="Min Volatility"
                    ))

                    fig.update_layout(
                        xaxis_title="Volatility (%)",
                        yaxis_title="Expected Return (%)",
                        height=500,
                        legend=dict(orientation="h", y=-0.2),
                        hovermode="closest"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Portfolio optimization failed: {e}")


def display_sec_filings():
    """Display SEC filings ingestion and question-answering."""
    st.header("📄 SEC Filings Intelligence")

    portfolio = PortfolioManager(POSITIONS_PATH)
    positions = portfolio.get_positions()
    default_symbol = positions[0]['symbol'] if positions else "AAPL"

    config = get_config_manager()
    sec_api_key = config.get("sec_api_key") or os.getenv("SEC_API_KEY")
    contact_email = (
        config.get("sec_contact_email")
        or config.get("contact_email")
        or config.get("user_email")
        or config.get("email")
    )
    default_user_agent = config.get("sec_user_agent") or (
        f"StockTrackerCLI/1.0 ({contact_email})" if contact_email else "StockTrackerCLI/1.0 (your-email@example.com)"
    )

    user_agent = st.text_input(
        "SEC User-Agent",
        value=default_user_agent,
        help="SEC requires a descriptive user-agent string that includes your contact information."
    ).strip()

    ticker = st.text_input(
        "Ticker Symbol",
        value=default_symbol,
        max_chars=10,
        help="Company ticker to fetch filings for."
    ).upper().strip()

    form_type = st.selectbox(
        "Form Type",
        options=["10-K", "10-Q", "8-K", "13F", "S-1", "All"],
        index=0
    )

    filings_limit = st.slider("Recent Filings", min_value=1, max_value=5, value=3)

    if not sec_api_key:
        st.info("Tip: add SEC_API_KEY to enable sec-api for faster, more reliable filing search.")

    if st.button("📥 Fetch Filings", type="primary"):
        if not user_agent or "your-email" in user_agent.lower():
            st.warning("Provide a valid SEC-compliant user-agent (e.g., 'YourApp/1.0 (you@example.com)').")
        elif not ticker:
            st.warning("Enter a ticker symbol to continue.")
        else:
            with st.spinner(f"Downloading {form_type} filings for {ticker}..."):
                try:
                    sec_client = get_sec_client(user_agent, sec_api_key)
                    normalized_form = None if form_type == "All" else form_type
                    filings = sec_client.fetch_filings(
                        symbol=ticker,
                        form_type=normalized_form,
                        limit=filings_limit
                    )
                except Exception as e:
                    st.error(f"Unable to download filings: {e}")
                    filings = []

            if filings:
                vector_store = get_vector_store()
                documents, metadatas, ids = [], [], []

                for filing in filings:
                    metadata = sanitize_metadata({
                        "symbol": filing['symbol'],
                        "form_type": filing['form_type'],
                        "filing_date": filing['filing_date'],
                        "report_date": filing.get('report_date'),
                        "accession_number": filing['accession_number'],
                        "source": f"{filing['symbol']}_{filing['accession_number']}",
                        "url": filing['document_url']
                    })
                    processed_docs = DocumentProcessor.process_document(
                        content=filing['content'],
                        metadata=metadata,
                        chunk_size=1200
                    )
                    for doc in processed_docs:
                        documents.append(doc['text'])
                        metadatas.append(doc['metadata'])
                        ids.append(doc['id'])

                if documents:
                    vector_store.add_documents(
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids,
                        collection_name="sec_filings"
                    )
                    st.success(f"Ingested {len(documents)} chunks from {len(filings)} filing(s).")

                rows = []
                for filing in filings:
                    snippet = DocumentProcessor.clean_text(filing['content'][:800])[:400]
                    rows.append({
                        "Form": filing['form_type'],
                        "Filing Date": filing['filing_date'],
                        "Report Date": filing.get('report_date', '—'),
                        "Accession #": filing['accession_number'],
                        "Primary Document": filing['document_url'],
                        "Summary": snippet + ("…" if len(snippet) == 400 else "")
                    })

                st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')

                for filing in filings:
                    with st.expander(f"{filing['form_type']} • {filing['filing_date']} • {filing['accession_number']}"):
                        st.markdown(f"[View Filing Index]({filing['filing_url']})")
                        st.markdown(f"[Primary Document]({filing['document_url']})")
                        st.write(DocumentProcessor.clean_text(filing['content'][:1200]) + "…")

                st.session_state["sec_filings_indexed"] = True
            else:
                st.info("No filings were returned for the selected filters.")

    st.markdown("---")
    st.subheader("Ask Questions About Indexed Filings")
    question = st.text_area(
        "What would you like to know?",
        placeholder="e.g., What risks did the latest 10-K highlight?"
    )
    symbol_filter = st.text_input(
        "Limit answers to ticker (optional)",
        value=ticker,
        help="Results will be filtered to filings for this symbol if provided."
    ).strip().upper()

    if st.button("🧠 Analyze Filings"):
        if not question.strip():
            st.warning("Enter a question to analyze.")
            return

        groq_key = config.get("groq_api_key")
        if not groq_key:
            st.error("Groq API key not configured. Run `stock-tracker setup-ai` first.")
            return

        try:
            vector_store = get_vector_store()
            search_results = vector_store.query_similar(
                query=question,
                n_results=8,
                collection_name="sec_filings"
            )
        except Exception as e:
            st.error(f"Unable to search indexed filings: {e}")
            return

        documents = search_results.get("documents", [[]])
        metadatas = search_results.get("metadatas", [[]])

        if symbol_filter and documents and documents[0]:
            filtered_docs = []
            filtered_metadatas = []
            for doc, meta in zip(documents[0], metadatas[0]):
                if meta and meta.get("symbol", "").upper() == symbol_filter:
                    filtered_docs.append(doc)
                    filtered_metadatas.append(meta)
            documents = [filtered_docs]
            metadatas = [filtered_metadatas]

        if not documents or not documents[0]:
            st.warning("No SEC filings have been indexed yet. Fetch filings first.")
            return

        contexts = []
        for doc, meta in zip(documents[0], metadatas[0]):
            symbol = meta.get("symbol", "Unknown")
            form = meta.get("form_type", "Form")
            filing_date = meta.get("filing_date", "N/A")
            contexts.append(f"{symbol} {form} ({filing_date}):\n{doc}")

        context_text = "\n\n---\n\n".join(contexts)

        groq_client = Groq(api_key=groq_key)
        with st.spinner("Generating analysis..."):
            try:
                completion = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    temperature=0.2,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a professional equity analyst. "
                                "Use the provided SEC filing excerpts to answer the user's question. "
                                "Cite the relevant forms and filing dates when possible."
                            )
                        },
                        {
                            "role": "user",
                            "content": f"Context:\n{context_text}\n\nQuestion: {question}"
                        }
                    ]
                )
                st.markdown(completion.choices[0].message.content)
            except Exception as e:
                st.error(f"Groq analysis failed: {e}")


def display_chat():
    """Display AI Chat tab."""
    st.header("💬 AI Financial Assistant")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.markdown('<div class="chat-history">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        role = message.get("role", "assistant")
        role_class = "user" if role == "user" else "assistant"
        role_label = "You" if role == "user" else "Advisor"
        raw_content = message.get("content", "")

        if role == "assistant" and md_to_html:
            rendered_content = md_to_html(raw_content)
        else:
            rendered_content = html.escape(raw_content).replace("\n", "<br>")
        st.markdown(
            f"""
            <div class="chat-bubble {role_class}">
                <div class="chat-role">{role_label}</div>
                <div class="chat-text">{rendered_content}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    status_placeholder = st.empty()

    with st.form("chat_form", clear_on_submit=True):
        st.markdown('<div class="chat-input-wrapper">', unsafe_allow_html=True)
        cols = st.columns([5, 1])
        with cols[0]:
            prompt = st.text_area(
                "Your message",
                key="chat_input_text",
                placeholder="Ask about your portfolio or market trends...",
                height=100,
                label_visibility="collapsed",
            )
        with cols[1]:
            submitted = st.form_submit_button("Send", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        prompt = (prompt or "").strip()
        if not prompt:
            status_placeholder.warning("Please enter a question to chat.")
            return

        st.session_state.messages.append({"role": "user", "content": prompt})

        with status_placeholder, st.spinner("Assistant is thinking..."):
            try:
                config = get_config_manager()
                groq_key = config.get("groq_api_key")
                
                if not groq_key:
                    st.error("Groq API key not configured.")
                    return

                tavily_key = config.get("tavily_api_key") or os.getenv("TAVILY_API_KEY")
                if tavily_key:
                    os.environ.setdefault("TAVILY_API_KEY", tavily_key)
                else:
                    st.info(
                        "Tavily API key not set. Web search context will be limited."
                    )

                twelvedata_key = config.get("twelvedata_api_key") or os.getenv("TWELVE_DATA_API_KEY")
                data_fetcher = None
                if twelvedata_key:
                    data_fetcher = DataFetcher(twelvedata_api_key=twelvedata_key)
                else:
                    st.info(
                        "Twelve Data API key not set. Live price lookups unavailable in chat responses."
                    )
                    
                # Initialize components
                groq_client = Groq(api_key=groq_key)
                vector_store = get_vector_store()
                orchestrator = AgentOrchestrator(
                    model_client=groq_client,
                    vector_store=vector_store,
                    tavily_api_key=tavily_key,
                    data_fetcher=data_fetcher,
                )
                
                response = orchestrator.run(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
                try:
                    st.rerun()
                except AttributeError:
                    st.experimental_rerun()
                
            except Exception as e:
                st.error(f"Error generating response: {e}")


def main():
    """Main Streamlit application."""
    # Sidebar
    with st.sidebar:
        st.title("📊 Stock Tracker")
        st.markdown("---")

        st.markdown("### 🔧 Quick Actions")
        st.markdown("""
        **CLI Commands:**
        ```bash
        # Add position
        stock-tracker add SYMBOL QTY PRICE

        # View report
        stock-tracker report

        # Setup APIs
        stock-tracker setup-alpha-vantage
        stock-tracker setup-ai
        ```
        """)

        st.markdown("---")

        # Display last updated time
        st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if st.button("🔄 Clear Cache"):
            st.cache_data.clear()
            st.success("Cache cleared!")

    # Main content area
    st.title("📊 Stock Portfolio Dashboard")

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Portfolio Overview",
        "Stock Analysis",
        "🤖 ML Predictions",
        "📊 Backtesting",
        "📄 SEC Filings",
        "Watchlist",
        "💬 AI Chat"
    ])

    with tab1:
        display_portfolio_overview()

    with tab2:
        display_stock_analysis()

    with tab3:
        display_ml_predictions()

    with tab4:
        display_backtesting()

    with tab5:
        display_sec_filings()

    with tab6:
        display_watchlist()
        
    with tab7:
        display_chat()



def launch_ui():
    """Launch the Streamlit application."""
    import sys
    import os
    from streamlit.web import cli as stcli

    # Get the absolute path of this file
    file_path = os.path.abspath(__file__)

    # Construct the command
    sys.argv = ["streamlit", "run", file_path]

    # Run streamlit
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
