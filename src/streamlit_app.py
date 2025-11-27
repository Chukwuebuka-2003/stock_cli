"""
Streamlit UI for Stock Tracker CLI

Interactive dashboard for portfolio analysis, stock charts, and technical indicators.
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.backtesting import Backtester
from src.config import ConfigManager
from src.data_fetcher import DataFetcher
from src.feature_engineering import FeatureEngineer
from src.ml_models import MLPredictor, ProphetPredictor, train_ensemble_models
from src.portfolio import PortfolioManager
from src.technical_indicators import TechnicalIndicators
from src.watchlist import WatchlistManager
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
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_config_manager():
    """Get cached ConfigManager instance."""
    return ConfigManager(CONFIG_PATH)


@st.cache_resource
def get_data_fetcher():
    """Get cached DataFetcher instance."""
    config = get_config_manager()
    api_key = config.get("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        st.error("⚠️ Alpha Vantage API key not configured. Run `stock-tracker setup-alpha-vantage` first.")
        st.stop()
    return DataFetcher(api_key)


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
    """Create a pie chart showing portfolio composition."""
    symbols = [p["symbol"] for p in portfolio_data]
    values = [p["currentValue"] for p in portfolio_data]

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
    """Create a bar chart showing individual stock performance."""
    symbols = [p["symbol"] for p in portfolio_data]
    gains = [p["gainLoss"] for p in portfolio_data]
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
        positions = portfolio.get_positions()

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
            if quote:
                current_price = quote["currentPrice"]
                quantity = data["quantity"]
                purchase_price = data["purchasePrice"]
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
                use_container_width=True
            )

        with col2:
            st.plotly_chart(
                create_performance_chart(portfolio_data),
                use_container_width=True
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

        st.dataframe(styled_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error loading portfolio: {e}")
        logger.error(f"Portfolio overview error: {e}", exc_info=True)


def display_stock_analysis():
    """Display individual stock analysis tab."""
    st.header("📈 Stock Analysis")

    # Stock selector
    portfolio = PortfolioManager(POSITIONS_PATH)
    positions = portfolio.get_positions()

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
            gain_loss = (quote["currentPrice"] - position["purchasePrice"]) * position["quantity"]
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
    st.plotly_chart(fig, use_container_width=True)

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
        stocks = watchlist.list_stocks()

        if not stocks:
            st.info("📝 No stocks in watchlist. Add stocks using: `stock-tracker watchlist add SYMBOL`")
            return

        # Fetch current data for watchlist
        watchlist_data = []

        for symbol, data in stocks.items():
            quote = get_stock_quote(symbol)
            if quote:
                watchlist_data.append({
                    "Symbol": symbol,
                    "Current Price": quote["currentPrice"],
                    "Change": quote["change"],
                    "Change %": quote["changePercent"],
                    "Previous Close": quote["previousClose"],
                    "Note": data.get("note", "")
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

            st.dataframe(styled_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.info("💡 Use the CLI to add or remove watchlist items: `stock-tracker watchlist add/remove SYMBOL`")

    except Exception as e:
        st.error(f"Error loading watchlist: {e}")
        logger.error(f"Watchlist error: {e}", exc_info=True)


def display_ml_predictions():
    """Display ML predictions tab."""
    st.header("🤖 ML Price Predictions")

    portfolio = PortfolioManager(POSITIONS_PATH)
    positions = portfolio.get_positions()

    if not positions:
        st.info("📝 No positions in portfolio. Add stocks using the CLI.")
        return

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
                    st.error("Insufficient data for training. Try a longer period.")
                    return

                st.info(f"Training on {len(df)} days of historical data...")

                # Train ensemble models
                results = train_ensemble_models(df, selected_symbol, horizon)

                if not results:
                    st.error("Failed to train models.")
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

                    st.plotly_chart(fig, use_container_width=True)

                    # Show prediction table
                    st.subheader("Forecast Values")
                    st.dataframe(
                        forecast_df.head(10).style.format({
                            "Predicted_Price": "${:.2f}",
                            "Lower_Bound": "${:.2f}",
                            "Upper_Bound": "${:.2f}"
                        }),
                        use_container_width=True,
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

                    st.plotly_chart(fig, use_container_width=True)

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

                        st.plotly_chart(fig, use_container_width=True)

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

                    st.plotly_chart(fig, use_container_width=True)

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
                        use_container_width=True,
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

                    st.plotly_chart(fig, use_container_width=True)

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
    positions = portfolio.get_positions()

    if not positions:
        st.info("📝 No positions in portfolio. Add stocks using the CLI.")
        return

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

                st.plotly_chart(fig, use_container_width=True)

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
                            use_container_width=True,
                            hide_index=True
                        )

            except Exception as e:
                st.error(f"Error during backtesting: {e}")
                logger.error(f"Backtesting error: {e}", exc_info=True)

    else:
        st.info("👆 Click the button above to run backtest")


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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Portfolio Overview",
        "Stock Analysis",
        "🤖 ML Predictions",
        "📊 Backtesting",
        "Watchlist"
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
        display_watchlist()


if __name__ == "__main__":
    main()
