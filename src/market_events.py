"""
Market Event Detection using Tavily API.

This module monitors financial news and market events to trigger automated reports
when significant events are detected related to portfolio holdings.
"""
import json
import logging
import os
import re
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class MarketEventDetector:
    """Detect market events using Tavily API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Market Event Detector.

        Args:
            api_key: Tavily API key. If None, will try to read from TAVILY_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            logger.warning("Tavily API key not configured. Market event detection unavailable.")
            self.client = None
        else:
            try:
                from tavily import TavilyClient
                self.client = TavilyClient(api_key=self.api_key)
                logger.info("Tavily client initialized successfully")
            except ImportError:
                logger.error("tavily-python package not installed. Install with: pip install tavily-python")
                self.client = None

    def check_events_for_symbols(
        self,
        symbols: List[str],
        days: int = 1,
        search_depth: str = "advanced"
    ) -> Dict[str, List[Dict]]:
        """
        Check for market events related to specific stock symbols.

        Args:
            symbols: List of stock symbols to monitor (e.g., ["AAPL", "GOOGL"])
            days: Number of days back to search (default: 1)
            search_depth: Search depth - "basic" or "advanced" (default: "advanced")

        Returns:
            Dictionary mapping symbols to their relevant news events
        """
        if not self.client:
            logger.error("Tavily client not initialized. Cannot check market events.")
            return {}

        events_by_symbol = {}

        for symbol in symbols:
            try:
                # Search for news about this specific stock
                query = f"{symbol} stock market news earnings"

                response = self.client.search(
                    query=query,
                    topic="news",
                    days=days,
                    search_depth=search_depth,
                    max_results=5
                )

                if events := self._extract_events(response, symbol):
                    events_by_symbol[symbol] = events
                    logger.info(f"Found {len(events)} events for {symbol}")

            except Exception as e:
                logger.error(f"Error checking events for {symbol}: {e}")
                continue

        return events_by_symbol

    def check_general_market_events(
        self,
        days: int = 1,
        search_depth: str = "advanced"
    ) -> List[Dict]:
        """
        Check for general market-moving events.

        Args:
            days: Number of days back to search (default: 1)
            search_depth: Search depth - "basic" or "advanced" (default: "advanced")

        Returns:
            List of significant market events
        """
        if not self.client:
            logger.error("Tavily client not initialized. Cannot check market events.")
            return []

        try:
            # Search for general market news
            query = "stock market breaking news major events Fed interest rates"

            response = self.client.search(
                query=query,
                topic="news",
                days=days,
                search_depth=search_depth,
                max_results=10
            )

            events = self._extract_events(response, "MARKET")
            logger.info(f"Found {len(events)} general market events")
            return events

        except Exception as e:
            logger.error(f"Error checking general market events: {e}")
            return []

    def _extract_events(self, response: Dict, symbol: str) -> List[Dict]:
        """
        Extract and format events from Tavily API response.

        Args:
            response: Raw response from Tavily API
            symbol: Stock symbol associated with these events

        Returns:
            List of formatted event dictionaries
        """
        events = []

        # Validate response is a dict
        if not isinstance(response, dict):
            logger.error(f"Invalid response type: expected dict, got {type(response)}")
            return events

        # Extract results from response
        results = response.get("results", [])

        for result in results:
            event = {
                "symbol": symbol,
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "score": result.get("score", 0.0),
                "published_date": result.get("published_date", "")
            }

            # Only include events with meaningful content
            if event["title"] and event["content"]:
                events.append(event)

        return events

    def should_trigger_report(
        self,
        events_by_symbol: Dict[str, List[Dict]],
        threshold: float = 0.7
    ) -> bool:
        """
        Determine if detected events warrant triggering an automated report.

        Args:
            events_by_symbol: Dictionary of events detected per symbol
            threshold: Minimum relevance score to trigger report (0.0 to 1.0)

        Returns:
            True if report should be triggered, False otherwise
        """
        if not events_by_symbol:
            return False

        # Check if any events exceed the threshold
        for symbol, events in events_by_symbol.items():
            for event in events:
                if event.get("score", 0.0) >= threshold:
                    logger.info(f"High-relevance event detected for {symbol}: {event['title']}")
                    return True

        return False

    def format_events_summary(self, events_by_symbol: Dict[str, List[Dict]]) -> str:
        """
        Format detected events into a readable summary.

        Args:
            events_by_symbol: Dictionary of events detected per symbol

        Returns:
            Formatted string summary of events
        """
        if not events_by_symbol:
            return "No significant market events detected."

        summary_lines = ["📰 Market Events Detected:", ""]

        for symbol, events in events_by_symbol.items():
            summary_lines.append(f"**{symbol}**:")
            for event in events[:3]:  # Limit to top 3 events per symbol
                summary_lines.extend(
                    (
                        f"  • {event['title']}",
                        f"    {event['content'][:150]}...",
                        f"    🔗 {event['url']}",
                        "",
                    )
                )

        return "\n".join(summary_lines)


def _is_valid_symbol(symbol: str) -> bool:
    """
    Validate stock symbol format.

    Args:
        symbol: Stock symbol to validate

    Returns:
        True if symbol is valid, False otherwise
    """
    # Allow uppercase alphanumeric, 1-5 chars (standard US stock symbols)
    # Can be adjusted for international symbols if needed
    return bool(re.fullmatch(r"[A-Z0-9]{1,5}", symbol))


def check_portfolio_events(portfolio_positions: List[Dict]) -> Optional[Dict]:
    """
    Convenience function to check for events related to portfolio positions.

    Args:
        portfolio_positions: List of portfolio positions with 'symbol' keys

    Returns:
        Dictionary containing events and whether to trigger report, or None if unavailable
    """
    detector = MarketEventDetector()

    if not detector.client:
        return None

    # Extract symbols from positions, validate and deduplicate
    raw_symbols = [pos["symbol"].upper() for pos in portfolio_positions if "symbol" in pos]

    # Deduplicate and validate
    symbols = list({s for s in raw_symbols if _is_valid_symbol(s)})

    # Log any invalid symbols
    invalid_symbols = set(raw_symbols) - set(symbols)
    if invalid_symbols:
        logger.warning(f"Skipping invalid symbols: {invalid_symbols}")

    if not symbols:
        logger.warning("No valid symbols found in portfolio positions")
        return None

    # Check for events
    events_by_symbol = detector.check_events_for_symbols(symbols, days=1)
    general_events = detector.check_general_market_events(days=1)

    should_trigger = detector.should_trigger_report(events_by_symbol, threshold=0.7)

    return {
        "symbol_events": events_by_symbol,
        "general_events": general_events,
        "should_trigger_report": should_trigger,
        "summary": detector.format_events_summary(events_by_symbol)
    }
