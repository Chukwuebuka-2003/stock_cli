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

# Company name mapping for better search accuracy
COMPANY_NAMES = {
    "NVDA": "Nvidia",
    "AAPL": "Apple",
    "GOOGL": "Google Alphabet",
    "MSFT": "Microsoft",
    "TSLA": "Tesla",
    "AMZN": "Amazon",
    "META": "Meta Facebook",
    "POET": "POET Technologies",
    "AMPX": "Amprius Technologies",
    "RZLV": "Rezolve AI",
    "WULF": "TeraWulf",
    # Add more as needed
}

# Trusted financial news domains for quality filtering
TRUSTED_DOMAINS = [
    'bloomberg.com',
    'reuters.com',
    'cnbc.com',
    'marketwatch.com',
    'seekingalpha.com',
    'fool.com',
    'barrons.com',
    'wsj.com',
    'finance.yahoo.com',
    'investing.com',
    'benzinga.com',
    'tipranks.com',
    'zacks.com',
    'morningstar.com',
    'thestreet.com',
    'businesswire.com',
    'prnewswire.com',
    'globenewswire.com',
]


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
                # Get company name for better search
                company_name = COMPANY_NAMES.get(symbol, symbol)

                # Improved search query with company name and proper grouping
                query = f'("{company_name}" OR {symbol}) stock earnings news announcement'

                response = self.client.search(
                    query=query,
                    topic="news",
                    days=days,
                    search_depth=search_depth,
                    max_results=10,  # Fetch 10 results to apply quality filtering before selecting top 3
                    include_domains=TRUSTED_DOMAINS  # Only trusted sources
                )

                # Extract and filter events for relevance
                if events := self._extract_and_filter_events(response, symbol, company_name):
                    events_by_symbol[symbol] = events
                    logger.info(f"Found {len(events)} relevant events for {symbol}")

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

    def _extract_and_filter_events(self, response: Dict, symbol: str, company_name: str) -> List[Dict]:
        """
        Extract and filter events from Tavily API response for quality and relevance.

        Args:
            response: Raw response from Tavily API
            symbol: Stock symbol associated with these events
            company_name: Company name for relevance checking

        Returns:
            List of high-quality, relevant event dictionaries
        """
        events = []

        # Validate response is a dict
        if not isinstance(response, dict):
            logger.error(f"Invalid response type: expected dict, got {type(response)}")
            return events

        # Extract results from response
        results = response.get("results", [])

        for result in results:
            title = result.get("title", "")
            content = result.get("content", "")
            url = result.get("url", "")
            score = result.get("score", 0.0)

            # Quality checks
            if not self._is_quality_content(title, content, url):
                logger.debug(f"Skipping low-quality content: {title[:50]}...")
                continue

            # Relevance check - must mention the stock or company
            if not self._is_relevant_to_stock(title, content, symbol, company_name):
                logger.debug(f"Skipping irrelevant article for {symbol}: {title[:50]}...")
                continue

            # Clean the content
            cleaned_content = self._clean_content(content)

            event = {
                "symbol": symbol,
                "title": title,
                "url": url,
                "content": cleaned_content,
                "score": score,
                "published_date": result.get("published_date", "")
            }

            events.append(event)

        # Sort by relevance score and return top 3
        events.sort(key=lambda x: x['score'], reverse=True)
        return events[:3]

    def _is_quality_content(self, title: str, content: str, url: str) -> bool:
        """
        Check if content meets quality standards.

        Args:
            title: Article title
            content: Article content
            url: Article URL

        Returns:
            True if content is high quality, False otherwise
        """
        # Must have title and substantial content
        if not title or len(content) < 50:
            return False

        # Filter out common junk patterns
        junk_patterns = [
            r'password.*must be',
            r'create.*account',
            r'sign up',
            r'email this page',
            r'print this page',
            r'\(Ad\)',
            r'Image \d+:',
            r'tc pixel',
            r'Accept cookies',
            r'Subscribe now',
            r'Free trial',
        ]

        content_lower = content.lower()
        for pattern in junk_patterns:
            if re.search(pattern, content_lower, re.IGNORECASE):
                return False

        # Check for navigation elements
        # Only fail if these appear in the first 100 characters (likely navigation)
        nav_keywords = ['calendar', 'dividend calculator', 'earnings calendar', 'market holidays']
        if any(keyword in content_lower for keyword in nav_keywords) and any(keyword in content_lower[:100] for keyword in nav_keywords):
            return False

        return True

    def _is_relevant_to_stock(self, title: str, content: str, symbol: str, company_name: str) -> bool:
        """
        Verify that the article is actually about the specified stock.

        Args:
            title: Article title
            content: Article content
            symbol: Stock symbol
            company_name: Company name

        Returns:
            True if article is relevant, False otherwise
        """
        title_lower = title.lower()
        content_lower = content.lower()
        symbol_lower = symbol.lower()
        company_lower = company_name.lower()

        # Must mention symbol OR company name in title or first 300 characters of content
        content_preview = content_lower[:300]
        return (symbol_lower in title_lower or company_lower in title_lower or
                symbol_lower in content_preview or company_lower in content_preview)

    def _clean_content(self, content: str) -> str:
        """
        Clean HTML artifacts and unnecessary text from content.

        Args:
            content: Raw content text

        Returns:
            Cleaned content
        """
        # Remove common HTML artifacts
        cleaned = re.sub(r'Image \d+:', '', content)
        cleaned = re.sub(r'\(Ad\)', '', cleaned)
        cleaned = re.sub(r'tc pixel', '', cleaned)

        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)

        # Trim and return
        return cleaned.strip()

    def should_trigger_report(
        self,
        events_by_symbol: Dict[str, List[Dict]],
        threshold: float = 0.75
    ) -> bool:
        """
        Determine if detected events warrant triggering an automated report.

        Args:
            events_by_symbol: Dictionary of events detected per symbol
            threshold: Minimum relevance score to trigger report (0.0 to 1.0, default: 0.75)

        Returns:
            True if report should be triggered, False otherwise
        """
        if not events_by_symbol:
            return False

        # Check if any events exceed the threshold
        for symbol, events in events_by_symbol.items():
            for event in events:
                if event.get("score", 0.0) >= threshold:
                    logger.info(f"High-relevance event detected for {symbol} (score: {event['score']:.2f}): {event['title']}")
                    return True

        # Also trigger if we have at least 2 events for any symbol (indicates significant activity)
        for symbol, events in events_by_symbol.items():
            if len(events) >= 2:
                logger.info(f"Multiple events detected for {symbol}, triggering report")
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

    should_trigger = detector.should_trigger_report(events_by_symbol, threshold=0.75)

    return {
        "symbol_events": events_by_symbol,
        "general_events": general_events,
        "should_trigger_report": should_trigger,
        "summary": detector.format_events_summary(events_by_symbol)
    }
