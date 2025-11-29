import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from tavily import TavilyClient

from .base import BaseAgent

logger = logging.getLogger(__name__)

class ResearchAgent(BaseAgent):
    """
    Agent responsible for executing research steps using various tools.
    """
    def __init__(
        self,
        model_client,
        vector_store=None,
        alpha_vantage_key=None,
        tavily_api_key: Optional[str] = None,
        tavily_client: Optional[TavilyClient] = None,
        data_fetcher=None,
    ):
        super().__init__("ResearchAgent", model_client)
        self.vector_store = vector_store
        self.alpha_vantage_key = alpha_vantage_key
        self.tavily_api_key = tavily_api_key or os.environ.get("TAVILY_API_KEY")
        self.tavily_client = tavily_client
        self.data_fetcher = data_fetcher

        if self.tavily_api_key and self.tavily_client is None:
            try:
                self.tavily_client = TavilyClient(api_key=self.tavily_api_key)
            except Exception as exc:
                logger.error("Failed to initialize Tavily client: %s", exc)
                self.tavily_client = None

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the research plan.
        """
        plan = context.get("plan", [])
        findings = []
        
        quotes_store = context.setdefault("quotes", {})
        
        for step in plan:
            self._log_step(f"Executing step {step['step_id']}: {step['description']}")
            tool = step.get("tool")
            query = step.get("query")
            
            result = None
            try:
                if tool == "search_web":
                    result = self._search_web(query)
                elif tool == "search_rag":
                    result = self._search_rag(query)
                elif tool == "get_stock_price":
                    result, quote_meta = self._get_stock_price(query)
                    if quote_meta:
                        quotes_store[quote_meta["symbol"]] = quote_meta
                else:
                    result = f"Unknown tool: {tool}"
            except Exception as e:
                result = f"Error executing {tool}: {str(e)}"
                
            findings.append({
                "step_id": step["step_id"],
                "description": step["description"],
                "tool": tool,
                "result": result
            })
            
        return {"findings": findings}

    def _search_web(self, query: str) -> str:
        """Search the web using Tavily."""
        if not self.tavily_client:
            return "Web search unavailable: configure TAVILY_API_KEY."
        
        try:
            response = self.tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=5,
                include_answer=True,
            )
        except Exception as e:
            return f"Web search failed: {e}"

        results = response.get("results", [])
        if not results:
            return "Web search returned no results."

        formatted: List[str] = []
        answer = response.get("answer")
        if answer:
            formatted.append(f"Tavily Summary: {answer}")
            formatted.append("---")

        for idx, item in enumerate(results, start=1):
            title = item.get("title", "Untitled")
            url = item.get("url", "")
            content = item.get("content") or item.get("snippet") or ""
            content = content.strip()
            if len(content) > 320:
                content = content[:317] + "..."
            formatted.append(f"{idx}. {title}\n   {url}\n   Insight: {content}")

        return "\n".join(formatted)

    def _search_rag(self, query: str) -> str:
        """Search local vector store."""
        if not self.vector_store:
            return "RAG not initialized."
        try:
            results = self.vector_store.query_similar(query)
            return str(results)
        except Exception as e:
            return f"RAG search failed: {e}"

    def _get_stock_price(self, symbol: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Get live stock price using the configured data fetcher."""
        if not symbol:
            return "No symbol provided for price lookup.", None

        if not self.data_fetcher:
            return "Live quote unavailable: configure Twelve Data API key.", None

        try:
            data = self.data_fetcher.get_stock_data(symbol)
        except Exception as exc:
            return f"Failed to retrieve price for {symbol.upper()}: {exc}", None

        if not data:
            return f"No price data returned for {symbol.upper()}.", None

        price = data.get("currentPrice")
        prev_close = data.get("previousClose")
        change = data.get("change")
        change_pct_raw = data.get("changePercent")
        change_pct_value = None
        if isinstance(change_pct_raw, (int, float)):
            change_pct_value = float(change_pct_raw)
        elif isinstance(change_pct_raw, str):
            try:
                cleaned = change_pct_raw.strip().replace("%", "")
                change_pct_value = float(cleaned)
            except ValueError:
                change_pct_value = None

        if price is None:
            return f"Price data unavailable for {symbol.upper()}.", None

        arrow = ""
        if isinstance(change, (int, float)):
            if change > 0:
                arrow = "▲"
            elif change < 0:
                arrow = "▼"

        direction_word = "flat"
        if isinstance(change, (int, float)):
            if change > 0:
                direction_word = "up"
            elif change < 0:
                direction_word = "down"

        ticker_prefix = f"{symbol.upper()} {arrow}".strip()
        if not arrow:
            ticker_prefix = symbol.upper()
        price_line = f"{ticker_prefix}: ${price:,.2f}"
        if isinstance(change, (int, float)) and change != 0:
            price_line += f" {direction_word} {abs(change):.2f}"
        if isinstance(change_pct_value, (int, float)) and change_pct_value != 0:
            price_line += f" ({abs(change_pct_value):.2f}%)"
        elif isinstance(change_pct_raw, str) and change_pct_raw.strip():
            price_line += f" ({change_pct_raw.strip()})"

        if isinstance(prev_close, (int, float)):
            price_line += f" • Prev close ${prev_close:,.2f}"

        summary = price_line
        meta = {
            "symbol": symbol.upper(),
            "price": price,
            "change": change,
            "change_percent": change_pct_value if change_pct_value is not None else change_pct_raw,
            "previous_close": prev_close,
            "summary": summary,
        }
        return summary, meta
