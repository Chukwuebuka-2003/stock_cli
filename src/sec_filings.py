"""
Utilities for fetching and caching SEC filings.
"""

import json
import logging
from datetime import datetime, timedelta
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

try:
    from sec_api import QueryApi
except ImportError:  # pragma: no cover - optional dependency
    QueryApi = None

logger = logging.getLogger(__name__)

# Public ticker → CIK mapping published by SEC as a pipe-delimited text file.
TICKER_URL = "https://www.sec.gov/Archives/edgar/cik-lookup-data.txt"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
FILING_BASE_URL = "https://www.sec.gov/Archives/edgar/data"


class SECFilingsClient:
    """Client for downloading SEC filings metadata and documents."""

    def __init__(self, cache_dir: str, user_agent: str, cache_ttl_days: int = 7, sec_api_key: Optional[str] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ticker_cache_path = self.cache_dir / "ticker_map.json"
        self.cache_ttl = timedelta(days=cache_ttl_days)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/json,text/html"
        })
        self._ticker_map: Dict[str, str] = {}
        self.query_api = None
        if sec_api_key:
            if QueryApi is None:
                logger.warning("sec-api package not installed; SEC_API_KEY will be ignored.")
            else:
                try:
                    self.query_api = QueryApi(api_key=sec_api_key)
                    logger.info("Initialized sec-api QueryApi client.")
                except Exception as e:  # pragma: no cover - network init
                    logger.error(f"Failed to initialize sec-api client: {e}")
                    self.query_api = None

    def fetch_filings(self, symbol: str, form_type: Optional[str] = None, limit: int = 3) -> List[Dict]:
        """
        Fetch recent filings (metadata + raw content) for a ticker.
        """
        if self.query_api:
            filings = self._get_filings_via_sec_api(symbol, form_type=form_type, limit=limit)
        else:
            cik = self._get_cik(symbol)
            if not cik:
                raise ValueError(f"Unable to find CIK for symbol '{symbol}'.")
            filings = self._get_recent_filings(cik, form_type=form_type, limit=limit)
        results = []
        for filing in filings:
            content = self._download_document(filing['document_url'])
            if not content:
                continue
            filing['content'] = content
            filing['symbol'] = symbol.upper()
            results.append(filing)

        return results

    def _get_cik(self, symbol: str) -> Optional[str]:
        symbol = symbol.upper()
        ticker_map = self._load_ticker_map()
        cik = ticker_map.get(symbol)
        if cik:
            return cik

        cik = self._lookup_cik_online(symbol)
        if cik:
            ticker_map[symbol] = cik
            self._ticker_map = ticker_map
            try:
                with open(self.ticker_cache_path, "w") as f:
                    json.dump(ticker_map, f)
            except IOError:
                logger.warning("Unable to persist updated ticker map to cache.")
        return cik

    def _load_ticker_map(self) -> Dict[str, str]:
        if self._ticker_map:
            return self._ticker_map

        if self.ticker_cache_path.exists():
            modified = datetime.fromtimestamp(self.ticker_cache_path.stat().st_mtime)
            if datetime.utcnow() - modified < self.cache_ttl:
                try:
                    with open(self.ticker_cache_path, "r") as f:
                        self._ticker_map = json.load(f)
                        return self._ticker_map
                except json.JSONDecodeError:
                    logger.warning("Cached SEC ticker map is corrupted. Refreshing...")

        try:
            response = self.session.get(TICKER_URL, timeout=30)
            response.raise_for_status()
            ticker_map: Dict[str, str] = {}
            for line in response.text.splitlines():
                parts = line.strip().split("|")
                if len(parts) < 2:
                    continue
                ticker = parts[0].strip().upper()
                cik = parts[1].strip()
                if not ticker or not cik:
                    continue
                try:
                    ticker_map[ticker] = str(int(cik)).zfill(10)
                except ValueError:
                    continue
            with open(self.ticker_cache_path, "w") as f:
                json.dump(ticker_map, f)
            self._ticker_map = ticker_map
        except requests.RequestException as e:
            logger.error(f"Failed to refresh SEC ticker map: {e}")
            if not self._ticker_map:
                raise
        return self._ticker_map

    def _get_recent_filings(self, cik: str, form_type: Optional[str], limit: int) -> List[Dict]:
        filings: List[Dict] = []
        try:
            padded_cik = cik.zfill(10)
            response = self.session.get(SUBMISSIONS_URL.format(cik=padded_cik), timeout=30)
            response.raise_for_status()
            data = response.json()
            recent = data.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            accession_numbers = recent.get("accessionNumber", [])
            filing_dates = recent.get("filingDate", [])
            report_dates = recent.get("reportDate", [])
            primary_documents = recent.get("primaryDocument", [])

            for idx, form in enumerate(forms):
                if form_type and form != form_type:
                    continue
                accession = accession_numbers[idx]
                accession_slug = accession.replace("-", "")
                primary_doc = primary_documents[idx]
                filing_url = f"{FILING_BASE_URL}/{int(cik)}/{accession_slug}-index.htm"
                document_url = f"{FILING_BASE_URL}/{int(cik)}/{accession_slug}/{primary_doc}"

                filings.append({
                    "form_type": form,
                    "filing_date": filing_dates[idx],
                    "report_date": report_dates[idx],
                    "accession_number": accession,
                    "filing_url": filing_url,
                    "document_url": document_url
                })

                if len(filings) >= limit:
                    break

        except requests.RequestException as e:
            logger.error(f"Failed to fetch filings for CIK {cik}: {e}")
            raise

        return filings

    def _get_filings_via_sec_api(self, symbol: str, form_type: Optional[str], limit: int) -> List[Dict]:
        if not self.query_api:
            raise ValueError("sec-api client not configured.")

        query_terms = [f"ticker:{symbol.upper()}"]
        if form_type and form_type.lower() != "all":
            query_terms.append(f'formType:"{form_type}"')

        query = {
            "query": {
                "query_string": {
                    "query": " AND ".join(query_terms)
                }
            },
            "from": 0,
            "size": limit,
            "sort": [{"filedAt": {"order": "desc"}}]
        }

        try:
            data = self.query_api.get_filings(query)
        except Exception as e:
            logger.error(f"sec-api filing query failed: {e}")
            raise

        if isinstance(data, str):
            data = json.loads(data)

        filings: List[Dict] = []
        for filing in data.get("filings", []):
            document_url = self._select_primary_document_url(filing)
            if not document_url:
                document_url = self._build_primary_document_url(filing)
            if not document_url:
                document_url = (
                    filing.get("linkToHtml")
                    or filing.get("linkToText")
                    or filing.get("filingUrl")
                    or filing.get("linkToFilingDocuments")
                )
            if not document_url:
                continue

            filing_url = (
                filing.get("linkToFilingDetails")
                or filing.get("linkToFilingDocuments")
                or filing.get("filingUrl")
                or document_url
            )

            filings.append({
                "form_type": filing.get("formType"),
                "filing_date": filing.get("filingDate") or (filing.get("filedAt") or "")[:10],
                "report_date": filing.get("periodOfReport"),
                "accession_number": filing.get("accessionNo") or filing.get("accessionNumber") or "",
                "filing_url": filing_url,
                "document_url": document_url
            })

        return filings

    def _lookup_cik_online(self, symbol: str) -> Optional[str]:
        """Fallback lookup using the EDGAR browse endpoint."""
        lookup_strategies = [
            self._lookup_cik_via_browse_ticker,
            self._lookup_cik_via_company_search
        ]
        for strategy in lookup_strategies:
            cik = strategy(symbol)
            if cik:
                return cik
        return None

    def _lookup_cik_via_browse_ticker(self, symbol: str) -> Optional[str]:
        try:
            params = {
                "CIK": symbol,
                "owner": "exclude",
                "count": "1",
                "action": "getcompany",
                "output": "atom"
            }
            response = self.session.get(
                "https://www.sec.gov/cgi-bin/browse-edgar",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            text = response.text
            match = re.search(r"<cik>(\d+)</cik>", text, re.IGNORECASE)
            if not match:
                match = re.search(r"CIK#?:\s*([0-9]+)", text)
            if match:
                cik_numeric = match.group(1)
                return str(int(cik_numeric)).zfill(10)
        except requests.RequestException as e:
            logger.error(f"Failed to look up CIK for {symbol}: {e}")
        except (ValueError, AttributeError):
            logger.error(f"Invalid CIK format received for {symbol}")
        return None

    def _lookup_cik_via_company_search(self, symbol: str) -> Optional[str]:
        """Fallback by searching the company name field."""
        try:
            params = {
                "company": symbol,
                "owner": "exclude",
                "count": "1",
                "action": "getcompany",
                "output": "atom"
            }
            response = self.session.get(
                "https://www.sec.gov/cgi-bin/browse-edgar",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            match = re.search(r"<cik>(\d+)</cik>", response.text, re.IGNORECASE)
            if match:
                cik_numeric = match.group(1)
                return str(int(cik_numeric)).zfill(10)
        except requests.RequestException as e:
            logger.error(f"Company search CIK lookup failed for {symbol}: {e}")
        except (ValueError, AttributeError):
            logger.error(f"Invalid company search CIK format for {symbol}")
        return None


    def _download_document(self, url: str) -> Optional[str]:
        try:
            response = self.session.get(url, timeout=45)
            response.raise_for_status()
            response.encoding = response.encoding or "utf-8"
            return response.text
        except requests.RequestException as e:
            logger.error(f"Failed to download SEC document {url}: {e}")
            return None

    def _select_primary_document_url(self, filing: Dict[str, Any]) -> Optional[str]:
        documents = filing.get("filingDocuments")
        if documents is None:
            documents = filing.get("documentFormatFiles")
        if documents is None:
            documents = filing.get("documentFiles")
        if documents is None:
            return None
        preferred_types = {"10-K", "10-Q", "8-K", "S-1", "20-F", "40-F"}

        for doc in documents:
            doc_type = (doc.get("documentType") or doc.get("type") or "").upper()
            url = (
                doc.get("documentUrl")
                or doc.get("documentUrlHtml")
                or doc.get("primaryDocument")
                or doc.get("url")
            )
            if doc_type in preferred_types and url:
                return url

        for doc in documents:
            url = (
                doc.get("documentUrl")
                or doc.get("documentUrlHtml")
                or doc.get("primaryDocument")
                or doc.get("url")
            )
            if url and url.lower().endswith((".htm", ".html")):
                return url

        return None

    def _build_primary_document_url(self, filing: Dict[str, Any]) -> Optional[str]:
        cik_value = filing.get("cik") or filing.get("cik_str")
        accession = filing.get("accessionNo") or filing.get("accessionNumber")
        primary_doc = filing.get("primaryDocument")

        if not primary_doc:
            alternate = filing.get("primaryDocument") or filing.get("primaryHtml")
            if alternate:
                primary_doc = alternate

        if not (cik_value and accession and primary_doc):
            return None

        try:
            cik_numeric = str(int(cik_value))
            accession_slug = accession.replace("-", "")
            return f"{FILING_BASE_URL}/{int(cik_numeric)}/{accession_slug}/{primary_doc}"
        except (ValueError, TypeError):
            logger.warning("Unable to construct primary document URL for CIK %s", cik_value)
            return None
