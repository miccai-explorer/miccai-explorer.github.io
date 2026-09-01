"""Shared HTTP fetch helpers with retry and rate-limiting."""

import logging
import time

import requests

logger = logging.getLogger(__name__)


def fetch_html(url: str, cfg: dict) -> str:
    """Fetch URL with retry backoff. Returns HTML string."""
    headers = {"User-Agent": cfg["user_agent"]}
    max_retries = cfg.get("max_retries", 3)
    retry_delay = cfg.get("retry_delay", 5.0)

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            # Force UTF-8: requests defaults to ISO-8859-1 for text/html when the
            # server omits charset, but these sites serve UTF-8.
            resp.encoding = "utf-8"
            return resp.text
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"Retry {attempt + 1}/{max_retries} for {url}: {e}"
                )
                time.sleep(retry_delay)
            else:
                raise
