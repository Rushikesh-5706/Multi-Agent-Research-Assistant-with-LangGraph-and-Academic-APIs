from __future__ import annotations

import os
import tempfile
from typing import Optional

import pdfplumber
import requests
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

_MAX_CHARS = 3500
_MAX_PAGES = 3


class PDFParser:
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=8),
        stop=stop_after_attempt(2),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        reraise=False,
    )
    def extract_text_from_url(self, pdf_url: str) -> Optional[str]:
        logger.debug(f"Downloading PDF | url={pdf_url}")
        try:
            response = requests.get(
                pdf_url,
                timeout=30,
                headers={"User-Agent": "ResearchAgent/1.0 (academic-use)"},
            )
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name
            text = self._extract_from_path(tmp_path)
            os.unlink(tmp_path)
            return text
        except requests.exceptions.RequestException:
            raise
        except Exception as exc:
            logger.warning(f"PDF extraction failed | url={pdf_url} | reason={exc}")
            return None

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=8),
        stop=stop_after_attempt(2),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        reraise=False,
    )
    def find_open_access_url(self, doi: str) -> Optional[str]:
        """Query Unpaywall for an open-access PDF URL for the given DOI."""
        logger.debug(f"Querying Unpaywall for open-access PDF | doi={doi}")
        try:
            url = f"https://api.unpaywall.org/v2/{doi}"
            params = {"email": "research-agent@example.com"}
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            oa_location = data.get("best_oa_location")
            if oa_location:
                pdf_url = oa_location.get("url_for_pdf")
                if pdf_url:
                    logger.debug(
                        f"Unpaywall found open-access PDF | doi={doi} | url={pdf_url}"
                    )
                    return pdf_url
        except requests.exceptions.RequestException:
            raise
        except Exception as exc:
            logger.warning(f"Unpaywall lookup failed | doi={doi} | reason={exc}")
        return None

    def _extract_from_path(self, path: str) -> Optional[str]:
        try:
            with pdfplumber.open(path) as pdf:
                parts: list[str] = []
                for page in pdf.pages[:_MAX_PAGES]:
                    page_text = page.extract_text()
                    if page_text:
                        parts.append(page_text)
                full_text = "\n".join(parts).strip()
                return full_text[:_MAX_CHARS] or None
        except Exception as exc:
            logger.warning(f"pdfplumber extraction error | path={path} | reason={exc}")
            return None
