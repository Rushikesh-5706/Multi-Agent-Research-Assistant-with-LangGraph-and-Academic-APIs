from __future__ import annotations

import re
from typing import Optional

import requests
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.models.schemas import Paper, WorkflowState


class CitationAgent:
    def compile_citations(self, state: WorkflowState) -> WorkflowState:
        logger.info(
            f"Citation Agent: Compiling citations | papers={len(state.papers)}"
        )
        seen_keys: set[str] = set()
        for paper in state.papers:
            base_key = self._generate_key(paper)
            key = base_key
            counter = 1
            while key in seen_keys:
                key = f"{base_key}_{counter}"
                counter += 1
            seen_keys.add(key)
            paper.bibtex_key = key
            paper.bibtex = self._fetch_bibtex(paper)
        state.stage = "cited"
        logger.info("Citation Agent: Citation compilation complete")
        return state

    def _generate_key(self, paper: Paper) -> str:
        first_author = (
            re.sub(r"[^a-z0-9]", "", paper.authors[0].split()[-1].lower())
            if paper.authors
            else "unknown"
        )
        year = str(paper.year) if paper.year else "0000"
        first_word = (
            re.sub(r"[^a-z]", "", paper.title.lower().split()[0])
            if paper.title
            else "paper"
        )
        return f"{first_author}{year}{first_word}"

    def _fetch_bibtex(self, paper: Paper) -> str:
        if paper.doi:
            bibtex = self._fetch_crossref_bibtex(paper.doi)
            if bibtex:
                return bibtex
        return self._construct_bibtex(paper)

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        reraise=False,
    )
    def _fetch_crossref_bibtex(self, doi: str) -> Optional[str]:
        logger.debug(f"Citation Agent: Fetching BibTeX from Crossref | doi={doi}")
        url = f"https://doi.org/{doi}"
        headers = {"Accept": "application/x-bibtex"}
        try:
            response = requests.get(
                url, headers=headers, timeout=15, allow_redirects=True
            )
            response.raise_for_status()
            text = response.text.strip()
            if text.startswith("@"):
                logger.debug(f"Citation Agent: Crossref BibTeX retrieved | doi={doi}")
                return text
        except requests.exceptions.RequestException:
            raise
        except Exception as exc:
            logger.warning(
                f"Citation Agent: Crossref BibTeX fetch failed | doi={doi} | reason={exc}"
            )
        return None

    def _construct_bibtex(self, paper: Paper) -> str:
        key = paper.bibtex_key or "unknown2000paper"
        authors = " and ".join(paper.authors) if paper.authors else "Unknown Author"
        title = paper.title or "Untitled"
        year = str(paper.year) if paper.year else "2000"
        journal = "arXiv preprint" if paper.source == "arxiv" else "Manuscript"
        lines = [f"@article{{{key},"]
        lines.append(f"  title = {{{title}}},")
        lines.append(f"  author = {{{authors}}},")
        lines.append(f"  journal = {{{journal}}},")
        lines.append(f"  year = {{{year}}},")
        if paper.arxiv_id:
            lines.append(f"  note = {{arXiv:{paper.arxiv_id}}},")
        if paper.doi:
            lines.append(f"  doi = {{{paper.doi}}},")
        lines.append("}")
        return "\n".join(lines)
