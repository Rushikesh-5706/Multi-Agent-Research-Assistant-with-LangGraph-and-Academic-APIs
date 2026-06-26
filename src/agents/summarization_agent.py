from __future__ import annotations

import os

import httpx
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.models.schemas import Paper, WorkflowState
from src.tools.pdf_parser import PDFParser

_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
_MODEL = "llama3.1:8b"
_TIMEOUT_SECONDS = 120.0

_PROMPT_TEMPLATE = (
    "You are a research assistant with expertise in academic literature analysis. "
    "Read the following text from an academic paper and provide a summary covering: "
    "(1) the core research problem, "
    "(2) the proposed methodology or approach, "
    "(3) the key findings or contributions. "
    "Write three concise bullet points. Do not speculate beyond what is stated in the text.\n\n"
    "Paper title: {title}\n\n"
    "Text:\n{text}"
)


class SummarizationAgent:
    def __init__(self) -> None:
        self._pdf_parser = PDFParser()

    def summarize_all(self, state: WorkflowState) -> WorkflowState:
        logger.info(f"Summarization Agent: Starting | papers={len(state.papers)}")
        for i, paper in enumerate(state.papers, start=1):
            logger.info(
                f"Summarization Agent: Summarizing paper {i}/{len(state.papers)} "
                f"| title={paper.title!r}"
            )
            paper.summary = self._summarize_paper(paper)
        state.stage = "summarized"
        logger.info("Summarization Agent: All papers processed")
        return state

    def _summarize_paper(self, paper: Paper) -> str:
        source_text = self._get_source_text(paper)
        try:
            return self._call_ollama(source_text, paper.title)
        except Exception as exc:
            logger.error(
                f"Summarization Agent: Ollama call failed after retries "
                f"| paper={paper.title!r} | error={exc}"
            )
            return paper.abstract

    def _get_source_text(self, paper: Paper) -> str:
        if paper.pdf_url:
            pdf_text = self._pdf_parser.extract_text_from_url(paper.pdf_url)
            if pdf_text:
                logger.debug(f"Using PDF text | paper={paper.title!r}")
                return pdf_text
            logger.warning(
                f"PDF inaccessible, falling back to abstract | paper={paper.title!r}"
            )
        return paper.abstract

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=15),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.ConnectError)),
        reraise=True,
    )
    def _call_ollama(self, text: str, title: str) -> str:
        prompt = _PROMPT_TEMPLATE.format(title=title, text=text)
        payload = {
            "model": _MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 350},
        }
        logger.debug(
            f"Calling Ollama | model={_MODEL} | endpoint={_OLLAMA_HOST}/api/generate"
        )
        with httpx.Client(timeout=_TIMEOUT_SECONDS) as client:
            response = client.post(f"{_OLLAMA_HOST}/api/generate", json=payload)
            response.raise_for_status()
        result = response.json()
        summary = result.get("response", "").strip()
        logger.debug(f"Ollama response received | length={len(summary)} chars")
        return summary if summary else "Summary not available."
