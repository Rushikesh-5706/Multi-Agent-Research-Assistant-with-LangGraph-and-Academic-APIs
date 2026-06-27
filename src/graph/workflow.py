from __future__ import annotations

import os
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph
from loguru import logger

from src.agents.citation_agent import CitationAgent
from src.agents.search_agent import SearchAgent
from src.agents.summarization_agent import SummarizationAgent
from src.agents.supervisor_agent import SupervisorAgent
from src.models.schemas import WorkflowState
from src.state.redis_manager import RedisManager

# Canonical display names for paper sources
_SOURCE_LABELS: dict[str, str] = {
    "arxiv": "arXiv",
    "semantic_scholar": "Semantic Scholar",
}


class GraphState(TypedDict):
    run_id: str
    topic: str
    stage: str
    error: Optional[str]


def build_workflow(redis_manager: RedisManager):
    supervisor = SupervisorAgent()
    search = SearchAgent()
    summarization = SummarizationAgent()
    citation = CitationAgent()

    def supervisor_node(state: GraphState) -> dict:
        logger.info("Supervisor: Planning workflow")
        ws = redis_manager.load_state(state["run_id"])
        ws = supervisor.plan(ws)
        redis_manager.save_state(state["run_id"], ws)
        return {"stage": ws.stage, "error": ws.error}

    def search_node(state: GraphState) -> dict:
        logger.info("Search Agent: Retrieving papers from arXiv and Semantic Scholar")
        try:
            ws = redis_manager.load_state(state["run_id"])
            ws = search.search(ws)
            if not ws.papers:
                ws.error = "Search returned zero papers from all sources after retries."
            redis_manager.save_state(state["run_id"], ws)
            return {"stage": ws.stage, "error": ws.error}
        except Exception as exc:
            logger.error(f"Search Agent: Fatal error | error={exc}")
            return {"stage": "search_failed", "error": str(exc)}

    def summarize_node(state: GraphState) -> dict:
        logger.info("Summarization Agent: Generating LLM summaries via Ollama")
        try:
            ws = redis_manager.load_state(state["run_id"])
            ws = summarization.summarize_all(ws)
            redis_manager.save_state(state["run_id"], ws)
            return {"stage": ws.stage, "error": ws.error}
        except Exception as exc:
            logger.error(f"Summarization Agent: Fatal error | error={exc}")
            return {"stage": "summarize_failed", "error": str(exc)}

    def citation_node(state: GraphState) -> dict:
        logger.info("Citation Agent: Compiling BibTeX citations")
        try:
            ws = redis_manager.load_state(state["run_id"])
            ws = citation.compile_citations(ws)
            redis_manager.save_state(state["run_id"], ws)
            return {"stage": ws.stage, "error": ws.error}
        except Exception as exc:
            logger.error(f"Citation Agent: Fatal error | error={exc}")
            return {"stage": "citation_failed", "error": str(exc)}

    def report_node(state: GraphState) -> dict:
        logger.info("Report: Generating output files")
        ws = redis_manager.load_state(state["run_id"])
        _write_markdown(ws)
        _write_bibtex(ws)
        ws.stage = "complete"
        redis_manager.save_state(state["run_id"], ws)
        return {"stage": "complete", "error": None}

    def error_node(state: GraphState) -> dict:
        logger.error(
            f"Workflow routed to error state | stage={state.get('stage')} | error={state.get('error')}"
        )
        return {"stage": "error"}

    def _route_or_error(next_node: str):
        def _route(state: GraphState) -> str:
            return "error" if state.get("error") else next_node
        return _route

    graph = StateGraph(GraphState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("search", search_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("citation", citation_node)
    graph.add_node("report", report_node)
    graph.add_node("error_node", error_node)

    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "search")
    graph.add_conditional_edges(
        "search",
        _route_or_error("summarize"),
        {"summarize": "summarize", "error": "error_node"},
    )
    graph.add_conditional_edges(
        "summarize",
        _route_or_error("citation"),
        {"citation": "citation", "error": "error_node"},
    )
    graph.add_conditional_edges(
        "citation",
        _route_or_error("report"),
        {"report": "report", "error": "error_node"},
    )
    graph.add_edge("report", END)
    graph.add_edge("error_node", END)

    return graph.compile()


def _write_markdown(state: WorkflowState) -> None:
    os.makedirs("./output", exist_ok=True)
    path = "./output/literature_review.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Literature Review: {state.topic}\n\n")
        f.write("---\n\n")
        f.write("## Introduction\n\n")
        f.write(
            f"This literature review examines recent research on {state.topic}. "
            f"Papers were retrieved from arXiv and Semantic Scholar. "
            f"Each {state.topic} paper was processed using a locally hosted language model (Ollama llama3.1:8b) "
            f"to extract its core contribution, methodology, and results. "
            f"The review maps the current research landscape and identifies recurring themes "
            f"and open problems in the field.\n\n"
        )
        f.write("---\n\n")
        f.write("## Related Work\n\n")
        for i, paper in enumerate(state.papers, start=1):
            authors_str = ", ".join(paper.authors[:3])
            if len(paper.authors) > 3:
                authors_str += " et al."
            year_str = f" ({paper.year})" if paper.year else ""
            f.write(f"### {i}. {paper.title}\n\n")
            f.write(f"**Authors:** {authors_str}{year_str}")
            if paper.doi:
                f.write(f" | **DOI:** {paper.doi}")
            source_label = _SOURCE_LABELS.get(
                paper.source, paper.source.replace("_", " ").title()
            )
            f.write(f" | **Source:** {source_label}\n\n")
            f.write("**Summary:**\n\n")
            summary_text = paper.summary if paper.summary else paper.abstract
            f.write(f"{summary_text}\n\n")
            f.write("---\n\n")

        # Topic-aware conclusion derived from the actual papers retrieved
        f.write("## Conclusion\n\n")
        paper_count = len(state.papers)
        source_counts: dict[str, int] = {}
        for p in state.papers:
            source_counts[p.source] = source_counts.get(p.source, 0) + 1
        years = [p.year for p in state.papers if p.year]
        year_range = f" spanning {min(years)} to {max(years)}" if years else ""
        source_summary = " and ".join(
            f"{count} from {_SOURCE_LABELS.get(src, src)}"
            for src, count in source_counts.items()
        )
        f.write(
            f"This review examined {paper_count} papers on {state.topic}{year_range}, "
            f"retrieved from {source_summary}. "
            f"The surveyed work reflects a field that is actively advancing on multiple fronts, "
            f"with contributions spanning theoretical foundations and applied systems. "
            f"The papers cited above provide entry points into the primary literature "
            f"and form a foundation for researchers entering this area.\n\n"
        )
        f.write("---\n\n")
        f.write("## References\n\n")
        for i, paper in enumerate(state.papers, start=1):
            authors_str = ", ".join(paper.authors[:3])
            if len(paper.authors) > 3:
                authors_str += " et al."
            year_str = str(paper.year) if paper.year else "n.d."
            ref = f"[{i}] {authors_str} ({year_str}). {paper.title}."
            if paper.doi:
                ref += f" DOI: {paper.doi}."
            elif paper.arxiv_id:
                ref += f" arXiv:{paper.arxiv_id}."
            f.write(f"{ref}\n\n")
    logger.info(
        f"Report: literature_review.md written | path={path} | papers={len(state.papers)}"
    )


def _write_bibtex(state: WorkflowState) -> None:
    os.makedirs("./output", exist_ok=True)
    path = "./output/references.bib"
    entries = [p.bibtex for p in state.papers if p.bibtex]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(entries))
        f.write("\n")
    logger.info(
        f"Report: references.bib written | path={path} | entries={len(entries)}"
    )
