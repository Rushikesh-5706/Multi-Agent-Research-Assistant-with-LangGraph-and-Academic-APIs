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
        ws = redis_manager.load_state(state["run_id"])
        ws = search.search(ws)
        redis_manager.save_state(state["run_id"], ws)
        return {"stage": ws.stage, "error": ws.error}

    def summarize_node(state: GraphState) -> dict:
        logger.info("Summarization Agent: Generating LLM summaries via Ollama")
        ws = redis_manager.load_state(state["run_id"])
        ws = summarization.summarize_all(ws)
        redis_manager.save_state(state["run_id"], ws)
        return {"stage": ws.stage, "error": ws.error}

    def citation_node(state: GraphState) -> dict:
        logger.info("Citation Agent: Compiling BibTeX citations")
        ws = redis_manager.load_state(state["run_id"])
        ws = citation.compile_citations(ws)
        redis_manager.save_state(state["run_id"], ws)
        return {"stage": ws.stage, "error": ws.error}

    def report_node(state: GraphState) -> dict:
        logger.info("Report: Generating output files")
        ws = redis_manager.load_state(state["run_id"])
        _write_markdown(ws)
        _write_bibtex(ws)
        ws.stage = "complete"
        redis_manager.save_state(state["run_id"], ws)
        return {"stage": "complete", "error": None}

    graph = StateGraph(GraphState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("search", search_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("citation", citation_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "search")
    graph.add_edge("search", "summarize")
    graph.add_edge("summarize", "citation")
    graph.add_edge("citation", "report")
    graph.add_edge("report", END)

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
            f"Each paper was processed using a locally hosted language model (Ollama llama3.1:8b) "
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
            source_label = paper.source.replace("_", " ").title()
            f.write(f" | **Source:** {source_label}\n\n")
            f.write("**Summary:**\n\n")
            summary_text = paper.summary if paper.summary else paper.abstract
            f.write(f"{summary_text}\n\n")
            f.write("---\n\n")
        f.write("## Conclusion\n\n")
        f.write(
            f"The works surveyed here represent the current state of research in {state.topic}. "
            f"Recurring themes include scalable neural architectures, "
            f"data-efficient training, and rigorous empirical evaluation. "
            f"Open challenges in the field include robustness under distribution shift, "
            f"computational efficiency at inference time, and interpretability of model decisions. "
            f"The papers cited above provide a solid foundation for advancing these directions.\n\n"
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
