# Multi-Agent Research Assistant

A command-line tool that generates academic literature reviews from a research topic. The system uses a four-agent pipeline orchestrated by LangGraph, retrieves papers from arXiv and Semantic Scholar, summarizes them using a locally hosted language model via Ollama, compiles BibTeX citations, and writes a structured Markdown report.

The full stack runs in Docker. No cloud API keys are required for the default configuration.

---

## System Architecture

```
Agent Workflow (LangGraph StateGraph)

┌─────────────────┐     ┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│ Supervisor Agent│────▶│  Search Agent   │────▶│ Summarization Agent  │────▶│ Citation Agent  │────▶│ Report Generator │
│                 │     │                 │     │                      │     │                 │     │                  │
│ Generates search│     │ Queries arXiv   │     │ Downloads PDFs       │     │ Fetches BibTeX  │     │ Writes .md       │
│ queries from    │     │ and Semantic    │     │ Calls Ollama         │     │ via Crossref    │     │ and .bib files   │
│ input topic     │     │ Scholar APIs    │     │ llama3.1:8b          │     │ Constructs      │     │                  │
│                 │     │                 │     │                      │     │ fallback entries│     │                  │
└─────────────────┘     └─────────────────┘     └──────────────────────┘     └─────────────────┘     └──────────────────┘
         │                       │                          │                          │                        │
         └───────────────────────┴──────────────────────────┴──────────────────────────┴──── Redis ────────────┘
                                              State persisted and read at each transition
```

Docker Compose Services

```
┌────────────────────────────────────────────────────┐
│                Docker Compose Network               │
│                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │
│  │     app      │  │    redis     │  │  ollama  │ │
│  │              │  │              │  │          │ │
│  │  LangGraph   │◀─│  7.2-alpine  │  │ llama3.1 │ │
│  │  workflow    │  │  Port 6379   │  │  :8b     │ │
│  │              │─▶│  State store │  │ Port     │ │
│  │              │  └──────────────┘  │ 11434    │ │
│  │              │─────────────────────▶          │ │
│  └──────────────┘                  └──────────┘ │
└────────────────────────────────────────────────────┘
```

---

## Project Structure

```
.
├── main.py                          CLI entrypoint (click)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── output/
│   ├── literature_review.md         Generated on each run
│   ├── references.bib               Generated on each run
│   └── agent_run.log                Written by loguru during run
└── src/
    ├── agents/
    │   ├── supervisor_agent.py      Generates search queries from topic
    │   ├── search_agent.py          Queries arXiv and Semantic Scholar
    │   ├── summarization_agent.py   PDF parsing and Ollama LLM calls
    │   └── citation_agent.py        BibTeX via Crossref or constructed
    ├── graph/
    │   └── workflow.py              LangGraph StateGraph definition
    ├── models/
    │   └── schemas.py               Pydantic models: Paper, WorkflowState
    ├── state/
    │   └── redis_manager.py         Redis serialization layer
    ├── tools/
    │   ├── arxiv_tool.py            arXiv Python library wrapper
    │   ├── semantic_scholar_tool.py Semantic Scholar REST API wrapper
    │   └── pdf_parser.py            pdfplumber-based extraction
    └── utils/
        └── logger.py                loguru configuration
```

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Docker Engine | 24+ |
| Docker Compose | v2 |
| Available disk space | 10 GB (Ollama model) |
| Internet access | Required for API queries and model download |

---

## Setup

**Clone the repository**

```bash
git clone https://github.com/Rushikesh-5706/Multi-Agent-Research-Assistant-with-LangGraph-and-Academic-APIs.git
cd Multi-Agent-Research-Assistant-with-LangGraph-and-Academic-APIs
```

**Configure environment**

```bash
cp .env.example .env
```

No changes are needed for the default configuration. Optional API keys documented in `.env.example` can be added to improve rate limits.

**Start all services**

```bash
docker-compose up --build
```

On first startup, Ollama downloads the llama3.1:8b model (~5 GB). Subsequent starts use the cached volume. All three services must show `(healthy)` in `docker ps` before the application accepts commands.

---

## Usage

```bash
docker-compose run --rm app python main.py --topic "transformer architectures in natural language processing"
```

Replace the topic string with any research area. The tool accepts multi-word topics with spaces inside quotes.

**Run with a different topic**

```bash
docker-compose run --rm app python main.py --topic "graph neural networks for molecular property prediction"
```

**Check health status of services**

```bash
docker ps
```

All three containers should show `(healthy)` before running the application.

---

## Output Files

All output is written to `./output/` on the host machine via a Docker volume mount.

| File | Description |
|------|-------------|
| `literature_review.md` | Markdown report with Introduction, Related Work, Conclusion, and References sections |
| `references.bib` | BibTeX file containing one `@article` entry per paper |
| `agent_run.log` | Structured log of the full agent execution with timestamps |

**Report structure**

```markdown
# Literature Review: [topic]
## Introduction
## Related Work
### 1. [Paper Title]
### 2. [Paper Title]
...
## Conclusion
## References
```

---

## Environment Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `REDIS_HOST` | Yes | `redis` | Redis service hostname |
| `REDIS_PORT` | Yes | `6379` | Redis service port |
| `OLLAMA_HOST` | No | `http://ollama:11434` | Ollama inference endpoint |
| `SEMANTIC_SCHOLAR_API_KEY` | No | — | Improves Semantic Scholar rate limits |
| `ANTHROPIC_API_KEY` | No | — | Not used in default config |

---

## Agent Design

| Agent | Responsibility | Input | Output | External Calls |
|-------|---------------|-------|--------|----------------|
| Supervisor | Plans search strategy | Topic string | List of search queries | None |
| Search | Retrieves academic papers | Query list | List of Paper objects | arXiv API, Semantic Scholar API |
| Summarization | Generates paper summaries | Paper list | Paper list with summaries | Ollama /api/generate, PDF URLs |
| Citation | Compiles BibTeX references | Paper list | Paper list with BibTeX | Crossref DOI endpoint |

---

## State Management

Each agent reads the full workflow state from Redis at the start of its execution and writes the updated state back on completion. The state is a Pydantic `WorkflowState` model serialized to JSON. LangGraph manages the node transition logic; Redis provides persistence and observability across transitions.

Redis key format: `research:{run_id}:state` with a 24-hour TTL.

---

## Error Handling

All external API calls use `tenacity` retry logic with exponential backoff. Specific behavior by failure type:

| Failure | Behavior |
|---------|---------|
| arXiv API error | Retried 3 times; logged as error if exhausted; other sources continue |
| Semantic Scholar API error | Retried 3 times; logged as error if exhausted |
| PDF download failure | Falls back to abstract-level summarization; warning logged |
| Ollama timeout or HTTP error | Retried 3 times; falls back to raw abstract if exhausted |
| Crossref BibTeX unavailable | BibTeX constructed from paper metadata |

---

## Limitations

- llama3.1:8b produces functional summaries. Papers with dense mathematical notation may receive less precise summaries than larger proprietary models would produce.
- Some conference papers are not open-access. The summarization agent falls back to abstract text for these.
- Semantic Scholar's unauthenticated tier is rate-limited. The free API key available at semanticscholar.org/product/api significantly increases the limit.

---

## License

MIT
