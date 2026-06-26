from unittest.mock import patch

from src.agents.citation_agent import CitationAgent
from src.models.schemas import Paper, WorkflowState


def test_generate_key_first_last_format():
    agent = CitationAgent()
    p = Paper(
        paper_id="a:1", title="Attention Is All", authors=["Ashish Vaswani"],
        abstract="x", year=2017, source="arxiv",
    )
    key = agent._generate_key(p)
    assert key == "vaswani2017attention"


def test_generate_key_last_first_format():
    agent = CitationAgent()
    p = Paper(
        paper_id="a:2", title="BERT Pretraining", authors=["Devlin, Jacob"],
        abstract="x", year=2019, source="semantic_scholar",
    )
    key = agent._generate_key(p)
    assert key == "devlin2019bert"


def test_construct_bibtex_no_trailing_comma():
    agent = CitationAgent()
    p = Paper(
        paper_id="a:1", title="Test Paper", authors=["Smith Alice"],
        abstract="x", year=2023, source="arxiv", arxiv_id="2301.00001",
    )
    p.bibtex_key = "smith2023test"
    bib = agent._construct_bibtex(p)
    lines = bib.strip().split("\n")
    # Last line before closing brace must NOT end with comma
    second_to_last = lines[-2].rstrip()
    assert not second_to_last.endswith(","), f"Trailing comma found: {second_to_last!r}"


def test_key_deduplication():
    agent = CitationAgent()
    papers = [
        Paper(paper_id=f"a:{i}", title=f"Test Paper {i}", authors=["Smith Alice"],
              abstract="x", year=2023, source="arxiv")
        for i in range(3)
    ]
    ws = WorkflowState(run_id="r1", topic="nlp", papers=papers)
    with patch.object(agent, "_fetch_bibtex", return_value="@article{key, title={T}}"):
        ws = agent.compile_citations(ws)
    keys = [p.bibtex_key for p in ws.papers]
    assert len(set(keys)) == len(keys), f"Duplicate keys: {keys}"


def test_bibtex_output_parses():
    import bibtexparser
    agent = CitationAgent()
    p = Paper(
        paper_id="a:1", title="A Test Paper", authors=["Brown Bob", "Lee Carol"],
        abstract="x", year=2022, source="semantic_scholar",
    )
    p.bibtex_key = "brown2022test"
    bib = agent._construct_bibtex(p)
    parsed = bibtexparser.loads(bib)
    assert len(parsed.entries) == 1
    assert parsed.entries[0]["ID"] == "brown2022test"
