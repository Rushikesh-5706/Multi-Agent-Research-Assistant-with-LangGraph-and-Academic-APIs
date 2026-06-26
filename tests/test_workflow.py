from unittest.mock import MagicMock, patch

from src.graph.workflow import build_workflow, _write_markdown, _write_bibtex
from src.models.schemas import Paper, WorkflowState


def _make_papers():
    return [
        Paper(
            paper_id="arxiv:001",
            title="Attention Is All You Need",
            authors=["Vaswani Ashish", "Shazeer Noam"],
            abstract="We propose the Transformer architecture.",
            year=2017,
            arxiv_id="1706.03762",
            source="arxiv",
            summary="- Self-attention\n- Transformer\n- SOTA MT",
            bibtex_key="vaswani2017attention",
            bibtex="@article{vaswani2017attention,\n  title = {Attention Is All You Need},\n  author = {Vaswani Ashish},\n  journal = {arXiv preprint},\n  year = {2017}\n}",
        ),
        Paper(
            paper_id="ss:002",
            title="BERT: Bidirectional Transformers",
            authors=["Devlin Jacob", "Chang Ming-Wei"],
            abstract="We introduce BERT.",
            year=2019,
            doi="10.18653/v1/N19-1423",
            source="semantic_scholar",
            summary="- Bidirectional encoder\n- Masked LM\n- SOTA GLUE",
            bibtex_key="devlin2019bert",
            bibtex="@article{devlin2019bert,\n  title = {BERT},\n  author = {Devlin Jacob},\n  journal = {Manuscript},\n  year = {2019}\n}",
        ),
    ]


def test_write_markdown_sections(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "output").mkdir()
    ws = WorkflowState(
        run_id="r1", topic="transformer models", papers=_make_papers(), stage="cited"
    )
    _write_markdown(ws)
    content = (tmp_path / "output" / "literature_review.md").read_text()
    assert "# Literature Review: transformer models" in content
    assert "## Introduction" in content
    assert "## Related Work" in content
    assert "## Conclusion" in content
    assert "## References" in content
    assert "Attention Is All You Need" in content
    assert "BERT" in content
    assert "arXiv" in content          # source label correctly rendered
    assert "Arxiv" not in content      # not incorrectly rendered


def test_write_bibtex_produces_valid_entries(tmp_path, monkeypatch):
    import bibtexparser
    monkeypatch.chdir(tmp_path)
    (tmp_path / "output").mkdir()
    ws = WorkflowState(
        run_id="r1", topic="nlp", papers=_make_papers(), stage="cited"
    )
    _write_bibtex(ws)
    content = (tmp_path / "output" / "references.bib").read_text()
    parsed = bibtexparser.loads(content)
    assert len(parsed.entries) == 2


def test_build_workflow_compiles():
    mock_redis = MagicMock()
    mock_redis.load_state.return_value = WorkflowState(run_id="r1", topic="test")
    workflow = build_workflow(mock_redis)
    assert workflow is not None
