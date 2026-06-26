from src.models.schemas import Paper, WorkflowState


def test_paper_defaults():
    p = Paper(
        paper_id="arxiv:001",
        title="Test Paper",
        authors=["Smith Alice"],
        abstract="An abstract.",
        source="arxiv",
    )
    assert p.summary is None
    assert p.bibtex is None
    assert p.year is None


def test_workflow_state_serialization():
    ws = WorkflowState(run_id="abc", topic="machine learning")
    serialized = ws.model_dump_json()
    restored = WorkflowState.model_validate_json(serialized)
    assert restored.run_id == ws.run_id
    assert restored.topic == ws.topic
    assert restored.stage == "init"
    assert restored.papers == []


def test_workflow_state_with_papers():
    p = Paper(
        paper_id="ss:001",
        title="Paper One",
        authors=["Jones Bob"],
        abstract="Abstract.",
        source="semantic_scholar",
        year=2023,
    )
    ws = WorkflowState(run_id="r1", topic="nlp", papers=[p])
    restored = WorkflowState.model_validate_json(ws.model_dump_json())
    assert len(restored.papers) == 1
    assert restored.papers[0].title == "Paper One"
