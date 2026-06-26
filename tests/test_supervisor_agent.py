from src.agents.supervisor_agent import SupervisorAgent
from src.models.schemas import WorkflowState


def test_plan_generates_queries():
    agent = SupervisorAgent()
    ws = WorkflowState(run_id="r1", topic="graph neural networks")
    result = agent.plan(ws)
    assert result.stage == "planned"
    assert len(result.search_queries) >= 1
    assert "graph neural networks" in result.search_queries[0]


def test_plan_strips_whitespace():
    agent = SupervisorAgent()
    ws = WorkflowState(run_id="r2", topic="  transformer models  ")
    result = agent.plan(ws)
    assert "transformer models" in result.search_queries[0]
