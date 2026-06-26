from __future__ import annotations

from loguru import logger

from src.models.schemas import WorkflowState


class SupervisorAgent:
    def plan(self, state: WorkflowState) -> WorkflowState:
        logger.info(f"Supervisor: Planning workflow | topic={state.topic!r}")
        topic = state.topic.strip()
        # Two queries: direct topic and a survey framing — both consumed by SearchAgent
        queries = [
            topic,
            f"survey {topic}",
        ]
        state.search_queries = queries
        state.stage = "planned"
        logger.info(
            f"Supervisor: Generated {len(queries)} search queries | queries={queries}"
        )
        return state
