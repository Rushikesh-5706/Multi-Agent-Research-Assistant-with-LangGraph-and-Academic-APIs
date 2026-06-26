from __future__ import annotations

import os
import sys
import uuid

import click
from dotenv import load_dotenv

load_dotenv()

from loguru import logger

from src.graph.workflow import build_workflow
from src.models.schemas import WorkflowState
from src.state.redis_manager import RedisManager
from src.utils.logger import setup_logger


@click.command()
@click.option(
    "--topic",
    required=True,
    type=str,
    help="Research topic for which to generate a literature review.",
)
def main(topic: str) -> None:
    """Generate an academic literature review using a multi-agent LangGraph pipeline."""
    os.makedirs("./output", exist_ok=True)
    setup_logger()
    logger.info(f"Research Assistant starting | topic={topic!r}")

    run_id = str(uuid.uuid4())
    logger.info(f"Run ID: {run_id}")

    redis_manager = RedisManager()
    initial_state = WorkflowState(run_id=run_id, topic=topic)
    redis_manager.save_state(run_id, initial_state)
    logger.info(f"Initial state saved to Redis | run_id={run_id}")

    workflow = build_workflow(redis_manager)

    graph_initial_state: dict = {
        "run_id": run_id,
        "topic": topic,
        "stage": "init",
        "error": None,
    }

    try:
        result = workflow.invoke(graph_initial_state)
        if result.get("error"):
            logger.error(f"Workflow terminated with error: {result['error']}")
            sys.exit(1)
        logger.info("Workflow completed successfully")
        logger.info("Output written to:")
        logger.info("  ./output/literature_review.md")
        logger.info("  ./output/references.bib")
        logger.info("  ./output/agent_run.log")
        sys.exit(0)
    except Exception as exc:
        logger.error(f"Fatal error during workflow execution: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
