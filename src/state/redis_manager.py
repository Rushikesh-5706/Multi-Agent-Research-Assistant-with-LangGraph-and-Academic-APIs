from __future__ import annotations

import os

import redis
from loguru import logger

from src.models.schemas import WorkflowState

_STATE_TTL_SECONDS = 86400  # 24 hours


class RedisManager:
    def __init__(self) -> None:
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", "6379"))
        self._client = redis.Redis(host=host, port=port, decode_responses=True)
        self._verify_connection()

    def _verify_connection(self) -> None:
        try:
            self._client.ping()
            conn = self._client.connection_pool.connection_kwargs
            logger.info(
                f"Redis connection established | host={conn.get('host')} port={conn.get('port')}"
            )
        except redis.ConnectionError as exc:
            logger.error(f"Redis connection failed: {exc}")
            raise

    def save_state(self, run_id: str, state: WorkflowState) -> None:
        key = f"research:{run_id}:state"
        self._client.set(key, state.model_dump_json(), ex=_STATE_TTL_SECONDS)
        logger.debug(
            f"State saved to Redis | run_id={run_id} | stage={state.stage} | papers={len(state.papers)}"
        )

    def load_state(self, run_id: str) -> WorkflowState:
        key = f"research:{run_id}:state"
        raw = self._client.get(key)
        if raw is None:
            raise KeyError(f"No state found in Redis for run_id={run_id}")
        state = WorkflowState.model_validate_json(raw)
        logger.debug(f"State loaded from Redis | run_id={run_id} | stage={state.stage}")
        return state

    def delete_state(self, run_id: str) -> None:
        key = f"research:{run_id}:state"
        self._client.delete(key)
        logger.debug(f"State deleted from Redis | run_id={run_id}")
