import pytest
from unittest.mock import MagicMock, patch

from src.models.schemas import WorkflowState
from src.state.redis_manager import RedisManager


@pytest.fixture
def mock_redis_client():
    with patch("src.state.redis_manager.redis.Redis") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.ping.return_value = True
        mock_instance.connection_pool.connection_kwargs = {"host": "localhost", "port": 6379}
        mock_cls.return_value = mock_instance
        yield mock_instance


def test_save_and_load_state(mock_redis_client):
    manager = RedisManager()
    ws = WorkflowState(run_id="test-123", topic="deep learning", stage="planned")
    serialized = ws.model_dump_json()

    manager.save_state("test-123", ws)
    mock_redis_client.set.assert_called_once()
    call_args = mock_redis_client.set.call_args
    assert "research:test-123:state" in call_args[0]

    mock_redis_client.get.return_value = serialized
    loaded = manager.load_state("test-123")
    assert loaded.run_id == "test-123"
    assert loaded.topic == "deep learning"
    assert loaded.stage == "planned"


def test_load_state_missing_key_raises(mock_redis_client):
    manager = RedisManager()
    mock_redis_client.get.return_value = None
    with pytest.raises(KeyError, match="No state found"):
        manager.load_state("nonexistent-run-id")
