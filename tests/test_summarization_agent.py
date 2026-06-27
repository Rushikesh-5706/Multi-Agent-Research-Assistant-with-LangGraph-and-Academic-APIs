import pytest
from unittest.mock import MagicMock, patch
import httpx
from src.agents.summarization_agent import SummarizationAgent
from src.models.schemas import Paper

def test_get_source_text_primary_pdf():
    agent = SummarizationAgent()
    agent._pdf_parser = MagicMock()
    agent._pdf_parser.extract_text_from_url.return_value = "PDF Text"
    
    p = Paper(paper_id="1", title="Test", authors=["A"], source="test", pdf_url="http://pdf", abstract="Abstract")
    text = agent._get_source_text(p)
    assert text == "PDF Text"
    agent._pdf_parser.extract_text_from_url.assert_called_once_with("http://pdf")

def test_get_source_text_unpaywall_fallback():
    agent = SummarizationAgent()
    agent._pdf_parser = MagicMock()
    # Primary PDF extract fails, Unpaywall URL extract succeeds
    agent._pdf_parser.extract_text_from_url.side_effect = [None, "Unpaywall Text"]
    agent._pdf_parser.find_open_access_url.return_value = "http://oa-pdf"
    
    p = Paper(paper_id="1", title="Test", authors=["A"], source="test", pdf_url="http://pdf", doi="10.1234/test", abstract="Abstract")
    text = agent._get_source_text(p)
    assert text == "Unpaywall Text"
    agent._pdf_parser.find_open_access_url.assert_called_once_with("10.1234/test")
    assert agent._pdf_parser.extract_text_from_url.call_count == 2

def test_get_source_text_abstract_fallback():
    agent = SummarizationAgent()
    agent._pdf_parser = MagicMock()
    agent._pdf_parser.extract_text_from_url.return_value = None
    agent._pdf_parser.find_open_access_url.return_value = None
    
    p = Paper(paper_id="1", title="Test", authors=["A"], source="test", pdf_url="http://pdf", doi="10.1234/test", abstract="Abstract text")
    text = agent._get_source_text(p)
    assert text == "Abstract text"

@patch("src.agents.summarization_agent.httpx.Client")
def test_call_ollama_success(mock_client):
    mock_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "LLM Summary"}
    mock_instance.post.return_value = mock_response
    mock_client.return_value.__enter__.return_value = mock_instance
    
    agent = SummarizationAgent()
    summary = agent._call_ollama("Some text", "Test Title")
    assert summary == "LLM Summary"

@patch("src.agents.summarization_agent.httpx.Client")
def test_call_ollama_failure(mock_client):
    mock_instance = MagicMock()
    mock_instance.post.side_effect = httpx.ConnectError("Failed to connect")
    mock_client.return_value.__enter__.return_value = mock_instance
    
    agent = SummarizationAgent()
    with pytest.raises(Exception):
        agent._call_ollama("Some text", "Test Title")
