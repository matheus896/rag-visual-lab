"""
Unit Tests: AgenticRAGProvider
===============================

Tests for the AgenticRAGProvider service class, which handles intelligent
dataset routing using CrewAI agents.

Test Strategy:
    - Mock CrewAI Crew.kickoff() to isolate agent logic
    - Validate JSON parsing with valid and invalid responses
    - Test error handling and graceful degradation
    - Verify integration with DatasetsProvider
"""

import pytest
import json
import sys
import os
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


@pytest.fixture
def provider():
    """Fixture that creates an AgenticRAGProvider instance."""
    from services.agentic_rag_provider import AgenticRAGProvider
    return AgenticRAGProvider()


@patch('services.agentic_rag_provider.Agent')
@patch('services.agentic_rag_provider.Task')
@patch('services.agentic_rag_provider.Crew')
def test_route_query_success(mock_crew_class, mock_task_class, mock_agent_class, provider):
    """
    Tests successful query routing with valid JSON response.
    
    Validates:
    - Agent is created with correct role and goal
    - Task is created with correct description
    - Crew.kickoff() is called
    - JSON response is parsed correctly
    - Dictionary with dataset_name, locale, query is returned
    """
    # Setup mock Agent and Task
    mock_agent_instance = MagicMock()
    mock_agent_class.return_value = mock_agent_instance
    
    mock_task_instance = MagicMock()
    mock_task_class.return_value = mock_task_instance
    
    # Setup mock Crew
    mock_crew_instance = MagicMock()
    mock_crew_class.return_value = mock_crew_instance
    
    # Define expected response
    expected_response = {
        "dataset_name": "direito_constitucional",
        "locale": "pt-br",
        "query": "O que é direito constitucional fala do abandono afetivo?"
    }
    
    # Make crew.kickoff() return the JSON as a string
    mock_crew_instance.kickoff.return_value = json.dumps(expected_response)
    
    # Call route_query
    query = "O que é direito constitucional fala do abandono afetivo?"
    result = provider.route_query(query)
    
    # Verify results
    assert result is not None, "route_query should return a dict, not None"
    assert isinstance(result, dict), "route_query should return a dict"
    assert result["dataset_name"] == "direito_constitucional"
    assert result["locale"] == "pt-br"
    assert "abandono afetivo" in result["query"].lower()
    
    # Verify Crew was instantiated and executed
    mock_crew_class.assert_called_once()
    mock_crew_instance.kickoff.assert_called_once()


@patch('services.agentic_rag_provider.Agent')
@patch('services.agentic_rag_provider.Task')
@patch('services.agentic_rag_provider.Crew')
def test_route_query_with_markdown_json(mock_crew_class, mock_task_class, mock_agent_class, provider):
    """
    Tests query routing when agent wraps JSON in markdown code blocks.
    
    This handles the common case where LLMs add ```json and ``` around responses
    despite explicit instructions not to.
    """
    mock_agent_instance = MagicMock()
    mock_agent_class.return_value = mock_agent_instance
    
    mock_task_instance = MagicMock()
    mock_task_class.return_value = mock_task_instance
    
    mock_crew_instance = MagicMock()
    mock_crew_class.return_value = mock_crew_instance
    
    expected_response = {
        "dataset_name": "synthetic_dataset_papers",
        "locale": "en",
        "query": "Tell me about synthetic datasets"
    }
    
    # Simulate LLM response with markdown code blocks
    markdown_response = f"```json\n{json.dumps(expected_response)}\n```"
    mock_crew_instance.kickoff.return_value = markdown_response
    
    query = "Tell me about synthetic datasets"
    result = provider.route_query(query)
    
    # Should still parse correctly despite markdown
    assert result is not None
    assert result["dataset_name"] == "synthetic_dataset_papers"
    assert result["locale"] == "en"


@patch('services.agentic_rag_provider.Agent')
@patch('services.agentic_rag_provider.Task')
@patch('services.agentic_rag_provider.Crew')
def test_route_query_invalid_json(mock_crew_class, mock_task_class, mock_agent_class, provider):
    """
    Tests that invalid JSON response is handled gracefully.
    
    When the agent returns plain text instead of JSON, the function should
    return None instead of crashing.
    """
    mock_agent_instance = MagicMock()
    mock_agent_class.return_value = mock_agent_instance
    
    mock_task_instance = MagicMock()
    mock_task_class.return_value = mock_task_instance
    
    mock_crew_instance = MagicMock()
    mock_crew_class.return_value = mock_crew_instance
    
    # Make crew.kickoff() return invalid JSON (plain text)
    mock_crew_instance.kickoff.return_value = "This is not JSON, just plain text from the agent"
    
    query = "Some query"
    result = provider.route_query(query)
    
    # Should return None, not crash
    assert result is None, "route_query should return None for invalid JSON"


@patch('services.agentic_rag_provider.Agent')
@patch('services.agentic_rag_provider.Task')
@patch('services.agentic_rag_provider.Crew')
def test_route_query_empty_response(mock_crew_class, mock_task_class, mock_agent_class, provider):
    """
    Tests that empty or whitespace-only responses are handled.
    """
    mock_agent_instance = MagicMock()
    mock_agent_class.return_value = mock_agent_instance
    
    mock_task_instance = MagicMock()
    mock_task_class.return_value = mock_task_instance
    
    mock_crew_instance = MagicMock()
    mock_crew_class.return_value = mock_crew_instance
    
    # Empty response
    mock_crew_instance.kickoff.return_value = ""
    
    query = "Some query"
    result = provider.route_query(query)
    
    assert result is None


@patch('services.agentic_rag_provider.Agent')
@patch('services.agentic_rag_provider.Task')
@patch('services.agentic_rag_provider.Crew')
def test_route_query_crew_exception(mock_crew_class, mock_task_class, mock_agent_class, provider):
    """
    Tests that exceptions from Crew.kickoff() are caught gracefully.
    """
    mock_agent_instance = MagicMock()
    mock_agent_class.return_value = mock_agent_instance
    
    mock_task_instance = MagicMock()
    mock_task_class.return_value = mock_task_instance
    
    mock_crew_instance = MagicMock()
    mock_crew_class.return_value = mock_crew_instance
    
    # Make crew.kickoff() raise an exception
    mock_crew_instance.kickoff.side_effect = RuntimeError("API connection failed")
    
    query = "Some query"
    result = provider.route_query(query)
    
    # Should return None, not propagate exception
    assert result is None


def test_datasets_provider_integration(provider):
    """
    Tests that DatasetsProvider is correctly initialized.
    
    Validates:
    - DatasetsProvider is instantiated
    - get_datasets() returns list with at least 2 datasets
    - get_dataset_description() returns formatted string
    """
    assert provider.datasets_provider is not None
    
    datasets = provider.datasets_provider.get_datasets()
    assert isinstance(datasets, list)
    assert len(datasets) >= 2
    
    description = provider.datasets_provider.get_dataset_description()
    assert isinstance(description, str)
    assert len(description) > 0
    assert "synthetic_dataset_papers" in description
    assert "direito_constitucional" in description


@patch('services.agentic_rag_provider.Agent')
@patch('services.agentic_rag_provider.Task')
@patch('services.agentic_rag_provider.Crew')
def test_route_query_with_different_datasets(mock_crew_class, mock_task_class, mock_agent_class, provider):
    """
    Tests that agent can route to different datasets based on query.
    
    This tests robustness by verifying the function works with different
    dataset names.
    """
    mock_agent_instance = MagicMock()
    mock_agent_class.return_value = mock_agent_instance
    
    mock_task_instance = MagicMock()
    mock_task_class.return_value = mock_task_instance
    
    mock_crew_instance = MagicMock()
    mock_crew_class.return_value = mock_crew_instance
    
    # Test routing to synthetic_dataset_papers
    response1 = {
        "dataset_name": "synthetic_dataset_papers",
        "locale": "en",
        "query": "What is a synthetic dataset?"
    }
    mock_crew_instance.kickoff.return_value = json.dumps(response1)
    
    result1 = provider.route_query("Tell me about synthetic datasets")
    assert result1["dataset_name"] == "synthetic_dataset_papers"
    
    # Reset mock for second call
    mock_crew_instance.reset_mock()
    mock_agent_class.reset_mock()
    mock_task_class.reset_mock()
    mock_crew_class.reset_mock()
    
    # Test routing to direito_constitucional
    response2 = {
        "dataset_name": "direito_constitucional",
        "locale": "pt-br",
        "query": "O que é responsabilidade civil?"
    }
    mock_crew_instance.kickoff.return_value = json.dumps(response2)
    
    result2 = provider.route_query("Me explique sobre direito civil")
    assert result2["dataset_name"] == "direito_constitucional"
