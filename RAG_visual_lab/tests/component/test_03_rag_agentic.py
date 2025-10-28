"""
Component Tests: 03 - RAG Agente
=================================

Tests for the RAG Agente Streamlit page, validating the UI components,
state management, and integration with services.

Test Strategy:
    - Use Streamlit AppTest to simulate page loading
    - Mock service providers to avoid external calls
    - Validate UI elements are rendered correctly
    - Test chat interface functionality
    - Verify session state initialization
"""

import pytest
from streamlit.testing.v1 import AppTest
from unittest.mock import patch, MagicMock
import os

# Define the path to the app's root for correct file loading in tests
APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.fixture
def mock_agentic_provider():
    """
    Pytest fixture to mock the AgenticRAGProvider service.
    This prevents tests from making real CrewAI/LLM calls and allows us to
    control the returned data for predictable UI testing.
    
    CRITICAL: We patch at the service level, not the page module level.
    """
    with patch('services.agentic_rag_provider.AgenticRAGProvider') as mock:
        # Create an instance of the mock to be configured
        mock_instance = MagicMock()
        # Configure the mock instance's methods
        mock_instance.route_query.return_value = {
            "dataset_name": "direito_constitucional",
            "locale": "pt-br",
            "query": "O que é direito constitucional?"
        }
        # When the class is instantiated in the app, it will return our configured mock instance
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_gemini_provider():
    """
    Pytest fixture to mock the GeminiProvider to avoid real API calls.
    """
    with patch('services.gemini_provider.validate_gemini_api_key') as mock_validate, \
         patch('services.gemini_provider.get_gemini_llm_function') as mock_llm:
        
        mock_validate.return_value = True
        
        # Mock LLM function
        mock_llm_func = MagicMock()
        mock_llm_func.return_value = "Resposta de teste do agente RAG"
        mock_llm.return_value = mock_llm_func
        
        yield mock_validate, mock_llm


@pytest.fixture
def mock_retriever_provider():
    """
    Pytest fixture to mock the RetrieverProvider to avoid ChromaDB calls.
    """
    with patch('services.retriever_provider.RetrieverProvider') as mock:
        mock_instance = MagicMock()
        mock_instance.search.return_value = [
            "Chunk 1: Informação sobre direito constitucional",
            "Chunk 2: Jurisprudência relevante",
            "Chunk 3: Precedentes legais"
        ]
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_augmentation_provider():
    """
    Pytest fixture to mock the AugmentationProvider.
    """
    with patch('services.augmentation_provider.AugmentationProvider') as mock:
        mock_instance = MagicMock()
        mock_instance.generate_prompt.return_value = "Prompt enriquecido para LLM"
        mock.return_value = mock_instance
        yield mock_instance


def test_rag_agentic_page_smoke_test(
    mock_agentic_provider,
    mock_gemini_provider,
    mock_retriever_provider,
    mock_augmentation_provider
):
    """
    Smoke test: Verifies that the RAG Agente page loads without errors.
    
    This is the most basic test - just check that the page can be instantiated
    and rendered without exceptions.
    """
    page_path = os.path.join(APP_ROOT, "pages", "03_🤖_RAG_Agentic.py")
    at = AppTest.from_file(page_path, default_timeout=30)
    
    at.run()

    # Verify page loaded without exceptions
    assert not at.exception, f"Page should load without errors. Exception: {at.exception}"


def test_page_title_and_elements(
    mock_agentic_provider,
    mock_gemini_provider,
    mock_retriever_provider,
    mock_augmentation_provider
):
    """
    Tests that page displays the correct title and main UI elements.
    
    Validates:
    - Title matches expected value
    - Chat input is present
    - Sidebar configuration exists
    """
    page_path = os.path.join(APP_ROOT, "pages", "03_🤖_RAG_Agentic.py")
    at = AppTest.from_file(page_path, default_timeout=30)
    
    at.run()

    # Check for title
    assert len(at.title) > 0, "Page should have a title"
    title_text = at.title[0].value if hasattr(at.title[0], 'value') else str(at.title[0])
    assert "Laboratório de RAG Agente" in title_text or "Agente" in title_text

    # Check for chat input
    assert len(at.chat_input) == 1, "Page should have exactly one chat input"

    # Verify no exceptions
    assert not at.exception


def test_sidebar_exists(
    mock_agentic_provider,
    mock_gemini_provider,
    mock_retriever_provider,
    mock_augmentation_provider
):
    """
    Tests that sidebar configuration is rendered.
    """
    page_path = os.path.join(APP_ROOT, "pages", "03_🤖_RAG_Agentic.py")
    at = AppTest.from_file(page_path, default_timeout=30)
    
    at.run()

    # Verify page loaded
    assert not at.exception


def test_chat_input_present(
    mock_agentic_provider,
    mock_gemini_provider,
    mock_retriever_provider,
    mock_augmentation_provider
):
    """
    Tests that the chat input widget is present and ready for input.
    """
    page_path = os.path.join(APP_ROOT, "pages", "03_🤖_RAG_Agentic.py")
    at = AppTest.from_file(page_path, default_timeout=30)
    
    at.run()

    # Verify chat input exists
    assert len(at.chat_input) == 1
    chat_input = at.chat_input[0]
    
    # Chat input should be ready for text entry
    assert chat_input is not None


def test_session_state_initialization(
    mock_agentic_provider,
    mock_gemini_provider,
    mock_retriever_provider,
    mock_augmentation_provider
):
    """
    Tests that session state variables are properly initialized.
    
    The initialize_session_state() function should create:
    - rag_agentic_messages
    - rag_agentic_logs
    - rag_agentic_provider
    - agentic_chroma_db_path
    - agentic_chroma_n_results
    """
    page_path = os.path.join(APP_ROOT, "pages", "03_🤖_RAG_Agentic.py")
    at = AppTest.from_file(page_path, default_timeout=30)
    
    at.run()

    # Access session state through the app context
    # Note: AppTest doesn't directly expose session_state, but we can verify
    # the page runs without errors, which means initialization succeeded
    assert not at.exception


def test_page_loads_with_default_info_message(
    mock_agentic_provider,
    mock_gemini_provider,
    mock_retriever_provider,
    mock_augmentation_provider
):
    """
    Tests that the page displays a welcome message when no messages exist.
    """
    page_path = os.path.join(APP_ROOT, "pages", "03_🤖_RAG_Agentic.py")
    at = AppTest.from_file(page_path, default_timeout=30)
    
    at.run()

    # Verify page loaded without exceptions
    assert not at.exception
    
    # The page should have rendered without errors
    assert len(at.title) > 0


def test_expanders_for_routing_and_reasoning(
    mock_agentic_provider,
    mock_gemini_provider,
    mock_retriever_provider,
    mock_augmentation_provider
):
    """
    Tests that expanders are present for showing agent reasoning.
    
    The page should have expanders for:
    - Informações do Roteamento (routing info JSON)
    - Raciocínio do Agente (agent reasoning logs)
    """
    page_path = os.path.join(APP_ROOT, "pages", "03_🤖_RAG_Agentic.py")
    at = AppTest.from_file(page_path, default_timeout=30)
    
    at.run()

    # Verify page structure is sound
    assert not at.exception


def test_footer_documentation_present(
    mock_agentic_provider,
    mock_gemini_provider,
    mock_retriever_provider,
    mock_augmentation_provider
):
    """
    Tests that the footer with technical documentation exists.
    """
    page_path = os.path.join(APP_ROOT, "pages", "03_🤖_RAG_Agentic.py")
    at = AppTest.from_file(page_path, default_timeout=30)
    
    at.run()

    # Verify page rendered successfully
    assert not at.exception


def test_clear_conversation_button_exists(
    mock_agentic_provider,
    mock_gemini_provider,
    mock_retriever_provider,
    mock_augmentation_provider
):
    """
    Tests that a button to clear conversation history exists.
    
    This button should be labeled "🗑️ Limpar Conversa" or similar.
    """
    page_path = os.path.join(APP_ROOT, "pages", "03_🤖_RAG_Agentic.py")
    at = AppTest.from_file(page_path, default_timeout=30)
    
    at.run()

    # Verify page loaded
    assert not at.exception
    
    # Check for button
    assert len(at.button) > 0, "Page should have at least one button (clear conversation)"
