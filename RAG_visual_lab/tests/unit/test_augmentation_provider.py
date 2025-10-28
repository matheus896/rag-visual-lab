"""
Unit Tests: AugmentationProvider
=================================

Tests for the AugmentationProvider service class, which orchestrates
the combination of document chunks (from ChromaDB) with conversational
memory (from Redis) to create enriched prompts for the LLM.

Test Strategy:
    - Mock MemoryProvider to isolate AugmentationProvider logic
    - Validate prompt formatting matches reference implementation
    - Verify memory persistence operations
    - Test error handling and edge cases

Autor: James (Developer)
Data: 2025-10-26
Task: Task 3.2 - RAG Pipeline with Memory Integration
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def test_augmentation_provider_initialization():
    """
    Tests if the AugmentationProvider initializes correctly with a valid talk_id.
    
    Validates:
    - Instance is created successfully
    - talk_id is stored correctly
    - MemoryProvider is instantiated
    - last_prompt starts empty
    """
    with patch('services.augmentation_provider.MemoryProvider') as mock_memory:
        from services.augmentation_provider import AugmentationProvider
        
        augmenter = AugmentationProvider(talk_id="test-session")
        
        assert augmenter.talk_id == "test-session"
        assert augmenter.last_prompt == ""
        mock_memory.assert_called_once_with(talk_id="test-session")


def test_augmentation_provider_rejects_empty_talk_id():
    """
    Tests that AugmentationProvider raises ValueError for empty talk_id.
    
    This prevents bugs where conversations would be lost or mixed due to
    missing identifiers.
    """
    from services.augmentation_provider import AugmentationProvider
    
    with pytest.raises(ValueError, match="talk_id não pode ser vazio"):
        AugmentationProvider(talk_id="")
    
    with pytest.raises(ValueError, match="talk_id não pode ser vazio"):
        AugmentationProvider(talk_id=None)


def test_generate_prompt_includes_chunks():
    """
    Tests that the generated prompt includes all provided chunks.
    
    This validates the core functionality of the Augmentation step:
    combining retrieved document chunks into the prompt.
    """
    with patch('services.augmentation_provider.MemoryProvider') as mock_memory:
        from services.augmentation_provider import AugmentationProvider
        
        # Setup mock to return empty history
        mock_memory_instance = mock_memory.return_value
        mock_memory_instance.get_conversation.return_value = None
        
        augmenter = AugmentationProvider(talk_id="test")
        
        chunks = ["First chunk about RAG", "Second chunk about embeddings"]
        query = "What is RAG?"
        
        prompt = augmenter.generate_prompt(query=query, chunks=chunks)
        
        # Verify all chunks are in the prompt
        assert "First chunk about RAG" in prompt
        assert "Second chunk about embeddings" in prompt
        
        # Verify prompt structure (XML-like tags from reference)
        assert "<chunks>" in prompt
        assert "</chunks>" in prompt


def test_generate_prompt_includes_query():
    """
    Tests that the query is included in the generated prompt.
    
    The query should be clearly delimited in the prompt structure.
    """
    with patch('services.augmentation_provider.MemoryProvider') as mock_memory:
        from services.augmentation_provider import AugmentationProvider
        
        mock_memory_instance = mock_memory.return_value
        mock_memory_instance.get_conversation.return_value = None
        
        augmenter = AugmentationProvider(talk_id="test")
        
        query = "What are the main concepts of RAG?"
        chunks = ["RAG stands for Retrieval-Augmented Generation"]
        
        prompt = augmenter.generate_prompt(query=query, chunks=chunks)
        
        # Verify query is in the prompt
        assert query in prompt
        
        # Verify query delimiters
        assert "<query>" in prompt
        assert "</query>" in prompt


def test_generate_prompt_includes_memory_history():
    """
    Tests that conversational history from Redis is included in the prompt.
    
    This is the key integration point that differentiates RAG with Memory
    from classic RAG - the prompt must include previous interactions.
    """
    with patch('services.augmentation_provider.MemoryProvider') as mock_memory:
        from services.augmentation_provider import AugmentationProvider
        
        # Setup mock to return conversation history
        mock_memory_instance = mock_memory.return_value
        mock_memory_instance.get_conversation.return_value = [
            {"role": "assistant", "content": "Previous response"},
            {"role": "user", "content": "Previous question"}
        ]
        
        augmenter = AugmentationProvider(talk_id="test")
        
        query = "Follow-up question"
        chunks = ["Some context"]
        
        prompt = augmenter.generate_prompt(query=query, chunks=chunks)
        
        # Verify history is included
        assert "Previous question" in prompt
        assert "Previous response" in prompt
        
        # Verify history delimiters
        assert "<historico>" in prompt
        assert "</historico>" in prompt


def test_generate_prompt_handles_empty_history():
    """
    Tests that prompt generation works correctly when there's no history.
    
    First interaction in a conversation should still generate a valid prompt.
    """
    with patch('services.augmentation_provider.MemoryProvider') as mock_memory:
        from services.augmentation_provider import AugmentationProvider
        
        mock_memory_instance = mock_memory.return_value
        mock_memory_instance.get_conversation.return_value = None
        
        augmenter = AugmentationProvider(talk_id="test")
        
        query = "First question"
        chunks = ["Context chunk"]
        
        prompt = augmenter.generate_prompt(query=query, chunks=chunks)
        
        # Should still generate valid prompt
        assert query in prompt
        assert "Context chunk" in prompt
        assert "<historico>" in prompt
        
        # Should have placeholder text
        assert "Nenhum histórico disponível" in prompt


def test_generate_prompt_follows_reference_format():
    """
    Tests that the prompt follows the exact format from code-sandeco-rag-memory.txt.
    
    This ensures compatibility with the reference implementation and proper
    LLM instruction structure.
    """
    with patch('services.augmentation_provider.MemoryProvider') as mock_memory:
        from services.augmentation_provider import AugmentationProvider
        
        mock_memory_instance = mock_memory.return_value
        mock_memory_instance.get_conversation.return_value = None
        
        augmenter = AugmentationProvider(talk_id="test")
        
        prompt = augmenter.generate_prompt(
            query="Test query",
            chunks=["chunk1", "chunk2"]
        )
        
        # Verify priority is stated
        assert "query=1, chunks=2, historico=3" in prompt
        
        # Verify response language instruction
        assert "pt-br" in prompt.lower()
        assert "markdown" in prompt.lower()
        
        # Verify knowledge insufficiency instruction
        assert "não temos conhecimento suficiente" in prompt


def test_add_response_to_memory_saves_correctly():
    """
    Tests that add_response_to_memory correctly persists the interaction.
    
    Validates:
    - Both user prompt and assistant response are saved
    - Correct roles are assigned
    - MemoryProvider.add_message is called with correct parameters
    """
    with patch('services.augmentation_provider.MemoryProvider') as mock_memory:
        from services.augmentation_provider import AugmentationProvider
        
        mock_memory_instance = mock_memory.return_value
        mock_memory_instance.get_conversation.return_value = None
        
        augmenter = AugmentationProvider(talk_id="test")
        
        # Generate a prompt first (sets last_prompt)
        augmenter.generate_prompt(query="Test?", chunks=["context"])
        
        # Add response to memory
        llm_response = "This is the LLM response"
        result = augmenter.add_response_to_memory(llm_response)
        
        # Verify it returned True (success)
        assert result is True
        
        # Verify add_message was called twice (user + assistant)
        assert mock_memory_instance.add_message.call_count == 2
        
        # Verify calls have correct roles
        calls = mock_memory_instance.add_message.call_args_list
        assert calls[0][0][0] == "user"  # First call: user role
        assert calls[1][0][0] == "assistant"  # Second call: assistant role
        assert calls[1][0][1] == llm_response  # Assistant message content


def test_add_response_to_memory_fails_without_prompt():
    """
    Tests that add_response_to_memory fails gracefully if no prompt was generated.
    
    This prevents saving incomplete interactions to memory.
    """
    with patch('services.augmentation_provider.MemoryProvider') as mock_memory:
        from services.augmentation_provider import AugmentationProvider
        
        augmenter = AugmentationProvider(talk_id="test")
        
        # Try to add response without generating a prompt first
        result = augmenter.add_response_to_memory("Some response")
        
        # Should return False
        assert result is False
        
        # Should not call add_message
        mock_memory.return_value.add_message.assert_not_called()


def test_clear_memory_calls_delete_conversation():
    """
    Tests that clear_memory properly delegates to MemoryProvider.
    """
    with patch('services.augmentation_provider.MemoryProvider') as mock_memory:
        from services.augmentation_provider import AugmentationProvider
        
        mock_memory_instance = mock_memory.return_value
        
        augmenter = AugmentationProvider(talk_id="test")
        augmenter.clear_memory()
        
        # Verify delete_conversation was called
        mock_memory_instance.delete_conversation.assert_called_once()


def test_get_conversation_returns_history():
    """
    Tests that get_conversation retrieves history from MemoryProvider.
    """
    with patch('services.augmentation_provider.MemoryProvider') as mock_memory:
        from services.augmentation_provider import AugmentationProvider
        
        mock_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        mock_memory_instance = mock_memory.return_value
        mock_memory_instance.get_conversation.return_value = mock_history
        
        augmenter = AugmentationProvider(talk_id="test")
        history = augmenter.get_conversation()
        
        # Verify history is returned correctly
        assert history == mock_history
        mock_memory_instance.get_conversation.assert_called_once()


def test_generate_prompt_limits_history_to_5_messages():
    """
    Tests that only the last 5 messages from history are included in the prompt.
    
    This prevents prompt bloat and keeps context focused on recent conversation.
    """
    with patch('services.augmentation_provider.MemoryProvider') as mock_memory:
        from services.augmentation_provider import AugmentationProvider
        
        # Create history with 10 messages
        mock_history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
            for i in range(10)
        ]
        
        mock_memory_instance = mock_memory.return_value
        mock_memory_instance.get_conversation.return_value = mock_history
        
        augmenter = AugmentationProvider(talk_id="test")
        prompt = augmenter.generate_prompt(query="Test", chunks=["chunk"])
        
        # Verify only last 5 messages are in the prompt
        # Messages 0-4 should be in the prompt
        for i in range(5):
            assert f"Message {i}" in prompt
        
        # Messages 5-9 should NOT be in the prompt
        for i in range(5, 10):
            assert f"Message {i}" not in prompt
