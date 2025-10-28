"""
Unit Tests: RetrieverProvider
==============================

Tests for the RetrieverProvider service class, which handles document chunk
retrieval from ChromaDB using semantic search.

Test Strategy:
    - Mock ChromaDB and SentenceTransformer to isolate RetrieverProvider logic
    - Validate initialization, error handling, and search functionality
    - Test integration with ChromaDB query interface
    - Verify proper exception handling for missing collections

Autor: James (Developer)
Data: 2025-10-26
Task: Task 3.3 - ChromaDB Integration
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock, Mock

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def test_retriever_provider_initialization_success():
    """
    Tests successful initialization of RetrieverProvider.
    
    Validates:
    - Instance is created successfully
    - db_path and collection_name are stored correctly
    - ChromaDB client is initialized
    - SentenceTransformer model is loaded
    """
    with patch('services.retriever_provider.chromadb.PersistentClient') as mock_client_class, \
         patch('services.retriever_provider.SentenceTransformer') as mock_model_class:
        
        # Setup mocks
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 139
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        
        from services.retriever_provider import RetrieverProvider
        
        # Initialize RetrieverProvider
        retriever = RetrieverProvider(
            db_path="./test_db",
            collection_name="test_collection"
        )
        
        # Verify initialization
        assert retriever.db_path == "./test_db"
        assert retriever.collection_name == "test_collection"
        assert retriever.client is not None
        assert retriever.collection is not None
        assert retriever.modelo is not None
        
        # Verify ChromaDB client was initialized with correct path
        mock_client_class.assert_called_once_with(path="./test_db")
        
        # Verify collection was retrieved
        mock_client.get_collection.assert_called_once_with(name="test_collection")
        
        # Verify model was loaded
        mock_model_class.assert_called_once_with('paraphrase-multilingual-MiniLM-L12-v2')


def test_retriever_provider_rejects_empty_collection_name():
    """
    Tests that RetrieverProvider raises ValueError for empty collection_name.
    
    This prevents bugs where queries would fail due to missing collection.
    """
    from services.retriever_provider import RetrieverProvider
    
    with pytest.raises(ValueError, match="collection_name não pode ser vazio"):
        RetrieverProvider(collection_name="")
    
    with pytest.raises(ValueError, match="collection_name não pode ser vazio"):
        RetrieverProvider(collection_name=None)  # type: ignore


def test_retriever_provider_handles_missing_collection():
    """
    Tests that RetrieverProvider raises Exception when collection doesn't exist.
    
    This ensures proper error messaging when user specifies non-existent collection.
    """
    with patch('services.retriever_provider.chromadb.PersistentClient') as mock_client_class:
        
        # Setup mock to raise ValueError (collection not found)
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = ValueError("Collection not found")
        mock_client_class.return_value = mock_client
        
        from services.retriever_provider import RetrieverProvider
        
        with pytest.raises(Exception, match="Coleção 'nonexistent' não encontrada"):
            RetrieverProvider(
                db_path="./test_db",
                collection_name="nonexistent"
            )


def test_retriever_search_returns_chunks():
    """
    Tests that search() returns list of document chunks.
    
    Validates the core functionality: querying ChromaDB and returning results.
    """
    with patch('services.retriever_provider.chromadb.PersistentClient') as mock_client_class, \
         patch('services.retriever_provider.SentenceTransformer') as mock_model_class:
        
        # Setup ChromaDB mock
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 10
        
        # Mock query response - ChromaDB returns nested list structure
        mock_collection.query.return_value = {
            'documents': [['chunk1', 'chunk2', 'chunk3']],
            'distances': [[0.1, 0.2, 0.3]],
            'metadatas': [[{}, {}, {}]]
        }
        
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        # Setup SentenceTransformer mock
        mock_model = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.tolist.return_value = [[0.1, 0.2, 0.3]]
        mock_model.encode.return_value = mock_embedding
        mock_model_class.return_value = mock_model
        
        from services.retriever_provider import RetrieverProvider
        
        retriever = RetrieverProvider(collection_name="test")
        chunks = retriever.search("What is RAG?", n_results=3)
        
        # Verify results
        assert isinstance(chunks, list)
        assert len(chunks) == 3
        assert chunks[0] == 'chunk1'
        assert chunks[1] == 'chunk2'
        assert chunks[2] == 'chunk3'
        
        # Verify model.encode was called with query
        mock_model.encode.assert_called_once_with(["What is RAG?"])
        
        # Verify ChromaDB query was called correctly
        mock_collection.query.assert_called_once()
        call_args = mock_collection.query.call_args
        assert call_args.kwargs['n_results'] == 3
        assert 'documents' in call_args.kwargs['include']
        assert 'distances' in call_args.kwargs['include']


def test_retriever_search_with_custom_n_results():
    """
    Tests that search() respects the n_results parameter.
    
    Validates that the number of results can be configured.
    """
    with patch('services.retriever_provider.chromadb.PersistentClient') as mock_client_class, \
         patch('services.retriever_provider.SentenceTransformer') as mock_model_class:
        
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 100
        
        # Mock 10 results
        mock_collection.query.return_value = {
            'documents': [[f'chunk{i}' for i in range(10)]],
            'distances': [[0.1 * i for i in range(10)]],
            'metadatas': [[{} for _ in range(10)]]
        }
        
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        mock_model = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.tolist.return_value = [[0.1, 0.2]]
        mock_model.encode.return_value = mock_embedding
        mock_model_class.return_value = mock_model
        
        from services.retriever_provider import RetrieverProvider
        
        retriever = RetrieverProvider(collection_name="test")
        chunks = retriever.search("test query", n_results=10)
        
        assert len(chunks) == 10
        
        # Verify n_results was passed to ChromaDB
        call_args = mock_collection.query.call_args
        assert call_args.kwargs['n_results'] == 10


def test_retriever_search_handles_empty_results():
    """
    Tests that search() handles empty results gracefully.
    
    When ChromaDB returns no results, should return empty list.
    """
    with patch('services.retriever_provider.chromadb.PersistentClient') as mock_client_class, \
         patch('services.retriever_provider.SentenceTransformer') as mock_model_class:
        
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        
        # Mock empty results
        mock_collection.query.return_value = {
            'documents': [[]],
            'distances': [[]],
            'metadatas': [[]]
        }
        
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        mock_model = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.tolist.return_value = [[0.1, 0.2]]
        mock_model.encode.return_value = mock_embedding
        mock_model_class.return_value = mock_model
        
        from services.retriever_provider import RetrieverProvider
        
        retriever = RetrieverProvider(collection_name="test")
        chunks = retriever.search("query with no matches", n_results=5)
        
        assert chunks == []


def test_retriever_search_handles_query_error():
    """
    Tests that search() handles ChromaDB query errors gracefully.
    
    Should return empty list instead of crashing when query fails.
    """
    with patch('services.retriever_provider.chromadb.PersistentClient') as mock_client_class, \
         patch('services.retriever_provider.SentenceTransformer') as mock_model_class:
        
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 10
        
        # Mock query error
        mock_collection.query.side_effect = Exception("ChromaDB connection error")
        
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        mock_model = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.tolist.return_value = [[0.1, 0.2]]
        mock_model.encode.return_value = mock_embedding
        mock_model_class.return_value = mock_model
        
        from services.retriever_provider import RetrieverProvider
        
        retriever = RetrieverProvider(collection_name="test")
        chunks = retriever.search("test query", n_results=5)
        
        # Should return empty list instead of raising exception
        assert chunks == []


def test_retriever_get_collection_info():
    """
    Tests get_collection_info() returns correct metadata.
    
    Validates debugging utility function.
    """
    with patch('services.retriever_provider.chromadb.PersistentClient') as mock_client_class, \
         patch('services.retriever_provider.SentenceTransformer') as mock_model_class:
        
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_collection.count.return_value = 42
        mock_collection.metadata = {"description": "Test data"}
        
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        
        from services.retriever_provider import RetrieverProvider
        
        retriever = RetrieverProvider(collection_name="test_collection")
        info = retriever.get_collection_info()
        
        assert info['name'] == "test_collection"
        assert info['count'] == 42
        assert info['metadata']['description'] == "Test data"


def test_retriever_uses_custom_model():
    """
    Tests that RetrieverProvider can use a custom embedding model.
    
    Validates flexibility in model selection.
    """
    with patch('services.retriever_provider.chromadb.PersistentClient') as mock_client_class, \
         patch('services.retriever_provider.SentenceTransformer') as mock_model_class:
        
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 10
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        
        from services.retriever_provider import RetrieverProvider
        
        retriever = RetrieverProvider(
            collection_name="test",
            model_name="custom-model-name"
        )
        
        # Verify custom model was loaded
        mock_model_class.assert_called_once_with("custom-model-name")
        assert retriever.model_name == "custom-model-name"
