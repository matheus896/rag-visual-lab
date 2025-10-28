import pytest
from unittest.mock import Mock, patch, MagicMock
import os
import sys

# Adiciona o diretório raiz do projeto ao sys.path
APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, APP_ROOT)

from services.gemini_provider import GeminiEmbeddingConfig, GeminiConfig, validate_gemini_api_key


@pytest.fixture
def gemini_embedding_config():
    """Fixture que cria uma configuração de embedding válida."""
    return GeminiEmbeddingConfig(
        model_name="gemini-embedding-001",
        api_key="test-api-key",
        output_dimensionality=768,
        task_type="RETRIEVAL_DOCUMENT"
    )


@pytest.fixture
def gemini_llm_config():
    """Fixture que cria uma configuração de LLM válida."""
    return GeminiConfig(
        model_name="gemini-2.5-flash",
        api_key="test-api-key",
        temperature=0.7,
        max_tokens=2000
    )


def test_gemini_embedding_config_validation():
    """
    Testa se a configuração valida corretamente a dimensionalidade.
    """
    # Dimensionalidade válida
    config = GeminiEmbeddingConfig(
        api_key="test-key",
        output_dimensionality=768
    )
    assert config.output_dimensionality == 768
    
    # Dimensionalidade muito baixa deve levantar erro
    with pytest.raises(ValueError, match="deve estar entre 128 e 3072"):
        GeminiEmbeddingConfig(
            api_key="test-key",
            output_dimensionality=50  # Menor que 128
        )
    
    # Dimensionalidade muito alta deve levantar erro
    with pytest.raises(ValueError, match="deve estar entre 128 e 3072"):
        GeminiEmbeddingConfig(
            api_key="test-key",
            output_dimensionality=4000  # Maior que 3072
        )


def test_gemini_llm_config_initialization(gemini_llm_config):
    """
    Testa se a configuração do LLM é inicializada corretamente.
    """
    assert gemini_llm_config.model_name == "gemini-2.5-flash"
    assert gemini_llm_config.api_key == "test-api-key"
    assert gemini_llm_config.temperature == 0.7
    assert gemini_llm_config.max_tokens == 2000


def test_gemini_embedding_config_initialization(gemini_embedding_config):
    """
    Testa se a configuração de embedding é inicializada corretamente.
    """
    assert gemini_embedding_config.model_name == "gemini-embedding-001"
    assert gemini_embedding_config.api_key == "test-api-key"
    assert gemini_embedding_config.output_dimensionality == 768
    assert gemini_embedding_config.task_type == "RETRIEVAL_DOCUMENT"


def test_validate_gemini_api_key_with_valid_key():
    """
    Testa validação de API key válida.
    """
    assert validate_gemini_api_key("valid-key-123") is True


def test_validate_gemini_api_key_with_none():
    """
    Testa validação de API key None.
    """
    # Mock da variável de ambiente para garantir que não existe
    with patch.dict(os.environ, {}, clear=True):
        assert validate_gemini_api_key(None) is False


def test_validate_gemini_api_key_with_empty_string():
    """
    Testa validação de API key vazia.
    """
    assert validate_gemini_api_key("") is False
