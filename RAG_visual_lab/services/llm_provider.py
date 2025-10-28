"""
Provedor de LLM
===============

Abstração para integração com diferentes provedores de LLM e modelos de embedding.

Este módulo desacopla a lógica da aplicação dos detalhes de implementação
dos provedores de IA, seguindo o princípio de Inversão de Dependência.

Provedores Suportados:
- OpenAI (GPT-4, GPT-3.5, Embeddings)
- Ollama (modelos locais)
- HuggingFace (modelos open-source)
"""

import os
from typing import Optional, List, Dict, Any, Callable
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


class LLMConfig:
    """Configuração para o provedor de LLM."""
    
    def __init__(
        self,
        provider: str = "openai",
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        """
        Inicializa a configuração do LLM.
        
        Args:
            provider: Nome do provedor (openai, ollama, huggingface)
            model_name: Nome do modelo a usar
            api_key: Chave de API (se necessário)
            base_url: URL base para API (se necessário)
            temperature: Temperatura para geração
            max_tokens: Número máximo de tokens na resposta
        """
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens


class EmbeddingConfig:
    """Configuração para o modelo de embeddings."""
    
    def __init__(
        self,
        provider: str = "openai",
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        embedding_dim: int = 1536
    ):
        """
        Inicializa a configuração de embeddings.
        
        Args:
            provider: Nome do provedor
            model_name: Nome do modelo de embedding
            api_key: Chave de API
            embedding_dim: Dimensão dos vetores de embedding
        """
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.embedding_dim = embedding_dim


@st.cache_resource
def get_llm_function(config: LLMConfig) -> Callable:
    """
    Retorna a função de LLM configurada.
    
    Args:
        config: Configuração do LLM
        
    Returns:
        Função callable para completions
    """
    if config.provider == "openai":
        return _get_openai_llm(config)
    elif config.provider == "ollama":
        return _get_ollama_llm(config)
    else:
        raise ValueError(f"Provedor não suportado: {config.provider}")


@st.cache_resource
def get_embedding_function(config: EmbeddingConfig) -> Callable:
    """
    Retorna a função de embedding configurada.
    
    Args:
        config: Configuração do embedding
        
    Returns:
        Função callable para embeddings
    """
    if config.provider == "openai":
        return _get_openai_embeddings(config)
    else:
        raise ValueError(f"Provedor de embedding não suportado: {config.provider}")


def _get_openai_llm(config: LLMConfig) -> Callable:
    """
    Cria função de LLM para OpenAI.
    
    Nota: Implementação simplificada. Para produção, considere
    adicionar retry logic, rate limiting, etc.
    """
    try:
        from openai import OpenAI
        
        client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        
        def llm_function(
            prompt: str,
            system_prompt: Optional[str] = None,
            **kwargs
        ) -> str:
            """Função de completion para OpenAI."""
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            response = client.chat.completions.create(
                model=config.model_name,
                messages=messages,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                **kwargs
            )
            
            return response.choices[0].message.content
        
        return llm_function
        
    except ImportError:
        st.error("Biblioteca 'openai' não instalada. Execute: pip install openai")
        raise


def _get_ollama_llm(config: LLMConfig) -> Callable:
    """
    Cria função de LLM para Ollama.
    
    Placeholder para implementação futura.
    """
    def llm_function(prompt: str, **kwargs) -> str:
        raise NotImplementedError("Suporte Ollama em desenvolvimento")
    
    return llm_function


def _get_openai_embeddings(config: EmbeddingConfig) -> Callable:
    """
    Cria função de embeddings para OpenAI.
    """
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=config.api_key)
        
        def embedding_function(texts: List[str]) -> List[List[float]]:
            """Função de embedding para OpenAI."""
            response = client.embeddings.create(
                model=config.model_name,
                input=texts
            )
            
            return [item.embedding for item in response.data]
        
        return embedding_function
        
    except ImportError:
        st.error("Biblioteca 'openai' não instalada. Execute: pip install openai")
        raise


def validate_api_key(provider: str, api_key: Optional[str] = None) -> bool:
    """
    Valida se a chave de API está disponível.
    
    Args:
        provider: Nome do provedor
        api_key: Chave de API a validar
        
    Returns:
        True se a chave é válida, False caso contrário
    """
    if provider == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY")
        return key is not None and len(key) > 0
    
    return True  # Outros provedores podem não precisar de API key
