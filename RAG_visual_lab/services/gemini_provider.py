"""
Provedor Gemini
===============

Integração com Google Gemini para LLM e Embeddings.

Baseado na documentação oficial do Gemini API:
- https://github.com/google-gemini/cookbook/blob/main/quickstarts/Embeddings.ipynb
- https://ai.google.dev/gemini-api/docs/embeddings

Modelos Suportados:
- LLM: gemini-2.5-flash, gemini-2.0-flash, gemini-1.5-pro
- Embeddings: gemini-embedding-001 (dimensões: 128-3072, recomendado: 768, 1536, 3072)
"""

import os
from typing import List, Optional, Callable
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


class GeminiConfig:
    """Configuração para Gemini LLM."""
    
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000
    ):
        """
        Inicializa configuração do Gemini LLM.
        
        Args:
            model_name: Nome do modelo Gemini
            api_key: Chave de API do Gemini
            temperature: Temperatura para geração (0.0-1.0)
            max_tokens: Máximo de tokens na resposta (aumentado para 4000 para evitar truncamento)
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens


class GeminiEmbeddingConfig:
    """Configuração para Gemini Embeddings."""
    
    def __init__(
        self,
        model_name: str = "gemini-embedding-001",
        api_key: Optional[str] = None,
        output_dimensionality: int = 768,
        task_type: str = "RETRIEVAL_DOCUMENT"
    ):
        """
        Inicializa configuração de embeddings Gemini.
        
        Args:
            model_name: Modelo de embedding
            api_key: Chave de API do Gemini
            output_dimensionality: Dimensão dos vetores (128-3072, recomendado: 768, 1536, 3072)
            task_type: Tipo de tarefa (RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, SEMANTIC_SIMILARITY, etc.)
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.output_dimensionality = output_dimensionality
        self.task_type = task_type
        
        # Valida dimensionalidade
        if not (128 <= output_dimensionality <= 3072):
            raise ValueError(f"output_dimensionality deve estar entre 128 e 3072, recebido: {output_dimensionality}")


@st.cache_resource
def get_gemini_llm_function(_config: GeminiConfig) -> Callable:
    """
    Retorna função de LLM configurada para Gemini.
    
    Args:
        _config: Configuração do Gemini (prefixo _ para evitar hashing)
        
    Returns:
        Função callable para completions
    """
    try:
        from google import genai
        from google.genai import types
        
        client = genai.Client(api_key=_config.api_key)
        
        def llm_function(
            prompt: str,
            system_prompt: Optional[str] = None,
            max_retries: int = 2,
            **kwargs
        ) -> str:
            """
            Função de completion para Gemini com retry automático.
            
            Args:
                prompt: Prompt do usuário
                system_prompt: Instrução de sistema (opcional)
                max_retries: Máximo de tentativas se resposta for muito curta (padrão: 2)
                **kwargs: Argumentos adicionais
                
            Returns:
                Texto gerado pelo modelo
            """
            # 🤖 Logging: Início da geração
            print(f"\n🤖 [GENERATION] Chamando {_config.model_name}...")
            
            response_text = ""  # Inicializa variável
            
            for attempt in range(max_retries):
                # Monta o conteúdo
                parts = []
                
                # Gemini não tem system_prompt separado, então concatenamos
                if system_prompt:
                    full_prompt = f"{system_prompt}\n\n{prompt}"
                else:
                    full_prompt = prompt
                
                parts.append(types.Part(text=full_prompt))
                
                # Configura geração
                generation_config = types.GenerateContentConfig(
                    temperature=_config.temperature,
                    max_output_tokens=_config.max_tokens,
                    **kwargs
                )
                
                # Gera conteúdo
                response = client.models.generate_content(
                    model=_config.model_name,
                    contents=types.Content(parts=parts),
                    config=generation_config
                )
                
                # Extrai texto e informações de parada
                response_text = response.text if response.text else ""
                stop_reason = response.candidates[0].finish_reason if response.candidates else "UNKNOWN"
                
                # 📊 Logging diagnóstico
                print(f"   └─ Stop reason: {stop_reason} | Tentativa: {attempt + 1}/{max_retries}")
                print(f"✅ [GENERATION] Resposta gerada ({len(response_text)} caracteres)")
                
                # ✅ Se a resposta parece completa, retorna
                if len(response_text) > 100 or attempt == max_retries - 1:
                    return response_text if response_text else ""
                
                # ⚠️ Se resposta é muito curta (<100 chars) e não é última tentativa, retry
                if len(response_text) < 100 and attempt < max_retries - 1:
                    print(f"   ⚠️  Resposta anormalmente curta ({len(response_text)} chars). Tentando novamente...")
                    continue
            
            return response_text if response_text else ""
        
        return llm_function
        
    except ImportError:
        st.error("Biblioteca 'google-genai' não instalada. Execute: pip install google-genai")
        raise


@st.cache_resource
def get_gemini_embedding_function(_config: GeminiEmbeddingConfig) -> Callable:
    """
    Retorna função de embeddings configurada para Gemini.
    
    Args:
        _config: Configuração de embeddings Gemini (prefixo _ para evitar hashing)
        
    Returns:
        Função callable para embeddings
    """
    try:
        from google import genai
        from google.genai import types
        
        client = genai.Client(api_key=_config.api_key)
        
        def embedding_function(texts: List[str]) -> List[List[float]]:
            """
            Função de embedding para Gemini com batching automático.
            
            Args:
                texts: Lista de textos para gerar embeddings
                
            Returns:
                Lista de vetores de embedding
            """
            # Configura embeddings
            embed_config = types.EmbedContentConfig(
                task_type=_config.task_type,
                output_dimensionality=_config.output_dimensionality
            )
            
            # Gemini tem limite de 100 textos por batch
            BATCH_SIZE = 100
            all_embeddings = []
            total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
            
            # 📊 Logging: Início do processamento
            print(f"\n📊 [EMBEDDINGS] Processando {len(texts)} textos em {total_batches} batch(es)...")
            
            # Processa em batches de 100 textos
            for batch_idx, i in enumerate(range(0, len(texts), BATCH_SIZE), 1):
                batch = texts[i:i + BATCH_SIZE]
                
                print(f"   └─ Batch {batch_idx}/{total_batches}: {len(batch)} textos")
                
                # Gera embeddings para o batch
                result = client.models.embed_content(
                    model=_config.model_name,
                    contents=batch,  # type: ignore - Gemini aceita List[str]
                    config=embed_config
                )
                
                # Extrai valores dos embeddings do batch
                if result.embeddings is None:
                    all_embeddings.extend([[] for _ in batch])
                else:
                    batch_embeddings = [
                        embedding.values if embedding.values is not None else []
                        for embedding in result.embeddings
                    ]
                    all_embeddings.extend(batch_embeddings)
            
            print(f"✅ [EMBEDDINGS] Total de {len(all_embeddings)} embeddings gerados com sucesso!")
            
            return all_embeddings
        
        return embedding_function
        
    except ImportError:
        st.error("Biblioteca 'google-genai' não instalada. Execute: pip install google-genai")
        raise


def validate_gemini_api_key(api_key: Optional[str] = None) -> bool:
    """
    Valida se a chave de API do Gemini está disponível.
    
    Args:
        api_key: Chave de API a validar
        
    Returns:
        True se a chave é válida, False caso contrário
    """
    key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY")
    return key is not None and len(key) > 0
