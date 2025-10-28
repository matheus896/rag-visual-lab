"""
RAG com Memória - Laboratório Visual de RAG
============================================

Demonstração de RAG com memória conversacional persistente usando Redis.

Este módulo implementa um chatbot que mantém o contexto da conversa,
permitindo interações mais naturais e contextualizadas. A memória é
armazenada em Redis, garantindo persistência entre sessões.

Funcionalidades:
- Chat interativo com histórico persistente
- Memória conversacional com Redis
- Integração com LLM (Gemini) para respostas contextualizadas
- Botão para limpar histórico
"""

import streamlit as st
import os
import sys
from typing import List, Dict, Any, Optional

# Adiciona o diretório raiz ao path para imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.memory_provider import MemoryProvider
from services.augmentation_provider import AugmentationProvider
from services.retriever_provider import RetrieverProvider
from services.gemini_provider import (
    GeminiConfig,
    get_gemini_llm_function,
    validate_gemini_api_key
)


# ==================== CONFIGURAÇÃO DA PÁGINA ====================

st.set_page_config(
    page_title="RAG com Memória | Lab Visual",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Laboratório de RAG com Memória")
st.markdown("""
Converse com um assistente que **lembra do contexto** da sua conversa.
A memória é armazenada em Redis, permitindo que você retome conversas mesmo após fechar o navegador.
""")

st.divider()


# ==================== INICIALIZAÇÃO DO STATE ====================

def initialize_session_state():
    """Inicializa as variáveis de estado da sessão."""
    if "rag_memoria_talk_id" not in st.session_state:
        # Gera um ID único para esta sessão de conversa
        import uuid
        st.session_state.rag_memoria_talk_id = str(uuid.uuid4())
    
    if "rag_memoria_provider" not in st.session_state:
        # Inicializa o provedor de memória
        st.session_state.rag_memoria_provider = MemoryProvider(
            talk_id=st.session_state.rag_memoria_talk_id
        )
    
    # Configurações padrão do ChromaDB
    if "chroma_db_path" not in st.session_state:
        # Caminho relativo: ../chroma_db (um nível acima de RAG_visual_lab)
        st.session_state.chroma_db_path = "./chroma_db"
    
    if "chroma_collection_name" not in st.session_state:
        st.session_state.chroma_collection_name = "synthetic_dataset_papers"
    
    if "chroma_n_results" not in st.session_state:
        st.session_state.chroma_n_results = 10
    
    # Lista de coleções disponíveis no ChromaDB
    if "available_collections" not in st.session_state:
        st.session_state.available_collections = [
            "synthetic_dataset_papers",
            "direito_constitucional"
        ]

initialize_session_state()


# ==================== SIDEBAR - CONFIGURAÇÕES ====================

with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Validação da API Key do Gemini
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=os.getenv("GEMINI_API_KEY", ""),
        help="Obtenha sua chave em https://ai.google.dev/"
    )
    
    api_key_valid = validate_gemini_api_key(api_key)
    
    if api_key_valid:
        st.success("✅ API Key válida")
    else:
        st.error("❌ API Key inválida ou não fornecida")
    
    st.divider()
    
    # Configurações do ChromaDB
    st.subheader("🗄️ Configuração do ChromaDB")
    
    st.session_state.chroma_db_path = st.text_input(
        "Caminho do Banco",
        value=st.session_state.chroma_db_path,
        help="Caminho para o diretório do ChromaDB persistente"
    )
    
    st.session_state.chroma_collection_name = st.selectbox(
        "Nome da Coleção",
        options=st.session_state.available_collections,
        index=st.session_state.available_collections.index(st.session_state.chroma_collection_name),
        help="Selecione a coleção de documentos no ChromaDB"
    )
    
    st.session_state.chroma_n_results = st.slider(
        "Número de Chunks",
        min_value=1,
        max_value=20,
        value=st.session_state.chroma_n_results,
        help="Quantidade de chunks a recuperar por consulta"
    )
    
    st.divider()
    
    # Informações da sessão
    st.subheader("📊 Informações da Sessão")
    st.code(f"Talk ID: {st.session_state.rag_memoria_talk_id[:8]}...")
    
    # Botão para limpar histórico
    if st.button("🗑️ Limpar Histórico", use_container_width=True):
        st.session_state.rag_memoria_provider.delete_conversation()
        st.success("Histórico limpo com sucesso!")
        st.rerun()
    
    st.divider()
    
    # Informações sobre Redis
    st.subheader("ℹ️ Sobre a Memória")
    st.info("""
    **Redis**: Banco de dados em memória usado para armazenar o histórico.
    
    **Expiração**: Conversas expiram após 24 horas de inatividade.
    
    **Persistência**: O histórico sobrevive ao reinício da aplicação.
    """)


# ==================== FUNÇÃO AUXILIAR - GERAÇÃO DE RESPOSTA ====================

def build_rag_with_memory_pipeline(query: str, talk_id: str, api_key: str) -> tuple[str, str]:
    """
    Pipeline RAG + Memória que combina recuperação de documentos com contexto conversacional.
    
    Este é o coração da integração Task 3.3, replicando o fluxo de:
    code-sandeco-rag-memory.txt (main.py lines 366-374)
    
    Fluxo:
    1. Retriever → Busca chunks relevantes no ChromaDB
    2. AugmentationProvider → Combina chunks + memória Redis em prompt enriquecido
    3. Generation → LLM gera resposta baseada no prompt aumentado
    4. Persist → Salva interação na memória Redis
    
    Args:
        query: Pergunta do usuário
        talk_id: Identificador da conversa
        api_key: Chave da API Gemini
        
    Returns:
        Uma tupla contendo (resposta_do_llm, prompt_completo)
    """
    prompt = ""
    try:
        # ===== ETAPA 1: RETRIEVAL (R do RAG) =====
        # Inicializa o RetrieverProvider com configurações da sessão
        retriever = RetrieverProvider(
            db_path=st.session_state.chroma_db_path,
            collection_name=st.session_state.chroma_collection_name
        )
        
        # Busca chunks relevantes no ChromaDB
        chunks = retriever.search(
            query_text=query,
            n_results=st.session_state.chroma_n_results
        )
        
        # Fallback para chunks de exemplo se ChromaDB estiver vazio ou houver erro
        if not chunks:
            st.warning("""
            ⚠️ Nenhum chunk recuperado do ChromaDB. 
            Verifique se a coleção existe e contém documentos.
            Usando chunks de exemplo para demonstração.
            """)
            chunks = [
                "RAG (Retrieval-Augmented Generation) é uma técnica que combina recuperação de informação com geração de linguagem natural.",
                "O componente de memória permite que o sistema mantenha contexto conversacional entre interações.",
                "Redis é usado para persistir o histórico de conversas com expiração de 24 horas."
            ]
        
        # ===== ETAPA 2: AUGMENTATION (A do RAG) =====
        print(f"\n📝 [AUGMENTATION] Aumentando prompt com chunks + histórico...")
        
        # Inicializa o orquestrador que combina chunks + memória
        augmenter = AugmentationProvider(talk_id=talk_id)
        
        # Gera prompt enriquecido com chunks + histórico
        prompt = augmenter.generate_prompt(query=query, chunks=chunks)
        
        print(f"✅ [AUGMENTATION] Prompt enriquecido gerado ({len(prompt)} caracteres)")
        
        # ===== ETAPA 3: GENERATION (G do RAG) =====
        # Configura e chama o LLM
        llm_config = GeminiConfig(api_key=api_key, temperature=0.7, max_tokens=2000)
        llm_function = get_gemini_llm_function(llm_config)
        
        response = llm_function(prompt)
        
        # ===== ETAPA 4: PERSIST =====
        print(f"\n💾 [PERSISTENCE] Salvando interação no Redis...")
        
        # Salva a interação (query + response) na memória
        augmenter.add_response_to_memory(response)
        
        print(f"✅ [PERSISTENCE] Conversa salva com sucesso!")
        
        return response, prompt
    
    except Exception as e:
        error_msg = str(e)
        
        
        if "does not exist" in error_msg:
            st.error(f"""
            ❌ **Erro de Configuração do ChromaDB**
            
            A coleção **'{st.session_state.chroma_collection_name}'** não foi encontrada em **'{st.session_state.chroma_db_path}'**.
            
            **Possíveis soluções**:
            1. Verifique se o caminho está correto (deve apontar para o diretório do ChromaDB)
            2. Verifique se a coleção foi criada (use `semantic_encoder.py` para criar)
            3. Tente usar o caminho absoluto ao invés de relativo
            
            **Caminho atual configurado**: `{st.session_state.chroma_db_path}`
            """)
        else:
            st.error(f"❌ Erro ao gerar resposta: {error_msg}")
        
        return f"Erro: {error_msg}", prompt


def generate_response_with_context(query: str, history: List[Dict[str, Any]]) -> str:
    """
    DEPRECATED: Esta função foi substituída por build_rag_with_memory_pipeline().
    
    Mantida temporariamente para compatibilidade com testes existentes.
    Será removida após migração completa para o novo pipeline.
    
    Args:
        query: Pergunta do usuário
        history: Histórico da conversa
        
    Returns:
        Resposta gerada pelo LLM
    """
    # Constrói o prompt com contexto do histórico
    context_messages = []
    
    # Adiciona as últimas 5 mensagens do histórico (já estão em ordem reversa)
    for msg in history[:5]:
        context_messages.append(f"{msg['role']}: {msg['content']}")
    
    # Monta o prompt final
    if context_messages:
        context_text = "\n".join(reversed(context_messages))  # Inverte para ordem cronológica
        prompt = f"""Você é um assistente útil. Com base no histórico da conversa abaixo, responda à pergunta do usuário.

Histórico da Conversa:
{context_text}

Pergunta Atual: {query}

Responda de forma natural e contextualizada, considerando o histórico acima:"""
    else:
        prompt = f"""Você é um assistente útil. Responda à seguinte pergunta:

{query}"""
    
    # Gera resposta usando Gemini
    try:
        llm_config = GeminiConfig(api_key=api_key, temperature=0.7)
        llm_function = get_gemini_llm_function(llm_config)
        response = llm_function(prompt)
        return response
    except Exception as e:
        return f"Erro ao gerar resposta: {str(e)}"


# ==================== INTERFACE DE CHAT ====================

# Recupera o histórico da conversa
conversation_history = st.session_state.rag_memoria_provider.get_conversation()

# Exibe o histórico de mensagens
if conversation_history:
    for message in reversed(conversation_history):  # Reverte para mostrar ordem cronológica
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
else:
    st.info("👋 Olá! Sou um assistente com memória. Faça uma pergunta para começarmos!")

# Input de chat
if user_query := st.chat_input("Digite sua mensagem..."):
    # Verifica se a API key é válida
    if not api_key_valid:
        st.error("⚠️ Por favor, configure uma API Key válida na barra lateral.")
    else:
        # Exibe a mensagem do usuário
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Gera a resposta do assistente usando o pipeline RAG + Memória
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                response, full_prompt = build_rag_with_memory_pipeline(
                    query=user_query,
                    talk_id=st.session_state.rag_memoria_talk_id,
                    api_key=api_key,
                )
                st.markdown(response)

                # Adiciona o expander para visualização didática do prompt
                if full_prompt:
                    with st.expander("🔍 Ver o prompt completo enviado ao LLM"):
                        st.code(full_prompt, language="markdown")
        
        # Rerun para atualizar a interface com o histórico que a pipeline salvou
        st.rerun()


# ==================== FOOTER ====================

st.divider()

with st.expander("🔍 Ver Informações Técnicas"):
    st.markdown("""
    ### Arquitetura RAG com Memória Conversacional
    
    1. **RetrieverProvider**: Busca chunks relevantes no ChromaDB usando embeddings semânticos
    2. **AugmentationProvider**: Orquestra a combinação de chunks + memória Redis
    3. **MemoryProvider**: Gerencia o histórico conversacional persistente
    4. **Gemini LLM**: Modelo de linguagem para gerar respostas contextualizadas
    
    ### Fluxo de Dados Completo
    
    ```
    Usuário digita query →
    ├─ [1] RetrieverProvider busca chunks no ChromaDB
    ├─ [2] MemoryProvider recupera histórico do Redis
    ├─ [3] AugmentationProvider combina chunks + histórico
    ├─ [4] LLM gera resposta com contexto enriquecido
    └─ [5] MemoryProvider salva interação no Redis
    ```
    
    ### Vantagens desta Abordagem
    
    - ✅ **RAG**: Respostas baseadas em documentos reais
    - ✅ **Memória**: Conversas contextualizadas e naturais
    - ✅ **Persistência**: Histórico sobrevive ao reinício da app
    - ✅ **Escalabilidade**: Redis + ChromaDB suportam milhões de interações
    - ✅ **Busca Semântica**: Embeddings capturam significado, não apenas palavras-chave
    """)

