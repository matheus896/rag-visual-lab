"""
RAG Agente - Laboratório Visual de RAG
======================================

Demonstração de RAG com roteamento de datasets inteligente usando CrewAI.

Este módulo implementa um chatbot que usa um agente de IA para decidir
qual base vetorial é mais apropriada para responder uma pergunta. O agente
analisa a query do usuário e roteia para o dataset correto (ex: direito_constitucional
ou synthetic_dataset_papers), depois executa o pipeline RAG completo naquele dataset.

Funcionalidades:
- Chat interativo com roteamento automático de datasets
- Agente CrewAI que analisa e decide dataset apropriado
- Visualização do raciocínio do agente (transparência didática)
- Pipeline RAG completo (Retrieval + Augmentation + Generation)
- Histórico de conversa com visualização dos logs do agente

Arquitetura (Task 4):
1. AgenticRAGProvider → Roteia query para dataset correto
2. RetrieverProvider → Busca chunks no dataset selecionado
3. AugmentationProvider → Enriquece prompt com chunks
4. GeminiProvider → Gera resposta final
"""

import streamlit as st
import os
import sys
from typing import Dict, Optional

# Adiciona o diretório raiz ao path para imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.agentic_rag_provider import AgenticRAGProvider
from services.retriever_provider import RetrieverProvider
from services.augmentation_provider import AugmentationProvider
from services.gemini_provider import (
    GeminiConfig,
    get_gemini_llm_function,
    validate_gemini_api_key
)


# ==================== CONFIGURAÇÃO DA PÁGINA ====================

st.set_page_config(
    page_title="RAG Agente | Lab Visual",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 Laboratório de RAG Agente")
st.markdown("""
Converse com um assistente **inteligente** que roteia automaticamente sua pergunta 
para a base de conhecimento mais apropriada. O agente analisa sua query e decide 
qual dataset usar, tornando o processo **transparente e didático**.
""")

st.divider()


# ==================== INICIALIZAÇÃO DO STATE ====================

def initialize_session_state():
    """Inicializa as variáveis de estado da sessão."""
    if "rag_agentic_messages" not in st.session_state:
        st.session_state.rag_agentic_messages = []
    
    if "rag_agentic_logs" not in st.session_state:
        st.session_state.rag_agentic_logs = ""
    
    if "rag_agentic_provider" not in st.session_state:
        st.session_state.rag_agentic_provider = AgenticRAGProvider()
    
    # Configurações padrão do ChromaDB
    if "agentic_chroma_db_path" not in st.session_state:
        st.session_state.agentic_chroma_db_path = "./chroma_db"
    
    if "agentic_chroma_n_results" not in st.session_state:
        st.session_state.agentic_chroma_n_results = 10

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
    
    st.session_state.agentic_chroma_db_path = st.text_input(
        "Caminho do Banco",
        value=st.session_state.agentic_chroma_db_path,
        help="Caminho para o diretório do ChromaDB persistente"
    )
    
    st.session_state.agentic_chroma_n_results = st.slider(
        "Número de Chunks",
        min_value=1,
        max_value=20,
        value=st.session_state.agentic_chroma_n_results,
        help="Quantidade de chunks a recuperar por consulta"
    )
    
    st.divider()
    
    # Botão para limpar histórico
    if st.button("🗑️ Limpar Conversa", use_container_width=True):
        st.session_state.rag_agentic_messages = []
        st.session_state.rag_agentic_logs = ""
        st.success("Conversa limpa com sucesso!")
        st.rerun()
    
    st.divider()
    
    # Informações sobre o Agente
    st.subheader("ℹ️ Sobre o Agente")
    st.info("""
    **CrewAI**: Framework para construir agentes IA colaborativos.
    
    **Roteamento Automático**: O agente analisa sua query e escolhe:
    - `synthetic_dataset_papers` para perguntas sobre datasets sintéticos
    - `direito_constitucional` para perguntas sobre direito
    
    **Transparência**: Você vê o raciocínio do agente ao lado.
    """)


# ==================== FUNÇÃO AUXILIAR - PIPELINE AGENTE RAG ====================

def build_agentic_rag_pipeline(query: str, api_key: str) -> tuple[str, Dict, str]:
    """
    Pipeline Agentic RAG que roteia queries para datasets apropriados.
    
    Fluxo:
    1. AgenticRAGProvider → Roteia query para dataset correto via CrewAI
    2. RetrieverProvider → Busca chunks no dataset selecionado
    3. AugmentationProvider → Enriquece prompt com chunks
    4. Generation → LLM gera resposta final
    
    Args:
        query: Pergunta do usuário
        api_key: Chave da API Gemini
        
    Returns:
        Uma tupla contendo (resposta, routing_result, agent_reasoning)
    """
    agent_reasoning = ""
    routing_result = {}
    
    try:
        # ===== ETAPA 1: AGENTIC ROUTING =====
        print(f"\n🤖 [AGENTIC ROUTING] Analisando query para rotear dataset...")
        
        # Executar roteamento (os logs já vão para o terminal via TeeOutput no provider)
        routing_result = st.session_state.rag_agentic_provider.route_query(query)
        
        # Obter logs capturados do provider
        agent_reasoning = st.session_state.rag_agentic_provider.last_logs
        
        if not routing_result:
            return "❌ Erro: O agente não conseguiu rotear a query.", {}, agent_reasoning
        
        dataset_name = routing_result.get("dataset_name")
        locale = routing_result.get("locale")
        translated_query = routing_result.get("query", query)
        
        print(f"✅ [AGENTIC ROUTING] Dataset selecionado: {dataset_name}")
        print(f"   └─ Locale: {locale}")
        print(f"   └─ Query traduzida: {translated_query}")
        
        # ===== ETAPA 2: RETRIEVAL =====
        print(f"\n🔎 [RETRIEVAL] Buscando chunks em '{dataset_name}'...")
        
        retriever = RetrieverProvider(
            db_path=st.session_state.agentic_chroma_db_path,
            collection_name=dataset_name
        )
        
        chunks = retriever.search(
            query_text=translated_query,
            n_results=st.session_state.agentic_chroma_n_results
        )
        
        if not chunks:
            st.warning(f"""
            ⚠️ Nenhum chunk recuperado de '{dataset_name}'. 
            Verifique se a coleção existe e contém documentos.
            """)
            chunks = [f"Informação padrão sobre {dataset_name}"]
        
        print(f"✅ [RETRIEVAL] Encontrados {len(chunks)} chunks")
        
        # ===== ETAPA 3: AUGMENTATION =====
        print(f"\n📝 [AUGMENTATION] Enriquecendo prompt com chunks...")
        
        augmenter = AugmentationProvider(talk_id="agentic_session")
        prompt = augmenter.generate_prompt(query=translated_query, chunks=chunks)
        
        print(f"✅ [AUGMENTATION] Prompt gerado ({len(prompt)} caracteres)")
        
        # ===== ETAPA 4: GENERATION =====
        print(f"\n🤖 [GENERATION] Gerando resposta com Gemini...")
        
        llm_config = GeminiConfig(api_key=api_key, temperature=0.7, max_tokens=2000)
        llm_function = get_gemini_llm_function(llm_config)
        
        response = llm_function(prompt)
        
        print(f"✅ [GENERATION] Resposta gerada ({len(response)} caracteres)")
        
        return response, routing_result, agent_reasoning
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Erro no pipeline: {error_msg}")
        
        if "does not exist" in error_msg:
            return f"""
❌ **Erro de Configuração do ChromaDB**

A coleção não foi encontrada no caminho configurado.

**Possíveis soluções**:
1. Verifique se o caminho está correto
2. Verifique se a coleção foi criada
3. Tente usar o caminho absoluto

**Dataset esperado**: {routing_result.get('dataset_name', 'desconhecido')}
**Caminho configurado**: {st.session_state.agentic_chroma_db_path}
            """, routing_result, agent_reasoning
        else:
            return f"❌ Erro: {error_msg}", routing_result, agent_reasoning


# ==================== INTERFACE DE CHAT ====================

# Exibe o histórico de mensagens
if st.session_state.rag_agentic_messages:
    for message in st.session_state.rag_agentic_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
else:
    st.info("👋 Olá! Sou um assistente agente de RAG. Qual é sua dúvida?")

# Input de chat
if user_query := st.chat_input("Digite sua pergunta..."):
    # Verifica se a API key é válida
    if not api_key_valid:
        st.error("⚠️ Por favor, configure uma API Key válida na barra lateral.")
    else:
        # Exibe a mensagem do usuário
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Adiciona ao histórico
        st.session_state.rag_agentic_messages.append({
            "role": "user",
            "content": user_query
        })
        
        # Gera a resposta usando o pipeline agentic RAG
        with st.chat_message("assistant"):
            with st.spinner("Analisando e roteando sua pergunta..."):
                response, routing_info, agent_logs = build_agentic_rag_pipeline(
                    query=user_query,
                    api_key=api_key
                )
                
                # Exibe a resposta principal
                st.markdown(response)
                
                # Exibe informações do roteamento
                if routing_info:
                    col1, col2 = st.columns(2)
                    with col1:
                        with st.expander("📊 Informações do Roteamento"):
                            st.json(routing_info)
                    
                    with col2:
                        with st.expander("🧠 Raciocínio do Agente"):
                            st.code(agent_logs, language="text")
        
        # Adiciona resposta ao histórico
        st.session_state.rag_agentic_messages.append({
            "role": "assistant",
            "content": response
        })
        
        # Rerun para atualizar a interface
        st.rerun()


# ==================== FOOTER ====================

st.divider()

with st.expander("🔍 Ver Informações Técnicas - Pipeline Agentic RAG"):
    st.markdown("""
    ### Arquitetura Agentic RAG
    
    O diferencial desta abordagem é o **roteamento inteligente**:
    
    #### Componentes
    
    1. **AgenticRAGProvider**: Agent CrewAI que roteia queries
       - Analisa a intenção do usuário
       - Consulta lista de datasets disponíveis
       - Seleciona o mais apropriado
       - **Retorna**: `{dataset_name, locale, translated_query}`
    
    2. **RetrieverProvider**: Busca no dataset selecionado
       - Usa embeddings semânticos
       - Recupera chunks mais relevantes
    
    3. **AugmentationProvider**: Enriquecimento de prompt
       - Combina chunks + contexto
       - Gera prompt otimizado para LLM
    
    4. **GeminiProvider**: Geração de resposta
       - LLM processa prompt enriquecido
       - Retorna resposta contextualizada
    
    #### Fluxo Completo
    
    ```
    Pergunta do Usuário
         ↓
    🤖 AgenticRAGProvider.route_query()
         ├─ Analisa intent da query
         ├─ Consulta datasets disponíveis
         └─ Seleciona dataset correto
         ↓
    🔎 RetrieverProvider.search()
         ├─ Busca em dataset selecionado
         └─ Retorna chunks relevantes
         ↓
    📝 AugmentationProvider.generate_prompt()
         ├─ Combina chunks + contexto
         └─ Cria prompt otimizado
         ↓
    🤖 GeminiProvider.generate()
         ├─ LLM processa prompt
         └─ Retorna resposta final
         ↓
    Resposta ao Usuário
    ```
    
    #### Vantagens do Roteamento Agente
    
    - ✅ **Inteligência**: Agent aprende padrões de queries
    - ✅ **Escalabilidade**: Fácil adicionar novos datasets
    - ✅ **Precisão**: Dataset certo = respostas mais acuradas
    - ✅ **Transparência**: Raciocínio do agente é visível
    - ✅ **Didático**: Perfeito para ensinar RAG avançado
    
    #### Datasets Disponíveis
    
    | Dataset | Locale | Descrição |
    |---------|--------|-----------|
    | `synthetic_dataset_papers` | `en` | Sobre datasets sintéticos e IA |
    | `direito_constitucional` | `pt-br` | Direito, leis e jurisprudência |
    
    #### Exemplo de Uso
    
    **Query**: "O que é abandono afetivo no direito constitucional?"
    
    **Roteamento do Agent**:
    ```json
    {
      "dataset_name": "direito_constitucional",
      "locale": "pt-br",
      "query": "O que é direito constitucional fala do abandono afetivo?"
    }
    ```
    
    **Pipeline**: Busca em `direito_constitucional` → Retorna chunks jurídicos → LLM gera resposta especializada
    """)