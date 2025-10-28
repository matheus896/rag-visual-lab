"""
RAG Clássico - Laboratório Visual de RAG
=========================================

Demonstração passo a passo do funcionamento do RAG tradicional.

Este módulo ilustra as 4 etapas fundamentais do RAG:
1. Upload e Chunking: Carregar documento e dividir em chunks
2. Embedding & Storage: Gerar embeddings e armazenar vetores
3. Query & Retrieval: Buscar chunks relevantes por similaridade
4. Generation: Gerar resposta usando LLM com contexto recuperado
"""

import streamlit as st
import numpy as np
from typing import List, Dict, Any, Optional
import os
import sys

# Adiciona o diretório raiz ao path para imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.ui_components import (
    render_document_uploader,
    display_source_chunks,
    render_parameter_controls,
    display_metrics_cards,
    display_info_box,
    display_embedding_visualization_guide,
    display_pca_explainer,
    display_variance_explainer
)
from utils.text_processing import (
    extract_text_from_file,
    chunk_text,
    count_tokens_approximate
)
from services.llm_provider import (
    LLMConfig,
    EmbeddingConfig,
    get_llm_function,
    get_embedding_function,
    validate_api_key
)
from services.gemini_provider import (
    GeminiConfig,
    GeminiEmbeddingConfig,
    get_gemini_llm_function,
    get_gemini_embedding_function,
    validate_gemini_api_key
)


# ==================== CONFIGURAÇÃO DA PÁGINA ====================

st.set_page_config(
    page_title="RAG Clássico | Lab Visual",
    page_icon="🔰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔰 RAG Clássico - Passo a Passo")
st.markdown("""
Explore as **4 etapas fundamentais** do RAG tradicional de forma interativa.
Cada tab representa uma fase do processo, permitindo que você entenda 
como o RAG combina recuperação de informação com geração de linguagem natural.
""")

st.divider()


# ==================== INICIALIZAÇÃO DO STATE ====================

def initialize_session_state():
    """Inicializa as variáveis de estado da sessão."""
    if "rag_classic_document_text" not in st.session_state:
        st.session_state.rag_classic_document_text = None
    
    if "rag_classic_chunks" not in st.session_state:
        st.session_state.rag_classic_chunks = []
    
    if "rag_classic_embeddings" not in st.session_state:
        st.session_state.rag_classic_embeddings = None
    
    if "rag_classic_query_results" not in st.session_state:
        st.session_state.rag_classic_query_results = None

initialize_session_state()


# ==================== FUNÇÕES AUXILIARES ====================

def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calcula similaridade cosseno entre dois vetores.
    
    Args:
        vec1: Primeiro vetor
        vec2: Segundo vetor
        
    Returns:
        Similaridade cosseno (0 a 1)
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def search_similar_chunks(
    query_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
    chunks: List[str],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Busca chunks mais similares à query usando similaridade cosseno.
    
    Args:
        query_embedding: Embedding da query
        chunk_embeddings: Embeddings dos chunks
        chunks: Lista de chunks de texto
        top_k: Número de resultados a retornar
        
    Returns:
        Lista de dicionários com chunks e scores
    """
    similarities = []
    
    for i, chunk_emb in enumerate(chunk_embeddings):
        similarity = calculate_cosine_similarity(query_embedding, chunk_emb)
        similarities.append({
            'index': i,
            'chunk': chunks[i],
            'score': float(similarity)
        })
    
    # Ordena por score decrescente
    similarities.sort(key=lambda x: x['score'], reverse=True)
    
    return similarities[:top_k]


def generate_rag_response(
    query: str,
    context_chunks: List[str],
    llm_function
) -> str:
    """
    Gera resposta usando LLM com contexto recuperado.
    
    Args:
        query: Pergunta do usuário
        context_chunks: Chunks recuperados
        llm_function: Função do LLM
        
    Returns:
        Resposta gerada
    """
    # Monta o contexto
    context = "\n\n---\n\n".join(context_chunks)
    
    # Template do prompt
    system_prompt = """Você é um assistente útil que responde perguntas baseado apenas no contexto fornecido.
Seja preciso, claro e objetivo. Se a informação não estiver no contexto, diga isso claramente."""
    
    user_prompt = f"""CONTEXTO:
{context}

PERGUNTA:
{query}

INSTRUÇÕES:
- Responda APENAS com base no contexto acima
- Se a informação não estiver no contexto, diga "Não tenho informação suficiente no contexto fornecido"
- Seja claro e objetivo
- Cite trechos relevantes do contexto quando apropriado

RESPOSTA:"""
    
    try:
        response = llm_function(user_prompt, system_prompt=system_prompt)
        return response
    except Exception as e:
        return f"Erro ao gerar resposta: {str(e)}"


# ==================== TABS PRINCIPAIS ====================

tab1, tab2, tab3, tab4 = st.tabs([
    "📄 1. Upload & Chunking",
    "🔢 2. Embedding & Storage",
    "🔍 3. Query & Retrieval",
    "💬 4. Generation"
])


# ==================== TAB 1: UPLOAD & CHUNKING ====================

with tab1:
    st.header("📄 Upload e Processamento de Documento")
    
    st.markdown("""
    **Objetivo desta etapa:** Carregar um documento e dividi-lo em chunks (pedaços) menores.
    
    **Por quê chunking?**
    - Modelos têm limite de contexto
    - Chunks menores = busca mais precisa
    - Permite processar documentos grandes
    """)
    
    # Parâmetros na sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Parâmetros de Chunking")
        chunk_size = st.slider(
            "Tamanho do Chunk (caracteres)",
            min_value=200,
            max_value=2000,
            value=800,
            step=100,
            help="Tamanho aproximado de cada chunk de texto"
        )
        
        overlap = st.slider(
            "Overlap entre Chunks",
            min_value=0,
            max_value=300,
            value=100,
            step=50,
            help="Caracteres compartilhados entre chunks consecutivos"
        )
    
    # Upload de arquivo
    uploaded_file = render_document_uploader(
        accepted_types=["txt", "md", "pdf"],
        help_text="Carregue um documento de texto para processar",
        key="uploaded_file"
    )
    
    # Para suportar o teste, verificamos o session_state também
    processed_file = uploaded_file or st.session_state.get("uploaded_file")

    if processed_file:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 📖 Documento Carregado")
            
            # Extrai texto
            with st.spinner("Extraindo texto do documento..."):
                document_text = extract_text_from_file(processed_file)
            
            if document_text:
                st.session_state.rag_classic_document_text = document_text
                
                # Mostra preview
                with st.expander("👁️ Preview do Documento", expanded=False):
                    st.text(document_text[:1000] + "..." if len(document_text) > 1000 else document_text)
        
        with col2:
            if st.session_state.rag_classic_document_text:
                # Estatísticas do documento
                doc_text = st.session_state.rag_classic_document_text
                st.markdown("#### 📊 Estatísticas do Documento")
                
                display_metrics_cards({
                    "Caracteres": f"{len(doc_text):,}",
                    "Palavras": f"{len(doc_text.split()):,}",
                    "Tokens (aprox.)": f"{count_tokens_approximate(doc_text):,}"
                })
        
        # Botão para processar chunks
        if st.session_state.rag_classic_document_text:
            st.markdown("---")
            
            if st.button("🔪 Processar e Dividir em Chunks", type="primary", use_container_width=True):
                with st.spinner("Dividindo documento em chunks..."):
                    chunks = chunk_text(
                        st.session_state.rag_classic_document_text,
                        chunk_size=chunk_size,
                        overlap=overlap
                    )
                    st.session_state.rag_classic_chunks = chunks
                
                st.success(f"✅ Documento dividido em {len(chunks)} chunks!")
    
    # Exibição dos chunks
    if st.session_state.rag_classic_chunks:
        st.markdown("---")
        st.markdown("#### 📚 Chunks Gerados")
        
        chunks = st.session_state.rag_classic_chunks
        
        # Métricas dos chunks
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de Chunks", len(chunks))
        with col2:
            avg_size = np.mean([len(c) for c in chunks])
            st.metric("Tamanho Médio", f"{avg_size:.0f} chars")
        with col3:
            total_tokens = sum([count_tokens_approximate(c) for c in chunks])
            st.metric("Total de Tokens", f"{total_tokens:,}")
        
        # Tabela de chunks
        st.markdown("##### Preview dos Chunks")
        
        # Evita erro quando há apenas 1 chunk
        max_preview = min(10, len(chunks))
        default_preview = min(5, len(chunks))
        
        if max_preview > 1:
            num_preview = st.slider(
                "Chunks para visualizar", 
                min_value=1, 
                max_value=max_preview, 
                value=default_preview
            )
        else:
            num_preview = 1
            st.info("📝 Mostrando o único chunk disponível.")
        
        for i, chunk in enumerate(chunks[:num_preview], 1):
            with st.expander(f"Chunk {i} ({len(chunk)} caracteres)"):
                st.text(chunk)
        
        if len(chunks) > num_preview:
            st.info(f"📝 Mostrando {num_preview} de {len(chunks)} chunks. Ajuste o slider para ver mais.")


# ==================== TAB 2: EMBEDDING & STORAGE ====================

with tab2:
    st.header("🔢 Embeddings e Armazenamento Vetorial")
    
    st.markdown("""
    **Objetivo desta etapa:** Converter chunks de texto em vetores numéricos (embeddings).
    
    **O que são embeddings?**
    - Representação numérica do significado do texto
    - Textos similares têm embeddings próximos
    - Permite busca semântica (por significado, não apenas palavras)
    """)
    
    if not st.session_state.rag_classic_chunks:
        display_info_box(
            "Atenção",
            "Você precisa primeiro processar um documento na **Tab 1** antes de gerar embeddings.",
            box_type="warning"
        )
    else:
        # Configuração de embeddings
        st.markdown("### ⚙️ Configuração do Modelo de Embeddings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            embedding_provider = st.selectbox(
                "Provedor",
                ["openai", "gemini"],
                help="Provedor de embeddings a utilizar"
            )
        
        with col2:
            if embedding_provider == "openai":
                embedding_model = st.selectbox(
                    "Modelo",
                    ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
                    help="Modelo de embedding a utilizar"
                )
            else:  # gemini
                embedding_model = st.selectbox(
                    "Modelo",
                    ["gemini-embedding-001"],
                    help="Modelo de embedding a utilizar"
                )
        
        # Dimensões do embedding
        if embedding_provider == "openai":
            embedding_dims = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536
            }
            embedding_dim = embedding_dims[embedding_model]
        else:  # gemini
            # Gemini permite dimensões flexíveis de 128 a 3072
            embedding_dim = st.selectbox(
                "Dimensão do Vetor",
                [768, 1536, 3072, 128, 256, 512],
                index=0,  # Default: 768
                help="Dimensão dos vetores de embedding (recomendado: 768, 1536 ou 3072)"
            )
        
        st.info(f"📏 Dimensão dos vetores: **{embedding_dim}**")
        
        # Validação de API key
        if embedding_provider == "openai":
            api_key_valid = validate_api_key("openai")
            error_msg = "⚠️ Chave da API OpenAI não encontrada! Configure a variável de ambiente OPENAI_API_KEY"
        else:  # gemini
            api_key_valid = validate_gemini_api_key()
            error_msg = "⚠️ Chave da API Gemini não encontrada! Configure a variável de ambiente GEMINI_API_KEY"
        
        if not api_key_valid:
            st.error(error_msg)
        else:
            # Botão para gerar embeddings
            if st.button("🚀 Gerar Embeddings", type="primary", use_container_width=True):
                try:
                    with st.spinner("Gerando embeddings dos chunks..."):
                        # Configura função de embedding baseado no provedor
                        if embedding_provider == "openai":
                            embed_config = EmbeddingConfig(
                                provider=embedding_provider,
                                model_name=embedding_model,
                                embedding_dim=embedding_dim
                            )
                            embed_func = get_embedding_function(embed_config)
                        else:  # gemini
                            embed_config = GeminiEmbeddingConfig(
                                model_name=embedding_model,
                                output_dimensionality=embedding_dim,
                                task_type="RETRIEVAL_DOCUMENT"
                            )
                            embed_func = get_gemini_embedding_function(embed_config)
                        
                        # Gera embeddings
                        chunks = st.session_state.rag_classic_chunks
                        embeddings = embed_func(chunks)
                        
                        # Converte para numpy array
                        embeddings_array = np.array(embeddings)
                        st.session_state.rag_classic_embeddings = embeddings_array
                    
                    st.success(f"✅ Embeddings gerados com sucesso! Shape: {embeddings_array.shape}")
                
                except Exception as e:
                    st.error(f"Erro ao gerar embeddings: {str(e)}")
        
        # Visualização dos embeddings
        if st.session_state.rag_classic_embeddings is not None:
            st.markdown("---")
            st.markdown("### 📊 Visualização dos Embeddings")
            
            embeddings = st.session_state.rag_classic_embeddings
            
            # Estatísticas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Número de Vetores", embeddings.shape[0])
            with col2:
                st.metric("Dimensões", embeddings.shape[1])
            with col3:
                st.metric("Tamanho Total", f"{embeddings.nbytes / 1024:.2f} KB")
            
            # ==================== SEÇÃO EDUCACIONAL: POR QUE VISUALIZAR? ====================
            
            st.markdown("#### 🗺️ Por Que Visualizar Embeddings?")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                **Embeddings são vetores de {embeddings.shape[1]} dimensões** - impossível visualizar diretamente!
                
                Esta visualização ajuda você a:
                - 🔍 **Ver padrões**: Identificar chunks similares visualmente
                - 📊 **Detectar agrupamentos**: Encontrar tópicos ou temas
                - 🐛 **Debugar**: Verificar se o chunking está funcionando bem
                - 🧠 **Entender**: Como o modelo "enxerga" seus documentos
                """)
            
            with col2:
                st.info(f"""
                ⚠️ **Atenção**
                
                Vamos reduzir de **{embeddings.shape[1]} → 3 dimensões** usando PCA.
                
                Isso significa que **perdemos detalhes**, mas ganhamos a capacidade de **visualizar**!
                """)
            
            # Guias educacionais em expanders
            display_embedding_visualization_guide(embeddings.shape[1])
            display_pca_explainer()
            
            st.markdown("---")
            
            # ==================== VISUALIZAÇÃO 3D COM PCA ====================
            
            st.markdown("#### 🎨 Mapa Interativo de Embeddings (PCA 3D)")
            
            try:
                from sklearn.decomposition import PCA
                import plotly.express as px
                import pandas as pd
                
                # Reduz dimensionalidade para 3D
                pca = PCA(n_components=3)
                embeddings_3d = pca.fit_transform(embeddings)
                
                # Prepara dados para plotly com preview mais longo
                df_plot = pd.DataFrame({
                    'x': embeddings_3d[:, 0],
                    'y': embeddings_3d[:, 1],
                    'z': embeddings_3d[:, 2],
                    'chunk_id': [f"Chunk {i+1}" for i in range(len(embeddings_3d))],
                    'preview': [chunk[:100] + "..." if len(chunk) > 100 else chunk 
                               for chunk in st.session_state.rag_classic_chunks]
                })
                
                # Cria gráfico 3D com configurações educacionais
                fig = px.scatter_3d(
                    df_plot,
                    x='x', y='y', z='z',
                    hover_data={
                        'chunk_id': True,
                        'preview': True,
                        'x': ':.2f',
                        'y': ':.2f',
                        'z': ':.2f'
                    },
                    labels={
                        'x': '📊 Componente Principal 1 (mais importante)',
                        'y': '📊 Componente Principal 2',
                        'z': '📊 Componente Principal 3'
                    }
                )
                
                # Melhora aparência do gráfico
                fig.update_traces(
                    marker=dict(
                        size=8,
                        opacity=0.8,
                        color='steelblue',
                        line=dict(width=0.5, color='white')
                    ),
                    selector=dict(mode='markers')
                )
                
                fig.update_layout(
                    height=600,
                    scene=dict(
                        xaxis_title='📊 PC1 (Direção Principal)',
                        yaxis_title='📊 PC2 (Segunda Direção)',
                        zaxis_title='📊 PC3 (Terceira Direção)',
                        camera=dict(
                            eye=dict(x=1.5, y=1.5, z=1.3)
                        )
                    ),
                    # FIX 1: Tooltip adaptado para tema dark
                    hoverlabel=dict(
                        bgcolor="rgba(50, 50, 50, 0.95)",  # Fundo escuro semi-transparente
                        font_size=13,
                        font_family="monospace",
                        font_color="white",  # Texto branco para contraste
                        bordercolor="steelblue"  # Borda colorida
                    )
                )
                
                # Dica de uso antes do gráfico
                st.info("💡 **Dica:** Passe o mouse sobre os pontos para ver o conteúdo dos chunks. Clique e arraste para rotacionar!")
                
                # Renderiza o gráfico
                st.plotly_chart(fig, use_container_width=True)
                
                # ==================== EXPLICAÇÃO DE VARIÂNCIA ====================
                
                variance_explained = pca.explained_variance_ratio_
                total_variance = sum(variance_explained)
                
                # FIX 2: Exibe os percentuais de forma visível E chama a função educacional
                st.markdown("#### 📊 Variância Explicada")
                
                # Mostra os percentuais de forma destacada
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("PC1", f"{variance_explained[0]:.1%}", help="Componente Principal 1 - Mais importante")
                with col2:
                    st.metric("PC2", f"{variance_explained[1]:.1%}", help="Componente Principal 2")
                with col3:
                    st.metric("PC3", f"{variance_explained[2]:.1%}", help="Componente Principal 3")
                with col4:
                    st.metric("Total", f"{total_variance:.1%}", delta=None, help="Variância total capturada")
                
                # Exibe explicação educacional detalhada
                display_variance_explainer(variance_explained.tolist(), total_variance)
                
            except ImportError:
                st.warning("⚠️ Bibliotecas necessárias não instaladas: scikit-learn, plotly")
                st.code("pip install scikit-learn plotly", language="bash")
            except Exception as e:
                st.error(f"❌ Erro na visualização: {str(e)}")
                with st.expander("🐛 Detalhes técnicos do erro"):
                    st.exception(e)


# ==================== TAB 3: QUERY & RETRIEVAL ====================

with tab3:
    st.header("🔍 Busca e Recuperação")
    
    st.markdown("""
    **Objetivo desta etapa:** Buscar os chunks mais relevantes para uma pergunta.
    
    **Como funciona a busca vetorial?**
    1. Converte a pergunta em embedding
    2. Calcula similaridade com todos os chunks
    3. Retorna os Top-K mais similares
    """)
    
    if st.session_state.rag_classic_embeddings is None:
        display_info_box(
            "Atenção",
            "Você precisa gerar os embeddings na **Tab 2** antes de fazer buscas.",
            box_type="warning"
        )
    else:
        # Input da query
        st.markdown("### 💬 Sua Pergunta")
        query = st.text_area(
            "Digite sua pergunta sobre o documento:",
            placeholder="Ex: Quais são os principais temas do documento?",
            height=100
        )
        
        # Parâmetros de busca
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider(
                "Top-K Resultados",
                min_value=1,
                max_value=min(10, len(st.session_state.rag_classic_chunks)),
                value=5,
                help="Número de chunks mais relevantes a recuperar"
            )
        
        with col2:
            show_scores = st.checkbox("Mostrar Scores de Similaridade", value=True)
        
        # Botão de busca
        if query and st.button("🔍 Buscar Chunks Relevantes", type="primary", use_container_width=True):
            try:
                with st.spinner("Buscando chunks relevantes..."):
                    # Armazena a query
                    st.session_state.rag_classic_last_query = query
                    
                    # Detecta qual provedor foi usado nos embeddings baseado na dimensão
                    embedding_dim = st.session_state.rag_classic_embeddings.shape[1]
                    
                    # OpenAI usa dimensões específicas, Gemini pode usar várias
                    if embedding_dim == 1536 or embedding_dim == 3072:
                        # Provavelmente OpenAI, mas pode ser Gemini também
                        # Vamos tentar OpenAI primeiro
                        try:
                            embed_config = EmbeddingConfig(
                                provider="openai",
                                model_name="text-embedding-3-small" if embedding_dim == 1536 else "text-embedding-3-large",
                                embedding_dim=embedding_dim
                            )
                            embed_func = get_embedding_function(embed_config)
                        except:
                            # Se falhar, tenta Gemini
                            embed_config = GeminiEmbeddingConfig(
                                model_name="gemini-embedding-001",
                                output_dimensionality=embedding_dim,
                                task_type="RETRIEVAL_QUERY"
                            )
                            embed_func = get_gemini_embedding_function(embed_config)
                    else:
                        # Outras dimensões, provavelmente Gemini
                        embed_config = GeminiEmbeddingConfig(
                            model_name="gemini-embedding-001",
                            output_dimensionality=embedding_dim,
                            task_type="RETRIEVAL_QUERY"
                        )
                        embed_func = get_gemini_embedding_function(embed_config)
                    
                    query_embedding = np.array(embed_func([query])[0])
                    
                    # Busca chunks similares
                    results = search_similar_chunks(
                        query_embedding,
                        st.session_state.rag_classic_embeddings,
                        st.session_state.rag_classic_chunks,
                        top_k=top_k
                    )
                    
                    st.session_state.rag_classic_query_results = results
                
                st.success(f"✅ Encontrados {len(results)} chunks relevantes!")
            
            except Exception as e:
                st.error(f"Erro na busca: {str(e)}")
        
        # Exibição dos resultados
        if st.session_state.rag_classic_query_results:
            st.markdown("---")
            st.markdown("### 📊 Resultados da Busca")
            
            results = st.session_state.rag_classic_query_results
            
            # Gráfico de scores
            if show_scores and len(results) > 1:
                import plotly.graph_objects as go
                
                scores = [r['score'] for r in results]
                labels = [f"Chunk {r['index']+1}" for r in results]
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=labels,
                        y=scores,
                        marker_color='steelblue',
                        text=[f"{s:.3f}" for s in scores],
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    title="Scores de Similaridade por Chunk",
                    xaxis_title="Chunk",
                    yaxis_title="Score de Similaridade",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Lista de chunks recuperados
            st.markdown("#### 📄 Chunks Recuperados")
            
            for i, result in enumerate(results, 1):
                score_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📄"
                
                with st.expander(
                    f"{score_emoji} Rank {i} - Chunk {result['index']+1} " +
                    (f"(Score: {result['score']:.4f})" if show_scores else ""),
                    expanded=(i <= 3)
                ):
                    st.markdown(f"**Conteúdo:**")
                    st.text(result['chunk'])
                    
                    if show_scores:
                        st.progress(result['score'])


# ==================== TAB 4: GENERATION ====================

with tab4:
    st.header("💬 Geração da Resposta")
    
    st.markdown("""
    **Objetivo desta etapa:** Usar um LLM para gerar uma resposta baseada nos chunks recuperados.
    
    **Como funciona?**
    1. Monta um prompt com o contexto (chunks recuperados)
    2. Adiciona a pergunta do usuário
    3. Envia para o LLM gerar a resposta
    """)
    
    if not st.session_state.rag_classic_query_results:
        display_info_box(
            "Atenção",
            "Você precisa fazer uma busca na **Tab 3** antes de gerar uma resposta.",
            box_type="warning"
        )
    else:
        # Configuração do LLM
        st.markdown("### ⚙️ Configuração do LLM")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            llm_provider = st.selectbox(
                "Provedor LLM",
                ["openai", "gemini"],
                help="Provedor de LLM para geração"
            )
        
        with col2:
            if llm_provider == "openai":
                llm_model = st.selectbox(
                    "Modelo",
                    ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                    help="Modelo de LLM para geração"
                )
            else:  # gemini
                llm_model = st.selectbox(
                    "Modelo",
                    ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"],
                    help="Modelo Gemini para geração"
                )
        
        with col3:
            temperature = st.slider(
                "Temperatura",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="Controla a criatividade (0=determinístico, 1=criativo)"
            )
        
        # Mostra a query original
        if 'rag_classic_last_query' in st.session_state:
            st.info(f"**Pergunta:** {st.session_state.rag_classic_last_query}")
        
        # Botão para gerar resposta
        if st.button("🤖 Gerar Resposta com LLM", type="primary", use_container_width=True):
            # Valida API key do provedor selecionado
            if llm_provider == "openai":
                if not validate_api_key("openai"):
                    st.error("⚠️ Chave da API OpenAI não encontrada!")
                    api_key_valid = False
                else:
                    api_key_valid = True
            else:  # gemini
                if not validate_gemini_api_key():
                    st.error("⚠️ Chave da API Gemini não encontrada!")
                    api_key_valid = False
                else:
                    api_key_valid = True
            
            if api_key_valid:
                try:
                    with st.spinner("Gerando resposta..."):
                        # Configura LLM baseado no provedor
                        if llm_provider == "openai":
                            llm_config = LLMConfig(
                                provider="openai",
                                model_name=llm_model,
                                temperature=temperature,
                                max_tokens=4000
                            )
                            llm_func = get_llm_function(llm_config)
                        else:  # gemini
                            llm_config = GeminiConfig(
                                model_name=llm_model,
                                temperature=temperature,
                                max_tokens=4000
                            )
                            llm_func = get_gemini_llm_function(llm_config)
                        
                        # Extrai chunks para contexto
                        results = st.session_state.rag_classic_query_results
                        context_chunks = [r['chunk'] for r in results]
                        
                        # Usa a query armazenada
                        query_text = st.session_state.get('rag_classic_last_query', '')
                        
                        if not query_text:
                            st.error("Pergunta não encontrada. Por favor, faça uma busca primeiro na Tab 3.")
                        else:
                            # Gera resposta
                            response = generate_rag_response(
                                query_text,
                                context_chunks,
                                llm_func
                            )
                            
                            st.session_state.rag_classic_response = response
                
                except Exception as e:
                    st.error(f"Erro ao gerar resposta: {str(e)}")
        
        # Exibe resposta
        if 'rag_classic_response' in st.session_state:
            st.markdown("---")
            st.markdown("### 💡 Resposta Gerada")
            
            st.markdown(st.session_state.rag_classic_response)
            
            # Mostra contexto usado
            with st.expander("📚 Ver Contexto Enviado ao LLM"):
                st.markdown("**Chunks usados como contexto:**")
                for i, result in enumerate(st.session_state.rag_classic_query_results, 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.text(result['chunk'])
                    st.markdown("---")
            
            # Métricas
            st.markdown("### 📊 Métricas")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Chunks Usados", len(st.session_state.rag_classic_query_results))
            
            with col2:
                total_context = sum(len(r['chunk']) for r in st.session_state.rag_classic_query_results)
                st.metric("Tokens Contexto (aprox.)", count_tokens_approximate(str(total_context)))
            
            with col3:
                response_tokens = count_tokens_approximate(st.session_state.rag_classic_response)
                st.metric("Tokens Resposta (aprox.)", response_tokens)


# ==================== FOOTER ====================

st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>💡 <b>Dica:</b> Experimente diferentes documentos e parâmetros para entender como cada etapa afeta o resultado final!</p>
    <p style='font-size: 0.9em;'>Desenvolvido com ❤️ </p>
</div>
""", unsafe_allow_html=True)
