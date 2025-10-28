"""
Componentes de UI Reutilizáveis
================================

Este módulo contém funções para renderizar elementos de UI consistentes
em todas as páginas do Laboratório Visual de RAG.

Princípios:
- Componentes funcionais puros (sem efeitos colaterais)
- Nomenclatura clara e descritiva
- Encapsulamento de complexidade
- Consistência visual em todo o aplicativo
"""

import streamlit as st
from typing import Optional, List, Dict, Any


def render_document_uploader(
    accepted_types: List[str] = ["pdf", "txt", "md"],
    help_text: Optional[str] = None,
    key: Optional[str] = None
) -> Optional[Any]:
    """
    Renderiza um uploader de documentos com configurações padrão.
    
    Args:
        accepted_types: Lista de extensões de arquivo aceitas
        help_text: Texto de ajuda opcional
        key: Chave única para o widget
        
    Returns:
        Arquivo carregado ou None
    """
    default_help = f"Tipos aceitos: {', '.join(accepted_types).upper()}"
    
    uploaded_file = st.file_uploader(
        "📄 Carregar Documento",
        type=accepted_types,
        help=help_text or default_help,
        key=key
    )
    
    return uploaded_file


def display_source_chunks(
    chunks: List[Dict[str, Any]],
    max_display: int = 5
) -> None:
    """
    Exibe os chunks de texto recuperados de forma organizada.
    
    Args:
        chunks: Lista de dicionários com informações dos chunks
        max_display: Número máximo de chunks a exibir
    """
    st.markdown("### 📚 Chunks Recuperados")
    
    for i, chunk in enumerate(chunks[:max_display], 1):
        with st.expander(f"Chunk {i} - Score: {chunk.get('score', 'N/A')}"):
            st.markdown(f"**Conteúdo:**")
            st.text(chunk.get('content', ''))
            
            if 'metadata' in chunk:
                st.markdown("**Metadados:**")
                st.json(chunk['metadata'])


def render_parameter_controls(
    default_chunk_size: int = 1200,
    default_overlap: int = 100,
    default_top_k: int = 5
) -> Dict[str, int]:
    """
    Renderiza controles de parâmetros para o RAG.
    
    Args:
        default_chunk_size: Tamanho padrão do chunk
        default_overlap: Overlap padrão entre chunks
        default_top_k: Número padrão de resultados
        
    Returns:
        Dicionário com os valores dos parâmetros
    """
    st.sidebar.markdown("### ⚙️ Parâmetros")
    
    chunk_size = st.sidebar.slider(
        "Tamanho do Chunk",
        min_value=100,
        max_value=2000,
        value=default_chunk_size,
        step=100,
        help="Número de tokens por chunk de texto"
    )
    
    overlap = st.sidebar.slider(
        "Overlap entre Chunks",
        min_value=0,
        max_value=500,
        value=default_overlap,
        step=50,
        help="Tokens compartilhados entre chunks consecutivos"
    )
    
    top_k = st.sidebar.slider(
        "Top K Resultados",
        min_value=1,
        max_value=20,
        value=default_top_k,
        help="Número de chunks mais relevantes a recuperar"
    )
    
    return {
        "chunk_size": chunk_size,
        "overlap": overlap,
        "top_k": top_k
    }


def display_metrics_cards(metrics: Dict[str, Any]) -> None:
    """
    Exibe métricas em cards formatados.
    
    Args:
        metrics: Dicionário com as métricas a exibir
    """
    cols = st.columns(len(metrics))
    
    for col, (label, value) in zip(cols, metrics.items()):
        with col:
            st.metric(label=label, value=value)


def render_loading_message(message: str = "Processando...") -> None:
    """
    Exibe uma mensagem de carregamento padronizada.
    
    Args:
        message: Mensagem a exibir
    """
    with st.spinner(message):
        st.empty()


def display_info_box(
    title: str,
    content: str,
    box_type: str = "info"
) -> None:
    """
    Exibe uma caixa de informação formatada.
    
    Args:
        title: Título da caixa
        content: Conteúdo da caixa
        box_type: Tipo da caixa (info, success, warning, error)
    """
    box_functions = {
        "info": st.info,
        "success": st.success,
        "warning": st.warning,
        "error": st.error
    }
    
    box_func = box_functions.get(box_type, st.info)
    box_func(f"**{title}**\n\n{content}")


def display_embedding_visualization_guide(embedding_dim: int) -> None:
    """
    Exibe um guia educacional sobre visualização de embeddings.
    
    Args:
        embedding_dim: Dimensão original dos embeddings
    """
    with st.expander("🧭 Como Ler o Gráfico de Embeddings?", expanded=False):
        st.markdown(f"""
        ### 📚 Analogia da Biblioteca
        
        Imagine que cada chunk do seu documento é um **livro em uma biblioteca gigante**.
        
        **Problema:** Sua "biblioteca" tem **{embedding_dim} estantes** (dimensões)!  
        É impossível visualizar {embedding_dim} dimensões de uma vez.
        
        **Solução:** Usamos uma técnica chamada **PCA** (análise de componentes principais)
        que é como tirar uma **foto 3D** dessa biblioteca enorme.
        
        ---
        
        ### 🎯 O Que Cada Elemento Significa?
        
        - **🔵 Cada ponto azul** = 1 chunk do seu documento
        - **Proximidade** = Chunks similares ficam próximos
        - **Distância** = Chunks diferentes ficam afastados
        - **Agrupamentos** = Tópicos ou temas semelhantes
        
        ---
        
        ### 🖱️ Interatividade
        
        - **Passe o mouse** sobre um ponto para ver o conteúdo do chunk
        - **Clique e arraste** para rotacionar o gráfico 3D
        - **Scroll** para dar zoom
        - **Duplo clique** para resetar a visualização
        """)


def display_pca_explainer() -> None:
    """
    Explica o conceito de PCA (Principal Component Analysis) em linguagem simples.
    """
    with st.expander("🔬 O Que é PCA? (Explicação Simples)", expanded=False):
        st.markdown("""
        ### 🎬 Imagine Descrever Uma Pessoa
        
        Você poderia usar **centenas de características**:
        - Altura, peso, cor dos olhos, tamanho do pé, cor do cabelo...
        - Comprimento dos dedos, largura dos ombros, tom de voz...
        
        Mas se você precisasse resumir em **apenas 3 características principais**, 
        quais escolheria? Provavelmente:
        1. **Altura** (mais importante)
        2. **Peso** (segunda mais importante)
        3. **Idade** (terceira mais importante)
        
        ---
        
        ### 🧮 É Isso Que PCA Faz!
        
        **PCA** pega suas **768 dimensões** e encontra as **3 direções** que capturam
        **o máximo de informação possível**.
        
        Pense assim:
        ```
        768 dimensões  →  [PCA mágico]  →  3 dimensões principais
        (impossível ver)                    (você consegue ver!)
        ```
        
        ---
        
        ### ✅ Por Que Isso é Útil?
        
        - ✨ **Visualizar** padrões que existem em alta dimensão
        - 🔍 **Encontrar** agrupamentos de chunks similares
        - 📊 **Entender** como seus documentos estão organizados
        - 🎯 **Debugar** problemas no chunking ou embeddings
        
        ---
        
        ### ⚠️ Limitação Importante
        
        Ao reduzir de 768 → 3 dimensões, **perdemos detalhes**.  
        É como tirar uma foto 2D de um objeto 3D: captura a essência, mas não tudo.
        """)


def display_variance_explainer(
    variance_explained: List[float],
    total_variance: float
) -> None:
    """
    Explica o conceito de variância explicada de forma didática.
    
    Args:
        variance_explained: Lista com variância de cada componente
        total_variance: Variância total explicada
    """
    with st.expander("📊 O Que Significam Esses Percentuais?", expanded=False):
        st.markdown(f"""
        ### 🎯 Variância Explicada = "Quanta informação mantivemos?"
        
        Quando reduzimos **768 dimensões → 3 dimensões**, inevitavelmente **perdemos informação**.
        
        A **variância explicada** nos diz: *"Quão fiel é essa visualização 3D comparada aos dados originais?"*
        
        ---
        
        ### 📈 Seus Números:
        
        - **PC1 (Componente Principal 1):** {variance_explained[0]:.1%}
          - A direção mais importante! Captura {variance_explained[0]:.1%} da variação total
        
        - **PC2 (Componente Principal 2):** {variance_explained[1]:.1%}
          - Segunda direção mais importante
        
        - **PC3 (Componente Principal 3):** {variance_explained[2]:.1%}
          - Terceira direção mais importante
        
        - **📊 TOTAL:** {total_variance:.1%}
          - Capturamos {total_variance:.1%} da informação original!
        
        ---
        
        ### 🤔 {total_variance:.1%} é Bom ou Ruim?
        
        """)
        
        # Barra de progresso visual
        st.progress(total_variance)
        
        # Interpretação baseada no valor
        if total_variance >= 0.7:
            st.success(f"""
            ✅ **Excelente!** {total_variance:.1%} é muito bom!  
            Sua visualização 3D captura a maior parte da estrutura dos dados.
            """)
        elif total_variance >= 0.4:
            st.info(f"""
            ✔️ **Bom!** {total_variance:.1%} é adequado para visualização.  
            Você está vendo os padrões principais, embora alguns detalhes sejam perdidos.
            """)
        elif total_variance >= 0.2:
            st.warning(f"""
            ⚠️ **Razoável.** {total_variance:.1%} significa que muita informação foi comprimida.  
            A visualização mostra tendências gerais, mas perde bastante detalhe.
            """)
        else:
            st.warning(f"""
            ⚠️ **Limitado.** {total_variance:.1%} é baixo.  
            A visualização 3D é muito simplificada. Os dados originais são muito complexos.
            """)
        
        st.markdown("""
        ---
        
        ### 💡 Dica Prática
        
        **Para embeddings de texto**, valores entre **20-40%** são normais!  
        Embeddings são **intencionalmente de alta dimensão** para capturar nuances
        da linguagem, então é **impossível** capturar tudo em 3D.
        
        O importante é que você consiga **ver padrões** de agrupamento no gráfico! 🎯
        """)
