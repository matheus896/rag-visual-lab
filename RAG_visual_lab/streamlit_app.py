"""
Laboratório Visual de RAG
==========================

Aplicativo Streamlit educacional para visualização de conceitos de RAG.
Este é o arquivo entrypoint que configura a navegação e o layout global.

Autor: Matheus Almeida
Data: Outubro 2025
"""

import streamlit as st

# Configuração global da página - DEVE ser a primeira chamada do Streamlit
st.set_page_config(
    page_title="Laboratório Visual de RAG",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/matheus896/rag-visual-lab',
        'Report a bug': None,
        'About': """
        # Laboratório Visual de RAG
        
        Uma ferramenta educacional interativa para aprender sobre 
        Retrieval-Augmented Generation (RAG) e suas variações.
        
        Desenvolvido como material de apoio para a mentoria do 
        Professor Sandeco.
        """
    }
)

# Mensagem de boas-vindas na sidebar
st.sidebar.success("👆 Navegue pelos laboratórios acima para explorar diferentes conceitos de RAG.")

# Informações adicionais na sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📚 Sobre este Laboratório")
    st.markdown("""
    Este aplicativo demonstra visualmente:
    - RAG Clássico
    - RAG com Memória
    - RAG Agente
    - RAG Corretivo (em breve)
    - GraphRAG (em breve)
    - RAG Fusion (em breve)
    """)
    
    st.markdown("---")
    st.markdown("### 🛠️ Tecnologias")
    st.markdown("""
    - **Framework**: Streamlit
    - **LLM & Embeddings**: Google Gemini + paraphrase-multilingual
    - **Visualizações**: Plotly, Agraph
    """)
