"""
Página Home - Laboratório Visual de RAG
========================================

Página inicial do aplicativo com introdução ao projeto e orientações de uso.
"""

import streamlit as st

# Título principal
st.title("🔬 Bem-vindo ao Laboratório Visual de RAG")

# Introdução
st.markdown("""
### 🎯 O que é este Laboratório?

Este é um **aplicativo educacional interativo** desenvolvido para ajudar você a compreender 
os conceitos fundamentais de **RAG (Retrieval-Augmented Generation)** e suas principais variações.

RAG é uma técnica avançada de Inteligência Artificial que combina:
- 🔍 **Recuperação de Informação** (Retrieval)
- 🤖 **Geração de Texto com LLMs** (Generation)
- 📊 **Dados Externos** para enriquecer as respostas

O objetivo é **combater alucinações** dos modelos de linguagem, garantindo que as respostas 
sejam baseadas em informações concretas e rastreáveis.
""")

# Seção: Como Usar
st.markdown("---")
st.markdown("### 🚀 Como Usar este Laboratório")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### 1️⃣ Navegação
    - Use o menu lateral para acessar diferentes módulos
    - Cada módulo explora um conceito específico de RAG
    - Experimente diferentes configurações e parâmetros
    """)
    
    st.markdown("""
    #### 2️⃣ Upload de Documentos
    - Carregue seus próprios documentos (PDF, TXT, MD)
    - O sistema processará e indexará o conteúdo
    - Use os documentos como base de conhecimento
    """)

with col2:
    st.markdown("""
    #### 3️⃣ Experimentação
    - Ajuste parâmetros como `chunk_size` e `top_k`
    - Observe o impacto em tempo real
    - Compare diferentes abordagens de RAG
    """)
    
    st.markdown("""
    #### 4️⃣ Visualizações
    - Explore mapas de embeddings interativos
    - Visualize o grafo de conhecimento
    - Acompanhe o processo de raciocínio do sistema
    """)

# Seção: Módulos Disponíveis
st.markdown("---")
st.markdown("### 📚 Módulos Disponíveis")

# Grid de módulos usando tabs
tab1, tab2, tab3 = st.tabs(["🔰 Básico", "🚀 Avançado", "🔬 Experimental"])

with tab1:
    st.markdown("""
    #### 🔹 RAG Clássico
    Implementação tradicional do RAG com fluxo passo a passo:
    - Chunking de documentos
    - Embeddings e busca vetorial
    - Geração aumentada
    
    #### 🔹 RAG com Memória
    RAG conversacional que mantém contexto entre perguntas:
    - Interface de chat
    - Histórico de conversação
    - Respostas contextualizadas
    """)

with tab2:
    st.markdown("""
    #### 🔸 RAG Agente
    Sistema autônomo que decide suas próprias ações:
    - Raciocínio em múltiplas etapas
    - Ferramentas especializadas (em breve)
    - Visualização do processo de pensamento
    
    #### 🔸 RAG Corretivo (em breve)
    Auto-correção e validação de respostas:
    - Verificação de relevância
    - Refinamento iterativo
    - Métricas de confiança
    """)

with tab3:
    st.markdown("""
    #### 🔬 GraphRAG (em breve)
    Utiliza grafos de conhecimento para relações complexas:
    - Extração de entidades e relações
    - Visualização de grafo interativo
    - Busca baseada em grafos
    
    #### 🔬 RAG Fusion (em breve)
    Combinação de múltiplas estratégias de busca:
    - Fusão de resultados
    - Re-ranking inteligente
    - Máxima cobertura de informação
    """)

# Seção: Recursos Adicionais
st.markdown("---")
st.markdown("### 📖 Recursos Adicionais")

resources_col1, resources_col2 = st.columns(2)

with resources_col1:
    st.info("""
    **📚 Documentação Streamlit**

    Explore a documentação oficial da biblioteca Streamlit
    que serve como base visual para este laboratório.
    
    [Acessar Documentação](https://docs.streamlit.io/)
    """)

with resources_col2:
    st.success("""
    **🎓 Material da Mentoria**
    
    Este laboratório é baseado no livro de RAG da mentoria 
    do Professor Sandeco.
    
    Consulte o material complementar para aprofundamento.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Desenvolvido com ❤️ para a comunidade de mentorados</p>
    <p><small>Professor Sandeco | 2025</small></p>
</div>
""", unsafe_allow_html=True)
