# 🔬 Laboratório Visual de RAG

Uma ferramenta educacional interativa para visualização e aprendizado de conceitos de RAG (Retrieval-Augmented Generation).

## 📋 Sobre o Projeto

Este aplicativo Streamlit foi desenvolvido como material de apoio para a mentoria do Professor Sandeco, com o objetivo de tornar o aprendizado de RAG mais prático e visual.

### 🎯 Módulos Disponíveis

- **RAG Clássico**: Implementação tradicional passo a passo
- **RAG com Memória**: Interface conversacional com histórico
- **RAG Agente**: Sistema autônomo com raciocínio em múltiplas etapas
- **RAG Corretivo**: Auto-correção e validação de respostas
- **GraphRAG**: Utilização de grafos de conhecimento
- **RAG Fusion**: Combinação de múltiplas estratégias

## 🚀 Instalação

### Pré-requisitos

- Python 3.10 ou superior
- pip ou uv para gerenciamento de pacotes

### Passos de Instalação

1. **Clone o repositório:**
   ```bash
   git clone <seu-repositorio>
   cd RAG_visual_lab
   ```

2. **Crie um ambiente virtual:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure as variáveis de ambiente:**
   ```bash
   cp .env.example .env
   # Edite o arquivo .env e adicione sua OPENAI_API_KEY
   ```

## 🎮 Uso

### Executar o Aplicativo

```bash
streamlit run streamlit_app.py
```

O aplicativo será aberto automaticamente em `http://localhost:8501`

### Navegação

1. Use o menu lateral para acessar diferentes módulos
2. Carregue seus documentos (PDF, TXT, MD)
3. Experimente com diferentes parâmetros
4. Observe as visualizações interativas

## 📁 Estrutura do Projeto

```
RAG_visual_lab/
├── streamlit_app.py        # Entrypoint principal
├── pages/                  # Páginas do aplicativo
│   └── 00_🏠_Home.py
├── utils/                  # Componentes reutilizáveis
│   ├── ui_components.py
│   └── text_processing.py
├── services/               # Integração com LLMs
│   └── llm_provider.py
├── requirements.txt        # Dependências
├── .env.example           # Exemplo de configuração
└── README.md              # Este arquivo
```

## 🛠️ Tecnologias

- **Framework**: Streamlit
- **RAG Engine**: LightRAG
- **Visualizações**: Plotly, Streamlit-Agraph
- **UI Avançada**: Streamlit-Elements
- **LLM**: OpenAI (configurável para outros provedores)

## 📚 Documentação

Para mais informações sobre RAG e LightRAG:
- [LightRAG GitHub](https://github.com/HKUDS/LightRAG)
- [Documentação Streamlit](https://docs.streamlit.io)

## 👥 Contribuindo

Este projeto é parte de uma mentoria educacional. Contribuições são bem-vindas!

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto é desenvolvido para fins educacionais.

## ✨ Agradecimentos

- Professor Sandeco e comunidade de mentorados
- Desenvolvedores do LightRAG
- Comunidade Streamlit

---

Desenvolvido com ❤️ para a comunidade de mentorados | 2024
