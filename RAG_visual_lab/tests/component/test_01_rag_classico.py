import pytest
from streamlit.testing.v1 import AppTest
import os
from unittest.mock import Mock

# Adiciona o diretório raiz do projeto ao sys.path
APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def test_rag_classico_page_smoke_test():
    """
    Teste básico para garantir que a página carrega sem erros.
    """
    page_script_path = os.path.join(APP_ROOT, "pages", "01_🔰_RAG_Clássico.py")
    assert os.path.exists(page_script_path)
    
    at = AppTest.from_file(page_script_path, default_timeout=30)
    at.run()
    
    assert not at.exception
    assert at.title[0].value == "🔰 RAG Clássico - Passo a Passo"

def test_file_processing_logic_within_app_context():
    """
    Simula o estado pós-upload e verifica se o session_state é 
    corretamente populado, usando o timeout para detectar travamentos.
    """
    page_script_path = os.path.join(APP_ROOT, "pages", "01_🔰_RAG_Clássico.py")
    # Usamos um timeout de 30 segundos. Se at.run() travar, o pytest falhará aqui.
    at = AppTest.from_file(page_script_path, default_timeout=30)

    # 1. Prepara o estado inicial da aplicação.
    #    A chave 'uploaded_file' corresponde ao `key` do st.file_uploader.
    mock_file = Mock()
    mock_file.name = "test.txt"
    mock_file.getvalue.return_value = b"Este e o conteudo do arquivo."
    at.session_state["uploaded_file"] = mock_file
    
    # 2. Executa o script.
    #    Se houver um loop infinito no processamento do arquivo, o teste falhará
    #    aqui devido ao timeout de 30 segundos.
    at.run()

    # 3. Verifica o resultado final no session_state.
    #    Se o teste não travou, verificamos se a operação foi bem-sucedida.
    assert "rag_classic_document_text" in at.session_state, \
        "O estado 'rag_classic_document_text' não foi definido após a execução."
    
    assert at.session_state.rag_classic_document_text == "Este e o conteudo do arquivo.", \
        "O conteúdo extraído para o session_state está incorreto."

    # 4. Verifica se a UI foi atualizada para refletir o estado processado.
    #    Isso confirma que o bloco 'if document_text:' foi executado.
    assert any("Estatísticas do Documento" in m.value for m in at.markdown), \
        "A seção 'Estatísticas do Documento' não foi renderizada após o processamento."
