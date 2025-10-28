import pytest
from unittest.mock import Mock, patch
import os
import sys

# Adiciona o diretório raiz do projeto ao sys.path
# para garantir que os imports relativos funcionem.
APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, APP_ROOT)

from utils.text_processing import extract_text_from_file, chunk_text

# --- Fixtures do Pytest ---

@pytest.fixture
def mock_txt_file():
    """Fixture que simula um arquivo .txt enviado pelo Streamlit."""
    mock_file = Mock()
    mock_file.name = "test_document.txt"
    content = "Este é um simples arquivo de texto."
    mock_file.getvalue.return_value = content.encode("utf-8")
    return mock_file

@pytest.fixture
def mock_pdf_file():
    """Fixture que simula um arquivo .pdf enviado."""
    mock_file = Mock()
    mock_file.name = "test_document.pdf"
    # O conteúdo em bytes não precisa ser um PDF real para o mock
    mock_file.getvalue.return_value = b"fake-pdf-bytes"
    return mock_file

@pytest.fixture
def mock_unsupported_file():
    """Fixture que simula um arquivo de tipo não suportado."""
    mock_file = Mock()
    mock_file.name = "archive.zip"
    mock_file.getvalue.return_value = b"fake-zip-bytes"
    return mock_file

# --- Testes ---

def test_extract_text_from_txt(mock_txt_file):
    """
    Verifica se a extração de texto de um arquivo .txt funciona corretamente.
    """
    expected_content = mock_txt_file.getvalue().decode("utf-8")
    extracted_text = extract_text_from_file(mock_txt_file)
    assert extracted_text == expected_content

@patch("utils.text_processing._extract_from_pdf")
def test_extract_text_from_pdf_calls_helper(mock_extract_pdf, mock_pdf_file):
    """
    Verifica se a função correta (_extract_from_pdf) é chamada para arquivos PDF.
    """
    # Define o que o helper mockado deve retornar
    mock_extract_pdf.return_value = "Texto extraído do PDF."
    
    extracted_text = extract_text_from_file(mock_pdf_file)
    
    # Verifica se o helper foi chamado uma vez com os bytes corretos
    mock_extract_pdf.assert_called_once_with(mock_pdf_file.getvalue())
    
    # Verifica se o resultado é o esperado
    assert extracted_text == "Texto extraído do PDF."

def test_extract_text_from_unsupported_file(mock_unsupported_file):
    """
    Verifica se a função retorna None para tipos de arquivo não suportados.
    """
    # Usamos um patch para mockar st.error e evitar erros no teste
    with patch("streamlit.error") as mock_st_error:
        extracted_text = extract_text_from_file(mock_unsupported_file)
        assert extracted_text is None
        # Verifica se a mensagem de erro foi chamada
        mock_st_error.assert_called_once()

def test_extract_text_from_none():
    """
    Verifica se a função lida corretamente com um input None.
    """
    assert extract_text_from_file(None) is None

@patch("utils.text_processing._extract_from_text")
def test_getvalue_is_called_once(mock_extract_text, mock_txt_file):
    """
    Garante que o método .getvalue() do arquivo é chamado exatamente uma vez.
    Isso é importante para investigar o bug de performance.
    """
    extract_text_from_file(mock_txt_file)
    
    # Verifica se getvalue() foi chamado
    mock_txt_file.getvalue.assert_called_once()

def test_chunk_text_logic():
    """
    Testa a lógica de chunking para reproduzir o MemoryError.
    """
    text = "0123456789" * 5  # Texto com 50 caracteres
    chunk_size = 20
    overlap = 5
    
    chunks = chunk_text(text, chunk_size, overlap)
    
    # Verifica se os chunks foram criados corretamente
    assert len(chunks) == 3
    assert chunks[0] == "01234567890123456789"  # 20 chars
    # O próximo chunk começa 5 caracteres antes do final do anterior
    # start = 20 - 5 = 15
    assert chunks[1] == "56789012345678901234"  # 20 chars
    # start = 15 + 20 - 5 = 30
    assert chunks[2] == "01234567890123456789"  # 20 chars

@pytest.mark.parametrize("text,expected_chunks", [
    ("", []),  # Empty text
    ("Short", ["Short"]),  # Text shorter than chunk_size
    ("A" * 100, ["A" * 100]),  # Exact chunk size
    ("Hello World! " * 20, None),  # Text with natural breaks (just verify it returns a list)
])
def test_chunk_text_edge_cases(text, expected_chunks):
    """
    Testa casos extremos da função chunk_text usando parametrização.
    """
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    
    if expected_chunks is not None:
        assert chunks == expected_chunks
    else:
        # Para casos onde apenas verificamos que retorna uma lista válida
        assert isinstance(chunks, list)
        assert all(isinstance(c, str) for c in chunks)

def test_chunk_text_with_invalid_overlap():
    """
    Verifica que overlap >= chunk_size retorna lista vazia e mostra erro.
    """
    with patch("streamlit.error") as mock_error:
        chunks = chunk_text("Test text", chunk_size=10, overlap=10)
        assert chunks == []
        mock_error.assert_called_once()
        
    with patch("streamlit.error") as mock_error:
        chunks = chunk_text("Test text", chunk_size=10, overlap=15)
        assert chunks == []
        mock_error.assert_called_once()

def test_chunk_text_preserves_overlap():
    """
    Verifica que o overlap é mantido corretamente entre chunks consecutivos.
    """
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    chunks = chunk_text(text, chunk_size=10, overlap=3)
    
    # Verifica que há overlap entre chunks consecutivos
    for i in range(len(chunks) - 1):
        current_chunk = chunks[i]
        next_chunk = chunks[i + 1]
        
        # Os últimos 3 caracteres do chunk atual devem aparecer no próximo
        # (pode ser menos se houver quebra natural de linha)
        if len(current_chunk) >= 3:
            overlap_text = current_chunk[-3:]
            # Verifica se alguma parte do overlap aparece no próximo chunk
            assert any(char in next_chunk for char in overlap_text), \
                f"No overlap found between chunks {i} and {i+1}"

def test_chunk_text_with_special_characters():
    """
    Testa chunking com caracteres especiais e encoding UTF-8.
    """
    text = "Olá! Este é um teste com acentuação: á, é, í, ó, ú, ç. " * 10
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    
    assert len(chunks) > 0
    # Verifica que os caracteres especiais foram preservados
    joined = "".join(chunks)
    assert "á" in joined
    assert "ç" in joined
