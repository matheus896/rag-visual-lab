"""
Processamento de Texto
=======================

Funções para extração e processamento de texto de diferentes formatos de arquivo.

Formatos Suportados:
- PDF (.pdf)
- Texto puro (.txt)
- Markdown (.md)
"""

from typing import Optional, Any
from io import BytesIO
import streamlit as st


def extract_text_from_file(uploaded_file: Any) -> Optional[str]:
    """
    Extrai texto de um arquivo carregado.
    
    Args:
        uploaded_file: Arquivo carregado via st.file_uploader
        
    Returns:
        Texto extraído ou None em caso de erro
    """
    if uploaded_file is None:
        return None
    
    try:
        filename = uploaded_file.name.lower()
        raw_bytes = uploaded_file.getvalue()
        
        # Processa PDF
        if filename.endswith('.pdf'):
            return _extract_from_pdf(raw_bytes)
        
        # Processa texto puro (TXT, MD, MARKDOWN)
        elif filename.endswith(('.txt', '.md', '.markdown')):
            return _extract_from_text(raw_bytes)
        
        else:
            st.error(f"Formato de arquivo não suportado: {filename.split('.')[-1]}")
            return None
            
    except Exception as e:
        st.error(f"Erro ao processar arquivo: {str(e)}")
        return None


def _extract_from_text(raw_bytes: bytes) -> str:
    """Extrai texto de arquivo TXT ou MD."""
    try:
        decoded = raw_bytes.decode('utf-8')
    except UnicodeDecodeError:
        decoded = raw_bytes.decode('latin-1', errors='ignore')
    
    return decoded


def _extract_from_pdf(raw_bytes: bytes) -> str:
    """
    Extrai texto de arquivo PDF usando PyPDF2.
    
    Nota: Usa PyPDF2 para compatibilidade com chunks_streamlit_app.py
    """
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        st.error("Biblioteca 'PyPDF2' não instalada. Execute: pip install PyPDF2")
        return ""
    
    text_parts = []
    try:
        reader = PdfReader(BytesIO(raw_bytes))
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    except Exception as exc:
        st.error(f"Não foi possível extrair texto do PDF: {exc}")
        return ""
    
    return "\n\n".join(text_parts)


def chunk_text(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 100
) -> list[str]:
    """
    Divide texto em chunks com overlap.
    
    Args:
        text: Texto a ser dividido
        chunk_size: Tamanho aproximado de cada chunk (em caracteres)
        overlap: Número de caracteres de overlap entre chunks
        
    Returns:
        Lista de chunks de texto
    """
    if not text:
        return []

    # Adiciona uma verificação para prevenir o loop infinito
    if overlap >= chunk_size:
        st.error(f"Configuração de Chunking Inválida: O 'Overlap' ({overlap}) deve ser menor que o 'Tamanho do Chunk' ({chunk_size}).")
        return []

    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        # Define o final do chunk atual
        end = start + chunk_size
        
        # Se não é o último chunk, tenta quebrar em uma posição melhor
        if end < text_length:
            # Procura por quebras naturais próximas ao final do chunk
            chunk_section = text[start:end]
            
            # Tenta quebrar em parágrafo (dupla quebra de linha)
            last_paragraph = chunk_section.rfind('\n\n')
            if last_paragraph > len(chunk_section) * 0.7:  # Se encontrou um parágrafo nos últimos 30%
                end = start + last_paragraph + 2
            
            # Se não encontrou parágrafo, tenta quebrar em frase (ponto + espaço)
            elif '. ' in chunk_section:
                last_sentence = chunk_section.rfind('. ')
                if last_sentence > len(chunk_section) * 0.7:  # Se encontrou uma frase nos últimos 30%
                    end = start + last_sentence + 2
            
            # Se não encontrou frase, tenta quebrar em palavra (espaço)
            elif ' ' in chunk_section:
                last_space = chunk_section.rfind(' ')
                if last_space > len(chunk_section) * 0.8:  # Se encontrou um espaço nos últimos 20%
                    end = start + last_space + 1
        
        # Extrai o chunk atual
        chunk = text[start:end].strip()
        if chunk:  # Só adiciona se o chunk não estiver vazio
            chunks.append(chunk)
        
        # Calcula a próxima posição inicial considerando o overlap
        if end >= text_length:
            break
        
        start = end - overlap
        
        # Garante que não voltamos para trás demais
        if start < 0:
            start = 0
    
    return chunks


def count_tokens_approximate(text: str) -> int:
    """
    Estima o número de tokens em um texto.
    
    Nota: Esta é uma estimativa simples. Para contagem precisa,
    use tiktoken ou a biblioteca específica do modelo.
    
    Args:
        text: Texto a ser contado
        
    Returns:
        Estimativa do número de tokens
    """
    # Estimativa simples: ~4 caracteres por token para inglês
    # Ajustado para ~3.5 para português
    return len(text) // 4
