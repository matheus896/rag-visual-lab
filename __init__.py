"""
RAG Visual Lab - Sistema de Recuperação Aumentada por Geração

Este pacote fornece ferramentas para processamento de documentos,
criação de chunks, embeddings semânticos e recuperação de informações.
"""

import sys
import os

# Adiciona o diretório raiz do projeto ao PYTHONPATH
# Isso permite importar módulos de qualquer lugar
_root_dir = os.path.dirname(os.path.abspath(__file__))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

# Exporta as classes principais para facilitar imports
__all__ = [
    'Chunks',
    'ReadFiles', 
    'SemanticEncoder',
    'Retriever',
    'Generation',
    'Augmentation',
]

# Importações opcionais - só importa se o módulo existir
try:
    from chunks import Chunks
except ImportError:
    Chunks = None

try:
    from read_files import ReadFiles
except ImportError:
    ReadFiles = None

try:
    from semantic_encoder import SemanticEncoder
except ImportError:
    SemanticEncoder = None

try:
    from retriever import Retriever
except ImportError:
    Retriever = None

try:
    from generation import Generation
except ImportError:
    Generation = None

try:
    from augmentation import Augmentation
except ImportError:
    Augmentation = None

__version__ = "1.0.0"
