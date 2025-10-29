import sys
import os

# Adiciona o diretório raiz ao PYTHONPATH
_current_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = _current_dir
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from semantic_encoder import SemanticEncoder

# Obter o diretório do script atual
script_dir = os.path.dirname(os.path.abspath(__file__))
docs_path = os.path.join(script_dir, "docs", "synthetic_dataset_papers")
chroma_db_path = os.path.join(script_dir, "RAG_visual_lab", "chroma_db")

encoder = SemanticEncoder(
    docs_dir=docs_path, #diretório dos documentos
    chunk_size=5000, #tamanho do chunk
    overlap_size=500, #tamanho da sobreposição
    db_path=chroma_db_path, #caminho do banco ChromaDB
)

# Construir base vetorial
stats = encoder.build(collection_name="synthetic_dataset_papers")

# Imprimir estatísticas
print(stats)



