# aquivo semantic_encoder.py

from chunks import Chunks
from read_files import ReadFiles
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
from typing import Any, Dict, List, Optional, cast
"""import os

script_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(script_dir, "RAG_visual_lab", "chroma_db")"""


class SemanticEncoder:
    """
    Constrói a base vetorial e popula o ChromaDB a partir de documentos em um diretório.

    Parâmetros:
    - docs_dir (str): diretório onde estão os documentos
    - chunk_size (int): tamanho de cada chunk
    - overlap_size (int): tamanho da sobreposição entre chunks
    - db_path (str): caminho do banco ChromaDB (default: "./chroma_db")
    - collection_name (str): nome da coleção no ChromaDB (default: "documentos_rag")
    """

    def __init__(
        self,
        docs_dir: str,
        chunk_size: int,
        overlap_size: int,
        db_path: str = "./chroma_db",
        collection_name: str = "documentos_rag",
    ) -> None:
        self.docs_dir = docs_dir
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.db_path = db_path
        self.collection_name = collection_name

        # Dependências
        self.rf = ReadFiles()
        self.chunker = Chunks(chunk_size=self.chunk_size, overlap_size=self.overlap_size)
        self.modelo = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = None

    def build(self, reset_collection: bool = True, collection_name: Optional[str] = None) -> dict:
        """
        Lê os documentos do diretório, cria chunks, gera embeddings e salva no ChromaDB.

        Args:
            reset_collection (bool): Se verdadeiro, apaga a coleção antes de recriá-la.
        Returns:
            dict: Estatísticas do processo (número de chunks salvos e total de documentos na coleção).
        """
        collection_name = collection_name or self.collection_name
        self.collection_name = collection_name

        # 1) Ler documentos e consolidar em markdown
        mds = self.rf.docs_to_markdown(self.docs_dir)

        # 2) Criar chunks
        text_chunks = self.chunker.create_chunks(mds)

        if not text_chunks:
            print("⚠️ Nenhum chunk foi gerado. Verifique o conteúdo em 'docs'.")
            try:
                existing_collection = self.client.get_collection(name=collection_name)
                total_docs = existing_collection.count()
            except Exception:
                total_docs = 0
            return {
                "chunks_salvos": 0,
                "colecao": collection_name,
                "total_documentos": total_docs,
            }

        # 3) Gerar embeddings
        base_vetorial_documentos = self.modelo.encode(text_chunks)
        embeddings = base_vetorial_documentos.tolist()  # ChromaDB requer lista

        # 4) (Re)criar/obter coleção
        if reset_collection:
            try:
                self.client.delete_collection(name=collection_name)
                print(f"Coleção '{collection_name}' existente foi deletada.")
            except Exception:
                pass

        try:
            self.collection = self.client.get_collection(name=collection_name)
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Coleção de chunks de documentos com embeddings"},
            )

        # 5) Inserir dados
        ids = [str(uuid.uuid4()) for _ in range(len(text_chunks))]
        metadatas: List[Dict[str, Any]] = [
            {
                "chunk_id": i,
                "chunk_size": len(chunk),
                "source": self.docs_dir,
            }
            for i, chunk in enumerate(text_chunks)
        ]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=text_chunks,
            metadatas=cast(Any, metadatas),
        )

        print(f"✅ Salvos {len(text_chunks)} chunks no ChromaDB!")
        print(
            f"📊 Coleção '{self.collection_name}' agora possui {self.collection.count()} documentos"
        )

        return {
            "chunks_salvos": len(text_chunks),
            "colecao": self.collection_name,
            "total_documentos": self.collection.count(),
        }



if __name__ == "__main__":
    

    from chunks import Chunks
    from read_files import ReadFiles
    from sentence_transformers import SentenceTransformer
    import chromadb
    import uuid
    import os

    # Obter o diretório do script atual
    script_dir = os.path.dirname(os.path.abspath(__file__))
    docs_path = os.path.join(script_dir, "docs", "synthetic_dataset_papers")
    chroma_db_path = os.path.join(script_dir, "RAG_visual_lab", "chroma_db")

    retriever = SemanticEncoder(
        docs_dir=docs_path, #diretório dos documentos
        chunk_size=2000, #tamanho do chunk
        overlap_size=500, #tamanho da sobreposição
        db_path=chroma_db_path, #caminho do banco ChromaDB
        collection_name="synthetic_dataset_papers"
    )
    
    # Construir base vetorial
    stats = retriever.build(collection_name="synthetic_dataset_papers")

    # Imprimir estatísticas
    print(stats)


