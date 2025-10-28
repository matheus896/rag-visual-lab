# services/retriever_provider.py

"""
RetrieverProvider Service
=========================

Encapsula a lógica de recuperação de chunks do ChromaDB para o pipeline RAG.
"""

import chromadb
from sentence_transformers import SentenceTransformer
from typing import Optional


class RetrieverProvider:
    """
    Gerencia a recuperação de chunks de documentos do ChromaDB usando busca semântica.
    
    Esta classe encapsula a lógica de:
    1. Conexão com ChromaDB em modo persistente
    2. Geração de embeddings usando SentenceTransformer
    3. Busca vetorial de chunks similares a uma query
    
    Arquitetura:
    - Segue o padrão Service do projeto (similar a MemoryProvider)
    - Retorna chunks no formato esperado pelo AugmentationProvider (list[str])
    - Isolado de dependências externas para facilitar testes
    
    Exemplo de uso:
        >>> retriever = RetrieverProvider(
        ...     db_path="./chroma_db",
        ...     collection_name="synthetic_dataset_papers"
        ... )
        >>> chunks = retriever.search("What is RAG?", n_results=5)
        >>> print(f"Found {len(chunks)} relevant chunks")
        Found 5 relevant chunks
    """
    
    DEFAULT_DB_PATH = "./chroma_db"
    DEFAULT_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'
    
    def __init__(
        self, 
        db_path: str = DEFAULT_DB_PATH, 
        collection_name: str = "",
        model_name: str = DEFAULT_MODEL
    ):
        """
        Inicializa o RetrieverProvider com conexão ao ChromaDB.
        
        Args:
            db_path: Caminho para o diretório do banco ChromaDB persistente.
                     Exemplo: "./chroma_db" ou "a:/RAG-Sandeco/01RAG/chroma_db"
            collection_name: Nome da coleção no ChromaDB. Deve existir previamente.
                           Exemplo: "synthetic_dataset_papers"
            model_name: Nome do modelo SentenceTransformer para gerar embeddings.
                       Default: 'paraphrase-multilingual-MiniLM-L12-v2'
        
        Raises:
            ValueError: Se collection_name estiver vazio ou for None
            Exception: Se a coleção não existir no ChromaDB ou houver erro de conexão
        
        Exemplo:
            >>> retriever = RetrieverProvider(
            ...     db_path="./chroma_db",
            ...     collection_name="my_documents"
            ... )
        """
        if not collection_name:
            raise ValueError("collection_name não pode ser vazio")
        
        self.db_path = db_path
        self.collection_name = collection_name
        self.model_name = model_name
        self.client = None  # chromadb.PersistentClient
        self.collection = None  # chromadb.Collection
        self.modelo = None  # SentenceTransformer
        
        self._initialize()
    
    def _initialize(self):
        """
        Inicializa o cliente ChromaDB, conecta à coleção e carrega o modelo de embeddings.
        
        Este método é chamado automaticamente durante __init__.
        Separa a lógica de inicialização para facilitar testes (pode ser mockado).
        
        Raises:
            Exception: Se houver erro ao conectar ao ChromaDB ou carregar o modelo
        """
        try:
            # Conectar ao ChromaDB em modo persistente
            # Referência: https://docs.trychroma.com/docs/run-chroma/persistent-client
            self.client = chromadb.PersistentClient(path=self.db_path)
            
            # Obter a coleção existente (não cria uma nova)
            # Lança exceção se a coleção não existir
            self.collection = self.client.get_collection(name=self.collection_name)
            
            # Carregar modelo de embeddings
            # Mesmo modelo usado no código de referência para garantir compatibilidade
            print(f"🔄 Carregando modelo de embeddings '{self.model_name}'...")
            self.modelo = SentenceTransformer(self.model_name)
            
            print(f"✅ Conectado à coleção '{self.collection_name}'")
            print(f"📊 Total de documentos: {self.collection.count()}")
            
        except ValueError as e:
            # Coleção não existe
            raise Exception(
                f"Coleção '{self.collection_name}' não encontrada em '{self.db_path}'. "
                f"Certifique-se de que a coleção foi criada primeiro. Erro: {e}"
            )
        except Exception as e:
            # Outros erros (rede, permissões, modelo inválido, etc.)
            raise Exception(f"Erro ao inicializar RetrieverProvider: {e}")
    
    def search(self, query_text: str, n_results: int = 10) -> list[str]:
        """
        Busca chunks de documentos similares à query usando busca vetorial.
        
        Processo:
        1. Gera embedding da query usando SentenceTransformer
        2. Busca os n_results chunks mais similares no ChromaDB (HNSW index)
        3. Retorna apenas os textos dos documentos (não metadados ou distâncias)
        
        Args:
            query_text: Texto da consulta do usuário.
                       Exemplo: "What are the main concepts of RAG?"
            n_results: Número máximo de chunks a retornar.
                      Padrão: 10 (balanceado entre contexto e custo de tokens)
        
        Returns:
            list[str]: Lista de chunks ordenados por similaridade (mais similar primeiro).
                      Retorna lista vazia se houver erro ou nenhum resultado.
        
        Raises:
            Exception: Se houver erro na geração de embeddings ou na query ao ChromaDB
        
        Exemplo:
            >>> retriever = RetrieverProvider(collection_name="docs")
            >>> chunks = retriever.search("Explain vector databases", n_results=3)
            >>> for i, chunk in enumerate(chunks, 1):
            ...     print(f"Chunk {i}: {chunk[:100]}...")
            Chunk 1: Vector databases store data as high-dimensional vectors...
            Chunk 2: ChromaDB is an AI-native open-source vector database...
            Chunk 3: Embeddings capture semantic meaning of text...
        """
        try:
            # 🔍 Logging: Início da busca
            print(f"\n🔎 [RETRIEVAL] Buscando chunks para query: '{query_text[:50]}...'")
            
            # 1. Gerar embedding da query
            # encode() retorna numpy array, convertemos para lista para ChromaDB
            assert self.modelo is not None, "Modelo não foi inicializado"
            query_embedding = self.modelo.encode([query_text])
            
            # 2. Buscar no ChromaDB usando busca vetorial
            # Referência: https://docs.trychroma.com/docs/querying-collections/query-and-get
            assert self.collection is not None, "Collection não foi inicializada"
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results,
                include=['documents', 'distances', 'metadatas']
            )
            
            # 3. Extrair apenas os textos dos documentos
            # results['documents'] é uma lista de listas: [[doc1, doc2, ...]]
            # Pegamos a primeira lista (única query)
            chunks = results['documents'][0] if results['documents'] else []
            
            # ✅ Logging: Resultado da busca
            print(f"✅ [RETRIEVAL] Encontrados {len(chunks)} chunks relevantes")
            
            return chunks
            
        except Exception as e:
            print(f"❌ [RETRIEVAL] Erro na busca: {e}")
            # Retorna lista vazia em caso de erro para não quebrar o pipeline
            return []
    
    def get_collection_info(self) -> dict:
        """
        Retorna informações sobre a coleção atual.
        
        Útil para debugging e validação de configuração.
        
        Returns:
            dict: Informações da coleção com keys:
                - name: Nome da coleção
                - count: Número total de documentos
                - metadata: Metadados da coleção (se houver)
        
        Exemplo:
            >>> retriever = RetrieverProvider(collection_name="docs")
            >>> info = retriever.get_collection_info()
            >>> print(f"Collection: {info['name']}, Documents: {info['count']}")
            Collection: docs, Documents: 139
        """
        if not self.collection:
            return {"name": None, "count": 0, "metadata": {}}
        
        return {
            "name": self.collection.name,
            "count": self.collection.count(),
            "metadata": self.collection.metadata or {}
        }
