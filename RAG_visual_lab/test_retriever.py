"""
Teste rápido do RetrieverProvider com ChromaDB real
"""
import sys
sys.path.insert(0, '.')

from services.retriever_provider import RetrieverProvider

print("="*60)
print("TESTE: RetrieverProvider com ChromaDB Real")
print("="*60)

# Inicializa o RetrieverProvider
print("\n1. Inicializando RetrieverProvider...")
retriever = RetrieverProvider(
    db_path='./chroma_db',
    collection_name='synthetic_dataset_papers'
)

# Obtém informações da coleção
print("\n2. Informações da coleção:")
info = retriever.get_collection_info()
print(f"   - Nome: {info['name']}")
print(f"   - Total de documentos: {info['count']}")

# Testa busca semântica
print("\n3. Testando busca semântica...")
query = "What is RAG and how does it work?"
chunks = retriever.search(query, n_results=5)

print(f"   - Query: '{query}'")
print(f"   - Chunks retornados: {len(chunks)}")

if chunks:
    print("\n4. Preview dos chunks encontrados:")
    for i, chunk in enumerate(chunks, 1):
        preview = chunk[:150].replace('\n', ' ')
        print(f"   [{i}] {preview}...")
else:
    print("\n4. ⚠️ Nenhum chunk encontrado")

print("\n" + "="*60)
print("✅ TESTE CONCLUÍDO")
print("="*60)
