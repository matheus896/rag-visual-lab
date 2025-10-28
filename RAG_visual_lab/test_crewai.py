"""
Teste do AgentRAGCrewAI - Similar ao main_busca.py
Teste do roteador de datasets usando CrewAI
"""
import sys
import os

# Adicionar a pasta pai (01RAG) ao sys.path para encontrar os módulos
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from retriever import Retriever
from augmentation import Augmentation
from generation import Generation
from services.agentic_rag_provider import AgenticRAGProvider

# Query de teste
query = "O que é direito constitucional fala do abandono afetivo?"

print("=" * 80)
print("TESTANDO AgentRAGCrewAI - ROTEADOR DE DATASETS")
print("=" * 80)
print(f"\n🔍 Query: {query}\n")

# Passo 1: Usar o AgentRAGCrewAI para rotear a query e escolher o dataset
print("📋 Passo 1: Roteando query para encontrar dataset apropriado...")
print("-" * 80)

try:
    agent_rag_crew = AgenticRAGProvider()
    dataset_info = agent_rag_crew.query(query)
    
    print("\n✅ Resposta do CrewAI:")
    print(f"   Tipo: {type(dataset_info)}")
    print(f"   Conteúdo: {dataset_info}\n")
    
    # Validar se é um dicionário com as chaves esperadas
    if isinstance(dataset_info, dict):
        dataset_escolhido = dataset_info.get('dataset_name')
        locale = dataset_info.get('locale')
        query_traduzida = dataset_info.get('query')
        
        print(f"   📊 Dataset Selecionado: {dataset_escolhido}")
        print(f"   🌐 Locale: {locale}")
        print(f"   🔄 Query Traduzida: {query_traduzida}\n")
    else:
        print(f"   ⚠️ Resposta não é um dicionário: {dataset_info}\n")
        dataset_escolhido = None
        
except Exception as e:
    print(f"❌ Erro ao executar AgentRAGCrewAI: {str(e)}\n")
    dataset_escolhido = None

# Passo 2: Se houver dataset, continuar com o pipeline RAG
if dataset_escolhido:
    print("\n" + "=" * 80)
    print("CONTINUANDO COM PIPELINE RAG COMPLETO")
    print("=" * 80)
    
    try:
        # Inicializar componentes RAG
        retriever = Retriever(collection_name=dataset_escolhido)
        augmentation = Augmentation()
        generation = Generation(model="gemini-2.5-flash-lite")
        
        print(f"\n📚 Passo 2: Buscando documentos do dataset '{dataset_escolhido}'...")
        print("-" * 80)
        
        # Buscar documentos
        chunks = retriever.search(query, n_results=10, show_metadata=False)
        print(f"✅ Encontrados {len(chunks)} chunks relevantes\n") # type: ignore
        
        # Gerar prompt aumentado
        print("📝 Passo 3: Gerando prompt aumentado...")
        print("-" * 80)
        prompt = augmentation.generate_prompt(query, chunks)
        print(f"✅ Prompt gerado (primeiros 200 caracteres): {prompt[:200]}...\n")
        
        # Gerar resposta final
        print("🤖 Passo 4: Gerando resposta com o modelo Gemini...")
        print("-" * 80)
        response = generation.generate(prompt)
        
        print("\n✅ RESPOSTA FINAL:")
        print("-" * 80)
        print(response)
        print("-" * 80)
        
    except Exception as e:
        print(f"❌ Erro no pipeline RAG: {str(e)}\n")
        import traceback
        traceback.print_exc()
else:
    print("\n⚠️ Não foi possível determinar o dataset. Pipeline RAG não executado.")

print("\n" + "=" * 80)
print("TESTE CONCLUÍDO")
print("=" * 80)
