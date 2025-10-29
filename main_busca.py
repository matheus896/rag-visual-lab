import sys
import os

# Adiciona o diretório raiz ao PYTHONPATH
_current_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = _current_dir
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from retriever import Retriever
from augmentation import Augmentation
from generation import Generation

retriever = Retriever(collection_name="synthetic_dataset_papers")
augmentation = Augmentation()
generation = Generation(model="gemini-2.5-flash")

query = "What's a synthetic dataset?"

# Buscar documentos
chunks = retriever.search(query, n_results=10, show_metadata=False)
prompt = augmentation.generate_prompt(query, chunks)

# Gerar resposta
response = generation.generate(prompt)

print(response)       
