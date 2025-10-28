"""
Augmentation Provider - Orquestrador de RAG com Memória
========================================================

Este módulo implementa o componente "A" (Augmentation) do RAG,
responsável por combinar o contexto de documentos recuperados (chunks)
com a memória conversacional persistente (Redis) para criar prompts
enriquecidos que serão enviados ao LLM.

Arquitetura:
    Query → Retriever (chunks) → AugmentationProvider (chunks + memory) → LLM

Funcionalidades:
    - Combina chunks recuperados do ChromaDB com histórico do Redis
    - Formata prompts seguindo o padrão do projeto de referência
    - Gerencia a persistência de prompts e respostas na memória
    - Prioriza informações: query > chunks > histórico
"""

from typing import List, Dict, Any, Optional
from services.memory_provider import MemoryProvider


class AugmentationProvider:
    """
    Orquestra a combinação de memória conversacional e contexto de documentos
    para criar prompts aumentados para o LLM.
    
    Esta classe é inspirada na implementação de referência em code-sandeco-rag-memory.txt,
    adaptada para a arquitetura do RAG Visual Lab.
    
    Attributes:
        talk_id (str): Identificador único da conversa
        memory_provider (MemoryProvider): Instância do provedor de memória Redis
        last_prompt (str): Último prompt gerado (usado para persistir na memória)
    
    Example:
        ```python
        augmenter = AugmentationProvider(talk_id="user-session-123")
        
        # Após recuperar chunks do Retriever
        chunks = ["chunk 1", "chunk 2", "chunk 3"]
        
        # Gera prompt combinado
        prompt = augmenter.generate_prompt(
            query="Qual é o tema principal?",
            chunks=chunks
        )
        
        # Envia para LLM e salva resposta
        llm_response = llm_function(prompt)
        augmenter.add_response_to_memory(llm_response)
        ```
    """
    
    def __init__(self, talk_id: str):
        """
        Inicializa o AugmentationProvider com um identificador de conversa.
        
        Args:
            talk_id: Identificador único da conversa (UUID ou string única)
            
        Raises:
            ValueError: Se talk_id for vazio ou None
        """
        if not talk_id:
            raise ValueError("talk_id não pode ser vazio.")
        
        self.talk_id = talk_id
        self.memory_provider = MemoryProvider(talk_id=self.talk_id)
        self.last_prompt = ""
        self.last_query = ""
    
    def generate_prompt(self, query: str, chunks: List[str]) -> str:
        """
        Gera um prompt combinado com histórico do Redis e chunks do ChromaDB.
        
        Este método segue o padrão da implementação de referência:
        1. Recupera o histórico conversacional do Redis (últimas 5 mensagens)
        2. Formata os chunks recuperados do ChromaDB
        3. Constrói o prompt final com delimitadores XML-like (<query>, <chunks>, <historico>)
        4. Define prioridade de informações: query=1, chunks=2, historico=3
        
        Args:
            query: Pergunta do usuário (string)
            chunks: Lista de chunks de texto recuperados do ChromaDB
            
        Returns:
            Prompt formatado pronto para ser enviado ao LLM
            
        Example:
            ```python
            prompt = augmenter.generate_prompt(
                query="O que é RAG?",
                chunks=["RAG significa...", "O conceito de RAG..."]
            )
            # Prompt contém query + chunks + histórico formatados
            ```
        """
        # Armazena a query original (apenas texto, sem chunks/histórico)
        self.last_query = query
        
        # 1. Recuperar histórico da conversa do Redis
        history = self.memory_provider.get_conversation() or []
        
        # Formata o histórico para ser incluído no prompt
        # Pegamos as últimas 5 mensagens e invertemos para ordem cronológica
        if history:
            history_text = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in reversed(history[:5])
            ])
        else:
            history_text = "Nenhum histórico disponível."
        
        # 2. Formatar os chunks recuperados
        separador = "\n\n------------------------\n\n"
        chunks_formatados = f"Conhecimento\n------------------------\n\n{separador.join(chunks)}"
        
        # 3. Construir o prompt final seguindo o template do projeto de referência
        # Padrão: Instruções + delimitadores <chunks> <query> <historico>
        # Prioridade: query=1, chunks=2, historico=3
        self.last_prompt = f"""Responda em pt-br e em markdown, a query do usuário delimitada por <query> 
usando apenas o conhecimento dos chunks delimitados por <chunks> 
e tenha em mente o historico das conversas anteriores delimitado por <historico>. 
Combine as informações para responder a query de forma unificada. A prioridade
das informações são: query=1, chunks=2, historico=3.

Se por acaso o conhecimento não for suficiente para responder a query, 
responda apenas que não temos conhecimento suficiente para responder 
a Pergunta.

<chunks>
{chunks_formatados}
</chunks>        

<query>
{query}
</query>

<historico>
{history_text}
</historico>

"""
        
        return self.last_prompt
    
    def add_response_to_memory(self, llm_response: str) -> bool:
        """
        Salva a query original e a resposta do LLM no Redis.
        
        Este método persiste a interação completa na memória,
        permitindo que conversas futuras tenham acesso a este contexto.
        
        ✅ CORRIGIDO: Salva apenas a QUERY original (não o prompt completo com chunks),
        garantindo que a UI não exiba chunks/histórico quando recarrega o histórico.
        
        O prompt completo é salvo como mensagem do "user" (contém a query original),
        e a resposta do LLM é salva como mensagem do "assistant".
        
        Args:
            llm_response: Resposta gerada pelo LLM
            
        Returns:
            True se salvou com sucesso, False caso contrário
            
        Example:
            ```python
            # Após gerar resposta com LLM
            response = llm_function(prompt)
            success = augmenter.add_response_to_memory(response)
            
            if success:
                print("Conversa salva no Redis!")
            ```
        """
        if not self.last_query:
            return False
        
        try:
            # Salva APENAS a query (não self.last_prompt que contém chunks)
            # Isso garante que ao recarregar o histórico, a UI exiba apenas a pergunta
            self.memory_provider.add_message("user", self.last_query)
            
            # Salva a resposta do LLM como mensagem do assistente
            self.memory_provider.add_message("assistant", llm_response)
            
            print(f"💬 Memória de '{self.talk_id[:8]}' atualizada com a mensagem de 'user'.")
            print(f"💬 Memória de '{self.talk_id[:8]}' atualizada com a mensagem de 'assistant'.")
            
            return True
        
        except Exception as e:
            print(f"Erro ao adicionar memória: {str(e)}")
            return False
    
    def clear_memory(self):
        """
        Limpa todo o histórico da conversa no Redis.
        
        Útil para reiniciar uma conversa ou limpar dados de teste.
        
        Example:
            ```python
            augmenter.clear_memory()
            print("Histórico limpo!")
            ```
        """
        self.memory_provider.delete_conversation()
    
    def get_conversation(self) -> Optional[List[Dict[str, Any]]]:
        """
        Recupera o histórico completo da conversa do Redis.
        
        Returns:
            Lista de mensagens (cada mensagem é um dict com 'role' e 'content')
            Retorna None se não houver histórico
            
        Example:
            ```python
            history = augmenter.get_conversation()
            if history:
                for msg in history:
                    print(f"{msg['role']}: {msg['content']}")
            ```
        """
        return self.memory_provider.get_conversation()
