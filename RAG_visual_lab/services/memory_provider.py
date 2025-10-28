"""
Memory Provider
===============

Gerenciador de memória conversacional usando Redis.

Este módulo fornece a classe MemoryProvider que gerencia o histórico
de conversas em um banco de dados Redis, permitindo persistência entre
sessões e suporte a múltiplas conversas simultâneas.

Funcionalidades:
- Adicionar mensagens ao histórico (upsert automático)
- Recuperar histórico completo de uma conversa
- Deletar conversas
- Expiração automática de conversas antigas (24 horas)
"""

import redis
import redis.exceptions
import json
from typing import List, Dict, Any, Optional


class MemoryProvider:
    """
    Gerenciador de memória conversacional com Redis.
    
    Esta classe encapsula toda a lógica de interação com Redis para
    armazenar e recuperar históricos de conversa. Cada conversa é
    identificada por um talk_id único.
    
    Características:
    - Operação upsert: cria ou atualiza conversas automaticamente
    - Expiração automática: conversas expiram após 24 horas de inatividade
    - Persistência: dados sobrevivem ao reinício da aplicação
    - Mensagens mais recentes primeiro: novo conteúdo é inserido no início da lista
    
    Example:
        >>> memory = MemoryProvider(talk_id="user-123")
        >>> memory.add_message("user", "Olá!")
        >>> memory.add_message("assistant", "Como posso ajudar?")
        >>> history = memory.get_conversation()
        >>> print(history)
        [
            {"role": "assistant", "content": "Como posso ajudar?"},
            {"role": "user", "content": "Olá!"}
        ]
    """
    
    DEFAULT_EXPIRATION_SECONDS = 24 * 60 * 60  # 24 horas

    def __init__(
        self, 
        talk_id: str,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        expiration_seconds: int = DEFAULT_EXPIRATION_SECONDS
    ):
        """
        Inicializa o provedor de memória.

        Args:
            talk_id: Identificador único da conversa
            host: Host do servidor Redis (padrão: localhost)
            port: Porta do servidor Redis (padrão: 6379)
            db: Número do database Redis (padrão: 0)
            expiration_seconds: Tempo em segundos para expiração da conversa (padrão: 24h)
        
        Raises:
            redis.exceptions.ConnectionError: Se não conseguir conectar ao Redis
        """
        self.talk_id = talk_id
        self.expiration = expiration_seconds
        
        # Inicializa cliente Redis com decode_responses=True para retornar strings
        # ao invés de bytes, simplificando o manuseio de dados JSON
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True
        )
        
        # Verifica conectividade (importante para fail-fast em produção)
        try:
            self.redis_client.ping()
        except redis.exceptions.ConnectionError as e:
            raise redis.exceptions.ConnectionError(
                f"Falha ao conectar ao Redis em {host}:{port}. "
                f"Certifique-se de que o servidor Redis está rodando. Erro: {e}"
            )

    def _get_key(self, talk_id: str) -> str:
        """
        Gera a chave padronizada do Redis para uma conversa.
        
        Args:
            talk_id: Identificador da conversa
            
        Returns:
            Chave formatada no padrão 'conversation:{talk_id}'
        """
        return f"conversation:{talk_id}"

    def add_message(self, role: str, message: str) -> None:
        """
        Adiciona uma nova mensagem ao histórico da conversa.

        Este método implementa uma operação de "upsert":
        - Se a conversa não existe, ela é criada com esta mensagem
        - Se já existe, a mensagem é adicionada ao histórico existente
        
        A mensagem mais recente é sempre inserida no INÍCIO da lista,
        mantendo um histórico cronológico reverso (útil para LLMs que
        processam contexto recente primeiro).
        
        A expiração da chave é RESETADA a cada nova mensagem, garantindo
        que conversas ativas nunca expirem.

        Args:
            role: Papel do emissor da mensagem ('user', 'assistant', 'system')
            message: Conteúdo textual da mensagem

        Example:
            >>> memory.add_message("user", "Qual é a capital do Brasil?")
            >>> memory.add_message("assistant", "A capital do Brasil é Brasília.")
        """
        key = self._get_key(self.talk_id)
        
        # 1. Tenta obter a conversa existente
        existing_history_json = self.redis_client.get(key)
        
        # 2. Se não existir, começa com lista vazia. Se existir, deserializa.
        if existing_history_json:
            history = json.loads(existing_history_json) # type: ignore
        else:
            history = []
            
        # 3. Adiciona a nova mensagem no INÍCIO (index 0)
        # Isso mantém as mensagens mais recentes primeiro
        history.insert(0, {"role": role, "content": message})
        
        # 4. Serializa e salva de volta no Redis, resetando a expiração
        # A flag 'ex' define o tempo de expiração em segundos
        updated_history_json = json.dumps(history)
        self.redis_client.set(key, updated_history_json, ex=self.expiration)

    def get_conversation(self) -> Optional[List[Dict[str, Any]]]:
        """
        Recupera o histórico completo da conversa.
        
        Returns:
            Lista de dicionários com as mensagens, ou None se a conversa não existir.
            Cada mensagem tem a estrutura: {"role": str, "content": str}
            As mensagens estão ordenadas da mais recente para a mais antiga.

        Example:
            >>> history = memory.get_conversation()
            >>> if history:
            ...     for msg in history:
            ...         print(f"{msg['role']}: {msg['content']}")
        """
        key = self._get_key(self.talk_id)
        history_json = self.redis_client.get(key)
        
        if history_json:
            return json.loads(history_json) # type: ignore
        
        return None

    def delete_conversation(self) -> None:
        """
        Deleta o histórico da conversa do Redis.
        
        Esta operação é irreversível. Use com cautela.
        É útil para implementar funcionalidades de "limpar histórico"
        ou para cumprir requisitos de privacidade (LGPD/GDPR).

        Example:
            >>> memory.delete_conversation()
            # A conversa foi permanentemente removida
        """
        key = self._get_key(self.talk_id)
        self.redis_client.delete(key)
