"""
Agentic RAG Provider 
====================

Serviço que encapsula a lógica de roteamento de queries usando CrewAI.
O agente decide qual dataset é mais apropriado para uma consulta do usuário.

Arquitetura:
- Encapsulação completa de CrewAI (nenhuma dependência na UI)
- Retorna dicionário limpo com {dataset_name, locale, query}
- Captura logs detalhados do agente para fins didáticos
- Tratamento robusto de erros e parsing JSON
"""

import json
import os
import sys
from io import StringIO
from contextlib import redirect_stdout
from typing import Optional, Dict, Tuple

from crewai import Agent, Task, Crew, Process
from .datasets_provider import DatasetsProvider
from dotenv import load_dotenv

load_dotenv()


class AgenticRAGProvider:
    """
    Provedor de RAG Agente que roteia queries para datasets apropriados.
    
    Responsabilidades:
    1. Gerenciar DatasetsProvider
    2. Definir Agent, Task e Crew do CrewAI
    3. Executar roteamento e capturar logs
    4. Parsear resposta JSON e retornar dicionário limpo
    """
    
    def __init__(self):
        """Inicializa o provedor com lista de datasets."""
        self.datasets_provider = DatasetsProvider()
        self.llm = "gemini/gemini-2.5-flash-lite"
        self.agent = None
        self.task = None
        self.crew = None
        self.last_logs = ""
    
    def _create_agent(self) -> Agent:
        """
        Cria o agente roteador de RAG.
        
        Responsabilidade: Analisar a query e escolher o dataset apropriado.
        """
        return Agent(
            role="Agente Roteador de RAG",
            goal=(
                "Decidir, com base na solicitação do usuário, "
                "qual base vetorial de conhecimento deve ser usada "
                "para responder da forma mais adequada."
            ),
            backstory=(
                "Você é um especialista em recuperação de informação e agente RAG. "
                "Sua função é interpretar a solicitação e determinar de forma precisa qual "
                "dataset de conhecimento deve ser consultado. "
            ),
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )
    
    def _create_task(self, query: str, agent: Agent) -> Task:
        """
        Cria a tarefa de roteamento.
        
        Responsabilidade: Instruir o agente a escolher um dataset e retornar JSON limpo.
        
        Args:
            query: Consulta do usuário
            agent: Agent que executará a tarefa
            
        Returns:
            Task configurada
        """
        datasets_desc = self.datasets_provider.get_dataset_description()
        
        task_description = f"""
Com base na solicitação do usuário "{query}" e na lista de datasets abaixo, escolha o mais apropriado.

Datasets disponíveis:
{datasets_desc}

Eu quero como saída um JSON com as seguintes informações do dataset escolhido:
- dataset_name: O nome exato do dataset
- locale: O locale do dataset (en ou pt-br)
- query: A query original, traduzida para o locale do dataset se necessário

Exemplo de saída:
{{"dataset_name": "direito_constitucional", "locale": "pt-br", "query": "O que é direito constitucional fala do abandono afetivo?"}}

É IMPERATIVO:
- Retorne APENAS o objeto JSON, sem nenhum texto adicional, explicação ou formatação markdown
- Não adicione ```json ou ``` envolvendo a resposta
- Não adicione caracteres especiais ou caracteres de escape
- Não escreva nada além do JSON de resposta
"""
        
        return Task(
            description=task_description,
            expected_output="Um objeto JSON com dataset_name, locale e query. Nada mais.",
            agent=agent
        )
    
    def query(self, query: str) -> Optional[Dict]:
        """
        Roteia uma query para o dataset apropriado (interface pública).
        
        Este método é a interface principal chamada externamente.
        Internamente chama route_query() para fazer o trabalho.
        
        Args:
            query: Consulta do usuário
            
        Returns:
            Dicionário com {dataset_name, locale, query} ou None se falhar
        """
        return self.route_query(query)
    
    def route_query(self, query: str) -> Optional[Dict]:
        """
        Roteia uma query para o dataset apropriado (implementação interna).
        
        Fluxo:
        1. Cria Agent e Task
        2. Executa Crew.kickoff() com captura de logs
        3. Parseia resposta JSON
        4. Retorna dicionário com {dataset_name, locale, query} ou None em caso de erro
        
        Args:
            query: Consulta do usuário
            
        Returns:
            Dicionário com {dataset_name, locale, query} ou None se falhar
            
        Raises:
            Nenhuma - trata todos os erros internamente
        """
        response_text = ""  # Inicializar para evitar UnboundLocalError
        
        try:
            # Criar Agent e Task
            agent = self._create_agent()
            task = self._create_task(query, agent)
            
            # Criar Crew com processamento sequencial
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True
            )
            
            # Executar crew e capturar output
            print(f"🤖 [AGENTIC] Roteando query: '{query[:50]}...'")
            result = crew.kickoff()
            
            # Converter resultado para string se necessário
            response_text = str(result)
            
            print(f"✅ [AGENTIC] Resposta do agente recebida ({len(response_text)} chars)")
            
            # Tentar parsear JSON
            # O agente pode retornar a resposta como .raw ou como string normal
            if hasattr(result, 'raw'):
                response_text = result.raw
            
            # Remover possíveis delimitadores markdown se presentes
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parsear JSON
            parsed_response = json.loads(response_text)
            
            print(f"✅ [AGENTIC] JSON parseado com sucesso")
            print(f"   └─ Dataset: {parsed_response.get('dataset_name', 'N/A')}")
            print(f"   └─ Locale: {parsed_response.get('locale', 'N/A')}")
            
            return parsed_response
            
        except json.JSONDecodeError as e:
            print(f"❌ [AGENTIC] Erro ao parsear JSON da resposta do agente")
            print(f"   └─ Raw response: {response_text[:200] if response_text else 'N/A'}")
            print(f"   └─ Erro: {str(e)}")
            return None
            
        except Exception as e:
            print(f"❌ [AGENTIC] Erro ao executar roteamento: {str(e)}")
            import traceback
            traceback.print_exc()
            return None