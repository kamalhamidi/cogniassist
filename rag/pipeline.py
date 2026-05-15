"""
rag/pipeline.py — Pipeline RAG principal.

Orchestre le flux complet de Retrieval-Augmented Generation :
1. Réception de la question utilisateur
2. Recherche des documents pertinents (retrieval)
3. Construction du prompt avec contexte
4. Génération de la réponse via LLM (Ollama)
5. Sauvegarde dans la mémoire de conversation
"""

from typing import Optional

from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage

from vectorstore.retriever import DocumentRetriever
from rag.prompt_builder import PromptBuilder
from rag.memory import ConversationMemory


class RAGPipeline:
    """
    Pipeline RAG complet.

    Intègre la recherche de documents, la construction de prompts
    et la génération de réponses via un LLM Ollama.
    """

    def __init__(
        self,
        retriever: Optional[DocumentRetriever] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        memory: Optional[ConversationMemory] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.3,
    ) -> None:
        """
        Initialise le pipeline RAG.

        Args:
            retriever: Retriever pour la recherche de documents.
            prompt_builder: Constructeur de prompts.
            memory: Mémoire de conversation.
            model_name: Nom du modèle Ollama.
            temperature: Température de génération.
        """
        from config import settings

        self.retriever = retriever or DocumentRetriever()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.memory = memory or ConversationMemory()

        model = model_name or settings.ollama_model
        self.llm = ChatOllama(
            model=model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
        )

    def query(
        self,
        question: str,
        user_context: Optional[str] = None,
        n_docs: Optional[int] = None,
    ) -> dict:
        """
        Exécute une requête RAG complète.

        Args:
            question: La question de l'utilisateur.
            user_context: Contexte utilisateur optionnel (profil, préférences).
            n_docs: Nombre de documents à récupérer.

        Returns:
            Dictionnaire avec la réponse, les sources et les métadonnées.
        """
        # 1. Recherche des documents pertinents
        retrieved_docs = self.retriever.retrieve(question, n_results=n_docs)
        context = self.retriever.retrieve_with_context(question, n_results=n_docs)

        # 2. Récupérer l'historique de conversation
        history = self.memory.get_history()

        # 3. Construire le prompt
        system_prompt = self.prompt_builder.build_system_prompt(
            context=context,
            user_context=user_context,
        )
        conversation_prompt = self.prompt_builder.build_conversation_prompt(
            question=question,
            history=history,
        )

        # 4. Générer la réponse
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=conversation_prompt),
        ]
        response = self.llm.invoke(messages)
        answer = response.content

        # 5. Sauvegarder dans la mémoire
        self.memory.add_exchange(question=question, answer=answer)

        # 6. Construire le résultat
        sources = [
            {
                "content": doc.content[:200] + "...",
                "score": doc.score,
                "metadata": doc.metadata,
            }
            for doc in retrieved_docs
        ]

        return {
            "answer": answer,
            "sources": sources,
            "num_sources": len(sources),
            "model": self.llm.model,
        }

    def reset_memory(self) -> None:
        """Réinitialise la mémoire de conversation."""
        self.memory.clear()
