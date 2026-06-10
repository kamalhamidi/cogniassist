"""
rag/pipeline.py — Pipeline RAG principal.

Orchestre le flux complet de Retrieval-Augmented Generation :
1. Recherche des documents pertinents (SmartRetriever)
2. Construction du prompt avec contexte (PromptBuilder)
3. Génération de la réponse via Ollama (ChatOllama)
4. Gestion de la mémoire de conversation (ConversationMemory)
5. Résumé de documents et analyse des lacunes
"""

import logging
import time
from typing import Generator, Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from config import settings
from vectorstore.store import VectorStore
from vectorstore.retriever import SmartRetriever, HybridRetriever
from rag.prompt_builder import PromptBuilder
from rag.memory import ConversationMemory

logger = logging.getLogger("cogniassist.rag")


class RAGPipeline:
    """
    Pipeline RAG complet avec Ollama.

    Intègre la recherche sémantique, la construction de prompts,
    la génération de réponses (synchrone et streaming), le résumé
    de documents et l'analyse des lacunes de connaissances.
    """

    def __init__(self) -> None:
        """
        Initialise tous les composants du pipeline.

        Utilise ChatOllama avec mistral:7b et les paramètres des settings.
        Vérifie la disponibilité d'Ollama au démarrage.
        """
        # LLM Ollama
        self.llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.3,
            num_predict=1024,
            top_k=40,
            top_p=0.9,
        )

        # Composants du pipeline
        self._vector_store = VectorStore()
        self._smart_retriever = SmartRetriever(self._vector_store)
        self.retriever = HybridRetriever(
            smart_retriever=self._smart_retriever,
            bm25_index=self._vector_store.bm25_index,
            dense_weight=settings.HYBRID_DENSE_WEIGHT,
            sparse_weight=settings.HYBRID_SPARSE_WEIGHT,
        )
        self.prompt_builder = PromptBuilder()
        self.memory = ConversationMemory(max_messages=10)

        # Vérifier la disponibilité
        self.is_ready: bool = False
        self._check_readiness()

    def _check_readiness(self) -> None:
        """
        Vérifie que Ollama est accessible en envoyant un prompt court.

        Met à jour self.is_ready en conséquence et log le résultat.
        """
        try:
            self.llm.invoke("test")
            self.is_ready = True
            logger.info("Pipeline RAG prêt ✅ (modèle : %s)", settings.ollama_model)
        except Exception as e:
            self.is_ready = False
            logger.error(
                "Pipeline RAG non prêt ❌ — Ollama indisponible : %s. "
                "Lancez : ollama run %s",
                str(e), settings.ollama_model,
            )

    def ask(
        self,
        question: str,
        user_id: str = "default",
        k: int = 5,
        filter_metadata: dict | None = None,
    ) -> dict:
        """
        Exécute une requête RAG complète avec personnalisation.

        Args:
            question: La question de l'utilisateur.
            user_id: Identifiant de l'utilisateur pour la personnalisation.
            k: Nombre de chunks à récupérer.
            filter_metadata: Filtre optionnel sur les métadonnées.

        Returns:
            Dictionnaire avec answer, sources, question, chunks_used,
            model_used, context_length et interaction_id.

        Raises:
            RuntimeError: Si Ollama n'est pas disponible.
        """
        if not self.is_ready:
            raise RuntimeError(
                "Le pipeline RAG n'est pas prêt. "
                f"Veuillez lancer Ollama : ollama run {settings.ollama_model}"
            )

        start_time = time.time()

        # 0. Personnalisation via le profil utilisateur
        from user import get_recommender, get_interaction_history
        rec = get_recommender(user_id)
        params = rec.adapt_rag_parameters()
        k = params["k"]
        user_profile = params["user_profile"]

        # 1. Récupérer les chunks pertinents
        chunks = self.retriever.retrieve(
            question, k=k, filter_metadata=filter_metadata,
        )

        # 2. Construire le contexte
        context = self.retriever.retrieve_with_context_window(question, k=k)

        # 3. Historique de conversation
        chat_history = self.memory.get_formatted_history()

        # 4. Construire le prompt
        prompt = self.prompt_builder.build_rag_prompt(
            question=question,
            context=context,
            chat_history=chat_history,
            user_profile=user_profile,
        )

        # 5. Appeler le LLM
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            answer = response.content
        except Exception as e:
            logger.error("Erreur lors de l'appel au LLM : %s", str(e))
            answer = (
                "Désolé, une erreur est survenue lors de la génération de la réponse. "
                f"Veuillez vérifier qu'Ollama est lancé : ollama run {settings.ollama_model}"
            )

        # 6. Mettre à jour la mémoire
        self.memory.add_user_message(question)
        self.memory.add_assistant_message(answer)

        # 7. Calculer le temps de réponse et sauvegarder dans l'historique
        response_time_ms = int((time.time() - start_time) * 1000)
        history = get_interaction_history(user_id)
        interaction_id = history.save_interaction(
            question=question,
            answer=answer,
            sources=[c.metadata.get("file_name", "") for c in chunks],
            chunks_used=len(chunks),
            response_time_ms=response_time_ms,
        )

        # 8. Construire les sources
        sources = [
            {
                "file_name": c.metadata.get("file_name", "inconnu"),
                "chunk_id": c.metadata.get("chunk_id", ""),
                "score": c.metadata.get("similarity_score", 0.0),
                "preview": c.page_content[:200],
            }
            for c in chunks
        ]

        return {
            "answer": answer,
            "sources": sources,
            "question": question,
            "chunks_used": len(chunks),
            "model_used": settings.ollama_model,
            "context_length": len(context),
            "interaction_id": interaction_id,
        }

    def ask_stream(
        self,
        question: str,
        user_id: str = "default",
        k: int = 5,
    ) -> Generator[str, None, None]:
        """
        Version streaming de ask() pour affichage temps réel dans Streamlit.

        Yield chaque token au fur et à mesure de la génération.
        Met à jour la mémoire après la fin du streaming.

        Args:
            question: La question de l'utilisateur.
            user_id: Identifiant de l'utilisateur pour la personnalisation.
            k: Nombre de chunks à récupérer.

        Yields:
            Tokens de la réponse un par un.

        Raises:
            RuntimeError: Si Ollama n'est pas disponible.
        """
        if not self.is_ready:
            raise RuntimeError(
                "Le pipeline RAG n'est pas prêt. "
                f"Veuillez lancer Ollama : ollama run {settings.ollama_model}"
            )

        # Personnalisation via le profil utilisateur
        from user import get_recommender
        rec = get_recommender(user_id)
        params = rec.adapt_rag_parameters()
        k = params["k"]
        user_profile = params["user_profile"]

        # Construire le prompt
        context = self.retriever.retrieve_with_context_window(question, k=k)
        chat_history = self.memory.get_formatted_history()
        prompt = self.prompt_builder.build_rag_prompt(
            question=question,
            context=context,
            chat_history=chat_history,
            user_profile=user_profile,
        )

        # Streaming token par token
        full_answer_parts: list[str] = []
        try:
            for chunk in self.llm.stream([HumanMessage(content=prompt)]):
                token = chunk.content
                full_answer_parts.append(token)
                yield token
        except Exception as e:
            logger.error("Erreur streaming LLM : %s", str(e))
            error_msg = "Erreur lors de la génération. Vérifiez Ollama."
            full_answer_parts.append(error_msg)
            yield error_msg

        # Mettre à jour la mémoire après streaming
        full_answer = "".join(full_answer_parts)
        self.memory.add_user_message(question)
        self.memory.add_assistant_message(full_answer)

    def summarize_document(self, file_name: str) -> str:
        """
        Génère un résumé automatique d'un document spécifique.

        Récupère tous les chunks du fichier et les concatène,
        puis utilise le LLM pour produire un résumé structuré.

        Args:
            file_name: Nom du fichier à résumer.

        Returns:
            Résumé structuré du document.
        """
        # Rechercher tous les chunks de ce fichier
        chunks = self.retriever.retrieve(
            query=file_name,
            k=20,
            filter_metadata={"file_name": file_name},
        )

        if not chunks:
            return f"Aucun document trouvé avec le nom '{file_name}'."

        # Concaténer le contenu
        content = "\n\n".join(c.page_content for c in chunks)

        # Construire et envoyer le prompt
        prompt = self.prompt_builder.build_summary_prompt(
            document_content=content,
            file_name=file_name,
        )

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            logger.error("Erreur résumé de '%s' : %s", file_name, str(e))
            return f"Erreur lors du résumé de '{file_name}' : {str(e)}"

    def analyze_knowledge_gaps(self, user_goals: str) -> str:
        """
        Analyse les lacunes de connaissances de l'utilisateur.

        Compare les documents disponibles avec les objectifs déclarés
        pour identifier les points forts et les manques.

        Args:
            user_goals: Objectifs de l'utilisateur.

        Returns:
            Analyse textuelle des lacunes.
        """
        stats = self._vector_store.get_collection_stats()
        doc_names = stats.get("unique_documents", [])

        if not doc_names:
            return "Aucun document indexé. Importez des documents pour analyser vos connaissances."

        documents_summary = f"Documents disponibles : {', '.join(doc_names)}"

        prompt = self.prompt_builder.build_knowledge_gap_prompt(
            documents_summary=documents_summary,
            user_goals=user_goals,
        )

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            logger.error("Erreur analyse lacunes : %s", str(e))
            return f"Erreur lors de l'analyse : {str(e)}"

    def clear_memory(self) -> None:
        """Vide la mémoire de conversation."""
        self.memory.clear()
        logger.info("Mémoire de conversation vidée.")

    def get_pipeline_status(self) -> dict:
        """
        Retourne l'état actuel du pipeline pour l'affichage dans l'UI.

        Returns:
            Dictionnaire avec is_ready, llm_model, ollama_url,
            message_count, embedding_model et total_documents_indexed.
        """
        try:
            stats = self._vector_store.get_collection_stats()
            total_indexed = stats.get("total_chunks", 0)
            emb_model = stats.get("embedding_model", "inconnu")
        except Exception:
            total_indexed = 0
            emb_model = "inconnu"

        return {
            "is_ready": self.is_ready,
            "llm_model": settings.ollama_model,
            "ollama_url": settings.ollama_base_url,
            "message_count": self.memory.get_message_count(),
            "embedding_model": emb_model,
            "total_documents_indexed": total_indexed,
        }
