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
from langchain_core.messages import HumanMessage, SystemMessage

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

        # ── Layer 2 — Identité (second cerveau) ──
        try:
            from user.style_analyzer import StyleAnalyzer
            from user.belief_extractor import BeliefExtractor
            from user.identity_prompt_builder import IdentityPromptBuilder
            from user import get_user_manager

            self.style_analyzer = StyleAnalyzer()
            self.belief_extractor = BeliefExtractor(
                llm=self.llm,
                embedder=self._vector_store.embedding_manager,
            )
            self.identity_builder = IdentityPromptBuilder(
                style_analyzer=self.style_analyzer,
                belief_extractor=self.belief_extractor,
                profile_manager=get_user_manager("default"),
            )
        except Exception as e:
            logger.error("Initialisation de la couche identité échouée : %s", e)
            self.style_analyzer = None
            self.belief_extractor = None
            self.identity_builder = None

        self.identity_mode_enabled: bool = settings.IDENTITY_MODE_DEFAULT

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

    def enable_identity_mode(self, enabled: bool) -> None:
        """Active ou désactive le mode identité (second cerveau)."""
        self.identity_mode_enabled = enabled
        logger.info("Mode identité %s.", "activé" if enabled else "désactivé")

    def _use_identity_mode(self) -> bool:
        """Indique si le mode identité doit être utilisé pour cette requête."""
        return bool(
            self.identity_mode_enabled
            and self.identity_builder is not None
            and self.identity_builder.is_identity_mode_ready()
        )

    def _build_messages(
        self,
        question: str,
        context: str,
        chat_history: str,
        user_profile: str,
    ) -> list:
        """
        Construit la liste de messages pour le LLM.

        En mode identité : prompt système « second cerveau » + message utilisateur.
        Sinon : comportement standard (prompt RAG unique) — aucun changement.
        """
        if self._use_identity_mode():
            logger.info("Mode identité (second cerveau) utilisé pour cette requête.")
            system_prompt = self.identity_builder.build_system_prompt(question)
            user_content = (
                f"Historique de la conversation :\n{chat_history or '(aucun)'}\n\n"
                f"Contexte issu de ses écrits et documents :\n{context}\n\n"
                f"Question :\n{question}"
            )
            return [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content),
            ]

        prompt = self.prompt_builder.build_rag_prompt(
            question=question,
            context=context,
            chat_history=chat_history,
            user_profile=user_profile,
        )
        return [HumanMessage(content=prompt)]

    def save_style_correction(
        self, query: str, generated: str, corrected: str,
    ) -> None:
        """
        Enregistre une correction stylistique de l'utilisateur et recalcule
        le profil de style en intégrant les corrections.
        """
        from user.identity_models import StyleCorrection
        from user.db import get_session

        session = get_session()
        try:
            rec = StyleCorrection(
                query=query, generated=generated, corrected=corrected,
            )
            session.add(rec)
            session.commit()
            logger.info("Correction stylistique enregistrée.")
        except Exception as e:
            session.rollback()
            logger.error("Erreur enregistrement correction : %s", e)

        # Recalcule (optionnel, non-bloquant) le profil de style
        try:
            if self.style_analyzer is not None:
                texts = [
                    c["text"]
                    for c in self._vector_store.get_personal_writing_chunks()
                ]
                corrections = [r.corrected for r in session.query(StyleCorrection).all()]
                all_texts = texts + corrections
                if all_texts:
                    metrics = self.style_analyzer.analyze(all_texts)
                    self.style_analyzer.save_profile(metrics, session)
        except Exception as e:
            logger.debug("Recalcul du style après correction échoué : %s", e)

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

        # 4. Construire les messages (mode identité ou standard)
        messages = self._build_messages(
            question=question,
            context=context,
            chat_history=chat_history,
            user_profile=user_profile,
        )

        # 5. Appeler le LLM
        try:
            response = self.llm.invoke(messages)
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
        source_names = [c.metadata.get("file_name", "") for c in chunks]
        interaction_id = history.save_interaction(
            question=question,
            answer=answer,
            sources=source_names,
            chunks_used=len(chunks),
            response_time_ms=response_time_ms,
        )

        # 7.5 — Évolution du profil ACPE (non-bloquant)
        try:
            from user.profile import UserProfileManager
            mgr = UserProfileManager(user_id)
            profile_data = mgr.get_profile()
            if profile_data.get("adaptive_learning_enabled", True):
                from user.profile_evolution import ProfileEvolutionEngine
                evo = ProfileEvolutionEngine(user_id)
                evo.analyze_interaction(
                    question=question,
                    answer=answer,
                    sources=source_names,
                    feedback=None,
                    response_time_ms=response_time_ms,
                )
        except Exception as e:
            logger.debug("ACPE evolution (non-bloquant) : %s", e)

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

        # Construire les messages (mode identité ou standard)
        context = self.retriever.retrieve_with_context_window(question, k=k)
        chat_history = self.memory.get_formatted_history()
        messages = self._build_messages(
            question=question,
            context=context,
            chat_history=chat_history,
            user_profile=user_profile,
        )

        # Streaming token par token
        full_answer_parts: list[str] = []
        try:
            for chunk in self.llm.stream(messages):
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

        # Sauvegarder l'interaction dans l'historique
        try:
            from user import get_interaction_history
            history = get_interaction_history(user_id)
            history.save_interaction(
                question=question,
                answer=full_answer,
                sources=[],
                chunks_used=0,
            )
        except Exception:
            pass

        # Évolution du profil ACPE (non-bloquant)
        try:
            from user.profile import UserProfileManager
            mgr = UserProfileManager(user_id)
            profile_data = mgr.get_profile()
            if profile_data.get("adaptive_learning_enabled", True):
                from user.profile_evolution import ProfileEvolutionEngine
                evo = ProfileEvolutionEngine(user_id)
                evo.analyze_interaction(
                    question=question,
                    answer=full_answer,
                    sources=[],
                    feedback=None,
                )
        except Exception as e:
            logger.debug("ACPE evolution streaming (non-bloquant) : %s", e)

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
