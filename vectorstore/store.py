"""
vectorstore/store.py — Gestion de la base vectorielle ChromaDB.

Fournit une interface complète pour créer, alimenter, rechercher et
gérer une collection ChromaDB persistante. Gère le batching, la
déduplication et les statistiques de la collection.
"""

import logging
import uuid
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain.schema import Document

from config import settings
from vectorstore.embedder import EmbeddingManager

logger = logging.getLogger("cogniassist.vectorstore")

# Taille maximale d'un batch pour l'ajout à ChromaDB
BATCH_SIZE = 50


class VectorStore:
    """
    Interface de gestion pour ChromaDB avec embeddings intégrés.

    Gère la persistance, l'ajout par batch avec déduplication,
    la recherche par similarité, et les statistiques de collection.
    """

    def __init__(self, persist_directory: Optional[str] = None) -> None:
        """
        Initialise le client ChromaDB persistant et le gestionnaire d'embeddings.

        Args:
            persist_directory: Répertoire de persistance ChromaDB.
                              Si None, utilise settings.chroma_persist_path.
        """
        if persist_directory is None:
            persist_directory = str(settings.chroma_persist_path)

        self.persist_directory = persist_directory
        self.collection_name = "cogniassist_documents"

        # Créer le répertoire si nécessaire
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialiser le client ChromaDB (télémétrie désactivée)
        self._client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Gestionnaire d'embeddings
        self.embedding_manager = EmbeddingManager()

        # Obtenir ou créer la collection
        self.collection = self._get_or_create_collection()

        logger.info(
            "VectorStore initialisé — collection '%s' (%d chunks), persist : %s",
            self.collection_name, self.collection.count(), self.persist_directory,
        )

    def _get_or_create_collection(self) -> chromadb.Collection:
        """
        Récupère la collection existante ou la crée si elle n'existe pas.

        Utilise la similarité cosinus comme métrique de distance.

        Returns:
            Collection ChromaDB prête à l'emploi.
        """
        existing_names = [c.name for c in self._client.list_collections()]

        if self.collection_name in existing_names:
            logger.debug("Collection '%s' existante chargée.", self.collection_name)
        else:
            logger.info("Collection '%s' créée.", self.collection_name)

        return self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, chunks: list[Document]) -> int:
        """
        Ajoute des chunks LangChain Document à ChromaDB.

        Étapes :
        1. Extraction des textes et métadonnées
        2. Génération des embeddings via EmbeddingManager
        3. Création d'IDs uniques (chunk_id ou uuid4)
        4. Vérification des doublons (chunk_id déjà existant)
        5. Ajout par batches de 50 pour éviter les problèmes mémoire

        Args:
            chunks: Liste de Documents LangChain à indexer.

        Returns:
            Nombre de chunks effectivement ajoutés (hors doublons).
        """
        if not chunks:
            return 0

        # Extraire textes, métadonnées et IDs
        texts: list[str] = []
        metadatas: list[dict] = []
        ids: list[str] = []

        for chunk in chunks:
            chunk_id = chunk.metadata.get("chunk_id", str(uuid.uuid4()))
            ids.append(chunk_id)
            texts.append(chunk.page_content)
            # ChromaDB n'accepte que str/int/float/bool dans les métadonnées
            clean_meta = {
                k: v for k, v in chunk.metadata.items()
                if isinstance(v, (str, int, float, bool))
            }
            metadatas.append(clean_meta)

        # Vérifier les doublons
        new_ids, new_texts, new_metas = self._filter_duplicates(ids, texts, metadatas)

        if not new_ids:
            logger.info("Aucun nouveau chunk à ajouter (tous des doublons).")
            return 0

        # Générer les embeddings
        logger.info("Génération des embeddings pour %d chunk(s)…", len(new_texts))
        embeddings = self.embedding_manager.embed_documents(new_texts)

        # Ajouter par batches
        total_added = 0
        for i in range(0, len(new_ids), BATCH_SIZE):
            batch_end = min(i + BATCH_SIZE, len(new_ids))
            self.collection.add(
                ids=new_ids[i:batch_end],
                documents=new_texts[i:batch_end],
                embeddings=embeddings[i:batch_end],
                metadatas=new_metas[i:batch_end],
            )
            batch_count = batch_end - i
            total_added += batch_count
            logger.debug("Batch %d–%d ajouté (%d chunks)", i, batch_end, batch_count)

        logger.info(
            "%d chunk(s) ajouté(s) à la collection '%s' (%d doublons ignorés)",
            total_added, self.collection_name, len(ids) - total_added,
        )
        return total_added

    def _filter_duplicates(
        self,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict],
    ) -> tuple[list[str], list[str], list[dict]]:
        """
        Filtre les chunks dont l'ID existe déjà dans la collection.

        Args:
            ids: Liste des IDs candidats.
            texts: Textes correspondants.
            metadatas: Métadonnées correspondantes.

        Returns:
            Tuple (new_ids, new_texts, new_metadatas) sans doublons.
        """
        if self.collection.count() == 0:
            return ids, texts, metadatas

        # Vérifier quels IDs existent déjà
        try:
            existing = self.collection.get(ids=ids, include=[])
            existing_ids = set(existing["ids"]) if existing and existing.get("ids") else set()
        except Exception:
            existing_ids = set()

        if not existing_ids:
            return ids, texts, metadatas

        new_ids, new_texts, new_metas = [], [], []
        for cid, text, meta in zip(ids, texts, metadatas):
            if cid not in existing_ids:
                new_ids.append(cid)
                new_texts.append(text)
                new_metas.append(meta)

        if len(existing_ids & set(ids)) > 0:
            logger.debug(
                "%d doublon(s) détecté(s) et ignoré(s)",
                len(ids) - len(new_ids),
            )

        return new_ids, new_texts, new_metas

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: dict | None = None,
    ) -> list[Document]:
        """
        Recherche les k chunks les plus similaires à la requête.

        Args:
            query: Requête en langage naturel.
            k: Nombre de résultats à retourner.
            filter_metadata: Filtre optionnel sur les métadonnées (clause where).

        Returns:
            Liste de Documents LangChain triés par similarité décroissante,
            avec 'similarity_score' ajouté aux métadonnées.
        """
        results_with_scores = self.similarity_search_with_score(
            query=query, k=k, filter_metadata=filter_metadata,
        )
        return [doc for doc, _ in results_with_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter_metadata: dict | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Recherche les k chunks les plus similaires avec leurs scores.

        Le score est la similarité cosinus entre 0 et 1 (1 = identique).

        Args:
            query: Requête en langage naturel.
            k: Nombre de résultats à retourner.
            filter_metadata: Filtre optionnel (clause where ChromaDB).

        Returns:
            Liste de tuples (Document, score) triés par score décroissant.
        """
        if self.collection.count() == 0:
            return []

        # Ajuster k si la collection est plus petite
        actual_k = min(k, self.collection.count())

        # Encoder la requête
        query_embedding = self.embedding_manager.embed_query(query)

        # Construire les arguments de la requête
        query_kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": actual_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filter_metadata:
            query_kwargs["where"] = filter_metadata

        results = self.collection.query(**query_kwargs)

        # Convertir en Documents LangChain
        documents_with_scores: list[tuple[Document, float]] = []

        if results and results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            distances = results["distances"][0] if results.get("distances") else []
            metas = results["metadatas"][0] if results.get("metadatas") else []

            for i, doc_text in enumerate(docs):
                # Cosine distance → similarité (1 - distance)
                distance = distances[i] if i < len(distances) else 0.0
                score = max(0.0, min(1.0, 1.0 - distance))

                metadata = dict(metas[i]) if i < len(metas) else {}
                metadata["similarity_score"] = score

                doc = Document(page_content=doc_text, metadata=metadata)
                documents_with_scores.append((doc, score))

        # Trier par score décroissant
        documents_with_scores.sort(key=lambda x: x[1], reverse=True)

        return documents_with_scores

    def delete_document(self, file_name: str) -> int:
        """
        Supprime tous les chunks d'un fichier spécifique.

        Args:
            file_name: Nom du fichier dont supprimer les chunks.

        Returns:
            Nombre de chunks supprimés.
        """
        # Compter avant suppression
        try:
            existing = self.collection.get(
                where={"file_name": file_name},
                include=[],
            )
            count_before = len(existing["ids"]) if existing and existing.get("ids") else 0
        except Exception:
            count_before = 0

        if count_before == 0:
            logger.info("Aucun chunk trouvé pour le fichier '%s'.", file_name)
            return 0

        # Supprimer
        self.collection.delete(where={"file_name": file_name})

        logger.info(
            "%d chunk(s) supprimé(s) pour le fichier '%s'.",
            count_before, file_name,
        )
        return count_before

    def get_collection_stats(self) -> dict:
        """
        Retourne des statistiques sur la collection actuelle.

        Returns:
            Dictionnaire avec : total_chunks, collection_name,
            embedding_model, persist_directory, unique_documents.
        """
        total = self.collection.count()

        # Récupérer les noms de fichiers uniques
        unique_docs: list[str] = []
        if total > 0:
            try:
                all_meta = self.collection.get(include=["metadatas"])
                if all_meta and all_meta.get("metadatas"):
                    file_names = {
                        m.get("file_name", "inconnu")
                        for m in all_meta["metadatas"]
                        if m
                    }
                    unique_docs = sorted(file_names)
            except Exception as e:
                logger.warning("Impossible de récupérer les documents uniques : %s", e)

        return {
            "total_chunks": total,
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_manager.get_active_model(),
            "persist_directory": self.persist_directory,
            "unique_documents": unique_docs,
        }

    def reset_collection(self) -> None:
        """
        Supprime et recrée la collection depuis zéro.

        ⚠️ Attention : cette opération supprime TOUS les embeddings stockés.
        """
        logger.warning(
            "Réinitialisation de la collection '%s' — "
            "tous les embeddings seront supprimés !",
            self.collection_name,
        )
        self._client.delete_collection(self.collection_name)
        self.collection = self._get_or_create_collection()
        logger.info("Collection '%s' réinitialisée.", self.collection_name)
