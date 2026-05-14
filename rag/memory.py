"""
rag/memory.py — Gestion de la mémoire de conversation.

Stocke l'historique des échanges question/réponse pour maintenir
le contexte conversationnel dans le pipeline RAG.
"""

from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Exchange:
    """Un échange question/réponse dans la conversation."""

    question: str
    answer: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = field(default_factory=dict)


class ConversationMemory:
    """
    Mémoire de conversation pour le pipeline RAG.

    Stocke les échanges passés et les fournit comme contexte
    pour les questions suivantes.
    """

    def __init__(self, max_exchanges: int = 20) -> None:
        """
        Initialise la mémoire de conversation.

        Args:
            max_exchanges: Nombre maximum d'échanges à conserver.
        """
        self.max_exchanges = max_exchanges
        self._history: list[Exchange] = []

    def add_exchange(
        self,
        question: str,
        answer: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Ajoute un échange à la mémoire.

        Args:
            question: La question posée.
            answer: La réponse générée.
            metadata: Métadonnées optionnelles.
        """
        exchange = Exchange(
            question=question,
            answer=answer,
            metadata=metadata or {},
        )
        self._history.append(exchange)

        # Limiter la taille de l'historique
        if len(self._history) > self.max_exchanges:
            self._history = self._history[-self.max_exchanges:]

    def get_history(self, last_n: Optional[int] = None) -> list[dict]:
        """
        Retourne l'historique des échanges.

        Args:
            last_n: Nombre d'échanges à retourner (les plus récents).
                    Si None, retourne tout l'historique.

        Returns:
            Liste de dictionnaires représentant les échanges.
        """
        history = self._history if last_n is None else self._history[-last_n:]
        return [
            {
                "question": ex.question,
                "answer": ex.answer,
                "timestamp": ex.timestamp,
            }
            for ex in history
        ]

    def get_context_string(self, last_n: int = 5) -> str:
        """
        Retourne l'historique formaté en texte pour le prompt.

        Args:
            last_n: Nombre d'échanges à inclure.

        Returns:
            Historique formaté en chaîne de caractères.
        """
        history = self.get_history(last_n=last_n)
        if not history:
            return ""

        parts: list[str] = []
        for ex in history:
            parts.append(f"Q: {ex['question']}")
            parts.append(f"R: {ex['answer']}")

        return "\n".join(parts)

    def clear(self) -> None:
        """Vide la mémoire de conversation."""
        self._history.clear()

    @property
    def size(self) -> int:
        """Nombre d'échanges en mémoire."""
        return len(self._history)
