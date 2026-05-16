"""
rag/memory.py — Gestion de la mémoire de conversation.

Stocke l'historique des messages utilisateur/assistant, gère la
compression automatique quand l'historique dépasse la limite,
et fournit des formats adaptés pour le prompt RAG et LangChain.
"""

import logging
from typing import Optional

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

logger = logging.getLogger("cogniassist.rag")


class ConversationMemory:
    """
    Mémoire de conversation avec compression automatique.

    Stocke les messages récents et compresse les plus anciens
    en un résumé textuel pour éviter les dépassements de contexte.
    """

    def __init__(self, max_messages: int = 10) -> None:
        """
        Initialise la mémoire de conversation.

        Args:
            max_messages: Nombre maximum de messages à conserver
                         avant compression.
        """
        self.messages: list[dict] = []
        self.max_messages = max_messages
        self.summary: str = ""

    def add_user_message(self, content: str) -> None:
        """
        Ajoute un message utilisateur à l'historique.

        Déclenche la compression si la limite est dépassée.

        Args:
            content: Contenu du message utilisateur.
        """
        self.messages.append({"role": "user", "content": content})

        if len(self.messages) > self.max_messages:
            self._compress_history()

    def add_assistant_message(self, content: str) -> None:
        """
        Ajoute un message assistant à l'historique.

        Args:
            content: Contenu de la réponse de l'assistant.
        """
        self.messages.append({"role": "assistant", "content": content})

    def get_formatted_history(self) -> str:
        """
        Formate l'historique de conversation pour injection dans le prompt.

        Inclut le résumé des messages compressés s'il existe,
        suivi des messages récents formatés.

        Returns:
            Historique formaté en chaîne de caractères,
            ou chaîne vide si aucun message.
        """
        if not self.messages and not self.summary:
            return ""

        parts: list[str] = []

        # Ajouter le résumé des anciens messages
        if self.summary:
            parts.append(self.summary)
            parts.append("")

        # Ajouter les messages récents
        for msg in self.messages:
            if msg["role"] == "user":
                parts.append(f"Utilisateur : {msg['content']}")
            else:
                parts.append(f"Assistant : {msg['content']}")

        return "\n".join(parts)

    def get_langchain_messages(self) -> list[BaseMessage]:
        """
        Convertit les messages en objets LangChain.

        Returns:
            Liste de HumanMessage / AIMessage LangChain.
        """
        lc_messages: list[BaseMessage] = []

        for msg in self.messages:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            else:
                lc_messages.append(AIMessage(content=msg["content"]))

        return lc_messages

    def _compress_history(self) -> None:
        """
        Compresse l'historique quand il dépasse max_messages.

        Conserve les 6 messages les plus récents et stocke un résumé
        textuel des messages plus anciens dans self.summary.
        """
        keep_count = 6
        old_messages = self.messages[:-keep_count]
        self.messages = self.messages[-keep_count:]

        # Construire un résumé des anciens messages
        old_summary_parts = [m["content"][:100] for m in old_messages]
        new_summary = "Résumé des échanges précédents : " + " | ".join(old_summary_parts)

        # Cumuler avec le résumé existant
        if self.summary:
            self.summary = self.summary + " | " + new_summary
        else:
            self.summary = new_summary

        logger.debug(
            "Historique compressé : %d anciens messages → résumé, %d conservés.",
            len(old_messages), len(self.messages),
        )

    def clear(self) -> None:
        """Réinitialise les messages et le résumé."""
        self.messages.clear()
        self.summary = ""
        logger.debug("Mémoire de conversation vidée.")

    def get_message_count(self) -> int:
        """
        Retourne le nombre total de messages dans l'historique courant.

        Returns:
            Nombre de messages.
        """
        return len(self.messages)
