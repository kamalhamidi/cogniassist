"""
ui — Système de design centralisé de CogniAssist.

Expose le thème global (CSS) et les composants d'interface réutilisables
(en-têtes héro, cartes, badges) pour une apparence cohérente sur toutes
les pages.
"""

from ui.theme import (
    PALETTE,
    apply_theme,
    page_header,
    section_title,
    stat_badge,
)

__all__ = [
    "PALETTE",
    "apply_theme",
    "page_header",
    "section_title",
    "stat_badge",
]
