"""
ui — Système de design centralisé de CogniAssist.

Expose le thème global (CSS) et les composants d'interface réutilisables
(en-têtes héro, cartes, badges, états vides, KPIs) pour une apparence
cohérente sur toutes les pages.
"""

from ui.theme import (
    PALETTE,
    apply_theme,
    chip,
    chips,
    empty_state,
    metric_card,
    page_header,
    section_title,
    stat_badge,
    status_pill,
)

__all__ = [
    "PALETTE",
    "apply_theme",
    "chip",
    "chips",
    "empty_state",
    "metric_card",
    "page_header",
    "section_title",
    "stat_badge",
    "status_pill",
]
