"""
ui — Système de design centralisé de CogniAssist.

Expose le thème global (CSS) et les composants d'interface réutilisables
(en-têtes héro, cartes, badges, états vides, KPIs) pour une apparence
cohérente sur toutes les pages.
"""

from ui.layout import setup_page
from ui.theme import (
    PALETTE,
    apply_theme,
    chip,
    chips,
    empty_state,
    get_theme_mode,
    img_data_uri,
    metric_card,
    page_header,
    section_title,
    stat_badge,
    status_pill,
    success_banner,
    suggestion_card,
    typing_indicator_html,
)

__all__ = [
    "PALETTE",
    "apply_theme",
    "setup_page",
    "chip",
    "chips",
    "empty_state",
    "get_theme_mode",
    "img_data_uri",
    "metric_card",
    "page_header",
    "section_title",
    "stat_badge",
    "status_pill",
    "success_banner",
    "suggestion_card",
    "typing_indicator_html",
]
