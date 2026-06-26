"""
ui/theme.py — Thème global et composants d'interface de CogniAssist.

Fournit :
- `apply_theme()`    : injecte le CSS global (typographie, couleurs, cartes,
                       boutons, sidebar, chat, onglets, etc.).
- `page_header()`    : en-tête « héro » dégradé réutilisable en haut de page.
- `section_title()`  : titre de section stylisé avec icône.
- `stat_badge()`     : petit badge/pilule coloré pour les statuts.

Le design vise une apparence moderne, douce et cohérente (palette violette
« cognitive »), tout en restant 100 % compatible avec les composants natifs
de Streamlit.
"""

from __future__ import annotations

import streamlit as st

# ═══════════════════════════════════════════════════════════════════════
# Palette de couleurs
# ═══════════════════════════════════════════════════════════════════════

PALETTE = {
    "primary": "#6C5CE7",
    "primary_soft": "#8E7BFF",
    "accent": "#A855F7",
    "ink": "#1E1B2E",
    "muted": "#6B6880",
    "bg": "#F6F5FC",
    "surface": "#FFFFFF",
    "border": "#ECEAF6",
    "success": "#16A34A",
    "warning": "#D97706",
    "danger": "#DC2626",
}


# ═══════════════════════════════════════════════════════════════════════
# CSS global
# ═══════════════════════════════════════════════════════════════════════

_GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --ca-primary: #6C5CE7;
    --ca-primary-soft: #8E7BFF;
    --ca-accent: #A855F7;
    --ca-ink: #1E1B2E;
    --ca-muted: #6B6880;
    --ca-bg: #F6F5FC;
    --ca-surface: #FFFFFF;
    --ca-border: #ECEAF6;
    --ca-shadow: 0 8px 24px rgba(30, 27, 46, 0.06);
}

/* ── Typographie & fond ─────────────────────────────────────────── */
html, body, [class*="css"], .stApp, button, input, textarea, select {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.stApp { background: var(--ca-bg); color: var(--ca-ink); }

/* Largeur & respiration du contenu principal */
.block-container {
    padding-top: 2.2rem;
    padding-bottom: 4rem;
    max-width: 1180px;
}

h1, h2, h3, h4 { color: var(--ca-ink); letter-spacing: -0.01em; }

/* ── En-tête héro ───────────────────────────────────────────────── */
.ca-hero {
    display: flex;
    align-items: center;
    gap: 18px;
    background: linear-gradient(120deg, #6C5CE7 0%, #8E7BFF 55%, #A855F7 100%);
    border-radius: 22px;
    padding: 22px 28px;
    margin-bottom: 18px;
    box-shadow: 0 14px 34px rgba(108, 92, 231, 0.28);
}
.ca-hero-icon {
    flex: 0 0 auto;
    font-size: 2.2rem;
    width: 62px;
    height: 62px;
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.18);
    display: flex;
    align-items: center;
    justify-content: center;
    backdrop-filter: blur(4px);
}
.ca-hero-title {
    color: #fff !important;
    margin: 0;
    font-size: 1.85rem;
    font-weight: 800;
    line-height: 1.1;
}
.ca-hero-sub {
    color: rgba(255, 255, 255, 0.88);
    margin: 0.3rem 0 0;
    font-size: 0.98rem;
    font-weight: 400;
}

/* ── Titres de section ──────────────────────────────────────────── */
.ca-section {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--ca-ink);
    margin: 6px 0 10px;
}
.ca-section .ca-section-bar {
    width: 4px;
    height: 20px;
    border-radius: 4px;
    background: linear-gradient(180deg, var(--ca-primary), var(--ca-accent));
}

/* ── Badges / pilules ───────────────────────────────────────────── */
.ca-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 999px;
    border: 1px solid transparent;
}

/* ── Métriques (KPI cards) ──────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--ca-surface);
    border: 1px solid var(--ca-border);
    border-radius: 18px;
    padding: 18px 20px;
    box-shadow: var(--ca-shadow);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(108, 92, 231, 0.12);
}
[data-testid="stMetricValue"] {
    color: var(--ca-primary);
    font-weight: 800;
}
[data-testid="stMetricLabel"] { color: var(--ca-muted); font-weight: 600; }

/* ── Boutons ────────────────────────────────────────────────────── */
button[data-testid^="stBaseButton"] {
    border-radius: 12px;
    font-weight: 600;
    transition: transform 0.12s ease, box-shadow 0.15s ease, background 0.15s ease;
}
button[data-testid^="stBaseButton"]:hover {
    transform: translateY(-1px);
}
/* Variantes secondaires (par défaut) */
button[data-testid="stBaseButton-secondary"],
button[data-testid="stBaseButton-secondaryFormSubmit"] {
    background: var(--ca-surface);
    border: 1px solid var(--ca-border);
    color: var(--ca-ink);
}
button[data-testid="stBaseButton-secondary"]:hover,
button[data-testid="stBaseButton-secondaryFormSubmit"]:hover {
    border-color: var(--ca-primary);
    color: var(--ca-primary);
    box-shadow: 0 6px 16px rgba(108, 92, 231, 0.12);
}
/* Variantes primaires */
button[data-testid="stBaseButton-primary"],
button[data-testid="stBaseButton-primaryFormSubmit"] {
    background: linear-gradient(135deg, var(--ca-primary), var(--ca-primary-soft));
    border: none;
    color: #fff;
    box-shadow: 0 6px 18px rgba(108, 92, 231, 0.30);
}
button[data-testid="stBaseButton-primary"]:hover,
button[data-testid="stBaseButton-primaryFormSubmit"]:hover {
    box-shadow: 0 10px 24px rgba(108, 92, 231, 0.40);
}
/* Bouton de téléchargement */
button[data-testid="stBaseButton-secondary"][kind] { border-radius: 12px; }

/* ── Champs de saisie ───────────────────────────────────────────── */
.stTextInput input,
.stTextArea textarea,
.stNumberInput input,
[data-baseweb="select"] > div,
[data-baseweb="input"] {
    border-radius: 12px !important;
}
.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: var(--ca-primary) !important;
    box-shadow: 0 0 0 3px rgba(108, 92, 231, 0.15) !important;
}

/* ── Sidebar ────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F7F6FE 0%, #EEEBFB 100%);
    border-right: 1px solid var(--ca-border);
}
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

/* ── Chat ───────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: var(--ca-surface);
    border: 1px solid var(--ca-border);
    border-radius: 18px;
    padding: 8px 16px;
    box-shadow: 0 2px 10px rgba(30, 27, 46, 0.04);
    margin-bottom: 6px;
}
[data-testid="stChatInput"] {
    border-radius: 16px;
    border: 1px solid var(--ca-border);
    box-shadow: var(--ca-shadow);
}

/* ── Expanders ──────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid var(--ca-border) !important;
    border-radius: 16px !important;
    box-shadow: 0 2px 10px rgba(30, 27, 46, 0.03);
    overflow: hidden;
    background: var(--ca-surface);
}

/* ── Onglets ────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: none; }
.stTabs [data-baseweb="tab"] {
    border-radius: 12px;
    padding: 8px 16px;
    font-weight: 600;
    color: var(--ca-muted);
}
.stTabs [data-baseweb="tab"]:hover { color: var(--ca-primary); }
.stTabs [aria-selected="true"] {
    background: rgba(108, 92, 231, 0.12);
    color: var(--ca-primary) !important;
}
.stTabs [data-baseweb="tab-highlight"] { background: var(--ca-primary); }

/* ── Conteneurs bordés (st.container(border=True)) ──────────────── */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 16px;
}

/* ── Barre de progression ───────────────────────────────────────── */
[data-testid="stProgress"] [role="progressbar"] > div {
    background: linear-gradient(90deg, var(--ca-primary), var(--ca-accent));
}

/* ── Alertes (info/success/warning/error) ───────────────────────── */
[data-testid="stAlert"] { border-radius: 14px; }

/* ── Navigation segmentée (top nav) ─────────────────────────────── */
div[data-testid="stSegmentedControl"] {
    display: flex;
    justify-content: center;
    width: 100%;
    margin: 2px 0 6px 0;
}
div[data-testid="stSegmentedControl"] [role="group"] {
    background: var(--ca-surface);
    border: 1px solid var(--ca-border);
    border-radius: 14px;
    padding: 5px;
    box-shadow: var(--ca-shadow);
    gap: 4px;
}
div[data-testid="stSegmentedControl"] button {
    padding: 8px 18px !important;
    font-size: 15px;
    font-weight: 600;
    border-radius: 10px !important;
    border: none !important;
}
div[data-testid="stSegmentedControl"] button[aria-checked="true"],
div[data-testid="stSegmentedControl"] button[aria-selected="true"] {
    background: linear-gradient(135deg, var(--ca-primary), var(--ca-primary-soft)) !important;
    color: #fff !important;
}

/* ── Dataframes ─────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid var(--ca-border);
}

/* ── Divider plus discret ───────────────────────────────────────── */
hr { border-color: var(--ca-border); opacity: 0.7; }

/* Masquer le menu/footer Streamlit pour un rendu plus « produit » */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
"""


# ═══════════════════════════════════════════════════════════════════════
# API publique
# ═══════════════════════════════════════════════════════════════════════

def apply_theme() -> None:
    """Injecte le thème global. À appeler une fois par run, tôt dans `main()`."""
    st.html(_GLOBAL_CSS)


def page_header(title: str, subtitle: str | None = None, icon: str = "🧠") -> None:
    """Affiche un en-tête « héro » dégradé en haut d'une page.

    Args:
        title:    Titre principal de la page.
        subtitle: Sous-titre / description courte (optionnel).
        icon:     Emoji affiché dans la pastille à gauche.
    """
    sub_html = f'<p class="ca-hero-sub">{subtitle}</p>' if subtitle else ""
    st.html(
        f"""
        <div class="ca-hero">
            <div class="ca-hero-icon">{icon}</div>
            <div>
                <h1 class="ca-hero-title">{title}</h1>
                {sub_html}
            </div>
        </div>
        """
    )


def section_title(title: str) -> None:
    """Affiche un titre de section stylisé (barre dégradée + texte)."""
    st.html(
        f"""
        <div class="ca-section">
            <span class="ca-section-bar"></span>
            <span>{title}</span>
        </div>
        """
    )


def stat_badge(label: str, kind: str = "primary") -> str:
    """Retourne le HTML d'un badge/pilule coloré (à passer à st.html).

    Args:
        label: Texte du badge.
        kind:  'primary' | 'success' | 'warning' | 'danger' | 'muted'.
    """
    colors = {
        "primary": ("rgba(108,92,231,.12)", "#6C5CE7"),
        "success": ("rgba(22,163,74,.12)", "#16A34A"),
        "warning": ("rgba(217,119,6,.14)", "#D97706"),
        "danger": ("rgba(220,38,38,.12)", "#DC2626"),
        "muted": ("rgba(107,104,128,.12)", "#6B6880"),
    }
    bg, fg = colors.get(kind, colors["primary"])
    return (
        f'<span class="ca-badge" style="background:{bg};color:{fg};">{label}</span>'
    )
