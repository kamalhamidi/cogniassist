"""
ui/theme.py — Thème global et composants d'interface de CogniAssist.

Fournit :
- `apply_theme()`      : injecte le CSS global (light/dark), la typographie,
                         les couleurs, cartes, boutons, sidebar, chat, etc.
- `page_header()`      : en-tête « héro » dégradé (avec illustration optionnelle).
- `section_title()`    : titre de section stylisé avec icône.
- `stat_badge()`       : petit badge/pilule coloré pour les statuts.
- `chip()` / `chips()` : pastilles discrètes (tags).
- `empty_state()`      : état vide illustré (icône + titre + texte).
- `metric_card()`      : carte KPI riche (icône + valeur + libellé + delta).
- `status_pill()`      : pilule de statut (online/offline) pour la sidebar.
- `success_banner()`   : bandeau de statut vert (style « toast » plein largeur).
- `suggestion_card()`  : carte d'invitation en dégradé (centre d'intérêt, etc.).
- `img_data_uri()`     : encode une image locale en data-URI (embeddable HTML).

Le design vise un rendu « produit SaaS » moderne : surfaces blanches arrondies,
ombres douces, palette violette « cognitive » et un mode sombre soigné.
"""

from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

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
    "bg": "#F5F4FB",
    "surface": "#FFFFFF",
    "border": "#ECEAF6",
    "success": "#16A34A",
    "warning": "#D97706",
    "danger": "#DC2626",
}


# ═══════════════════════════════════════════════════════════════════════
# Utilitaires
# ═══════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=16)
def img_data_uri(path: str) -> str:
    """Encode une image locale en data-URI (utilisable dans `st.html`)."""
    try:
        raw = Path(path).read_bytes()
        ext = Path(path).suffix.lstrip(".").lower() or "png"
        if ext == "svg":
            ext = "svg+xml"
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:image/{ext};base64,{b64}"
    except Exception:
        return ""


def get_theme_mode() -> str:
    """Retourne le mode de thème courant ('light' | 'dark')."""
    return st.session_state.get("theme_mode", "light")


# ═══════════════════════════════════════════════════════════════════════
# Variables CSS — light / dark
# ═══════════════════════════════════════════════════════════════════════

_LIGHT_VARS = """
    --ca-primary: #6C5CE7;
    --ca-primary-soft: #8E7BFF;
    --ca-accent: #A855F7;
    --ca-ink: #1E1B2E;
    --ca-muted: #6B6880;
    --ca-faint: #9690B0;
    --ca-bg: #F2F1FA;
    --ca-surface: #FFFFFF;
    --ca-surface-2: #FAF9FE;
    --ca-sidebar: #FFFFFF;
    --ca-border: #ECEAF6;
    --ca-border-soft: #F1EFFA;
    --ca-success: #16A34A;
    --ca-success-bg: #E9F8EF;
    --ca-warning: #D97706;
    --ca-danger: #DC2626;
    --ca-user-bubble: linear-gradient(135deg, #5B4DC7 0%, #7B6AE0 50%, #9B7AF5 100%);
    --ca-shadow: 0 10px 30px rgba(30, 27, 46, 0.06);
    --ca-shadow-sm: 0 4px 14px rgba(30, 27, 46, 0.05);
    --ca-shadow-lg: 0 22px 48px rgba(108, 92, 231, 0.16);
    --ca-radius: 18px;
"""

_DARK_VARS = """
    --ca-primary: #8E7BFF;
    --ca-primary-soft: #A78BFF;
    --ca-accent: #C084FC;
    --ca-ink: #F4F2FF;
    --ca-muted: #A7A2C4;
    --ca-faint: #837EA0;
    --ca-bg: #14131C;
    --ca-surface: #1E1C2A;
    --ca-surface-2: #232133;
    --ca-sidebar: #1A1825;
    --ca-border: #2C2940;
    --ca-border-soft: #262338;
    --ca-success: #34D399;
    --ca-success-bg: rgba(52, 211, 153, 0.12);
    --ca-warning: #FBBF24;
    --ca-danger: #F87171;
    --ca-user-bubble: linear-gradient(135deg, #4A3CB0 0%, #6B5AD0 50%, #8B6AF0 100%);
    --ca-shadow: 0 10px 30px rgba(0, 0, 0, 0.40);
    --ca-shadow-sm: 0 4px 14px rgba(0, 0, 0, 0.35);
    --ca-shadow-lg: 0 22px 48px rgba(0, 0, 0, 0.55);
    --ca-radius: 18px;
"""


# ═══════════════════════════════════════════════════════════════════════
# CSS global (composants)
# ═══════════════════════════════════════════════════════════════════════

_COMPONENT_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* ── Typographie & fond ─────────────────────────────────────────── */
html, body, [class*="css"], .stApp, button, input, textarea, select {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.stApp {
    background:
        radial-gradient(1100px 520px at 6% -10%, rgba(108, 92, 231, 0.10), transparent 60%),
        radial-gradient(900px 480px at 102% -4%, rgba(168, 85, 247, 0.10), transparent 55%),
        var(--ca-bg);
    color: var(--ca-ink);
}
.block-container {
    padding-top: 1.1rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}
h1, h2, h3, h4 { color: var(--ca-ink); letter-spacing: -0.01em; }
p, span, label, li, .stMarkdown { color: var(--ca-ink); }

/* ── Scrollbar ──────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(108, 92, 231, 0.30);
    border-radius: 999px;
    border: 2px solid transparent;
    background-clip: padding-box;
}
::-webkit-scrollbar-thumb:hover { background: rgba(108, 92, 231, 0.55); background-clip: padding-box; }

/* ── Animations ─────────────────────────────────────────────────── */
@keyframes caFadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes caTypingDot {
    0%, 80%, 100% { transform: translateY(0); opacity: 0.35; }
    40%           { transform: translateY(-5px); opacity: 1; }
}
.block-container > div > div[data-testid="stVerticalBlock"] { animation: caFadeUp 0.45s ease both; }

/* ── En-tête héro ───────────────────────────────────────────────── */
.ca-hero {
    position: relative;
    display: flex;
    align-items: center;
    gap: 20px;
    background: linear-gradient(110deg, #6C5CE7 0%, #7A6AF0 45%, #9B6BF5 100%);
    border-radius: 22px;
    padding: 26px 30px;
    margin-bottom: 16px;
    box-shadow: 0 18px 38px rgba(108, 92, 231, 0.30);
    overflow: hidden;
    min-height: 104px;
}
.ca-hero::after {
    content: "";
    position: absolute;
    top: -45%; right: -3%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(255,255,255,0.20), transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.ca-hero-art {
    position: absolute;
    right: 22px; top: 50%;
    transform: translateY(-50%);
    height: 130px;
    opacity: 0.70;
    pointer-events: none;
    filter: drop-shadow(0 8px 18px rgba(0,0,0,0.18));
}
.ca-hero-icon {
    flex: 0 0 auto;
    font-size: 1.7rem;
    width: 60px; height: 60px;
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.20);
    display: flex; align-items: center; justify-content: center;
    backdrop-filter: blur(4px);
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.28);
}
.ca-hero-title {
    color: #fff !important;
    margin: 0;
    font-size: 1.7rem;
    font-weight: 800;
    line-height: 1.1;
    position: relative;
}
.ca-hero-sub {
    color: rgba(255, 255, 255, 0.92);
    margin: 0.35rem 0 0;
    font-size: 0.95rem;
    font-weight: 400;
    position: relative;
}

/* ── Titres de section ──────────────────────────────────────────── */
.ca-section {
    display: flex;
    align-items: center;
    gap: 9px;
    font-size: 0.98rem;
    font-weight: 700;
    color: var(--ca-ink);
    margin: 4px 0 12px;
    white-space: nowrap;
}
.ca-section .ca-section-bar {
    width: 4px; height: 18px;
    border-radius: 4px;
    background: linear-gradient(180deg, var(--ca-primary), var(--ca-accent));
}

/* ── Badges / pilules ───────────────────────────────────────────── */
.ca-badge {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 0.8rem; font-weight: 600;
    padding: 4px 12px; border-radius: 999px;
    border: 1px solid transparent;
}
.ca-chip {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 0.78rem; font-weight: 600;
    padding: 5px 11px; border-radius: 999px;
    background: rgba(108, 92, 231, 0.10);
    color: var(--ca-primary);
    border: 1px solid rgba(108, 92, 231, 0.18);
    margin: 0 6px 6px 0;
}

/* ── Carte KPI riche ────────────────────────────────────────────── */
.ca-metric {
    position: relative;
    background: var(--ca-surface);
    border: 1px solid var(--ca-border);
    border-radius: 18px;
    padding: 18px 20px;
    box-shadow: var(--ca-shadow-sm);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    overflow: hidden; height: 100%;
}
.ca-metric::before {
    content: "";
    position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
    background: linear-gradient(180deg, var(--ca-primary), var(--ca-accent));
}
.ca-metric:hover { transform: translateY(-3px); box-shadow: var(--ca-shadow-lg); }
.ca-metric-top { display: flex; align-items: center; justify-content: space-between; }
.ca-metric-icon {
    font-size: 1.1rem; width: 38px; height: 38px;
    display: flex; align-items: center; justify-content: center;
    border-radius: 12px; background: rgba(108, 92, 231, 0.12);
}
.ca-metric-value { font-size: 1.8rem; font-weight: 800; color: var(--ca-ink); margin: 8px 0 2px; line-height: 1; }
.ca-metric-label { font-size: 0.85rem; color: var(--ca-muted); font-weight: 600; }
.ca-metric-delta { font-size: 0.78rem; font-weight: 700; margin-top: 4px; }

/* ── État vide ──────────────────────────────────────────────────── */
.ca-empty {
    text-align: center; padding: 38px 24px;
    border: 1.5px dashed var(--ca-border);
    border-radius: 20px;
    background: var(--ca-surface-2);
}
.ca-empty-icon {
    font-size: 2.4rem; width: 76px; height: 76px; margin: 0 auto 14px;
    display: flex; align-items: center; justify-content: center;
    border-radius: 22px; background: rgba(108, 92, 231, 0.12);
}
.ca-empty-title { font-size: 1.12rem; font-weight: 700; color: var(--ca-ink); margin: 0; }
.ca-empty-text { font-size: 0.92rem; color: var(--ca-muted); margin: 6px auto 0; max-width: 460px; }

/* ── Bandeau de succès (plein largeur) ──────────────────────────── */
.ca-banner {
    display: flex; align-items: center; gap: 10px;
    background: var(--ca-success-bg);
    border: 1px solid rgba(22, 163, 74, 0.22);
    color: var(--ca-success);
    border-radius: 14px;
    padding: 12px 16px;
    font-size: 0.9rem; font-weight: 600;
    margin: 2px 0 16px;
}
.ca-banner .ca-banner-dot {
    width: 20px; height: 20px; border-radius: 50%;
    background: var(--ca-success); color: #fff;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem; flex: 0 0 auto;
}

/* ── Carte « suggestion » en dégradé ────────────────────────────── */
.ca-suggest {
    position: relative; overflow: hidden;
    background: linear-gradient(140deg, #6C5CE7, #8B6BF2 60%, #A855F7);
    border-radius: 18px;
    padding: 18px 20px;
    box-shadow: 0 14px 30px rgba(108, 92, 231, 0.30);
    color: #fff;
}
.ca-suggest::after {
    content: ""; position: absolute; right: -30px; top: -30px;
    width: 130px; height: 130px;
    background: radial-gradient(circle, rgba(255,255,255,0.20), transparent 70%);
    border-radius: 50%;
}
.ca-suggest-title { font-weight: 800; font-size: 1rem; display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
.ca-suggest-text { font-size: 0.84rem; line-height: 1.5; color: rgba(255,255,255,0.92); position: relative; }

/* ── Métriques natives ──────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--ca-surface);
    border: 1px solid var(--ca-border);
    border-radius: 18px; padding: 18px 20px;
    box-shadow: var(--ca-shadow-sm);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
[data-testid="stMetric"]:hover { transform: translateY(-2px); box-shadow: var(--ca-shadow); }
[data-testid="stMetricValue"] { color: var(--ca-primary); font-weight: 800; }
[data-testid="stMetricLabel"] { color: var(--ca-muted); font-weight: 600; }

/* ── Boutons ────────────────────────────────────────────────────── */
button[data-testid^="stBaseButton"] {
    border-radius: 12px; font-weight: 600;
    transition: transform 0.12s ease, box-shadow 0.15s ease, background 0.15s ease, color 0.15s ease;
}
button[data-testid^="stBaseButton"]:hover { transform: translateY(-1px); }
button[data-testid^="stBaseButton"]:active { transform: translateY(0); }
button[data-testid="stBaseButton-secondary"],
button[data-testid="stBaseButton-secondaryFormSubmit"] {
    background: var(--ca-surface);
    border: 1px solid var(--ca-border);
    color: var(--ca-ink);
}
button[data-testid="stBaseButton-secondary"]:hover,
button[data-testid="stBaseButton-secondaryFormSubmit"]:hover {
    border-color: var(--ca-primary); color: var(--ca-primary);
    box-shadow: 0 6px 16px rgba(108, 92, 231, 0.14);
}
button[data-testid="stBaseButton-primary"],
button[data-testid="stBaseButton-primaryFormSubmit"] {
    background: linear-gradient(135deg, var(--ca-primary), var(--ca-primary-soft));
    border: none; color: #fff;
    box-shadow: 0 6px 18px rgba(108, 92, 231, 0.30);
}
button[data-testid="stBaseButton-primary"]:hover,
button[data-testid="stBaseButton-primaryFormSubmit"]:hover {
    box-shadow: 0 10px 24px rgba(108, 92, 231, 0.40);
}

/* ── Champs de saisie ───────────────────────────────────────────── */
.stTextInput input, .stTextArea textarea, .stNumberInput input,
[data-baseweb="select"] > div, [data-baseweb="input"] {
    border-radius: 12px !important;
    background: var(--ca-surface) !important;
    color: var(--ca-ink) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--ca-primary) !important;
    box-shadow: 0 0 0 3px rgba(108, 92, 231, 0.16) !important;
}

/* ── Sidebar ────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--ca-sidebar);
    border-right: 1px solid var(--ca-border);
}
[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.55rem; }

/* Boutons de navigation dans la sidebar */
[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] {
    background: transparent;
    border: 1px solid transparent;
    color: var(--ca-muted);
    justify-content: flex-start;
    text-align: left;
    font-weight: 600;
    padding: 9px 14px;
    box-shadow: none;
}
[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] p {
    color: var(--ca-muted); text-align: left; width: 100%;
}
[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"]:hover {
    background: rgba(108, 92, 231, 0.07);
    border-color: transparent;
    color: var(--ca-primary);
    transform: none;
    box-shadow: none;
}
[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"]:hover p { color: var(--ca-primary); }
/* État actif (page courante) */
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] {
    background: rgba(108, 92, 231, 0.12);
    border: 1px solid rgba(108, 92, 231, 0.20);
    color: var(--ca-primary);
    justify-content: flex-start; text-align: left;
    font-weight: 700; padding: 9px 14px;
    box-shadow: none;
}
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] p { color: var(--ca-primary); text-align: left; width: 100%; }
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"]:hover {
    box-shadow: none; transform: none;
    background: rgba(108, 92, 231, 0.16);
}

/* ── Chat ───────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: transparent;
    border: none;
    padding: 2px 0;
    box-shadow: none;
    margin-bottom: 2px;
    animation: caFadeUp 0.35s ease both;
}
/* Bulle assistant (blanche, coin haut-gauche net) */
[data-testid="stChatMessageContent"][aria-label="Chat message from assistant"] {
    background: var(--ca-surface);
    border: 1px solid var(--ca-border);
    border-radius: 6px 18px 18px 18px;
    padding: 12px 18px;
    box-shadow: var(--ca-shadow-sm);
    max-width: 88%;
}
/* Bulle utilisateur (alignée à droite, dégradé solide violet) */
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {
    flex-direction: row-reverse;
}
[data-testid="stChatMessageContent"][aria-label="Chat message from user"] {
    background: var(--ca-user-bubble);
    border: none;
    border-radius: 18px 6px 18px 18px;
    padding: 14px 20px;
    box-shadow: 0 8px 24px rgba(91, 77, 199, 0.30);
    max-width: 88%;
}
[data-testid="stChatMessageContent"][aria-label="Chat message from user"] p { color: #FFFFFF !important; }
[data-testid="stChatMessageContent"][aria-label="Chat message from user"] .stMarkdown p { color: #FFFFFF !important; }
/* Indicateur « en train d'écrire » */
.ca-typing-indicator {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 2px 0;
    min-height: 24px;
}
.ca-typing-label {
    font-size: 0.82rem;
    color: var(--ca-muted);
    font-weight: 500;
    letter-spacing: 0.01em;
}
.ca-typing-dots {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding-bottom: 1px;
}
.ca-typing-dots span {
    display: block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--ca-primary);
    animation: caTypingDot 1.2s ease-in-out infinite;
}
.ca-typing-dots span:nth-child(2) { animation-delay: 0.15s; }
.ca-typing-dots span:nth-child(3) { animation-delay: 0.3s; }
/* Avatars ronds */
[data-testid="stChatMessage"] > div:first-child:not([data-testid]) {
    border-radius: 50%;
    background: var(--ca-surface);
    border: 1px solid var(--ca-border);
    box-shadow: var(--ca-shadow-sm);
}
/* Masquer l'avatar utilisateur pour un rendu plus épuré */
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) > div:first-child:not([data-testid]) {
    display: none;
}
[data-testid="stChatInput"] {
    border-radius: 16px;
    border: 1px solid var(--ca-border);
    background: var(--ca-surface);
    box-shadow: var(--ca-shadow);
}
[data-testid="stChatInput"] textarea { color: var(--ca-ink); }
[data-testid="stChatInput"]:focus-within {
    border-color: var(--ca-primary);
    box-shadow: 0 0 0 3px rgba(108, 92, 231, 0.16);
}

/* Chips d'action sous les réponses (boutons compacts, scope par clé) */
[class*="st-key-act_"] button[data-testid^="stBaseButton"] {
    border-radius: 999px !important;
    padding: 3px 14px !important;
    min-height: 0 !important;
    height: 30px;
    width: auto !important;
    white-space: nowrap;
    font-size: 0.78rem !important;
    font-weight: 600;
    background: var(--ca-surface-2) !important;
    border: 1px solid var(--ca-border) !important;
    color: var(--ca-muted) !important;
    box-shadow: none !important;
}
[class*="st-key-act_"] button[data-testid^="stBaseButton"] p {
    color: var(--ca-muted);
    overflow: visible;
    text-overflow: clip;
    white-space: nowrap;
}
[class*="st-key-act_"] button[data-testid^="stBaseButton"]:hover {
    border-color: var(--ca-primary) !important; color: var(--ca-primary) !important;
    transform: none;
}
[class*="st-key-act_"] button[data-testid^="stBaseButton"]:hover p { color: var(--ca-primary); }

/* Lignes de suggestion (colonne droite) */
[class*="st-key-sug_"] button[data-testid^="stBaseButton"],
[class*="st-key-welcome_sug_"] button[data-testid^="stBaseButton"] {
    background: var(--ca-surface-2) !important;
    border: 1px solid var(--ca-border) !important;
    color: var(--ca-ink) !important;
    text-align: left;
    justify-content: flex-start;
    font-weight: 500;
    font-size: 0.82rem;
    box-shadow: none !important;
    padding: 10px 14px !important;
}
[class*="st-key-sug_"] button[data-testid^="stBaseButton"] p,
[class*="st-key-welcome_sug_"] button[data-testid^="stBaseButton"] p {
    text-align: left; width: 100%; color: var(--ca-ink);
}
[class*="st-key-sug_"] button[data-testid^="stBaseButton"]:hover,
[class*="st-key-welcome_sug_"] button[data-testid^="stBaseButton"]:hover {
    border-color: var(--ca-primary) !important;
    background: rgba(108,92,231,0.06) !important;
    transform: none;
}

/* ── Héro de profil (conteneur clé : profile_hero) ──────────────── */
.st-key-profile_hero {
    background: linear-gradient(115deg, #6C5CE7 0%, #7E6AF0 50%, #9B6BF5 100%);
    border-radius: 24px;
    padding: 24px 30px;
    box-shadow: 0 18px 40px rgba(108, 92, 231, 0.30);
    margin-bottom: 16px;
}
.st-key-profile_hero p, .st-key-profile_hero span, .st-key-profile_hero label,
.st-key-profile_hero h1, .st-key-profile_hero h2, .st-key-profile_hero h3 { color: #fff; }
.ca-pf-avatar {
    width: 116px; height: 116px; border-radius: 50%;
    background: #fff; display: flex; align-items: center; justify-content: center;
    margin: 0 auto 4px; position: relative;
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.20);
}
.ca-pf-avatar img { width: 76%; height: 76%; object-fit: contain; }
.ca-pf-pencil {
    position: absolute; right: 4px; bottom: 4px;
    width: 30px; height: 30px; border-radius: 50%;
    background: linear-gradient(135deg, #6C5CE7, #A855F7); color: #fff;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.78rem; border: 2px solid #fff;
}
.ca-pf-name { font-size: 1.9rem; font-weight: 800; margin: 0; line-height: 1.1; }
.ca-pf-sub { font-size: 0.9rem; color: rgba(255, 255, 255, 0.85); margin: 4px 0 16px; }
.ca-pf-badges { display: flex; gap: 12px; flex-wrap: wrap; }
.ca-pf-badge {
    display: flex; align-items: center; gap: 11px;
    background: rgba(255, 255, 255, 0.16);
    border: 1px solid rgba(255, 255, 255, 0.22);
    border-radius: 14px; padding: 9px 16px;
}
.ca-pf-badge .ic {
    font-size: 1rem; width: 30px; height: 30px; border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    background: rgba(255, 255, 255, 0.18);
}
.ca-pf-badge .lbl { font-size: 0.7rem; color: rgba(255, 255, 255, 0.8); line-height: 1.2; }
.ca-pf-badge .val { font-size: 0.9rem; font-weight: 700; color: #fff; line-height: 1.2; }

/* Bouton "Modifier le profil" (blanc) */
.st-key-pf_edit_btn button[data-testid^="stBaseButton"] {
    background: #fff !important; color: var(--ca-primary) !important;
    border: none !important; font-weight: 700; border-radius: 12px;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
}
.st-key-pf_edit_btn button[data-testid^="stBaseButton"] p { color: var(--ca-primary); }
/* Sélecteur d'avatar (pilule blanche) */
.st-key-pf_avatar_sel [data-baseweb="select"] > div {
    background: #fff !important; border-radius: 12px !important; border: none !important;
}
.st-key-pf_avatar_sel [data-baseweb="select"] div,
.st-key-pf_avatar_sel [data-baseweb="select"] span { color: var(--ca-ink) !important; }
.st-key-pf_avatar_sel label { color: rgba(255, 255, 255, 0.92) !important; font-weight: 600; }
/* Bouton "Mettre à jour l'avatar" (translucide) */
.st-key-pf_avatar_btn button[data-testid^="stBaseButton"] {
    background: rgba(255, 255, 255, 0.18) !important; color: #fff !important;
    border: 1px solid rgba(255, 255, 255, 0.32) !important; font-weight: 700;
    box-shadow: none;
}
.st-key-pf_avatar_btn button[data-testid^="stBaseButton"] p { color: #fff; }

/* ── Carte « Astuces » (colonne droite des préférences) ─────────── */
.ca-tips {
    background: linear-gradient(180deg, rgba(108, 92, 231, 0.08), rgba(168, 85, 247, 0.05));
    border: 1px solid var(--ca-border);
    border-radius: 18px; padding: 18px 20px;
}
.ca-tips-art { width: 92px; display: block; margin: 2px auto 8px; }
.ca-tips h4 { margin: 0 0 6px; font-size: 1rem; font-weight: 800; color: var(--ca-ink); }
.ca-tips-lead { font-size: 0.82rem; color: var(--ca-muted); line-height: 1.45; }
.ca-tip-row { display: flex; gap: 11px; align-items: flex-start; margin-top: 13px; }
.ca-tip-ic {
    width: 30px; height: 30px; border-radius: 9px; flex: 0 0 auto;
    display: flex; align-items: center; justify-content: center; font-size: 0.85rem;
}
.ca-tip-txt { font-size: 0.79rem; color: var(--ca-muted); line-height: 1.42; }
.ca-tip-txt b { color: var(--ca-ink); font-weight: 700; }

/* Slider (niveau d'expertise) */
[data-testid="stSlider"] [role="slider"] { background: var(--ca-primary) !important; }

/* ── Barre de saisie type messagerie (formulaire chatbar) ───────── */
[data-testid="stForm"]:has(.st-key-chatbar_q) {
    background: var(--ca-surface);
    border: 1px solid var(--ca-border);
    border-radius: 22px;
    box-shadow: var(--ca-shadow);
    padding: 16px 18px 14px;
    margin-top: 10px;
}
/* Champ de texte sans bordure, plus aéré */
.st-key-chatbar_q [data-baseweb="base-input"],
.st-key-chatbar_q [data-baseweb="input"],
.st-key-chatbar_q .stTextInput div[data-baseweb="input"] {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}
.st-key-chatbar_q .stTextInput input {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    font-size: 0.96rem;
    padding: 6px 4px !important;
}
.st-key-chatbar_q .stTextInput input:focus { box-shadow: none !important; }
/* Sélecteurs en pilule (modèle / mode RAG) */
.st-key-chatbar_model [data-baseweb="select"] > div,
.st-key-chatbar_rag [data-baseweb="select"] > div {
    border-radius: 999px !important;
    background: var(--ca-surface-2) !important;
    border: 1px solid var(--ca-border) !important;
    min-height: 36px !important;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--ca-ink);
}
.st-key-chatbar_model [data-baseweb="select"] svg,
.st-key-chatbar_rag [data-baseweb="select"] svg { color: var(--ca-muted); }
/* Bouton d'envoi rond (paper-plane) */
[data-testid="stForm"]:has(.st-key-chatbar_q) button[data-testid="stBaseButton-primaryFormSubmit"] {
    border-radius: 50% !important;
    width: 50px !important;
    height: 50px !important;
    min-height: 50px !important;
    padding: 0 !important;
    font-size: 1.15rem;
    margin-left: auto;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 10px 22px rgba(108, 92, 231, 0.38);
}
[data-testid="stForm"]:has(.st-key-chatbar_q) button[data-testid="stBaseButton-primaryFormSubmit"] p {
    font-size: 1.15rem; line-height: 1;
}

/* Zone de discussion défilante */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] [data-testid="stChatMessage"]) {
    background: transparent;
}

/* Popover « Ma pensée a évolué » : bouton discret */
.st-key-clear_chat button[data-testid^="stBaseButton"],
[data-testid="stPopover"] button[data-testid^="stBaseButton"] {
    border-radius: 12px;
    font-size: 0.84rem;
    font-weight: 600;
}

/* Ligne de document (colonne droite) */
.ca-doc-row {
    display: flex; align-items: center; justify-content: space-between;
    gap: 10px; padding: 9px 4px;
    border-bottom: 1px solid var(--ca-border-soft);
}
.ca-doc-row:last-child { border-bottom: none; }
.ca-doc-name { display: flex; align-items: center; gap: 8px; font-size: 0.82rem; color: var(--ca-ink); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ca-doc-meta { font-size: 0.72rem; color: var(--ca-faint); flex: 0 0 auto; }

/* ── Expanders ──────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid var(--ca-border) !important;
    border-radius: 14px !important;
    box-shadow: var(--ca-shadow-sm);
    overflow: hidden;
    background: var(--ca-surface);
}
[data-testid="stExpander"] summary { color: var(--ca-ink); }
[data-testid="stExpander"] summary:hover { color: var(--ca-primary); }

/* ── Onglets baseweb ────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: none; }
.stTabs [data-baseweb="tab"] {
    border-radius: 12px; padding: 8px 16px; font-weight: 600; color: var(--ca-muted);
}
.stTabs [data-baseweb="tab"]:hover { color: var(--ca-primary); }
.stTabs [aria-selected="true"] { background: rgba(108, 92, 231, 0.12); color: var(--ca-primary) !important; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--ca-primary); }

/* ── Conteneurs bordés ──────────────────────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 18px;
    transition: box-shadow 0.18s ease, border-color 0.18s ease;
}

/* ── File uploader ──────────────────────────────────────────────── */
[data-testid="stFileUploaderDropzone"] {
    border: 1.5px dashed rgba(108, 92, 231, 0.35) !important;
    border-radius: 16px !important;
    background: var(--ca-surface-2) !important;
    transition: border-color 0.18s ease, background 0.18s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--ca-primary) !important;
    background: rgba(108, 92, 231, 0.07) !important;
}

/* ── Progress ───────────────────────────────────────────────────── */
[data-testid="stProgress"] [role="progressbar"] > div {
    background: linear-gradient(90deg, var(--ca-primary), var(--ca-accent));
}

/* ── Alertes ────────────────────────────────────────────────────── */
[data-testid="stAlert"] { border-radius: 14px; }

/* ── Navigation segmentée (top nav) ─────────────────────────────── */
div[data-testid="stElementContainer"]:has(div[data-testid="stSegmentedControl"]) {
    position: sticky; top: 0; z-index: 1000;
    padding: 8px 0; margin-bottom: 2px;
}
div[data-testid="stSegmentedControl"] {
    display: flex; justify-content: flex-start; width: 100%; margin: 0;
}
div[data-testid="stSegmentedControl"] [role="group"] {
    background: var(--ca-surface);
    border: 1px solid var(--ca-border);
    border-radius: 14px; padding: 5px;
    box-shadow: var(--ca-shadow-sm); gap: 4px;
}
div[data-testid="stSegmentedControl"] button {
    padding: 8px 18px !important; font-size: 14px; font-weight: 600;
    border-radius: 10px !important; border: none !important;
    color: var(--ca-muted) !important;
    transition: background 0.15s ease, color 0.15s ease;
}
div[data-testid="stSegmentedControl"] button p { color: var(--ca-muted); }
div[data-testid="stSegmentedControl"] button[aria-checked="true"],
div[data-testid="stSegmentedControl"] button[aria-selected="true"] {
    background: rgba(108, 92, 231, 0.14) !important;
    color: var(--ca-primary) !important;
    box-shadow: none;
}
div[data-testid="stSegmentedControl"] button[aria-checked="true"] p,
div[data-testid="stSegmentedControl"] button[aria-selected="true"] p { color: var(--ca-primary); }

/* ── Dataframes ─────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 14px; overflow: hidden; border: 1px solid var(--ca-border);
}

/* ── Divider ────────────────────────────────────────────────────── */
hr { border-color: var(--ca-border); opacity: 0.7; }

/* Masquer chrome Streamlit */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stHeader"] { display: none; }
[data-testid="stToolbar"] { display: none; }
"""


# ═══════════════════════════════════════════════════════════════════════
# API publique
# ═══════════════════════════════════════════════════════════════════════

def _build_theme_css(mode: str) -> str:
    """Construit le bloc CSS complet pour un mode (mis en cache)."""
    variables = _DARK_VARS if mode == "dark" else _LIGHT_VARS
    return f"<style>:root {{{variables}}}\n{_COMPONENT_CSS}</style>"


@st.cache_data(show_spinner=False)
def _cached_theme_css(mode: str) -> str:
    """Cache le CSS par mode pour éviter de reconstruire à chaque rerun."""
    return _build_theme_css(mode)


def apply_theme() -> None:
    """Injecte le thème global (variables + composants). À appeler tôt dans `main()`."""
    mode = get_theme_mode()
    if st.session_state.get("_ca_theme_mode") != mode:
        st.session_state._ca_theme_mode = mode
    st.html(_cached_theme_css(mode))


def page_header(
    title: str,
    subtitle: str | None = None,
    icon: str = "🧠",
    art_image: str | None = None,
) -> None:
    """Affiche un en-tête « héro » dégradé en haut d'une page.

    Args:
        title:     Titre principal.
        subtitle:  Sous-titre / description courte (optionnel).
        icon:      Emoji affiché dans la pastille à gauche.
        art_image: Chemin d'une image décorative affichée à droite (optionnel).
    """
    sub_html = f'<p class="ca-hero-sub">{subtitle}</p>' if subtitle else ""
    art_html = ""
    if art_image:
        uri = img_data_uri(art_image)
        if uri:
            art_html = f'<img class="ca-hero-art" src="{uri}" alt="" />'
    st.html(
        f"""
        <div class="ca-hero">
            <div class="ca-hero-icon">{icon}</div>
            <div>
                <h1 class="ca-hero-title">{title}</h1>
                {sub_html}
            </div>
            {art_html}
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
    """Retourne le HTML d'un badge/pilule coloré (à passer à st.html)."""
    colors = {
        "primary": ("rgba(108,92,231,.12)", "#6C5CE7"),
        "success": ("rgba(22,163,74,.12)", "#16A34A"),
        "warning": ("rgba(217,119,6,.14)", "#D97706"),
        "danger": ("rgba(220,38,38,.12)", "#DC2626"),
        "muted": ("rgba(107,104,128,.12)", "#6B6880"),
    }
    bg, fg = colors.get(kind, colors["primary"])
    return f'<span class="ca-badge" style="background:{bg};color:{fg};">{label}</span>'


def chip(label: str) -> str:
    """Retourne le HTML d'une pastille discrète (tag)."""
    return f'<span class="ca-chip">{label}</span>'


def chips(labels: list[str]) -> None:
    """Affiche une rangée de pastilles à partir d'une liste de libellés."""
    if not labels:
        return
    inner = "".join(chip(str(label)) for label in labels)
    st.html(f'<div style="margin:2px 0 6px;">{inner}</div>')


def empty_state(icon: str, title: str, text: str = "") -> None:
    """Affiche un état vide illustré et centré."""
    text_html = f'<p class="ca-empty-text">{text}</p>' if text else ""
    st.html(
        f"""
        <div class="ca-empty">
            <div class="ca-empty-icon">{icon}</div>
            <p class="ca-empty-title">{title}</p>
            {text_html}
        </div>
        """
    )


def metric_card(
    label: str,
    value: str | int | float,
    icon: str = "📈",
    delta: str | None = None,
    delta_kind: str = "muted",
) -> None:
    """Affiche une carte KPI riche (icône + valeur + libellé + delta optionnel)."""
    delta_colors = {
        "success": "#16A34A", "warning": "#D97706",
        "danger": "#DC2626", "muted": "#6B6880",
    }
    delta_html = ""
    if delta:
        color = delta_colors.get(delta_kind, "#6B6880")
        delta_html = f'<div class="ca-metric-delta" style="color:{color};">{delta}</div>'
    st.html(
        f"""
        <div class="ca-metric">
            <div class="ca-metric-top">
                <span class="ca-metric-label">{label}</span>
                <span class="ca-metric-icon">{icon}</span>
            </div>
            <div class="ca-metric-value">{value}</div>
            {delta_html}
        </div>
        """
    )


def status_pill(label: str, kind: str = "success") -> None:
    """Affiche une pilule de statut compacte (idéale pour la sidebar)."""
    colors = {
        "success": ("rgba(22,163,74,.12)", "#16A34A", "#16A34A"),
        "danger": ("rgba(220,38,38,.12)", "#DC2626", "#DC2626"),
        "warning": ("rgba(217,119,6,.14)", "#D97706", "#D97706"),
        "muted": ("rgba(107,104,128,.12)", "#6B6880", "#9690b0"),
    }
    bg, fg, dot = colors.get(kind, colors["success"])
    st.html(
        f"""
        <div style="display:flex;align-items:center;gap:8px;background:{bg};
                    color:{fg};border-radius:12px;padding:8px 12px;
                    font-weight:600;font-size:0.86rem;">
            <span style="width:8px;height:8px;border-radius:50%;background:{dot};
                         box-shadow:0 0 0 3px {bg};"></span>
            <span>{label}</span>
        </div>
        """
    )


def success_banner(text: str, icon: str = "✓") -> None:
    """Affiche un bandeau de statut vert pleine largeur (style confirmation)."""
    st.html(
        f"""
        <div class="ca-banner">
            <span class="ca-banner-dot">{icon}</span>
            <span>{text}</span>
        </div>
        """
    )


def suggestion_card(title: str, text: str, icon: str = "🎯") -> None:
    """Affiche une carte d'invitation en dégradé (suggestion / centre d'intérêt)."""
    st.html(
        f"""
        <div class="ca-suggest">
            <div class="ca-suggest-title">{icon} {title}</div>
            <div class="ca-suggest-text">{text}</div>
        </div>
        """
    )


def typing_indicator_html(label: str = "CogniAssist écrit") -> str:
    """Retourne le HTML de l'indicateur « en train d'écrire » (points animés)."""
    return f"""
    <div class="ca-typing-indicator" aria-live="polite" aria-label="{label}">
        <span class="ca-typing-label">{label}</span>
        <span class="ca-typing-dots" aria-hidden="true">
            <span></span><span></span><span></span>
        </span>
    </div>
    """
