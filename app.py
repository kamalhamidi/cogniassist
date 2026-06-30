"""
CogniAssist — Point d'entrée principal de l'application Streamlit.

Lance l'interface multi-pages avec navigation sidebar :
- 💬 Chat : Conversation intelligente avec le RAG
- 📁 Documents : Import et gestion de documents
- 📊 Dashboard : Statistiques et visualisations
- 👤 Profil : Gestion du profil utilisateur
"""

from pathlib import Path

import streamlit as st

_ASSETS_DIR = Path(__file__).parent / "assets"
LOGO_PATH = str(_ASSETS_DIR / "logo.png")
ICON_PATH = str(_ASSETS_DIR / "cogniassist_icon.png")

# Navigation : (clé page, icône, libellé sidebar, libellé top-nav)
NAV_ITEMS = [
    ("💬 Chat", "💬", "Chat intelligent", "💬 Chat"),
    ("📁 Documents", "📁", "Documents indexés", "📁 Documents"),
    ("📊 Dashboard", "📊", "Tableau de bord", "📊 Dashboard"),
    ("👤 Profil", "👤", "Profil", "👤 Profil"),
]
PAGE_TO_SLUG = {
    "💬 Chat": "chat",
    "📁 Documents": "upload",
    "📊 Dashboard": "dashboard",
    "👤 Profil": "profile",
}
SLUG_TO_PAGE = {v: k for k, v in PAGE_TO_SLUG.items()}


def _goto(page: str) -> None:
    """Change de page et synchronise l'URL."""
    st.session_state.current_page = page
    st.query_params["page"] = PAGE_TO_SLUG[page]
    st.rerun()

st.set_page_config(
    page_title="CogniAssist",
    page_icon=ICON_PATH,
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state() -> None:
    """Initialise les variables de session Streamlit."""
    defaults: dict = {
        "user_id": "default",
        "messages": [],
        "current_page": "💬 Chat",
        "pipeline_ready": False,
        "last_interaction_id": None,
        "theme_mode": "light",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource
def _get_cached_pipeline():
    """Charge le pipeline RAG une seule fois."""
    from rag import get_pipeline
    return get_pipeline()


def _render_theme_toggle() -> None:
    """Affiche la bascule de thème clair / sombre (coin supérieur droit)."""
    mode = st.session_state.get("theme_mode", "light")
    c1, c2 = st.columns(2)
    with c1:
        if st.button(
            "🌙", key="theme_dark",
            type="primary" if mode == "dark" else "secondary",
            use_container_width=True, help="Mode sombre",
        ):
            if mode != "dark":
                st.session_state.theme_mode = "dark"
                st.rerun()
    with c2:
        if st.button(
            "☀️", key="theme_light",
            type="primary" if mode == "light" else "secondary",
            use_container_width=True, help="Mode clair",
        ):
            if mode != "light":
                st.session_state.theme_mode = "light"
                st.rerun()


def _render_sidebar() -> None:
    """Affiche la barre latérale : logo, statut, navigation et profil utilisateur."""
    from ui import img_data_uri

    # ── Statut du pipeline ──
    model_name, ready, doc_count = "mistral:7b", False, 0
    try:
        pipeline = _get_cached_pipeline()
        ready = bool(pipeline.is_ready)
        status = pipeline.get_pipeline_status()
        model_name = status.get("llm_model", model_name)
        doc_count = status.get("total_documents_indexed", 0)
        st.session_state.pipeline_ready = ready
    except Exception:
        st.session_state.pipeline_ready = False

    try:
        from user import get_user_manager
        docs = get_user_manager(st.session_state.user_id).get_user_documents()
        doc_count = len(docs)
    except Exception:
        pass

    with st.sidebar:
        # ── Logo ──
        logo_uri = img_data_uri(LOGO_PATH)
        st.html(
            f"""
            <div style="text-align:center;margin:2px 0 4px;">
                <img src="{logo_uri}" style="width:78%;max-width:190px;" alt="CogniAssist" />
                <div style="font-size:.78rem;color:var(--ca-muted);margin-top:2px;">
                    Assistant cognitif intelligent
                </div>
            </div>
            """
        )

        # ── Carte de statut ──
        dot = "#16A34A" if ready else "#DC2626"
        label = "Ollama connecté" if ready else "Ollama hors-ligne"
        st.html(
            f"""
            <div style="background:var(--ca-surface);border:1px solid var(--ca-border);
                        border-radius:14px;padding:11px 14px;margin:10px 0 14px;
                        box-shadow:var(--ca-shadow-sm);">
                <div style="display:flex;align-items:center;gap:8px;">
                    <span style="width:9px;height:9px;border-radius:50%;background:{dot};
                                 box-shadow:0 0 0 3px {dot}22;"></span>
                    <span style="font-weight:700;font-size:.86rem;color:var(--ca-ink);">{label}</span>
                </div>
                <div style="display:flex;align-items:center;justify-content:space-between;margin-top:8px;">
                    <span style="font-size:.78rem;color:var(--ca-muted);">Modèle actif</span>
                    <span style="font-size:.74rem;font-weight:700;color:var(--ca-primary);
                                 background:rgba(108,92,231,.12);padding:3px 9px;border-radius:999px;">
                        {model_name}
                    </span>
                </div>
            </div>
            """
        )

        # ── Navigation ──
        current = st.session_state.get("current_page", "💬 Chat")
        for page_key, icon, label, _ in NAV_ITEMS:
            badge = ""
            if page_key == "📁 Documents" and doc_count:
                badge = f"  ·  {doc_count}"
            if st.button(
                f"{icon}  {label}{badge}",
                key=f"nav_{page_key}",
                type="primary" if current == page_key else "secondary",
                use_container_width=True,
            ):
                if current != page_key:
                    _goto(page_key)

        # ── Liens secondaires (décoratifs) ──
        for icon, label in [("⚙️", "Paramètres"), ("❓", "Aide & support")]:
            if st.button(
                f"{icon}  {label}", key=f"nav_extra_{label}",
                type="secondary", use_container_width=True,
            ):
                st.toast(f"{label} — bientôt disponible ✨")

        # ── Carte profil utilisateur (bas) ──
        st.html(
            """
            <div style="margin-top:16px;display:flex;align-items:center;gap:11px;
                        background:var(--ca-surface);border:1px solid var(--ca-border);
                        border-radius:14px;padding:11px 13px;box-shadow:var(--ca-shadow-sm);">
                <div style="width:38px;height:38px;border-radius:11px;flex:0 0 auto;
                            background:linear-gradient(135deg,#6C5CE7,#A855F7);color:#fff;
                            display:flex;align-items:center;justify-content:center;
                            font-weight:800;font-size:.85rem;">HK</div>
                <div style="line-height:1.25;overflow:hidden;">
                    <div style="font-weight:700;font-size:.85rem;color:var(--ca-ink);">Hamidi Kamal</div>
                    <div style="font-size:.74rem;color:var(--ca-muted);">Master SID 2025–2026</div>
                </div>
            </div>
            """
        )


def main() -> None:
    """Point d'entrée principal de l'application."""
    init_session_state()

    # ═══ Thème global ═══
    from ui import apply_theme
    apply_theme()

    # ═══ Résolution de la page courante (URL ↔ session) ═══
    if "query_page_checked" not in st.session_state:
        query_page = st.query_params.get("page")
        if query_page in SLUG_TO_PAGE:
            st.session_state.current_page = SLUG_TO_PAGE[query_page]
        elif st.session_state.get("current_page") in PAGE_TO_SLUG:
            st.query_params["page"] = PAGE_TO_SLUG[st.session_state.current_page]
        else:
            st.session_state.current_page = "💬 Chat"
            st.query_params["page"] = "chat"
        st.session_state.query_page_checked = True

    # ═══ Sidebar ═══
    _render_sidebar()

    # ═══ Onboarding Gate ═══
    try:
        from user import get_user_manager
        _mgr = get_user_manager(st.session_state.user_id)
        _profile = _mgr.get_profile()
        if not _profile.get("onboarding_completed", False):
            from pages.onboarding import show_onboarding_page
            show_onboarding_page()
            st.stop()
    except Exception:
        pass  # Si erreur, continuer normalement

    # ═══ Top Nav Bar (onglets + bascule de thème) ═══
    nav_pages = [item[0] for item in NAV_ITEMS]
    col_nav, col_toggle = st.columns([8, 2], vertical_alignment="center")

    with col_nav:
        selected_page = st.segmented_control(
            "Navigation",
            options=nav_pages,
            default=st.session_state.current_page,
            label_visibility="collapsed",
            key="top_navbar",
        )

    with col_toggle:
        _render_theme_toggle()

    if selected_page and selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page
        st.query_params["page"] = PAGE_TO_SLUG[selected_page]
        st.rerun()
    elif not selected_page:
        st.rerun()

    page = st.session_state.current_page

    # ═══ Know Me Better Gate ═══
    try:
        from user import get_kmb_manager
        _kmb_mgr = get_kmb_manager(st.session_state.user_id)
        if not _kmb_mgr.has_seen_kmb_onboarding() and not st.session_state.get("kmb_skipped", False):
            if page == "💬 Chat":
                from pages.know_me_better import show_kmb_page
                show_kmb_page(is_onboarding=True)
                st.stop()
            else:
                # Si l'utilisateur clique sur une autre page de la sidebar, on considère qu'il passe/reporte l'onboarding KMB
                st.session_state.kmb_skipped = True
    except Exception:
        pass  # Si erreur, continuer normalement

    # ═══ Routage des pages ═══
    if page == "💬 Chat":
        from pages.chat import show_chat_page
        show_chat_page()
    elif page == "📁 Documents":
        from pages.upload import show_upload_page
        show_upload_page()
    elif page == "📊 Dashboard":
        from pages.dashboard import show_dashboard_page
        show_dashboard_page()
    elif page == "👤 Profil":
        from pages.profile import show_profile_page
        show_profile_page()


if __name__ == "__main__":
    main()
