"""
ui/layout.py — Layout partagé : session, thème, sidebar, gates.

Chaque page Streamlit appelle `setup_page()` une seule fois en tête de script.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

_ASSETS_DIR = Path(__file__).parent.parent / "assets"
LOGO_PATH = str(_ASSETS_DIR / "logo.png")
ICON_PATH = str(_ASSETS_DIR / "cogniassist_icon.png")

# Fichiers multipage Streamlit (ordre = ordre sidebar native)
PAGE_FILES = {
    "chat": "pages/1_💬_Chat.py",
    "upload": "pages/2_📁_Documents.py",
    "dashboard": "pages/3_📊_Dashboard.py",
    "profile": "pages/4_👤_Profil.py",
}

NAV_ITEMS = [
    ("chat", "💬", "Chat intelligent"),
    ("upload", "📁", "Documents indexés"),
    ("dashboard", "📊", "Tableau de bord"),
    ("profile", "👤", "Profil"),
]

_PAGE_CONFIGURED = "_ca_page_configured"


def init_session_state() -> None:
    """Initialise les variables de session Streamlit."""
    defaults: dict = {
        "user_id": "default",
        "messages": [],
        "pipeline_ready": False,
        "last_interaction_id": None,
        "theme_mode": "light",
        "kmb_skipped": False,
        "profile_tab": "⚙️ Préférences",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def configure_page() -> None:
    """Appelle st.set_page_config une seule fois par session."""
    if st.session_state.get(_PAGE_CONFIGURED):
        return
    st.set_page_config(
        page_title="CogniAssist",
        page_icon=ICON_PATH,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.session_state[_PAGE_CONFIGURED] = True


def _goto(page_id: str) -> None:
    """Navigue vers une page multipage Streamlit."""
    st.switch_page(PAGE_FILES[page_id])


@st.cache_data(ttl=30, show_spinner=False)
def _get_doc_count(user_id: str) -> int:
    """Compte les documents sans charger le pipeline RAG."""
    try:
        from user import get_user_manager
        return len(get_user_manager(user_id).get_user_documents())
    except Exception:
        return 0


@st.cache_data(ttl=15, show_spinner=False)
def _get_pipeline_status_light() -> dict:
    """Statut Ollama léger — sans initialiser le pipeline complet."""
    try:
        import json
        from urllib.error import URLError
        from urllib.request import Request, urlopen

        from config import settings
        base = settings.ollama_base_url.rstrip("/")
        req = Request(f"{base}/api/tags", headers={"Accept": "application/json"})
        with urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode())
        tags = data.get("models", [])
        names = {m.get("name", "") for m in tags}
        model = settings.ollama_model
        ready = any(model in n or n.startswith(model.split(":")[0]) for n in names)
        return {"ready": ready, "model": model}
    except (URLError, OSError, TimeoutError, ValueError, KeyError):
        return {"ready": False, "model": "mistral:7b"}


@st.cache_resource(show_spinner=False)
def _get_cached_pipeline():
    """Charge le pipeline RAG (Chat uniquement)."""
    from rag import get_pipeline
    return get_pipeline()


def _render_theme_toggle() -> None:
    """Bascule clair / sombre dans la sidebar."""
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


def _render_sidebar(current_page: str) -> None:
    """Barre latérale : logo, statut, navigation, profil."""
    from ui.theme import img_data_uri

    user_id = st.session_state.get("user_id", "default")
    doc_count = _get_doc_count(user_id)

    # Pipeline complet seulement sur Chat ; ping léger ailleurs
    if current_page == "chat":
        model_name, ready = "mistral:7b", False
        try:
            pipeline = _get_cached_pipeline()
            ready = bool(pipeline.is_ready)
            status = pipeline.get_pipeline_status()
            model_name = status.get("llm_model", model_name)
            doc_count = status.get("total_documents_indexed", doc_count)
            st.session_state.pipeline_ready = ready
        except Exception:
            st.session_state.pipeline_ready = False
            light = _get_pipeline_status_light()
            ready, model_name = light["ready"], light["model"]
    else:
        light = _get_pipeline_status_light()
        ready, model_name = light["ready"], light["model"]
        st.session_state.pipeline_ready = ready

    with st.sidebar:
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

        for page_id, icon, label in NAV_ITEMS:
            badge = f"  ·  {doc_count}" if page_id == "upload" and doc_count else ""
            if st.button(
                f"{icon}  {label}{badge}",
                key=f"nav_{page_id}",
                type="primary" if current_page == page_id else "secondary",
                use_container_width=True,
            ):
                if current_page != page_id:
                    _goto(page_id)

        st.markdown("---")
        _render_theme_toggle()

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


def _check_onboarding_gate() -> bool:
    """Retourne True si l'onboarding bloque la navigation."""
    try:
        from user import get_user_manager
        mgr = get_user_manager(st.session_state.user_id)
        profile = mgr.get_profile()
        if not profile.get("onboarding_completed", False):
            from views.onboarding import show_onboarding_page
            show_onboarding_page()
            return True
    except Exception:
        pass
    return False


def _check_kmb_gate(current_page: str) -> bool:
    """Retourne True si le gate Know Me Better bloque la page Chat."""
    if current_page != "chat":
        return False
    try:
        from user import get_kmb_manager
        kmb_mgr = get_kmb_manager(st.session_state.user_id)
        if not kmb_mgr.has_seen_kmb_onboarding() and not st.session_state.get("kmb_skipped", False):
            from views.know_me_better import show_kmb_page
            show_kmb_page(is_onboarding=True)
            return True
    except Exception:
        pass
    return False


def setup_page(current_page: str) -> None:
    """Initialise une page Streamlit : config, session, thème, sidebar, gates."""
    configure_page()
    init_session_state()

    from ui.theme import apply_theme, img_data_uri
    apply_theme()

    _render_sidebar(current_page)

    if _check_onboarding_gate():
        st.stop()

    if _check_kmb_gate(current_page):
        st.stop()
