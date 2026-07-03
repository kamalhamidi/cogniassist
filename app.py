"""
CogniAssist — Point d'entrée principal.

Redirige vers la page Chat. Chaque section de l'app est une page
Streamlit séparée dans `pages/` pour un chargement plus rapide.
"""

import streamlit as st

from ui.layout import configure_page, init_session_state

configure_page()
init_session_state()

st.switch_page("pages/1_💬_Chat.py")
