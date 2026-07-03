"""Page Profil — préférences, contexte RAG et identité."""

from ui.layout import setup_page

setup_page("profile")

from views.profile import show_profile_page

show_profile_page()
