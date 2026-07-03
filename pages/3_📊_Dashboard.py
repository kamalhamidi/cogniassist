"""Page Dashboard — statistiques et visualisations."""

from ui.layout import setup_page

setup_page("dashboard")

from views.dashboard import show_dashboard_page

show_dashboard_page()
