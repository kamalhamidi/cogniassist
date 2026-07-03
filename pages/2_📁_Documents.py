"""Page Documents — import et gestion de la base de connaissances."""

from ui.layout import setup_page

setup_page("upload")

from views.upload import show_upload_page

show_upload_page()
