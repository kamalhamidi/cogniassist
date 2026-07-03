"""Page Chat — conversation RAG intelligente."""

from ui.layout import setup_page

setup_page("chat")

from views.chat import show_chat_page

show_chat_page()
