# Install Ollama

1. Download Ollama : https://ollama.com/download
2. Install and launch it
3. Open a terminal and run :
   ollama pull mistral:7b
   ollama pull nomic-embed-text
4. Verify it works :
   ollama run mistral:7b "hello"
5. Then launch the app :
   streamlit run app.py
