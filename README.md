# 🧠 CogniAssist

**Assistant cognitif intelligent personnalisé basé sur le RAG (Retrieval-Augmented Generation)**

CogniAssist est un système qui permet aux utilisateurs d'importer leurs propres documents (PDF, DOCX, TXT), de les indexer sémantiquement, puis de poser des questions en langage naturel. Le système retrouve les passages pertinents et génère des réponses précises et contextualisées grâce à un pipeline RAG combinant LangChain et Ollama.

---

## 📋 Fonctionnalités

- 📄 **Ingestion multi-format** : Import de fichiers PDF, DOCX et TXT avec nettoyage et découpage automatique
- 🔍 **Recherche sémantique** : Embeddings via Ollama + base vectorielle ChromaDB
- 🤖 **Pipeline RAG** : LangChain + Ollama pour des réponses contextualisées
- 👤 **Personnalisation** : Profils utilisateurs avec SQLite/SQLAlchemy
- 💬 **Interface conversationnelle** : Chat intelligent avec historique et sources
- 📊 **Dashboard** : Statistiques d'utilisation et métriques
- 📈 **Évaluation** : Framework RAGAS pour mesurer la qualité des réponses

---

## 🏗️ Architecture

```
cogniassist/
├── app.py                    # Point d'entrée Streamlit
├── config.py                 # Configuration centralisée (Pydantic Settings)
│
├── ingestion/                # Module d'ingestion de documents
│   ├── loader.py             # Chargement PDF/DOCX/TXT
│   ├── chunker.py            # Découpage en chunks (LangChain)
│   └── cleaner.py            # Nettoyage et normalisation du texte
│
├── vectorstore/              # Module de gestion vectorielle
│   ├── embedder.py           # Génération d'embeddings (Ollama)
│   ├── store.py              # Interface ChromaDB
│   └── retriever.py          # Recherche par similarité sémantique
│
├── rag/                      # Module RAG principal
│   ├── pipeline.py           # Pipeline RAG complet
│   ├── prompt_builder.py     # Construction des prompts
│   ├── memory.py             # Mémoire de conversation
│   └── summarizer.py         # Résumé automatique (map-reduce)
│
├── user/                     # Module utilisateur
│   ├── profile.py            # Modèle profil (SQLAlchemy + Pydantic)
│   ├── history.py            # Historique des interactions
│   ├── recommender.py        # Recommandations personnalisées
│   └── db.py                 # Connexion SQLite/SQLAlchemy
│
├── evaluation/               # Module d'évaluation
│   ├── ragas_eval.py         # Évaluation RAGAS
│   ├── test_dataset.json     # Dataset de test
│   └── report.py             # Génération de rapports
│
├── pages/                    # Pages Streamlit
│   ├── chat.py               # 💬 Chat intelligent
│   ├── upload.py             # 📄 Import de documents
│   ├── dashboard.py          # 📊 Statistiques
│   └── profile.py            # 👤 Profil utilisateur
│
├── tests/                    # Tests unitaires (pytest)
│   ├── test_ingestion.py
│   ├── test_vectorstore.py
│   └── test_rag.py
│
├── data/                     # Données (ignoré par git)
│   ├── uploads/              # Documents uploadés
│   ├── chroma_db/            # Base vectorielle persistante
│   └── cogniassist.db        # Base SQLite
│
└── assets/                   # Ressources statiques
    └── logo.png
```

---

## 🚀 Installation

### Prérequis

- Python 3.11 ou supérieur
- pip (gestionnaire de paquets Python)
- Ollama (local)

### Étapes

1. **Cloner le dépôt**
   ```bash
   git clone <url-du-repo>
   cd cogniassist
   ```

2. **Créer un environnement virtuel**
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   # ou
   venv\Scripts\activate     # Windows
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurer les variables d'environnement**
   ```bash
   cp .env.example .env
   ```

5. **Installer Ollama**
   1. Download Ollama : https://ollama.com/download
   2. Install and launch it
   3. Open a terminal and run :
      ollama pull mistral:7b
      ollama pull nomic-embed-text
   4. Verify it works :
      ollama run mistral:7b "hello"
   5. Then launch the app :
      streamlit run app.py

L'application s'ouvre automatiquement dans votre navigateur à `http://localhost:8501`

---

## ⚙️ Variables d'environnement

| Variable | Description | Défaut |
|---|---|---|
| `OLLAMA_BASE_URL` | URL locale d'Ollama | `http://localhost:11434` |
| `OLLAMA_MODEL` | Modèle Ollama à utiliser | `mistral:7b` |
| `EMBEDDING_MODEL` | Modèle d'embeddings Ollama | `nomic-embed-text` |
| `CHROMA_PERSIST_DIR` | Répertoire de persistance ChromaDB | `./data/chroma_db` |
| `SQLITE_DB_PATH` | Chemin de la base SQLite | `./data/cogniassist.db` |
| `UPLOAD_DIR` | Répertoire des uploads | `./data/uploads` |
| `CHUNK_SIZE` | Taille des chunks (caractères) | `500` |
| `CHUNK_OVERLAP` | Chevauchement entre chunks | `50` |
| `MAX_RETRIEVED_DOCS` | Nombre max de documents retrouvés | `5` |
| `APP_NAME` | Nom de l'application | `CogniAssist` |
| `DEBUG` | Mode debug | `False` |

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## 🛠️ Technologies

| Composant | Technologie |
|---|---|
| Framework LLM | LangChain + LangChain-Ollama |
| LLM | Ollama (mistral:7b) |
| Embeddings | Ollama (nomic-embed-text) |
| Base vectorielle | ChromaDB |
| Base relationnelle | SQLite + SQLAlchemy |
| Interface | Streamlit |
| Validation | Pydantic |
| Évaluation | RAGAS |
| Visualisation | Plotly + Pandas |
| Tests | pytest |   

---

## 📄 Licence

Projet académique — PFE Master MSID.

---

## 👤 Auteur

**Kamal Hamidi** — Master MSID, Projet de Fin d'Études
