# 🧠 CogniAssist — Description Complète du Projet

> **Projet de Fin d'Études (PFE) — Master MSID**  
> **Auteur :** Kamal Hamidi  
> **Contexte académique :** Master SID 2025–2026

---

## 🎯 Vue d'ensemble

**CogniAssist** est un **assistant cognitif intelligent personnalisé** basé sur le paradigme **RAG (Retrieval-Augmented Generation)**. Il permet à un utilisateur d'importer ses propres documents (PDF, DOCX, TXT), de les indexer sémantiquement, puis de poser des questions en langage naturel. Le système retrouve les passages les plus pertinents et génère des réponses précises et contextualisées grâce à un pipeline IA complet fonctionnant entièrement **en local**, sans dépendance à des APIs cloud.

---

## 🏗️ Architecture Globale

```
Utilisateur
    │
    ▼
[Streamlit UI] ──── app.py (point d'entrée)
    │
    ├── pages/chat.py        ──→ [RAG Pipeline]
    ├── pages/upload.py      ──→ [Ingestion + VectorStore]
    ├── pages/dashboard.py   ──→ [User History + Stats]
    └── pages/profile.py     ──→ [User Profile]
         │
         ▼
[Modules Métier]
    ├── ingestion/    → Chargement, nettoyage, découpage de documents
    ├── vectorstore/  → Embeddings (Ollama) + ChromaDB (base vectorielle)
    ├── rag/          → Pipeline RAG complet (LangChain + Ollama/Mistral)
    ├── user/         → Profils, historique, recommandations (SQLite)
    └── evaluation/   → Métriques qualité (RAGAS)
```

---

## 📦 Modules Détaillés

---

### 1. `app.py` — Point d'entrée Streamlit

Le fichier principal qui orchestre l'application multi-pages.

| Élément | Description |
|---|---|
| Framework | **Streamlit** (UI web interactive en Python) |
| Navigation | Sidebar avec radio buttons : Chat / Documents / Dashboard / Profil |
| Session State | `user_id`, `messages`, `current_page`, `pipeline_ready` |
| Cache | `@st.cache_resource` pour charger le pipeline RAG **une seule fois** |
| Statut Ollama | Affiche en temps réel si le LLM est disponible et le nombre de documents indexés |

---

### 2. `config.py` — Configuration centralisée

Gestion de toute la configuration via **Pydantic Settings** + fichier `.env`.

| Variable | Valeur par défaut | Rôle |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL locale d'Ollama |
| `OLLAMA_MODEL` | `mistral:7b` | LLM utilisé pour la génération |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Modèle d'embeddings |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | Répertoire ChromaDB |
| `SQLITE_DB_PATH` | `./data/cogniassist.db` | Base de données utilisateurs |
| `UPLOAD_DIR` | `./data/uploads` | Dossier de fichiers uploadés |
| `CHUNK_SIZE` | `500` | Taille des chunks de texte (caractères) |
| `CHUNK_OVERLAP` | `50` | Chevauchement entre chunks |
| `MAX_RETRIEVED_DOCS` | `5` | Nombre de chunks récupérés par requête |

---

### 3. `ingestion/` — Module d'ingestion de documents

Responsable de transformer des fichiers bruts en chunks vectorisables.

#### `loader.py`
- Charge des fichiers **PDF** (via PyMuPDF/fitz), **DOCX** (via python-docx) et **TXT**
- Extrait le texte page par page ou section par section
- Attache des métadonnées (nom de fichier, type, nombre de pages)

#### `cleaner.py`
- Nettoyage et normalisation du texte : suppression des caractères spéciaux, espaces multiples, lignes vides
- Standardisation de l'encodage pour une vectorisation propre

#### `chunker.py`
- Découpage du texte en chunks de taille fixe avec chevauchement (`CHUNK_SIZE=500`, `CHUNK_OVERLAP=50`)
- Utilise **LangChain `RecursiveCharacterTextSplitter`**
- Chaque chunk conserve ses métadonnées (fichier source, index de chunk, `chunk_id`)

**Flux d'ingestion :**
```
Fichier (PDF/DOCX/TXT)
    → loader.py (extraction texte + métadonnées)
    → cleaner.py (nettoyage)
    → chunker.py (découpage)
    → vectorstore/ (embeddings + stockage ChromaDB)
```

---

### 4. `vectorstore/` — Module vectoriel

Gestion des embeddings et de la base vectorielle.

#### `embedder.py`
- Génère des vecteurs d'embeddings via **Ollama** (`nomic-embed-text`)
- Interface avec `langchain-ollama` ou `langchain-huggingface`

#### `store.py` (fichier le plus lourd : 13 Ko)
- Interface complète avec **ChromaDB** (base vectorielle persistante)
- CRUD : ajout, suppression, mise à jour de documents
- Méthodes : `add_documents()`, `delete_document()`, `get_collection_stats()`, `list_documents()`
- Persistance sur disque (`./data/chroma_db/`)

#### `retriever.py`
- **`SmartRetriever`** : recherche sémantique par similarité cosinus
- `retrieve(query, k, filter_metadata)` : retourne les k chunks les plus proches
- `retrieve_with_context_window()` : concatène les chunks en un contexte textuel structuré

---

### 5. `rag/` — Module RAG principal

Cœur du système, orchestre la génération augmentée par récupération.

#### `pipeline.py` — `RAGPipeline`

La classe centrale du projet. Flux de traitement d'une question :

```
Question utilisateur
    ↓
[0] Personnalisation : récupère le profil utilisateur (k adaptatif)
    ↓
[1] Retrieval : SmartRetriever → k chunks pertinents depuis ChromaDB
    ↓
[2] Context Window : concaténation des chunks en contexte
    ↓
[3] Chat History : récupère l'historique de conversation (mémoire)
    ↓
[4] Prompt Building : PromptBuilder construit le prompt enrichi
    ↓
[5] LLM Call : ChatOllama (mistral:7b) génère la réponse
    ↓
[6] Memory Update : ajoute question/réponse à la mémoire
    ↓
[7] History Save : sauvegarde l'interaction en base SQLite
    ↓
[8] Sources : retourne les documents sources utilisés
```

**Paramètres LLM :** `temperature=0.3`, `num_predict=1024`, `top_k=40`, `top_p=0.9`

**Méthodes clés :**
| Méthode | Description |
|---|---|
| `ask()` | Réponse synchrone complète |
| `ask_stream()` | Réponse en **streaming token par token** (pour l'UI Streamlit) |
| `summarize_document()` | Résumé automatique d'un document entier |
| `analyze_knowledge_gaps()` | Analyse des lacunes entre docs et objectifs utilisateur |
| `get_pipeline_status()` | Statut complet du pipeline pour l'UI |

#### `prompt_builder.py`
- Construit des prompts structurés selon 3 templates :
  - `build_rag_prompt()` : pour les questions normales (intègre contexte + historique + profil)
  - `build_summary_prompt()` : pour résumer un document
  - `build_knowledge_gap_prompt()` : pour analyser les lacunes

#### `memory.py` — `ConversationMemory`
- Maintient un historique glissant de **10 messages** (configurable)
- Format `[USER]: ... / [ASSISTANT]: ...` pour l'injection dans les prompts
- Méthodes : `add_user_message()`, `add_assistant_message()`, `get_formatted_history()`, `clear()`

#### `summarizer.py`
- Résumé **Map-Reduce** : résume d'abord chaque chunk séparément puis les combine
- Utile pour les très longs documents

---

### 6. `user/` — Module utilisateur

Gestion complète des profils et de la personnalisation.

#### `db.py`
- Connexion SQLite via **SQLAlchemy**
- Crée les tables au premier démarrage

#### `profile.py` — `UserProfile`
- Modèle complet du profil utilisateur (SQLAlchemy + Pydantic)
- Champs : `user_id`, `name`, `email`, `expertise_level` (débutant/intermédiaire/expert), `preferred_language`, `learning_goals`, `topics_of_interest`, `created_at`, etc.
- CRUD complet sur le profil

#### `history.py` — `InteractionHistory`
- Sauvegarde chaque interaction : `question`, `answer`, `sources`, `chunks_used`, `response_time_ms`, `feedback`
- Méthodes : `save_interaction()`, `save_feedback()`, `get_recent_interactions()`, `get_stats()`
- Statistiques : nombre total de questions, temps de réponse moyen, taux de satisfaction (feedbacks)

#### `recommender.py` — `Recommender`
- Système de **recommandations personnalisées** basé sur le profil et l'historique
- `adapt_rag_parameters()` : adapte le `k` (nombre de chunks) selon le niveau d'expertise
- `get_suggested_questions()` : suggère des questions basées sur les topics et l'historique

---

### 7. `pages/` — Interface utilisateur (4 pages Streamlit)

#### `chat.py` — 💬 Chat intelligent
- Layout 2 colonnes : **chat (70%)** + **suggestions (30%)**
- Streaming token par token avec `st.write_stream()`
- Affichage des sources utilisées (expandable)
- Boutons feedback 👍/👎 par réponse
- Questions suggérées cliquables

#### `upload.py` — 📁 Import de documents
- Upload de fichiers PDF, DOCX, TXT via `st.file_uploader`
- Barre de progression pendant l'ingestion
- Liste des documents déjà indexés avec option de suppression
- Résumé automatique à la demande

#### `dashboard.py` — 📊 Statistiques
- Métriques clés : nombre d'interactions, temps de réponse moyen, satisfaction
- Visualisations interactives avec **Plotly** :
  - Évolution temporelle des interactions
  - Distribution des feedbacks
  - Top des fichiers utilisés

#### `profile.py` — 👤 Profil utilisateur
- Formulaire de gestion du profil (nom, email, niveau d'expertise, objectifs, topics)
- Analyse des lacunes de connaissances (via `pipeline.analyze_knowledge_gaps()`)
- Historique des dernières interactions

---

### 8. `evaluation/` — Module d'évaluation qualité

#### `ragas_eval.py`
- Évaluation automatique du pipeline RAG avec le framework **RAGAS**
- Métriques calculées :
  - **Faithfulness** : la réponse est-elle fidèle au contexte récupéré ?
  - **Answer Relevancy** : la réponse est-elle pertinente par rapport à la question ?
  - **Context Precision / Recall** : qualité du retrieval

#### `test_dataset.json`
- Dataset de questions/réponses de référence pour l'évaluation automatique

#### `report.py`
- Génération de rapports d'évaluation structurés

---

### 9. `tests/` — Tests unitaires (pytest)

| Fichier | Module testé |
|---|---|
| `test_ingestion.py` | Loader, Cleaner, Chunker |
| `test_vectorstore.py` | Embedder, Store, Retriever |
| `test_rag.py` | Pipeline, PromptBuilder, Memory |

Lancement : `pytest tests/ -v`

---

## 🔄 Flux de données complet

```
[IMPORT]
Fichier → Loader → Cleaner → Chunker → Embedder (Ollama/nomic-embed-text)
→ ChromaDB (stockage persistant)

[QUESTION]
Question → SmartRetriever (ChromaDB) → Chunks pertinents
→ PromptBuilder (contexte + historique + profil)
→ ChatOllama (mistral:7b) → Réponse en streaming
→ SQLite (historique) + Mémoire de conversation
→ Streamlit (affichage + sources + feedback)
```

---

## 🛠️ Stack Technologique

| Couche | Technologie | Version |
|---|---|---|
| **LLM** | Ollama + Mistral 7B | Local |
| **Embeddings** | Ollama nomic-embed-text | Local |
| **Orchestration LLM** | LangChain + LangChain-Ollama | 0.3.25 |
| **Base vectorielle** | ChromaDB | 1.0.7 |
| **Base relationnelle** | SQLite + SQLAlchemy | 2.0.41 |
| **Interface utilisateur** | Streamlit | 1.45.1 |
| **Lecture PDF** | PyMuPDF (fitz) | 1.25.5 |
| **Lecture DOCX** | python-docx | 1.1.2 |
| **Validation config** | Pydantic + pydantic-settings | 2.11.3 |
| **Évaluation RAG** | RAGAS | 0.4.3 |
| **Visualisation** | Plotly + Pandas | 6.1.2 / 2.3.0 |
| **Tests** | pytest | 8.4.0 |
| **Tokenisation** | tiktoken | 0.9.0 |

---

## 🔐 Points clés de conception

| Aspect | Choix technique | Avantage |
|---|---|---|
| **Confidentialité** | Tout fonctionne **100% en local** (Ollama) | Aucune donnée envoyée dans le cloud |
| **Personnalisation** | Profil utilisateur + RAG adaptatif | Réponses adaptées au niveau d'expertise |
| **Streaming** | `ask_stream()` + `st.write_stream()` | Expérience utilisateur fluide |
| **Persistance** | ChromaDB (vecteurs) + SQLite (users) | Les données survivent au redémarrage |
| **Qualité** | Framework RAGAS intégré | Mesure objective de la qualité des réponses |
| **Mémoire** | `ConversationMemory` (10 tours) | Conversations contextuelles multi-tours |
| **Config** | Pydantic Settings + `.env` | Configuration sécurisée et validée |

---

## 🚀 Lancement

```bash
# 1. Prérequis : Ollama doit tourner localement
ollama pull mistral:7b
ollama pull nomic-embed-text

# 2. Installation des dépendances
pip install -r requirements.txt

# 3. Configuration
cp .env.example .env

# 4. Lancement
streamlit run app.py
# → http://localhost:8501
```

---

## 📊 Résumé statistique du projet

| Métrique | Valeur |
|---|---|
| **Nombre de modules Python** | 5 modules métier |
| **Nombre de fichiers `.py`** | ~22 fichiers |
| **Nombre de pages UI** | 4 (Chat, Upload, Dashboard, Profil) |
| **LLM utilisé** | Mistral 7B (via Ollama, 100% local) |
| **Formats supportés** | PDF, DOCX, TXT |
| **Type de stockage** | ChromaDB (vecteurs) + SQLite (données user) |
| **Mode de génération** | Synchrone (`ask`) + Streaming (`ask_stream`) |
| **Framework d'évaluation** | RAGAS (Faithfulness, Relevancy, Precision) |
