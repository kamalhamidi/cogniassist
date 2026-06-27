# 🧠 CogniAssist — Documentation complète des fonctionnalités

> **Projet de Fin d'Études (PFE) — Master MSID 2025–2026**
> **Auteur :** Kamal HAMIDI
> **Nature :** Assistant cognitif personnel basé sur le RAG (Retrieval-Augmented Generation), fonctionnant **100 % en local**.

Ce document recense **l'intégralité des fonctionnalités** réellement présentes dans le code source, module par module. Il sert de référence technique et fonctionnelle à jour.

---

## 📑 Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Architecture du projet](#2-architecture-du-projet)
3. [Fonctionnalités par domaine](#3-fonctionnalités-par-domaine)
   - [3.1 Ingestion de documents](#31-ingestion-de-documents)
   - [3.2 Vectorisation & recherche hybride](#32-vectorisation--recherche-hybride)
   - [3.3 Pipeline RAG](#33-pipeline-rag)
   - [3.4 Personnalisation & ACPE](#34-personnalisation--acpe-adaptive-cognitive-profiling-engine)
   - [3.5 Know Me Better (KMB)](#35-know-me-better-kmb)
   - [3.6 Onboarding](#36-onboarding)
   - [3.7 Interface & design system](#37-interface--design-system)
   - [3.8 Tableau de bord analytique](#38-tableau-de-bord-analytique)
   - [3.9 Profil, données & confidentialité](#39-profil-données--confidentialité)
   - [3.10 Évaluation qualité (RAGAS)](#310-évaluation-qualité-ragas)
   - [3.11 Tests & scripts utilitaires](#311-tests--scripts-utilitaires)
   - [3.12 Identité & mode « second cerveau » (Layer 2)](#312-identité--mode--second-cerveau--layer-2)
4. [Modèle de données](#4-modèle-de-données-sqlite)
5. [Stack technologique](#5-stack-technologique)
6. [Configuration](#6-configuration)
7. [Installation & lancement](#7-installation--lancement)

---

## 1. Vue d'ensemble

**CogniAssist** permet à un utilisateur d'importer ses propres documents (PDF, DOCX, TXT, MD), de les indexer sémantiquement, puis de poser des questions en langage naturel. Le système retrouve les passages les plus pertinents (recherche **hybride dense + sparse**) et génère des réponses contextualisées via un LLM local (Mistral 7B sous Ollama).

Au-delà du RAG classique, CogniAssist se distingue par son **moteur de profilage cognitif adaptatif (ACPE)** qui apprend du comportement de l'utilisateur pour personnaliser les réponses, recommander des contenus et faire évoluer un profil de connaissances par domaine — le tout sans jamais envoyer de données dans le cloud.

Une **couche d'identité (Layer 2)** va plus loin et fait de CogniAssist un véritable **second cerveau** : à partir des écrits personnels de l'utilisateur, le système extrait son **style d'écriture**, ses **croyances et positions**, puis peut répondre **avec sa voix et ses opinions** lorsque le « mode second cerveau » est activé (voir §3.12).

**Piliers du produit :**

| Pilier | Description |
|---|---|
| 🔒 **Confidentialité** | Tout tourne en local (Ollama + ChromaDB + SQLite). Aucune API cloud. |
| 🎯 **Personnalisation** | Profil utilisateur + RAG adaptatif + ACPE + Know Me Better. |
| ⚡ **Réactivité** | Réponses en streaming token par token. |
| 🧩 **Recherche hybride** | Fusion dense (embeddings) + sparse (BM25) via Reciprocal Rank Fusion. |
| 📊 **Transparence** | Sources affichées, paramètres RAG expliqués, insights comportementaux. |
| 🧪 **Qualité mesurable** | Évaluation automatique via RAGAS. |
| 🧬 **Second cerveau (Layer 2)** | Extraction d'identité : style d'écriture, croyances/positions et mode « second cerveau » qui répond comme l'utilisateur. |

---

## 2. Architecture du projet

```
cogniassist/
├── app.py                     # Point d'entrée Streamlit (navigation, gates, routage)
├── config.py                  # Configuration centralisée (Pydantic Settings + .env)
│
├── ui/                        # 🎨 Système de design (NOUVEAU)
│   ├── __init__.py            # Exports : apply_theme, page_header, section_title, stat_badge
│   └── theme.py               # CSS global + composants d'interface (palette, héros, cartes)
│
├── ingestion/                 # Transformation fichiers → chunks
│   ├── loader.py              # Chargement PDF (PyMuPDF) / DOCX / TXT / bytes
│   ├── cleaner.py             # Nettoyage, normalisation Unicode, filtrage
│   └── chunker.py             # Découpage (caractères ou tokens) + statistiques
│
├── vectorstore/               # Embeddings & recherche
│   ├── embedder.py            # Embeddings Ollama (nomic-embed-text) + fallback
│   ├── store.py               # ChromaDB (CRUD, dédup, stats) + collection personal_writing
│   ├── bm25_index.py          # Index sparse BM25 (build, add, remove, retrieve)
│   └── retriever.py           # SmartRetriever (dense+rerank) + HybridRetriever (RRF)
│
├── rag/                       # Cœur RAG
│   ├── pipeline.py            # RAGPipeline (ask, ask_stream, summarize, gaps, mode identité)
│   ├── prompt_builder.py      # Construction des prompts (RAG / résumé / lacunes)
│   ├── memory.py              # Mémoire de conversation glissante (10 tours)
│   └── summarizer.py          # Résumé par lot + rapport de connaissances
│
├── user/                      # Profils & personnalisation (ACPE + Identité)
│   ├── db.py                  # Connexion SQLite/SQLAlchemy + reset système
│   ├── profile.py             # UserProfile / Preferences / DocumentAccess + manager
│   ├── history.py             # Historique des interactions + stats + feedback
│   ├── recommender.py         # Recommandations + adaptation des paramètres RAG
│   ├── acpe_models.py         # Tables ACPE (KnowledgeProfile, UsagePattern, Prompt)
│   ├── knowledge_engine.py    # Profil de connaissances par domaine (EMA)
│   ├── profile_evolution.py   # Détection de tendances + génération d'insights
│   ├── progressive.py         # Suggestions de profiling progressif (cooldown)
│   ├── kmb_service.py         # Service « Know Me Better »
│   ├── identity_models.py     # 🧬 Tables Identité (StyleProfile, BeliefStore, StyleCorrection)
│   ├── style_analyzer.py      # 🧬 Empreinte stylistique (stdlib uniquement)
│   ├── belief_extractor.py    # 🧬 Extraction de croyances via LLM + détection de conflits
│   └── identity_prompt_builder.py  # 🧬 Prompt système « second cerveau »
│
├── evaluation/                # Mesure de qualité
│   ├── ragas_eval.py          # Évaluation RAGAS (faithfulness, relevancy, etc.)
│   ├── report.py              # Génération de rapports Markdown + DataFrames
│   └── test_dataset.json      # Dataset de référence Q/R
│
├── pages/                     # Pages Streamlit
│   ├── chat.py                # 💬 Chat intelligent
│   ├── upload.py              # 📁 Gestion des documents
│   ├── dashboard.py           # 📊 Tableau de bord
│   ├── profile.py             # 👤 Profil
│   ├── onboarding.py          # 🚀 Wizard d'onboarding (3 étapes)
│   └── know_me_better.py      # 🧠 Questionnaire KMB (5 sections)
│
├── scripts/                   # Outils CLI
│   ├── rebuild_bm25.py        # Reconstruction de l'index BM25
│   └── test_hybrid_retrieval.py
│
├── tests/                     # Tests pytest (ingestion, vectorstore, rag, user, acpe, kmb, eval, identity)
├── data/                      # ChromaDB, SQLite, uploads, bm25_index.pkl (ignoré par git)
└── assets/                    # logo.png
```

**Flux applicatif (app.py) :** thème global → sidebar (statut pipeline) → **gate onboarding** (si non complété) → barre de navigation segmentée → **gate Know Me Better** (avant le premier chat) → routage vers la page demandée. La navigation est synchronisée avec les `query_params` de l'URL (`?page=chat|upload|dashboard|profile`).

---

## 3. Fonctionnalités par domaine

### 3.1 Ingestion de documents

**Formats supportés :** PDF, DOCX/DOC, TXT, Markdown.

- **`DocumentLoader`** — Chargement depuis un chemin **ou directement depuis des bytes** (`load_from_bytes`, utilisé par l'upload Streamlit).
  - PDF via **PyMuPDF (fitz)** : extraction page par page avec numéro de page en métadonnée.
  - DOCX via **python-docx** : extraction paragraphe par paragraphe.
  - TXT/MD : lecture brute avec détection d'encodage.
  - Métadonnées attachées : `file_name`, `file_type`, `page_number`, etc.
- **`TextCleaner`** — Nettoyage et normalisation :
  - Suppression des espaces/sauts de ligne superflus.
  - Suppression des caractères spéciaux indésirables.
  - **Normalisation Unicode**.
  - **Filtrage des documents trop courts** (bruit).
- **`DocumentChunker`** — Découpage :
  - Par **caractères** (`RecursiveCharacterTextSplitter`, `CHUNK_SIZE=500`, `CHUNK_OVERLAP=50`).
  - Par **tokens** (`chunk_by_tokens`, via tiktoken) en option.
  - Génération de **statistiques** (`get_stats`) : nombre de chunks, moyenne de mots/chunk, taille moyenne.
  - Chaque chunk reçoit un `chunk_id` unique et conserve ses métadonnées source.

### 3.2 Vectorisation & recherche hybride

- **`EmbeddingManager`** — Génération d'embeddings :
  - Modèle principal : **Ollama `nomic-embed-text`**.
  - **Mécanisme de fallback** (`fallback_embeddings`) si Ollama indisponible.
  - `embed_documents`, `embed_query`, `get_active_model`, `test_connection`.
- **`VectorStore` (ChromaDB)** — Base vectorielle persistante :
  - `add_documents` avec **déduplication automatique** des chunks (`_filter_duplicates`).
  - `similarity_search` et `similarity_search_with_score`.
  - `delete_document`, `get_collection_stats`, `reset_collection`.
  - Persistance disque dans `./data/chroma_db/`.
- **`BM25Index`** — Index lexical sparse (rank_bm25) :
  - `build`, `add_documents`, `remove_document`, `retrieve`.
  - Tokenisation dédiée, persistance dans `./data/bm25_index.pkl`.
- **`SmartRetriever`** — Recherche dense intelligente :
  - Récupère `k×2` candidats, filtre par **score de similarité minimum (0.3)**.
  - **Reranking hybride** : `0.7 × similarité + 0.3 × chevauchement de mots-clés` (stopwords FR/EN ignorés).
  - `retrieve_with_context_window` : formate les chunks en contexte avec source + page.
- **`HybridRetriever`** — Fusion dense + sparse :
  - Récupère `k×3` candidats de chaque côté (ChromaDB + BM25).
  - **Reciprocal Rank Fusion (RRF)** pondérée : `dense_weight=0.6`, `sparse_weight=0.4`.
  - Déduplication par `chunk_id`, tri par score RRF.
  - C'est le retriever **utilisé en production par le pipeline RAG**.

### 3.3 Pipeline RAG

**`RAGPipeline`** (singleton, chargé une seule fois et mis en cache Streamlit).

- **LLM :** `ChatOllama` (mistral:7b) — `temperature=0.3`, `num_predict=1024`, `top_k=40`, `top_p=0.9`.
- **Vérification de disponibilité** au démarrage (`is_ready`) avec message d'aide si Ollama est éteint.
- **`ask()`** — Réponse synchrone complète :
  1. Récupère le profil et **adapte `k`** selon l'utilisateur.
  2. Retrieval hybride des chunks pertinents.
  3. Construction du contexte (context window).
  4. Injection de l'**historique de conversation**.
  5. Construction du prompt enrichi (contexte + historique + profil).
  6. Génération via le LLM.
  7. Mise à jour de la mémoire + **sauvegarde de l'interaction** (SQLite).
  8. **Évolution ACPE non-bloquante** (si apprentissage adaptatif activé).
  9. Retour : `answer`, `sources` (avec preview + score), `chunks_used`, `model_used`, `context_length`, `interaction_id`.
- **`ask_stream()`** — Version **streaming token par token** pour l'UI (`st.write_stream`).
- **`summarize_document()`** — Résumé automatique d'un document (récupère jusqu'à 20 chunks du fichier puis synthétise).
- **`analyze_knowledge_gaps()`** — Compare les documents disponibles aux **objectifs déclarés** pour identifier les lacunes.
- **`clear_memory()`** — Réinitialise la conversation.
- **`get_pipeline_status()`** — État complet pour l'UI (modèle, URL Ollama, nb messages, modèle d'embeddings, total indexé).
- **Mode identité (Layer 2)** — `enable_identity_mode(enabled)`, `_use_identity_mode()`, `_build_messages()` (bascule prompt système « second cerveau » vs prompt RAG standard), `save_style_correction(query, generated, corrected)`. Désactivé par défaut → **aucun changement de comportement** quand il est OFF.

**`PromptBuilder`** — 3 templates : `build_rag_prompt`, `build_summary_prompt`, `build_knowledge_gap_prompt`.

**`ConversationMemory`** — Mémoire glissante de **10 messages** : `add_user_message`, `add_assistant_message`, `get_formatted_history`, `get_langchain_messages`, **compression d'historique** (`_compress_history`), `clear`, `get_message_count`.

**`BatchSummarizer`** — `summarize_all_documents` (résumé par lot) et `generate_knowledge_report`.

### 3.4 Personnalisation & ACPE (Adaptive Cognitive Profiling Engine)

Le système ACPE apprend en continu du comportement de l'utilisateur. Trois moteurs coordonnés :

#### `KnowledgeProfileEngine` — Profil de connaissances par domaine
- Détecte automatiquement **10 domaines** par mots-clés : Python, Machine Learning, Deep Learning, NLP, RAG, Data Science, SQL & Bases de données, Cybersécurité, DevOps & Cloud, Web Development.
- Met à jour un **score de maîtrise (0–100)** via une **moyenne mobile exponentielle (EMA, α=0.3)**, pondérée par le feedback (👍 renforce, 👎 atténue).
- Apprend aussi **depuis les noms de documents uploadés** (`update_from_document`).
- Calcule un **niveau de confiance logarithmique** selon le nombre d'interactions.
- Expose `get_knowledge_profile`, `get_top_domains`, `get_weak_domains` (axes de progression).

#### `ProfileEvolutionEngine` — Évolution comportementale & insights
- `analyze_interaction` : point d'entrée appelé après chaque échange.
- Suit des **patterns d'utilisation** : total de questions, **longueur moyenne de réponse**, temps de réponse moyen, **heures d'activité**.
- **Détecte les changements d'intérêts** (domaines émergents sur les 20 dernières interactions).
- **Génère des insights lisibles** en français (domaines fréquents, axes de progression, préférence court/long, nouveaux intérêts, jalons d'activité).
- `get_evolution_summary` : agrège connaissances + patterns + insights pour le dashboard.

#### `ProgressiveProfilingEngine` — Suggestions contextuelles
- Propose des micro-suggestions **non intrusives** avec **cooldown intelligent** (≥10 interactions avant la 1ʳᵉ, ≥5 entre deux).
- Règles : « ajouter un domaine fréquent à vos intérêts », « changer de style de réponse selon votre comportement réel », « ajuster votre niveau d'expertise ».
- `check_for_prompts`, `accept_prompt` (applique l'action au profil), `decline_prompt`.
- Affiché dans la colonne droite du chat.

#### `PersonalizedRecommender` — Recommandations & RAG adaptatif
- `get_suggested_questions` : questions suggérées basées sur les topics et l'historique.
- `get_document_recommendations` : documents à (re)consulter.
- **`adapt_rag_parameters`** : adapte `k` (nombre de chunks) et la température **selon le niveau d'expertise** (débutant = plus de contexte, expert = plus ciblé).
- `get_learning_progress` : documents uploadés, score de connaissance, taux de feedback positif.

#### `InteractionHistory` — Historique
- `save_interaction`, `save_feedback` (👍/👎), `get_recent_history`, `get_interaction_stats`, `get_frequent_topics`, `search_history`, `clear_history`.

### 3.5 Know Me Better (KMB)

Questionnaire approfondi et progressif (`pages/know_me_better.py` + `KMBManager`) avec **sauvegarde automatique champ par champ** (callbacks `on_change`).

- **5 sections** :
  1. 👤 Informations personnelles (date de naissance, genre, pays, langues parlées).
  2. 🎨 Intérêts & loisirs (hobbies, curiosités, sujets favoris, types de contenu, communautés suivies).
  3. 📚 Apprentissage & croissance (compétences actuelles/futures, objectifs annuels, style et fréquence d'apprentissage).
  4. 💼 Travail & mode de vie (occupation, secteur, défis quotidiens, motivations).
  5. 💬 Préférences de communication (style, informations complémentaires).
- **Pourcentage de complétion** calculé dynamiquement (`_calculate_completion`) + barre de progression.
- **Deux modes** : wizard pas-à-pas (onboarding) ou expanders (dans la page Profil).
- `has_seen_kmb_onboarding` / `mark_kmb_onboarding_seen` pour ne le présenter qu'une fois.

### 3.6 Onboarding

Wizard d'accueil en **3 étapes** (`pages/onboarding.py`), affiché tant que l'onboarding n'est pas complété :

1. **Type d'utilisateur** : Individuel vs Entreprise (cartes cliquables stylisées).
2. **Questionnaire adapté** :
   - *Individuel* : nom, rôle, niveau d'expertise, objectifs, sujets d'intérêt, style de réponse, langue.
   - *Entreprise* : secteur, taille, cas d'usage, types de documents, niveau de confidentialité, style de communication, langue.
3. **Récapitulatif & confirmation**, avec sauvegarde dans le profil.
- Option **« Ignorer l'onboarding »** avec valeurs par défaut.
- Barre de progression et navigation avant/arrière.

### 3.7 Interface & design system

> Refonte UI récente : un **système de design centralisé** (`ui/theme.py`) applique une apparence moderne et cohérente sur **toutes** les pages.

- **`apply_theme()`** — Injecte un CSS global unique : police **Inter**, palette violette « cognitive », cartes arrondies et ombrées. Restyle automatiquement **tous** les widgets natifs Streamlit : métriques (cartes élevées), boutons (primaire dégradé / secondaire contouré), champs de saisie, bulles de chat, expanders, onglets, dataframes, barres de progression, alertes, sidebar et barre de navigation.
- **`page_header(title, subtitle, icon)`** — En-tête « héro » dégradé en haut de chaque page.
- **`section_title(title)`** — Titre de section avec barre dégradée.
- **`stat_badge(label, kind)`** — Pilules colorées (primary/success/warning/danger/muted).
- **Thème Streamlit** (`.streamlit/config.toml`) : `base=light`, `primaryColor=#6C5CE7`, fonds harmonisés, menu/footer masqués pour un rendu « produit ».
- **Sidebar** : logo de marque + statut Ollama en temps réel + modèle + nombre de documents indexés.
- **Navigation** : `st.segmented_control` stylisé en barre de pilules centrée, synchronisé avec l'URL.

### 3.8 Tableau de bord analytique

Page `dashboard.py`, alimentée par l'historique + l'ACPE :

- **KPIs** : documents, questions posées, score de connaissance /100, taux de satisfaction.
- **Activité récente** : histogramme des interactions par jour.
- **Sujets explorés** : barres de progression par topic.
- **Profil de connaissances** : **graphique radar Plotly** des domaines maîtrisés (fallback bar chart), domaines forts et axes de progression.
- **Insights comportementaux** : phrases générées par l'ACPE.
- **Activité par document** : tableau (chunks, consultations, présence de résumé, date).
- **Dernières interactions** : Q/R détaillées avec sources et feedback.
- **Recommandations** : documents à reconsulter.
- **Aperçu Enterprise** (conditionnel) : secteur, taille, confidentialité, cas d'usage.

### 3.9 Profil, données & confidentialité

Page `profile.py`, organisée en **5 onglets** :

- **⚙️ Préférences** : nom, niveau d'expertise, style de réponse, langue (FR/EN/AR), domaines d'intérêt, objectifs.
- **🎯 Contexte RAG** : prévisualisation **exacte** du contexte injecté dans les prompts (`get_personalization_context`), paramètres RAG adaptés (k, température) **avec explications de transparence**.
- **🔒 Données & confidentialité** :
  - Profil de connaissances détaillé + insights avec explications.
  - **Toggle d'apprentissage adaptatif** (désactive l'analyse automatique).
  - **Export des données** en JSON (`export_profile_data`).
  - Réinitialisation des données ACPE, relance de l'onboarding.
  - **Zone dangereuse** : effacer l'historique, réinitialiser le profil.
  - **Réinitialisation complète du système** (`reset_system`) avec double confirmation : profil + historique + documents + index ChromaDB & BM25.
- **🧠 Know Me Better** : le questionnaire KMB intégré en mode expanders.
- **🧠 Mon identité** (Layer 2) : profil de style (cartes de métriques + « votre voix »), croyances extraites (badges de confiance), résolution de conflits, et **toggle du mode second cerveau**. Voir §3.12.
- En-tête de profil : avatar personnalisable, badges (niveau, type individuel/entreprise, rôle).

### 3.10 Évaluation qualité (RAGAS)

Module `evaluation/` pour mesurer objectivement la qualité du RAG :

- **`RAGEvaluator`** :
  - `load_test_dataset` (depuis `test_dataset.json`).
  - `evaluate_single` et `evaluate` (batch).
  - **Catégorisation automatique** des questions (`_categorize_question`).
  - Métriques RAGAS : **Faithfulness**, **Answer Relevancy**, **Context Precision/Recall**.
- **`EvaluationReporter`** :
  - `get_summary_metrics`, `get_scores_dataframe`, `get_metrics_dataframe`.
  - **Génération d'un rapport Markdown** (`generate_markdown_report`, `export_report`).

### 3.11 Tests & scripts utilitaires

- **Tests pytest** : `test_ingestion`, `test_vectorstore`, `test_rag`, `test_user`, `test_acpe`, `test_kmb`, `test_evaluation`, `test_identity`.
  ```bash
  pytest tests/ -v
  ```
- **Scripts CLI** :
  - `scripts/rebuild_bm25.py` — Reconstruit l'index BM25 depuis ChromaDB.
  - `scripts/test_hybrid_retrieval.py` — Vérifie la recherche hybride.

### 3.12 Identité & mode « second cerveau » (Layer 2)

La **couche d'identité** transforme CogniAssist d'un assistant documentaire en
un **second cerveau** qui connaît la voix, les opinions et l'expertise de
l'utilisateur. Elle s'appuie sur les **écrits personnels** (journal, essais,
notes) importés dans une **collection ChromaDB séparée** (`personal_writing`),
sans jamais mélanger ces données avec les documents classiques.

> ⚙️ **Désactivé par défaut.** Quand le mode identité est OFF, le pipeline se comporte exactement comme avant (aucune régression). Dégradation gracieuse : si Ollama est éteint, l'extraction de croyances échoue silencieusement (toast d'avertissement) sans jamais bloquer l'upload.

#### `StyleAnalyzer` — Empreinte stylistique (`user/style_analyzer.py`)
Analyse rapide (bibliothèque standard uniquement, aucune dépendance NLP) des
écrits pour produire un profil de style :
- **Longueur moyenne de phrase**, **richesse lexicale** (type-token ratio).
- **Score de formalité** (0–1), **ratio de première personne**, **ratio d'atténuation** (hedging).
- **Préférence d'exemples** (`examples_first` / `theory_first` / `balanced`).
- **Longueur préférée** (`short` / `medium` / `long`).
- **Ton dominant** (`direct` / `analytical` / `diplomatic` / `enthusiastic`).
- **Compilation** en un *fragment de prompt* en langage naturel injectable dans le système.
- Persistance : upsert d'une **ligne unique** dans `style_profile` (`save_profile` / `get_profile`).

#### `BeliefExtractor` — Extraction de croyances (`user/belief_extractor.py`)
Passe LLM (Mistral) sur chaque chunk d'écriture personnelle pour extraire des
triplets **(topic, position, confidence)** :
- **Prompt strict** retournant un JSON (max 3 croyances/chunk, sujets spécifiques).
- **Parsing robuste** : tolère le markdown / le texte parasite, retourne `[]` en cas d'échec, **jamais de crash**, **jamais plus d'une tentative** par chunk.
- **Détection de conflits sémantiques** : similarité cosinus entre embeddings ; si même sujet (≥ `BELIEF_CONFLICT_THRESHOLD`) mais positions divergentes → les deux croyances passent au statut `conflicted`.
- **Statuts** : `active`, `conflicted`, `superseded`, `user_confirmed`.
- **Récupération au moment de la requête** : `get_beliefs_for_topic(query, top_k)` (similarité cosinus, uniquement `active`/`user_confirmed`).
- `get_all_beliefs` (pour l'UI), `resolve_conflict(keep, drop)` (l'utilisateur tranche), `extract_from_chunks` (batch avec barre de progression Streamlit).

#### `IdentityPromptBuilder` — Prompt « second cerveau » (`user/identity_prompt_builder.py`)
Assemble le prompt système d'identité au moment de la requête :
- **Style** (fragment depuis `style_profile`, avec fallback générique).
- **Positions pertinentes** sur le sujet de la question (via `BeliefExtractor`).
- **Domaines d'expertise** (via `KnowledgeProfileEngine`).
- **Règles absolues** : ne jamais inventer d'opinion, signaler l'absence de position, marquer l'incertain par `[non confirmé]`, répondre dans la langue de la question.
- `is_identity_mode_ready()` : `True` seulement si un `style_profile` existe **et** au moins `MIN_BELIEFS_FOR_IDENTITY_MODE` croyances sont présentes.

#### Intégration UI
- **Page Documents** → section **« 🧠 Alimenter mon identité »** : upload TXT/MD, date approximative optionnelle, indexation dans `personal_writing`, analyse du style + extraction des croyances (barres de progression), récapitulatif et aperçu des croyances.
- **Page Profil** → onglet **« 🧠 Mon identité »** : métriques de style, « votre voix », liste des croyances avec badges de confiance, **résolution des conflits** (boutons « C'est ma vision actuelle »), **toggle du mode second cerveau**.
- **Page Chat** : indicateur **« 🧠 Mode second cerveau actif »**, et bouton **« ✏️ Ce n'est pas ma façon de dire ça »** sous chaque réponse → zone d'édition → `save_style_correction` (enregistre dans `style_corrections` et recalcule le style).

---

## 4. Modèle de données (SQLite)

| Table | Rôle |
|---|---|
| `user_profiles` | Profil principal (nom, avatar, type, rôle, expertise, onboarding, apprentissage adaptatif…) |
| `user_preferences` | Préférences (style de réponse, langue, domaines, objectifs) |
| `document_access` | Documents enregistrés (chunks, consultations, résumé) |
| `interactions` | Historique Q/R (sources, chunks utilisés, temps de réponse, feedback) |
| `knowledge_profiles` | **ACPE** — maîtrise par domaine (score, confiance, nb interactions) |
| `usage_patterns` | **ACPE** — métriques comportementales (JSON) |
| `progressive_prompts` | **ACPE** — suggestions de profiling progressif (statut, action) |
| `personal_profile_data` (KMB) | Données « Know Me Better » + complétion |
| `style_profile` | **Identité** — empreinte stylistique (ligne unique) + fragment de prompt |
| `belief_store` | **Identité** — croyances (topic, position, confiance, statut, embedding JSON) |
| `style_corrections` | **Identité** — corrections stylistiques (query, généré, corrigé, diff) |

Base vectorielle : **ChromaDB** (`./data/chroma_db/`) — deux collections :
`cogniassist_documents` (documents) et **`personal_writing`** (écrits personnels — Layer 2).
Index lexical : **BM25** (`./data/bm25_index.pkl`).

---

## 5. Stack technologique

| Couche | Technologie | Version |
|---|---|---|
| LLM | Ollama + **Mistral 7B** | local |
| Embeddings | Ollama **nomic-embed-text** | local |
| Orchestration LLM | LangChain + langchain-ollama | 0.3.25 / 0.3.3 |
| Base vectorielle | ChromaDB | 1.0.7 |
| Recherche sparse | rank_bm25 | 0.2.2 |
| Base relationnelle | SQLite + SQLAlchemy | 2.0.41 |
| Interface | Streamlit | 1.45.1 |
| Lecture PDF / DOCX | PyMuPDF / python-docx | 1.25.5 / 1.1.2 |
| Tokenisation | tiktoken | 0.9.0 |
| Validation config | Pydantic + pydantic-settings | 2.11.3 / 2.9.1 |
| Évaluation | RAGAS | 0.4.3 |
| Visualisation | Plotly + Pandas + NumPy | 6.1.2 / 2.3.0 / 2.2.6 |
| Tests | pytest | 8.4.0 |

---

## 6. Configuration

Variables (via `.env`, gérées par `config.py`) :

| Variable | Défaut | Rôle |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL locale d'Ollama |
| `OLLAMA_MODEL` | `mistral:7b` | LLM de génération |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Modèle d'embeddings |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | Répertoire ChromaDB |
| `SQLITE_DB_PATH` | `./data/cogniassist.db` | Base SQLite |
| `UPLOAD_DIR` | `./data/uploads` | Dossier d'uploads |
| `CHUNK_SIZE` | `500` | Taille des chunks (caractères) |
| `CHUNK_OVERLAP` | `50` | Chevauchement entre chunks |
| `MAX_RETRIEVED_DOCS` | `5` | Chunks récupérés par requête |
| `BM25_INDEX_PATH` | `./data/bm25_index.pkl` | Index BM25 |
| `HYBRID_DENSE_WEIGHT` | `0.6` | Poids RRF dense |
| `HYBRID_SPARSE_WEIGHT` | `0.4` | Poids RRF sparse |
| `APP_NAME` | `CogniAssist` | Nom de l'application |
| `DEBUG` | `False` | Mode debug |
| `PERSONAL_WRITING_COLLECTION` | `personal_writing` | Collection ChromaDB des écrits personnels (Layer 2) |
| `IDENTITY_MODE_DEFAULT` | `False` | Mode second cerveau actif par défaut |
| `MIN_BELIEFS_FOR_IDENTITY_MODE` | `3` | Croyances minimales pour activer le mode identité |
| `BELIEF_CONFLICT_THRESHOLD` | `0.85` | Seuil de similarité pour détecter un conflit de croyances |
| `BELIEF_TOPIC_SIMILARITY_THRESHOLD` | `0.80` | Seuil de similarité de sujet pour le regroupement des croyances |

---

## 7. Installation & lancement

```bash
# 1. Environnement virtuel
python -m venv venv
source venv/bin/activate           # macOS/Linux  (venv\Scripts\activate sous Windows)

# 2. Dépendances
pip install -r requirements.txt

# 3. Configuration
cp .env.example .env

# 4. Ollama (prérequis local)
ollama pull mistral:7b
ollama pull nomic-embed-text

# 5. Lancement
streamlit run app.py
# → http://localhost:8501
```

> 💡 Si `streamlit` n'est pas trouvé ou que `langchain` manque, vérifiez que l'environnement virtuel du projet est bien activé (`source venv/bin/activate`) ou lancez directement `./venv/bin/streamlit run app.py`.

---

*Document généré à partir de l'analyse du code source — reflète l'état actuel du projet.*
