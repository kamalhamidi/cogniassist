# 🧠 CogniAssist — Dossier complet du projet (base pour le rapport PFE)

> **Projet de Fin d'Études (PFE) — Master MSID 2025–2026**
> **Auteur :** Kamal HAMIDI
> **Nature :** Second cerveau personnel basé sur le RAG (Retrieval-Augmented Generation), fonctionnant **100 % en local** (aucune donnée envoyée dans le cloud).

Ce document rassemble **toutes les informations nécessaires à la rédaction du rapport**.
Il est organisé selon l'**architecture en 6 couches (layers)** qui décrit le cycle de
vie complet d'une information dans le système : de son entrée jusqu'à l'amélioration
continue du modèle d'identité.

---

## Table des matières

1. [Résumé exécutif](#1-résumé-exécutif)
2. [Le modèle mental en 6 couches](#2-le-modèle-mental-en-6-couches)
3. [Architecture technique globale](#3-architecture-technique-globale)
4. [Layer 1 — Input : ce qui nourrit le cerveau](#4-layer-1--input--ce-qui-nourrit-le-cerveau)
5. [Layer 2 — Identity Extraction : qui es-tu](#5-layer-2--identity-extraction--qui-es-tu)
6. [Layer 3 — Identity Store : ton profil vivant](#6-layer-3--identity-store--ton-profil-vivant)
7. [Layer 4 — Retrieval : quoi récupérer au moment de la requête](#7-layer-4--retrieval--quoi-récupérer-au-moment-de-la-requête)
8. [Layer 5 — Generation : répondre comme toi](#8-layer-5--generation--répondre-comme-toi)
9. [Layer 6 — Feedback Loop : le cerveau s'améliore](#9-layer-6--feedback-loop--le-cerveau-saméliore)
10. [Couches transverses (ACPE, KMB, UI, Dashboard)](#10-couches-transverses)
11. [Modèle de données complet](#11-modèle-de-données-complet-sqlite--chromadb)
12. [Stack technologique](#12-stack-technologique)
13. [Configuration](#13-configuration-configpy)
14. [Tests & qualité](#14-tests--qualité)
15. [Installation & lancement](#15-installation--lancement)
16. [Plan de rapport suggéré](#16-plan-de-rapport-suggéré)

---

## 1. Résumé exécutif

**CogniAssist** est un assistant cognitif personnel qui transforme les documents et
les écrits d'un utilisateur en un **second cerveau** capable de répondre à ses
questions, dans **sa voix**, avec **ses opinions**, et de **s'améliorer** à chaque
interaction.

Le système combine :

- un **RAG local** (Mistral 7B + ChromaDB + recherche hybride dense/sparse) ;
- un **moteur de profilage cognitif adaptatif (ACPE)** qui apprend le comportement ;
- une **couche d'identité** qui extrait le style d'écriture et les croyances ;
- une **boucle de rétroaction** qui fait progresser le modèle d'identité en continu.

**Proposition de valeur :** *« Plus Kamal utilise CogniAssist, mieux CogniAssist
devient Kamal. »*

| Pilier | Description |
|---|---|
| 🔒 **Confidentialité** | 100 % local (Ollama + ChromaDB + SQLite). Aucune API cloud. |
| 🎯 **Personnalisation** | Profil + RAG adaptatif + ACPE + Know Me Better. |
| 🧬 **Identité (second cerveau)** | Style, croyances, mode « répond comme moi ». |
| 🔁 **Auto-amélioration** | Corrections, pouces, dérive, fidélité vocale. |
| ⚡ **Réactivité** | Réponses en streaming token par token. |
| 🧩 **Recherche hybride** | Dense (embeddings) + sparse (BM25) via RRF. |
| 🧪 **Qualité mesurable** | Évaluation RAGAS + score de fidélité d'identité. |

---

## 2. Le modèle mental en 6 couches

Chaque information traverse le système selon ce cycle :

```
                         ┌──────────────────────────────────────────────┐
                         │                                              │
   (1) INPUT  ──►  (2) IDENTITY  ──►  (3) IDENTITY   ──►  (4) RETRIEVAL  │
   documents       EXTRACTION         STORE              quoi récupérer  │
   écrits perso    style+croyances    profil vivant      au query time   │
                         │                                      │        │
                         │                                      ▼        │
                         │                              (5) GENERATION    │
                         │                              répondre comme toi│
                         │                                      │        │
                         └──────────────  (6) FEEDBACK LOOP  ◄──┘        │
                              le cerveau s'améliore en continu  ─────────┘
```

| Layer | Rôle | Question posée | Modules clés |
|---|---|---|---|
| **1 — Input** | Ingérer les sources | *Qu'est-ce qui nourrit le cerveau ?* | `ingestion/`, `pages/upload.py` |
| **2 — Identity Extraction** | Extraire l'identité | *Qui es-tu ?* | `style_analyzer.py`, `belief_extractor.py` |
| **3 — Identity Store** | Stocker le profil vivant | *Que sait-on de toi ?* | `identity_models.py`, SQLite, ChromaDB |
| **4 — Retrieval** | Récupérer le bon contexte | *Quoi tirer au query time ?* | `retriever.py`, `belief_extractor.get_beliefs_for_topic` |
| **5 — Generation** | Générer la réponse | *Comment répondre comme toi ?* | `pipeline.py`, `identity_prompt_builder.py` |
| **6 — Feedback Loop** | Apprendre des retours | *Comment s'améliorer ?* | `feedback_engine.py` |

---

## 3. Architecture technique globale

```
cogniassist/
├── app.py                     # Point d'entrée Streamlit (navigation, gates, routage)
├── config.py                  # Configuration centralisée (Pydantic Settings + .env)
│
├── ui/                        # 🎨 Système de design
│   ├── __init__.py            # Exports : apply_theme, page_header, section_title, stat_badge
│   └── theme.py               # CSS global + composants d'interface
│
├── ingestion/                 # LAYER 1 — Transformation fichiers → chunks
│   ├── loader.py              # PDF (PyMuPDF) / DOCX / TXT / bytes
│   ├── cleaner.py             # Nettoyage, normalisation Unicode, filtrage
│   └── chunker.py             # Découpage (caractères/tokens) + statistiques
│
├── vectorstore/               # LAYER 3 + 4 — Embeddings & recherche
│   ├── embedder.py            # Embeddings Ollama (nomic-embed-text) + fallback
│   ├── store.py               # ChromaDB : collections documents + personal_writing
│   ├── bm25_index.py          # Index sparse BM25
│   └── retriever.py           # SmartRetriever (dense) + HybridRetriever (RRF)
│
├── rag/                       # LAYER 5 — Cœur RAG (génération)
│   ├── pipeline.py            # RAGPipeline : ask, ask_stream, mode identité, feedback
│   ├── prompt_builder.py      # Prompts RAG / résumé / lacunes (identité = CogniAssist)
│   ├── memory.py              # Mémoire de conversation glissante (10 tours)
│   └── summarizer.py          # Résumé par lot + rapport de connaissances
│
├── user/                      # LAYERS 2, 3, 6 + ACPE — Identité & personnalisation
│   ├── db.py                  # SQLite/SQLAlchemy + migrations + reset système
│   ├── profile.py             # UserProfile / Preferences / DocumentAccess
│   ├── history.py             # Historique des interactions + feedback
│   ├── recommender.py         # Recommandations + adaptation des paramètres RAG
│   ├── acpe_models.py         # Tables ACPE (KnowledgeProfile, UsagePattern, Prompt)
│   ├── knowledge_engine.py    # Profil de connaissances par domaine (EMA)
│   ├── profile_evolution.py   # Détection de tendances + insights
│   ├── progressive.py         # Profiling progressif (suggestions, cooldown)
│   ├── kmb_service.py         # Service « Know Me Better »
│   ├── identity_models.py     # LAYER 3 — tables identité + feedback (7 tables)
│   ├── style_analyzer.py      # LAYER 2 — empreinte stylistique (stdlib)
│   ├── belief_extractor.py    # LAYER 2 + 4 — extraction & récupération de croyances
│   ├── identity_prompt_builder.py  # LAYER 5 — prompt « second cerveau »
│   └── feedback_engine.py     # LAYER 6 — boucle de rétroaction (6 signaux)
│
├── evaluation/                # Mesure de qualité
│   ├── ragas_eval.py          # RAGAS (faithfulness, relevancy, precision, recall)
│   ├── report.py              # Rapports Markdown + DataFrames
│   └── test_dataset.json      # Dataset de référence Q/R
│
├── pages/                     # Pages Streamlit
│   ├── chat.py                # 💬 Chat (modes normal / second cerveau)
│   ├── upload.py              # 📁 Documents + « Alimenter mon identité »
│   ├── dashboard.py           # 📊 Tableau de bord + section Apprentissage
│   ├── profile.py             # 👤 Profil + onglet « Mon identité »
│   ├── onboarding.py          # 🚀 Wizard d'onboarding (3 étapes)
│   └── know_me_better.py      # 🧠 Questionnaire KMB (5 sections)
│
├── scripts/                   # Outils CLI (rebuild_bm25, test_hybrid_retrieval)
├── tests/                     # Tests pytest (ingestion, rag, user, acpe, kmb,
│                              #   evaluation, identity, feedback_loop)
├── data/                      # ChromaDB, SQLite, uploads, bm25_index.pkl
└── assets/                    # logo.png
```

**Flux applicatif (`app.py`) :** thème global → sidebar (statut Ollama) → **gate
onboarding** → barre de navigation segmentée → **gate Know Me Better** → routage de
page. Navigation synchronisée avec les `query_params` de l'URL.

---

## 4. Layer 1 — Input : ce qui nourrit le cerveau

> **But :** transformer des fichiers et des écrits bruts en unités exploitables
> (chunks) prêtes à être vectorisées. C'est la porte d'entrée des connaissances
> ET de l'identité.

### Deux types d'entrées

| Type d'entrée | Source | Destination | Usage |
|---|---|---|---|
| **Documents** | PDF, DOCX, TXT, MD | collection ChromaDB `documents` | base de connaissances (RAG) |
| **Écrits personnels** | TXT, MD (journal, essais, notes) | collection ChromaDB `personal_writing` | extraction d'identité (Layer 2) |

La séparation est **stricte** : les écrits personnels ne polluent jamais la base
documentaire, et inversement.

### Composants (`ingestion/`)

- **`DocumentLoader`** — chargement depuis un chemin **ou depuis des bytes**
  (`load_from_bytes`, utilisé par l'upload Streamlit) :
  - PDF via **PyMuPDF (fitz)** : extraction page par page (n° de page en métadonnée) ;
  - DOCX via **python-docx** : paragraphe par paragraphe ;
  - TXT/MD : lecture brute avec détection d'encodage ;
  - métadonnées attachées : `file_name`, `file_type`, `page_number`…
- **`TextCleaner`** — nettoyage : espaces superflus, caractères indésirables,
  **normalisation Unicode**, **filtrage des textes trop courts** (bruit).
- **`DocumentChunker`** — découpage :
  - par **caractères** (`RecursiveCharacterTextSplitter`, `CHUNK_SIZE=500`,
    `CHUNK_OVERLAP=50`) ;
  - par **tokens** (`chunk_by_tokens`, via tiktoken) en option ;
  - **statistiques** (`get_stats`) : nb de chunks, mots/chunk, taille moyenne ;
  - chaque chunk reçoit un `chunk_id` unique + ses métadonnées source.

### Interfaces utilisateur

- **Page Documents (`pages/upload.py`)** : upload multi-fichiers, indexation,
  liste des documents, résumé automatique, suppression.
- **Section « 🧠 Alimenter mon identité »** : upload TXT/MD, **date approximative
  optionnelle**, indexation dans `personal_writing`, déclenchement de l'analyse de
  style + extraction de croyances avec barres de progression.

### Fonctionnalités clés (Layer 1)

- Multi-formats (PDF/DOCX/TXT/MD) + upload par bytes.
- Double pipeline d'ingestion (documents vs écrits personnels).
- Déduplication par `chunk_id`, métadonnées riches.
- Statistiques de découpage exposées à l'UI.

---

## 5. Layer 2 — Identity Extraction : qui es-tu

> **But :** à partir des écrits personnels, extraire **comment** l'utilisateur
> écrit (style) et **ce qu'il pense** (croyances). C'est la transformation
> « texte brut → identité structurée ».

### 5.1 Analyse de style — `StyleAnalyzer` (`user/style_analyzer.py`)

Analyse **sans aucune dépendance NLP externe** (uniquement `re`, `string`,
`collections`) — rapide et non bloquant. Métriques calculées :

| Métrique | Description |
|---|---|
| `avg_sentence_len` | nombre moyen de mots par phrase |
| `vocabulary_richness` | type-token ratio (mots uniques / total) |
| `formality_score` | 0 (familier) → 1 (formel), via lexiques FR+EN |
| `first_person_ratio` | fréquence des marqueurs de première personne |
| `hedging_ratio` | fréquence des formules d'atténuation (« peut-être »…) |
| `example_preference` | `examples_first` / `theory_first` / `balanced` |
| `preferred_length` | `short` / `medium` / `long` |
| `tone` | `direct` / `analytical` / `diplomatic` / `enthusiastic` |

Ces métriques sont **compilées en un fragment de prompt en langage naturel**
(`_compile_style_prompt`) injectable dans le système — *« Écris des phrases
concises… Adopte un ton direct… »*.

### 5.2 Extraction de croyances — `BeliefExtractor` (`user/belief_extractor.py`)

Passe LLM (Mistral) sur chaque chunk d'écriture pour extraire des triplets
**(topic, position, confidence)** :

- **Prompt strict** retournant un JSON (max 3 croyances/chunk, sujets spécifiques).
- **Parsing robuste** (`_parse_beliefs`) : tolère le markdown / le texte parasite,
  retourne `[]` en cas d'échec, **jamais de crash**, **une seule tentative** par chunk.
- **Embedding** de chaque croyance (`nomic-embed-text`) pour la similarité.
- **Détection de conflits sémantiques** (`_check_conflict`) : similarité cosinus ≥
  `BELIEF_CONFLICT_THRESHOLD` (0.85) sur le sujet mais positions divergentes → les
  deux croyances passent au statut `conflicted`.

### Fonctionnalités clés (Layer 2)

- Empreinte stylistique multi-dimensionnelle (8 métriques) FR + EN.
- Compilation automatique style → instruction en langage naturel.
- Extraction LLM de croyances avec confiance graduée.
- Détection de contradictions internes (conflits).
- Robustesse totale (dégradation gracieuse si Ollama est éteint).

---

## 6. Layer 3 — Identity Store : ton profil vivant

> **But :** persister l'identité de façon structurée et **évolutive**. Ce n'est pas
> un instantané figé mais un **profil vivant** qui se met à jour, garde un historique
> et trace son évolution.

### Statuts d'une croyance (cycle de vie)

```
   active ──────────────►  user_confirmed   (l'utilisateur valide)
     │                         ▲
     │ conflit détecté         │ 👍 / confirmation
     ▼                         │
  conflicted ─────────────► superseded       (remplacée / rejetée / archivée)
                  (résolution / 👎 répétés / changement d'avis)
```

### Tables d'identité (`user/identity_models.py`)

| Table | Rôle | Mutabilité |
|---|---|---|
| `style_profile` | empreinte stylistique (1 ligne) + fragment de prompt + `reinforcement_count` | mutable (upsert) |
| `belief_store` | croyances (topic, position, confiance, statut, embedding JSON) | mutable |
| `style_corrections` | corrections stylistiques de l'utilisateur (query, généré, corrigé, `processed`) | append + flag |

### Stockage vectoriel

- **ChromaDB** — collection dédiée `personal_writing` (séparée des documents).
- Embeddings des croyances stockés en JSON dans `belief_store.embedding_json`
  pour la similarité au moment de la requête.

### Fonctionnalités clés (Layer 3)

- Profil de style **unique et upserté** (toujours la dernière version).
- Croyances **versionnées par statut** (cycle de vie complet).
- Séparation physique des données d'identité (collection ChromaDB dédiée).
- Base d'un **historique immuable** (complété en Layer 6 : `belief_timeline`).

---

## 7. Layer 4 — Retrieval : quoi récupérer au moment de la requête

> **But :** au moment d'une question, sélectionner **le bon contexte** — à la fois
> documentaire (connaissances) et identitaire (croyances pertinentes).

### 7.1 Recherche documentaire hybride (`vectorstore/retriever.py`)

- **`EmbeddingManager`** — embeddings via **Ollama `nomic-embed-text`** + fallback.
- **`SmartRetriever`** — recherche dense (ChromaDB) + `retrieve_with_context_window`
  (formate les chunks en contexte avec source + page).
- **`HybridRetriever`** — fusion **dense + sparse** :
  - récupère `k×3` candidats de chaque côté (ChromaDB + BM25) ;
  - **Reciprocal Rank Fusion (RRF)** pondérée : `dense_weight=0.6`,
    `sparse_weight=0.4` ;
  - déduplication par `chunk_id`, tri par score RRF ;
  - **retriever utilisé en production**.

### 7.2 Récupération des croyances pertinentes

- `BeliefExtractor.get_beliefs_for_topic(query, top_k)` : embed la requête, calcule
  la similarité cosinus avec les croyances `active`/`user_confirmed`, retourne les
  plus proches (avec `id`, `topic`, `position`, `confidence`, `date_written`).

### 7.3 Adaptation des paramètres (ACPE)

- `PersonalizedRecommender.adapt_rag_parameters()` : ajuste **k** (nombre de chunks)
  et la **température** selon le niveau d'expertise (débutant = plus de contexte,
  expert = plus ciblé).

### Fonctionnalités clés (Layer 4)

- Recherche hybride dense + sparse (RRF) — robuste au vocabulaire et au sens.
- Récupération sémantique des opinions pertinentes au sujet.
- Paramètres de récupération adaptés au profil utilisateur.
- Fenêtre de contexte formatée avec traçabilité des sources.

---

## 8. Layer 5 — Generation : répondre comme toi

> **But :** générer la réponse. Deux personnalités selon le mode :
> **CogniAssist** (mode normal) ou **l'utilisateur lui-même** (mode second cerveau).

### 8.1 Pipeline RAG (`rag/pipeline.py`)

- **LLM :** `ChatOllama` (mistral:7b) — `temperature=0.3`, `num_predict=1024`,
  `top_k=40`, `top_p=0.9`.
- **`ask()`** (synchrone) et **`ask_stream()`** (streaming token par token).
- Vérification de disponibilité d'Ollama au démarrage (`is_ready`).
- Mémoire de conversation glissante (10 messages), sauvegarde de l'interaction,
  évolution ACPE non bloquante.

### 8.2 Deux identités, deux prompts

| Mode | Prompt système | Identité de la réponse |
|---|---|---|
| **Normal** (`prompt_builder.build_rag_prompt`) | « Tu es **CogniAssist**… » | « Bonjour, je suis CogniAssist » |
| **Second cerveau** (`identity_prompt_builder.build_system_prompt`) | « Tu **ES Kamal**, pas une IA… » | « Bonjour, je suis Kamal » |

Le routage est géré par `_build_messages()` : si le mode identité est **activé ET
prêt** (`_use_identity_mode()`), on injecte le prompt « second cerveau » ; sinon
comportement RAG standard (aucune régression).

### 8.3 Prompt « second cerveau » (`identity_prompt_builder.py`)

Assemblé au moment de la requête à partir de :

1. **Style** (fragment depuis `style_profile`, fallback générique) ;
2. **Positions pertinentes** sur le sujet (via Layer 4) ;
3. **Domaines d'expertise** (via `KnowledgeProfileEngine`).

**Règles absolues** injectées :
1. Ne jamais inventer d'opinion absente des écrits.
2. Signaler explicitement l'absence de position.
3. Marquer l'incertain par `[non confirmé]`.
4. Répondre dans la langue de la question.
5. Ne **jamais recopier** les annotations internes (`[conviction forte]`,
   `(écrit le …)`) dans la réponse — elles sont affichées séparément en petit.

`is_identity_mode_ready()` : vrai seulement si un `style_profile` existe **et** ≥
`MIN_BELIEFS_FOR_IDENTITY_MODE` (3) croyances sont présentes.

### 8.4 Transparence (page Chat)

- Indicateur **« 🧠 Mode second cerveau actif »**.
- **Score de fidélité vocale** affiché sous chaque réponse (🎯 / 📝 / ⚠️).
- Expander **« 🔍 Pourquoi cette réponse ? »** : croyances mobilisées (avec conviction
  + date en petit), fragment de style appliqué, score de fidélité.

### Fonctionnalités clés (Layer 5)

- Double personnalité (CogniAssist vs utilisateur) selon le mode.
- Génération en streaming.
- Prompt d'identité assemblé dynamiquement (style + croyances + expertise).
- Garde-fous anti-hallucination (5 règles absolues).
- Transparence complète du raisonnement.

---

## 9. Layer 6 — Feedback Loop : le cerveau s'améliore

> **But :** chaque interaction (correction, pouce, édition, changement d'avis)
> devient un **signal d'entraînement** qui affine le modèle d'identité.
> *Le cerveau devient meilleur à être Kamal plus Kamal l'utilise.*

### Les 6 signaux (`user/feedback_engine.py`)

| # | Signal | Déclencheur | Effet |
|---|---|---|---|
| 1 | **Correction stylistique** | l'utilisateur réécrit une réponse | sauvegarde + résumé du diff (LLM) + seuil → recalibration |
| 2 | **Confirmation / rejet de croyance** | boutons dans le profil | `user_confirmed` / `superseded` + timeline |
| 3 | **Amplification 👍 / 👎** | pouces sur une réponse | 👍 booste / 👎 décroît la confiance des croyances utilisées |
| 4 | **Détection de dérive** | tous les 20 interactions | compare corrections récentes au profil → recalibration |
| 5 | **Changement d'avis explicite** | « 💭 Ma pensée a évolué » | archive l'ancienne croyance, crée la nouvelle |
| 6 | **Score de fidélité d'identité** | après chaque réponse en mode identité | mesure la correspondance réponse ↔ style |

### Mécaniques détaillées

- **Recalibration du style** (`_recalibrate_style`) : recalcule le profil à partir
  des **textes corrigés** (ce que l'utilisateur voulait vraiment), combinés à la base
  d'écrits pour rester stable ; journalise *avant/après* dans `style_calibration_log`.
- **Confiance graduée** : 👍 `low→medium` (medium/high inchangés) ; 👎 `high→medium→low` ;
  après `BELIEF_CONFIDENCE_DECAY_STEPS` (3) 👎 sur une croyance → `conflicted`.
- **Fidélité vocale** (`score_response_fidelity`) : compare longueur de phrase,
  formalité et première personne de la réponse au profil → score 0–1 dans
  `identity_fidelity_log` (tendance : *improving / stable / declining*).
- **Historique immuable** : `belief_timeline` et `style_calibration_log` sont
  **append-only** (audit complet de l'évolution de la pensée).

### Tables Layer 6 (`user/identity_models.py`)

| Table | Rôle | Mutabilité |
|---|---|---|
| `feedback_signals` | chaque évènement de feedback + contexte + delta | append |
| `style_calibration_log` | recalculs de style (avant/après, delta_score, raison) | **append-only** |
| `belief_timeline` | historique des changements de croyances | **append-only** |
| `identity_fidelity_log` | score de fidélité par réponse | append |

### Intégration UI

- **Chat** : indicateur de fidélité, expander de transparence, pouces routés vers la
  boucle, bouton « Ce n'est pas ma façon de dire ça », input « Ma pensée a évolué ».
- **Profil → 🧠 Mon identité** : croyances éditables (✏️), cartes de conflit à deux
  versions, **historique d'évolution** (timeline), **courbe de fidélité vocale**
  (Plotly) + tendance.
- **Dashboard → 🔁 Apprentissage** : KPIs (corrections, calibrations, croyances
  confirmées, fidélité moyenne), tendance de fidélité, carte d'actions en attente
  (résoudre les conflits / recalibrer).

### Garanties de conception

- **Zéro régression** : purement additif, désactivé par défaut.
- **Non bloquant** : fidélité, dérive et recalibration s'exécutent **après**
  l'affichage de la réponse ; chaque handler est encapsulé dans try/except.
- **Source unique de vérité** : tous les seuils dans `config.py`.

---

## 10. Couches transverses

Ces systèmes ne sont pas des layers du cycle d'identité mais soutiennent l'ensemble.

### 10.1 ACPE — Adaptive Cognitive Profiling Engine

- **`KnowledgeProfileEngine`** : maîtrise par domaine (10 domaines), score 0–100 via
  **moyenne mobile exponentielle (EMA, α=0.3)** pondérée par le feedback ; apprend
  aussi des noms de documents ; confiance logarithmique.
- **`ProfileEvolutionEngine`** : patterns d'usage, détection de nouveaux intérêts,
  **insights** en français pour le dashboard.
- **`ProgressiveProfilingEngine`** : micro-suggestions non intrusives avec cooldown.
- **`PersonalizedRecommender`** : questions suggérées, recommandations de documents,
  **adaptation des paramètres RAG**.

### 10.2 Know Me Better (KMB)

Questionnaire en **5 sections** (infos perso, intérêts, apprentissage, travail,
communication) avec **sauvegarde champ par champ** et pourcentage de complétion.

### 10.3 Onboarding

Wizard en **3 étapes** (type individuel/entreprise → questionnaire adapté →
récapitulatif), avec option « ignorer ».

### 10.4 Interface & design system (`ui/theme.py`)

CSS global (police Inter, palette violette), composants réutilisables
(`page_header`, `section_title`, `stat_badge`), restylage de tous les widgets natifs,
navigation segmentée fixe, sidebar de statut.

### 10.5 Dashboard analytique

KPIs, activité, sujets explorés, **radar Plotly** des domaines, insights, activité
par document, dernières interactions, recommandations, **section Apprentissage
(Layer 6)**, aperçu Enterprise conditionnel.

### 10.6 Évaluation qualité (RAGAS)

`RAGEvaluator` (faithfulness, answer relevancy, context precision/recall) +
`EvaluationReporter` (rapports Markdown).

---

## 11. Modèle de données complet (SQLite + ChromaDB)

### Tables SQLite

| Table | Layer | Rôle |
|---|---|---|
| `user_profiles` | transverse | profil principal (nom, avatar, type, rôle, expertise…) |
| `user_preferences` | transverse | style de réponse, langue, domaines, objectifs |
| `document_access` | 1 | documents enregistrés (chunks, consultations, résumé) |
| `interactions` | 5/6 | historique Q/R (+ `beliefs_used_json`, feedback) |
| `knowledge_profiles` | ACPE | maîtrise par domaine (score, confiance) |
| `usage_patterns` | ACPE | métriques comportementales (JSON) |
| `progressive_prompts` | ACPE | suggestions de profiling progressif |
| `personal_profile_data` | KMB | données « Know Me Better » + complétion |
| `style_profile` | 3 | empreinte stylistique + fragment + `reinforcement_count` |
| `belief_store` | 3 | croyances (topic, position, confiance, statut, embedding) |
| `style_corrections` | 3/6 | corrections stylistiques (+ `processed`) |
| `feedback_signals` | 6 | évènements de feedback (contexte, delta, type) |
| `style_calibration_log` | 6 | recalculs de style (avant/après) — *append-only* |
| `belief_timeline` | 6 | historique des croyances — *append-only* |
| `identity_fidelity_log` | 6 | scores de fidélité par réponse |

### ChromaDB (`./data/chroma_db/`)

- `documents` (ou `cogniassist_documents`) — chunks des documents + embeddings.
- **`personal_writing`** — chunks des écrits personnels (Layer 1 → 2).

### Index lexical

- **BM25** (`./data/bm25_index.pkl`) — recherche sparse.

---

## 12. Stack technologique

| Couche | Technologie | Version |
|---|---|---|
| LLM | Ollama + **Mistral 7B** | local |
| Embeddings | Ollama **nomic-embed-text** | local |
| Orchestration LLM | LangChain + langchain-ollama | 0.3.x |
| Base vectorielle | ChromaDB | 1.0.x |
| Recherche sparse | rank_bm25 | 0.2.x |
| Base relationnelle | SQLite + SQLAlchemy | 2.0.x |
| Interface | Streamlit | 1.45.x |
| Lecture PDF / DOCX | PyMuPDF / python-docx | — |
| Tokenisation | tiktoken | — |
| Validation config | Pydantic + pydantic-settings | 2.x |
| Évaluation | RAGAS | 0.4.x |
| Visualisation | Plotly + Pandas + NumPy | — |
| Tests | pytest | 8.x |

> Aucune dépendance cloud. Tout s'exécute en local pour garantir la confidentialité.

---

## 13. Configuration (`config.py`)

Variables principales (gérées par Pydantic Settings, surchargées via `.env`) :

| Variable | Défaut | Layer | Rôle |
|---|---|---|---|
| `OLLAMA_MODEL` | `mistral:7b` | 5 | LLM de génération |
| `EMBEDDING_MODEL` | `nomic-embed-text` | 4 | modèle d'embeddings |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `50` | 1 | découpage |
| `MAX_RETRIEVED_DOCS` | `5` | 4 | chunks récupérés |
| `HYBRID_DENSE_WEIGHT` / `HYBRID_SPARSE_WEIGHT` | `0.6` / `0.4` | 4 | poids RRF |
| `PERSONAL_WRITING_COLLECTION` | `personal_writing` | 1/3 | collection écrits perso |
| `IDENTITY_MODE_DEFAULT` | `False` | 5 | mode second cerveau par défaut |
| `MIN_BELIEFS_FOR_IDENTITY_MODE` | `3` | 5 | croyances mini pour activer |
| `BELIEF_CONFLICT_THRESHOLD` | `0.85` | 2 | seuil de conflit |
| `BELIEF_TOPIC_SIMILARITY_THRESHOLD` | `0.80` | 6 | regroupement par sujet |
| `STYLE_RECALIBRATION_THRESHOLD` | `5` | 6 | corrections avant recalibration |
| `DRIFT_DETECTION_INTERVAL` | `20` | 6 | fréquence de détection de dérive |
| `DRIFT_THRESHOLD` | `0.25` | 6 | seuil de dérive |
| `FIDELITY_SCORE_WINDOW` | `30` | 6 | fenêtre de moyenne de fidélité |
| `BELIEF_CONFIDENCE_DECAY_STEPS` | `3` | 6 | 👎 avant `conflicted` |
| `THUMBS_CONFIDENCE_BOOST` | `True` | 6 | les 👍 boostent la confiance |

---

## 14. Tests & qualité

- **Suites pytest** : `test_ingestion`, `test_vectorstore`, `test_rag`, `test_user`,
  `test_acpe`, `test_kmb`, `test_evaluation`, **`test_identity`** (Layer 2/3/5),
  **`test_feedback_loop`** (Layer 6).
- **Approche** : base SQLite isolée + doubles (`FakeLLM`, `FakeEmbedder`) → aucun
  appel réseau réel.
- **Couverture Layer 6** : signal de correction, archivage timeline, rejet +
  remplacement, décroissance 👎 → conflicted, boost 👍, dérive nulle, log de fidélité,
  changement d'avis, log de recalibration, structure du résumé d'apprentissage.
- **État actuel** : **138 tests passent**, 0 régression ; l'application démarre
  (HTTP 200 en headless).

```bash
pytest tests/ -v
```

---

## 15. Installation & lancement

```bash
# 1. Environnement virtuel
python -m venv venv
source venv/bin/activate            # Windows : venv\Scripts\activate

# 2. Dépendances
pip install -r requirements.txt

# 3. Ollama (prérequis local)
ollama pull mistral:7b
ollama pull nomic-embed-text

# 4. Lancement
streamlit run app.py                # → http://localhost:8501
```

> 💡 Si `langchain`/`streamlit` est introuvable, activez le venv du projet
> (`source venv/bin/activate`) ou lancez `./venv/bin/streamlit run app.py`.

### Parcours de démonstration (end-to-end)

1. **Layer 1** — Importer des écrits personnels (Documents → Alimenter mon identité).
2. **Layer 2/3** — Vérifier le style + les croyances extraites (Profil → Mon identité).
3. **Layer 5** — Activer le mode second cerveau, poser une question → réponse « à la
   première personne » + score de fidélité.
4. **Layer 6** — 👎 une réponse 3× (la croyance devient `conflicted`), réécrire une
   réponse 5× (recalibration), résoudre un conflit (timeline mise à jour), consulter
   la section Apprentissage du dashboard.

---

## 16. Plan de rapport suggéré

1. **Introduction** — contexte, problématique (assistant générique vs second cerveau
   personnel), objectifs, confidentialité locale.
2. **État de l'art** — RAG, LLM locaux, mémoire/personnalisation des agents,
   profilage utilisateur.
3. **Architecture en 6 couches** — vue d'ensemble + justification du modèle mental.
4. **Layer 1 — Input** — ingestion multi-formats, double pipeline.
5. **Layer 2 — Identity Extraction** — analyse de style, extraction de croyances LLM.
6. **Layer 3 — Identity Store** — modèle de données, cycle de vie des croyances.
7. **Layer 4 — Retrieval** — recherche hybride (RRF), récupération de croyances.
8. **Layer 5 — Generation** — double identité, prompt builder, garde-fous.
9. **Layer 6 — Feedback Loop** — les 6 signaux, fidélité, dérive, historique immuable.
10. **Systèmes transverses** — ACPE, KMB, UI, dashboard, évaluation RAGAS.
11. **Implémentation & choix techniques** — stack 100 % locale, robustesse,
    non-blocage, source unique de vérité.
12. **Tests & évaluation** — stratégie de tests, RAGAS, score de fidélité d'identité.
13. **Résultats & démonstration** — captures d'écran, parcours end-to-end.
14. **Limites & perspectives** — passage multi-utilisateurs, modèles plus grands,
    extraction multimodale, métriques de fidélité enrichies.
15. **Conclusion**.

---

*Document de référence généré à partir de l'analyse du code source — reflète l'état
actuel du projet (architecture en 6 couches, Layers 1 à 6 implémentés).*
