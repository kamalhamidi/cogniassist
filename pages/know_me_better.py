"""
pages/know_me_better.py — Page du questionnaire "Know Me Better" (KMB).

Présente un wizard convivial et interactif pour apprendre à connaître
l'utilisateur de façon progressive. Gère la sauvegarde automatique.
"""

import streamlit as st
from datetime import date, datetime

# Options prédéfinies pour les questionnaires
GENDER_OPTIONS = ["Prefer not to say", "Male", "Female", "Other"]

HOBBY_OPTIONS = [
    "Reading", "Gaming", "Sports", "Music", "Movies and TV", "Traveling",
    "Cooking", "Photography", "Technology", "Art", "Writing", "Fitness",
    "Gardening", "Other"
]

CONTENT_OPTIONS = [
    "Books", "Articles", "YouTube", "Podcasts", "Online Courses",
    "Social Media", "Research Papers"
]

LEARNING_STYLE_OPTIONS = [
    "Reading", "Watching videos", "Practical exercises", "Discussions",
    "Step-by-step guidance", "Trial and error"
]

LEARNING_FREQUENCY_OPTIONS = [
    "", "Daily", "Weekly", "Monthly", "Occasionally"
]

OCCUPATION_OPTIONS = [
    "", "Student", "Employee", "Freelancer", "Entrepreneur", "Researcher",
    "Teacher", "Job Seeker", "Retired", "Other"
]

COMMUNICATION_OPTIONS = [
    "", "Short and direct", "Friendly and conversational",
    "Detailed and explanatory", "Step-by-step", "Professional and concise"
]

LANGUAGES_OPTIONS = [
    "Français", "Anglais", "Arabe", "Espagnol", "Allemand", "Italien", "Chinois", "Autre"
]


def save_kmb_field(field_name: str, key: str) -> None:
    """Callback de sauvegarde automatique appelée lors des changements de widgets."""
    val = st.session_state.get(key)
    
    # Gestion spéciale des dates
    if isinstance(val, date):
        val = val.isoformat()
        
    user_id = st.session_state.get("user_id", "default")
    from user import get_kmb_manager
    kmb_mgr = get_kmb_manager(user_id)
    kmb_mgr.update_kmb_field(field_name, val)


def show_kmb_page(is_onboarding: bool = False) -> None:
    """
    Affiche l'interface 'Know Me Better'.
    
    En mode onboarding: affiche une section à la fois avec navigation.
    En mode profil: n'affiche pas les boutons de navigation (géré par expanders).
    """
    user_id = st.session_state.get("user_id", "default")
    from user import get_kmb_manager
    kmb_mgr = get_kmb_manager(user_id)
    kmb_data = kmb_mgr.get_kmb_data()

    # Initialiser l'étape dans la session
    if "kmb_step" not in st.session_state:
        st.session_state.kmb_step = 0

    st.markdown("### 🧠 Know Me Better")
    st.markdown(
        "*Aidez CogniAssist à mieux vous comprendre afin de vous fournir "
        "des réponses, recommandations et une assistance personnalisées selon vos besoins.*"
    )

    # 1. Barre de progression et complétion
    completion = kmb_data.get("completion_percentage", 0)
    col_pct, col_prog = st.columns([1, 4])
    with col_pct:
        st.metric("Complétion", f"{completion}%")
    with col_prog:
        st.write("")
        st.write("")
        st.progress(completion / 100.0)

    # Définition des sections
    sections = [
        {"title": "👤 Informations personnelles", "desc": "Ces informations permettent de calibrer les réponses générales et la langue."},
        {"title": "🎨 Intérêts & Loisirs", "desc": "Vos centres d'intérêt aident CogniAssist à enrichir ses réponses avec des thèmes connexes."},
        {"title": "📚 Apprentissage & Croissance", "desc": "Comprendre comment vous apprenez permet d'adapter la pédagogie de l'assistant."},
        {"title": "💼 Travail & Mode de vie", "desc": "Savoir comment vous utilisez l'assistant au quotidien permet de contextualiser son utilité."},
        {"title": "💬 Préférences de communication", "desc": "Configurez le style et la tonalité de dialogue de votre choix."}
    ]

    current_step = st.session_state.kmb_step if is_onboarding else None

    # Bouton de saut d'onboarding
    if is_onboarding:
        col_title, col_skip = st.columns([3, 1])
        with col_title:
            st.markdown(f"#### Étape {st.session_state.kmb_step + 1} : {sections[st.session_state.kmb_step]['title']}")
        with col_skip:
            if st.button("Passer pour l'instant 🏃", key="btn_skip_kmb", use_container_width=True):
                st.session_state.kmb_skipped = True
                kmb_mgr.mark_kmb_onboarding_seen()
                st.rerun()
        st.info(f"💡 **Pourquoi remplir cette section ?** {sections[st.session_state.kmb_step]['desc']}")
        st.divider()

    # Rendu des formulaires
    # SECTION 1: Informations personnelles
    if not is_onboarding or current_step == 0:
        render_section_1(kmb_data, is_onboarding)

    # SECTION 2: Intérêts & Loisirs
    if not is_onboarding or current_step == 1:
        render_section_2(kmb_data, is_onboarding)

    # SECTION 3: Apprentissage & Croissance
    if not is_onboarding or current_step == 2:
        render_section_3(kmb_data, is_onboarding)

    # SECTION 4: Travail & Mode de vie
    if not is_onboarding or current_step == 3:
        render_section_4(kmb_data, is_onboarding)

    # SECTION 5: Préférences
    if not is_onboarding or current_step == 4:
        render_section_5(kmb_data, is_onboarding)

    # Navigation onboarding
    if is_onboarding:
        st.divider()
        col_prev, col_next = st.columns([1, 1])
        with col_prev:
            if st.session_state.kmb_step > 0:
                if st.button("← Précédent", key="btn_kmb_prev", use_container_width=True):
                    st.session_state.kmb_step -= 1
                    st.rerun()
        with col_next:
            if st.session_state.kmb_step < 4:
                if st.button("Suivant →", key="btn_kmb_next", use_container_width=True):
                    st.session_state.kmb_step += 1
                    st.rerun()
            else:
                if st.button("🚀 Commencer à utiliser l'application !", type="primary", key="btn_kmb_finish", use_container_width=True):
                    st.session_state.kmb_skipped = True
                    kmb_mgr.mark_kmb_onboarding_seen()
                    st.success("Configuration sauvegardée !")
                    st.rerun()


def render_section_1(data: dict, is_onboarding: bool) -> None:
    """Rendu de la section 1 : Informations personnelles."""
    container = st.container() if is_onboarding else st.expander("👤 1. Informations personnelles", expanded=False)
    with container:
        if not is_onboarding:
            st.caption("Ces informations permettent de calibrer les réponses générales et la langue.")
            
        # Date de naissance
        dob_val = data.get("date_of_birth", "")
        st.text_input(
            "Quelle est votre date de naissance ? (ex: AAAA-MM-JJ)",
            value=dob_val,
            placeholder="AAAA-MM-JJ",
            key="kmb_dob",
            on_change=save_kmb_field,
            args=("date_of_birth", "kmb_dob")
        )

        # Genre
        gender_val = data.get("gender", "Prefer not to say")
        gender_idx = GENDER_OPTIONS.index(gender_val) if gender_val in GENDER_OPTIONS else 0
        st.selectbox(
            "Quel est votre genre ?",
            options=GENDER_OPTIONS,
            index=gender_idx,
            key="kmb_gender",
            on_change=save_kmb_field,
            args=("gender", "kmb_gender")
        )

        # Pays
        country_val = data.get("country", "")
        st.text_input(
            "Dans quel pays résidez-vous actuellement ?",
            value=country_val,
            key="kmb_country",
            on_change=save_kmb_field,
            args=("country", "kmb_country")
        )

        # Langues parlées
        lang_vals = data.get("languages", [])
        st.multiselect(
            "Quelles langues parlez-vous confortablement ?",
            options=LANGUAGES_OPTIONS,
            default=[l for l in lang_vals if l in LANGUAGES_OPTIONS],
            key="kmb_languages",
            on_change=save_kmb_field,
            args=("languages", "kmb_languages")
        )


def render_section_2(data: dict, is_onboarding: bool) -> None:
    """Rendu de la section 2 : Intérêts & Loisirs."""
    container = st.container() if is_onboarding else st.expander("🎨 2. Intérêts & Loisirs", expanded=False)
    with container:
        if not is_onboarding:
            st.caption("Vos centres d'intérêt aident CogniAssist à enrichir ses réponses avec des thèmes connexes.")
            
        # Loisirs
        hobby_vals = data.get("hobbies", [])
        st.multiselect(
            "Quels sont vos loisirs et passe-temps ?",
            options=HOBBY_OPTIONS,
            default=[h for h in hobby_vals if h in HOBBY_OPTIONS],
            key="kmb_hobbies",
            on_change=save_kmb_field,
            args=("hobbies", "kmb_hobbies")
        )

        # Curiosité
        curious_val = data.get("interests", "")
        st.text_area(
            "Quels sujets vous rendent naturellement curieux ?",
            value=curious_val,
            height=80,
            placeholder="Ex : physique quantique, histoire médiévale, intelligence artificielle...",
            key="kmb_interests",
            on_change=save_kmb_field,
            args=("interests", "kmb_interests")
        )

        # Sujets de discussion
        talk_val = data.get("favorite_topics", "")
        st.text_area(
            "De quels sujets pourriez-vous parler pendant des heures ?",
            value=talk_val,
            height=80,
            placeholder="Ex : les tactiques d'échecs, la création de startups, les théories de films...",
            key="kmb_fav_topics",
            on_change=save_kmb_field,
            args=("favorite_topics", "kmb_fav_topics")
        )

        # Types de contenu
        content_vals = data.get("content_preferences", [])
        st.multiselect(
            "Quel type de contenu consommez-vous le plus souvent ?",
            options=CONTENT_OPTIONS,
            default=[c for c in content_vals if c in CONTENT_OPTIONS],
            key="kmb_content",
            on_change=save_kmb_field,
            args=("content_preferences", "kmb_content")
        )

        # Communautés suivies
        communities_val = data.get("followed_communities", [])
        st.text_input(
            "Quels domaines, communautés ou secteurs d'activité aimez-vous suivre ?",
            value=", ".join(communities_val) if isinstance(communities_val, list) else str(communities_val),
            placeholder="Ex : Reddit/r/science, GitHub, newsletters techniques, etc.",
            key="kmb_communities",
            on_change=lambda: save_kmb_field(
                "followed_communities", 
                "kmb_communities"
            ) if "," not in st.session_state.get("kmb_communities", "") else 
            st.session_state.update({"kmb_communities_parsed": [x.strip() for x in st.session_state.get("kmb_communities", "").split(",") if x.strip()]}) or
            save_kmb_field("followed_communities", "kmb_communities_parsed")
        )


def render_section_3(data: dict, is_onboarding: bool) -> None:
    """Rendu de la section 3 : Apprentissage & Croissance."""
    container = st.container() if is_onboarding else st.expander("📚 3. Apprentissage & Croissance", expanded=False)
    with container:
        if not is_onboarding:
            st.caption("Comprendre comment vous apprenez permet d'adapter la pédagogie de l'assistant.")
            
        # Compétences actuelles
        curr_skills = data.get("current_skills", "")
        st.text_input(
            "Quelles compétences essayez-vous d'améliorer actuellement ?",
            value=curr_skills,
            placeholder="Ex : programmation Python, communication, anglais commercial...",
            key="kmb_curr_skills",
            on_change=save_kmb_field,
            args=("current_skills", "kmb_curr_skills")
        )

        # Compétences futures
        fut_skills = data.get("future_skills", "")
        st.text_input(
            "Quelles compétences aimeriez-vous développer dans le futur ?",
            value=fut_skills,
            placeholder="Ex : management d'équipe, dessin numérique, soudure...",
            key="kmb_fut_skills",
            on_change=save_kmb_field,
            args=("future_skills", "kmb_fut_skills")
        )

        # Objectifs de l'année
        goals = data.get("yearly_goals", "")
        st.text_area(
            "Quels objectifs personnels sont importants pour vous cette année ?",
            value=goals,
            height=80,
            placeholder="Ex : Obtenir mon diplôme de Master, lancer un projet open source...",
            key="kmb_goals",
            on_change=save_kmb_field,
            args=("yearly_goals", "kmb_goals")
        )

        # Style d'apprentissage
        styles_val = data.get("learning_style", [])
        st.multiselect(
            "Comment préférez-vous apprendre de nouvelles choses ?",
            options=LEARNING_STYLE_OPTIONS,
            default=[s for s in styles_val if s in LEARNING_STYLE_OPTIONS],
            key="kmb_learn_style",
            on_change=save_kmb_field,
            args=("learning_style", "kmb_learn_style")
        )

        # Fréquence d'apprentissage
        freq_val = data.get("learning_frequency", "")
        freq_idx = LEARNING_FREQUENCY_OPTIONS.index(freq_val) if freq_val in LEARNING_FREQUENCY_OPTIONS else 0
        st.selectbox(
            "À quelle fréquence apprenez-vous activement quelque chose de nouveau ?",
            options=LEARNING_FREQUENCY_OPTIONS,
            index=freq_idx,
            key="kmb_learn_freq",
            on_change=save_kmb_field,
            args=("learning_frequency", "kmb_learn_freq")
        )


def render_section_4(data: dict, is_onboarding: bool) -> None:
    """Rendu de la section 4 : Travail & Mode de vie."""
    container = st.container() if is_onboarding else st.expander("💼 4. Travail & Mode de vie", expanded=False)
    with container:
        if not is_onboarding:
            st.caption("Savoir comment vous utilisez l'assistant au quotidien permet de contextualiser son utilité.")
            
        # Situation actuelle
        situation = data.get("occupation", "")
        sit_idx = OCCUPATION_OPTIONS.index(situation) if situation in OCCUPATION_OPTIONS else 0
        st.selectbox(
            "Quelle situation décrit le mieux votre statut actuel ?",
            options=OCCUPATION_OPTIONS,
            index=sit_idx,
            key="kmb_occupation",
            on_change=save_kmb_field,
            args=("occupation", "kmb_occupation")
        )

        # Secteur d'activité
        industry = data.get("industry", "")
        st.text_input(
            "Dans quel secteur d'activité ou domaine êtes-vous impliqué(e) ?",
            value=industry,
            placeholder="Ex : éducation, technologie, finance, santé...",
            key="kmb_industry",
            on_change=save_kmb_field,
            args=("industry", "kmb_industry")
        )

        # Défis quotidiens
        challenges = data.get("challenges", "")
        st.text_area(
            "Quels défis rencontrez-vous le plus souvent dans vos études ou votre travail ?",
            value=challenges,
            height=80,
            placeholder="Ex : gestion du temps, manque de documentation, résolution de bugs complexes...",
            key="kmb_challenges",
            on_change=save_kmb_field,
            args=("challenges", "kmb_challenges")
        )

        # Motivations
        motivations = data.get("motivations", "")
        st.text_input(
            "Qu'est-ce qui vous motive généralement à continuer de vous améliorer ?",
            value=motivations,
            placeholder="Ex : la curiosité intellectuelle, l'évolution de carrière, aider les autres...",
            key="kmb_motivations",
            on_change=save_kmb_field,
            args=("motivations", "kmb_motivations")
        )


def render_section_5(data: dict, is_onboarding: bool) -> None:
    """Rendu de la section 5 : Préférences de communication."""
    container = st.container() if is_onboarding else st.expander("💬 5. Préférences de communication", expanded=False)
    with container:
        if not is_onboarding:
            st.caption("Configurez le style et la tonalité de dialogue de votre choix.")
            
        # Style de communication
        comm = data.get("communication_style", "")
        comm_idx = COMMUNICATION_OPTIONS.index(comm) if comm in COMMUNICATION_OPTIONS else 0
        st.selectbox(
            "Comment aimeriez-vous que CogniAssist communique avec vous ?",
            options=COMMUNICATION_OPTIONS,
            index=comm_idx,
            key="kmb_comm_style",
            on_change=save_kmb_field,
            args=("communication_style", "kmb_comm_style")
        )

        # Informations complémentaires
        additional = data.get("additional_information", "")
        st.text_area(
            "Y a-t-il autre chose que vous aimeriez que CogniAssist sache sur vous ?",
            value=additional,
            height=100,
            placeholder="N'hésitez pas à partager d'autres éléments pour personnaliser votre assistant...",
            key="kmb_additional",
            on_change=save_kmb_field,
            args=("additional_information", "kmb_additional")
        )
