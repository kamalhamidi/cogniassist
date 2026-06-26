"""
pages/onboarding.py — Wizard d'onboarding ACPE.

Guide l'utilisateur à travers un processus d'onboarding en 3 étapes
pour configurer son profil cognitif initial. Adapte les questions
selon le type d'utilisateur (Individual / Enterprise).
"""

import streamlit as st


# ═══════════════════════════════════════════════════════════
# Données statiques pour les questionnaires
# ═══════════════════════════════════════════════════════════

INDIVIDUAL_ROLES = [
    "🎓 Étudiant", "👨‍🏫 Enseignant", "🔬 Chercheur",
    "💻 Développeur", "💼 Professionnel", "📋 Autre",
]
ROLE_VALUES = {
    "🎓 Étudiant": "student", "👨‍🏫 Enseignant": "teacher",
    "🔬 Chercheur": "researcher", "💻 Développeur": "developer",
    "💼 Professionnel": "professional", "📋 Autre": "other",
}

EXPERTISE_LEVELS = {
    "🟢 Débutant": "beginner",
    "🟡 Intermédiaire": "intermediate",
    "🔴 Expert": "expert",
}

INDIVIDUAL_GOALS = [
    "📖 Étudier des cours",
    "📝 Résumer des documents",
    "🎯 Préparer des examens",
    "🔬 Mener des recherches",
    "🗂️ Organiser mes connaissances",
    "⏱️ Gagner du temps au travail",
]

RESPONSE_STYLES = {
    "✂️ Court et direct": "concise",
    "📝 Explications détaillées": "detailed",
    "📋 Étape par étape": "step_by_step",
    "🎓 Pédagogique avec exemples": "educational",
}

LANGUAGES = {
    "🇫🇷 Français": "fr",
    "🇬🇧 English": "en",
    "🇸🇦 العربية": "ar",
}

ENTERPRISE_INDUSTRIES = [
    "🏥 Santé", "💰 Finance", "🎓 Éducation",
    "💻 Technologie", "🏛️ Gouvernement",
    "🏭 Industrie", "📋 Autre",
]
INDUSTRY_VALUES = {
    "🏥 Santé": "healthcare", "💰 Finance": "finance",
    "🎓 Éducation": "education", "💻 Technologie": "technology",
    "🏛️ Gouvernement": "government", "🏭 Industrie": "manufacturing",
    "📋 Autre": "other",
}

ORG_SIZES = ["1–10", "11–50", "51–250", "250+"]

ENTERPRISE_USE_CASES = [
    "📞 Support interne",
    "📚 Gestion des connaissances",
    "🎓 Formation des employés",
    "🤝 Assistance client",
    "🔬 Recherche",
]

ENTERPRISE_DOC_TYPES = [
    "📋 Politiques", "📝 Procédures", "📄 Contrats",
    "💻 Documentation technique", "📊 Rapports",
]

CONFIDENTIALITY_LEVELS = {
    "🟢 Standard": "standard",
    "🟡 Sensible": "sensitive",
    "🔴 Hautement sensible": "highly_sensitive",
}


def show_onboarding_page() -> None:
    """Affiche le wizard d'onboarding ACPE."""

    # Initialiser l'état de l'onboarding
    if "onboarding_step" not in st.session_state:
        st.session_state.onboarding_step = 1
    if "onboarding_data" not in st.session_state:
        st.session_state.onboarding_data = {}

    step = st.session_state.onboarding_step

    # Header
    st.html(
        """
        <div style="text-align:center; padding: 1.5rem 0 0.5rem;">
            <div style="font-size:3rem;width:84px;height:84px;border-radius:24px;
                        margin:0 auto 14px;display:flex;align-items:center;
                        justify-content:center;
                        background:linear-gradient(135deg,#6C5CE7,#A855F7);
                        box-shadow:0 16px 36px rgba(108,92,231,.35);">🧠</div>
            <h1 style="font-size:2.4rem;font-weight:800;margin:0;color:#1E1B2E;">
                Bienvenue sur CogniAssist</h1>
            <p style="font-size:1.05rem;color:#6B6880;margin-top:.4rem;">
                Configurons votre assistant cognitif en quelques étapes
            </p>
        </div>
        """
    )

    # Barre de progression
    progress = step / 3
    st.progress(progress, text=f"Étape {step} sur 3")

    st.divider()

    if step == 1:
        _step_1_user_type()
    elif step == 2:
        _step_2_questionnaire()
    elif step == 3:
        _step_3_confirmation()


# ═══════════════════════════════════════════════════════════
# Étape 1 : Type d'utilisateur
# ═══════════════════════════════════════════════════════════

def _step_1_user_type() -> None:
    """Étape 1 : Comment utiliserez-vous CogniAssist ?"""
    st.subheader("🎯 Comment utiliserez-vous CogniAssist ?")
    st.caption("Cela nous aidera à personnaliser votre expérience")

    col1, col2 = st.columns(2)

    with col1:
        st.html(
            """
            <div style="
                border: 1px solid #ECEAF6; border-radius: 18px;
                padding: 1.6rem; text-align: center; min-height: 200px;
                background:#FFFFFF; box-shadow:0 8px 24px rgba(30,27,46,.06);
                border-top:4px solid #6C5CE7;
            ">
                <div style="font-size: 3rem;">👤</div>
                <h3 style="margin:.4rem 0;color:#1E1B2E;">Individuel</h3>
                <p style="color: #6B6880; font-size: 0.9rem;">
                    Étudiant, chercheur, développeur...<br/>
                    Pour votre usage personnel
                </p>
            </div>
            """
        )
        if st.button(
            "👤 Choisir Individuel",
            use_container_width=True,
            type="primary",
            key="btn_individual",
        ):
            st.session_state.onboarding_data["user_type"] = "individual"
            st.session_state.onboarding_step = 2
            st.rerun()

    with col2:
        st.html(
            """
            <div style="
                border: 1px solid #ECEAF6; border-radius: 18px;
                padding: 1.6rem; text-align: center; min-height: 200px;
                background:#FFFFFF; box-shadow:0 8px 24px rgba(30,27,46,.06);
                border-top:4px solid #A855F7;
            ">
                <div style="font-size: 3rem;">🏢</div>
                <h3 style="margin:.4rem 0;color:#1E1B2E;">Entreprise / Organisation</h3>
                <p style="color: #6B6880; font-size: 0.9rem;">
                    Gestion de connaissances, support,<br/>
                    formation en entreprise
                </p>
            </div>
            """
        )
        if st.button(
            "🏢 Choisir Entreprise",
            use_container_width=True,
            type="secondary",
            key="btn_enterprise",
        ):
            st.session_state.onboarding_data["user_type"] = "enterprise"
            st.session_state.onboarding_step = 2
            st.rerun()

    # Option pour ignorer l'onboarding
    st.divider()
    if st.button("⏭️ Ignorer l'onboarding", key="btn_skip"):
        _skip_onboarding()


# ═══════════════════════════════════════════════════════════
# Étape 2 : Questionnaire adapté
# ═══════════════════════════════════════════════════════════

def _step_2_questionnaire() -> None:
    """Étape 2 : Questionnaire adapté au type d'utilisateur."""
    user_type = st.session_state.onboarding_data.get("user_type", "individual")

    if user_type == "individual":
        _individual_questionnaire()
    else:
        _enterprise_questionnaire()


def _individual_questionnaire() -> None:
    """Questionnaire pour les utilisateurs individuels."""
    st.subheader("👤 Parlons de vous")

    with st.form("individual_form"):
        # Nom
        name = st.text_input(
            "Comment souhaitez-vous être appelé ?",
            placeholder="Ex: Kamal",
        )

        # Rôle
        role = st.selectbox("Quel est votre rôle ?", INDIVIDUAL_ROLES)

        # Niveau d'expertise
        st.markdown("**Votre niveau d'expertise technique :**")
        expertise = st.select_slider(
            "Niveau d'expertise",
            options=list(EXPERTISE_LEVELS.keys()),
            value="🟡 Intermédiaire",
            label_visibility="collapsed",
        )

        # Objectifs
        st.markdown("**Quels sont vos objectifs principaux ?**")
        goals = []
        goal_cols = st.columns(2)
        for i, goal in enumerate(INDIVIDUAL_GOALS):
            with goal_cols[i % 2]:
                if st.checkbox(goal, key=f"goal_{i}"):
                    goals.append(goal.split(" ", 1)[1])

        # Sujets d'intérêt
        interests = st.text_input(
            "Sujets qui vous intéressent (séparés par des virgules)",
            placeholder="Ex: Python, Machine Learning, NLP",
        )

        # Style de réponse
        response_style = st.radio(
            "Comment préférez-vous recevoir les réponses ?",
            list(RESPONSE_STYLES.keys()),
            horizontal=True,
        )

        # Langue
        language = st.selectbox(
            "Langue préférée",
            list(LANGUAGES.keys()),
        )

        submitted = st.form_submit_button(
            "Continuer →", type="primary", use_container_width=True,
        )

        if submitted:
            data = st.session_state.onboarding_data
            data["name"] = name or "Utilisateur"
            data["role"] = ROLE_VALUES.get(role, "other")
            data["expertise_level"] = EXPERTISE_LEVELS.get(
                expertise, "intermediate",
            )
            data["goals"] = goals
            data["interests"] = [
                i.strip() for i in interests.split(",") if i.strip()
            ] if interests else []
            data["response_style"] = RESPONSE_STYLES.get(
                response_style, "detailed",
            )
            data["language"] = LANGUAGES.get(language, "fr")
            st.session_state.onboarding_step = 3
            st.rerun()

    # Bouton retour
    if st.button("← Retour", key="btn_back_2_ind"):
        st.session_state.onboarding_step = 1
        st.rerun()


def _enterprise_questionnaire() -> None:
    """Questionnaire pour les utilisateurs entreprise."""
    st.subheader("🏢 Votre organisation")

    with st.form("enterprise_form"):
        # Nom de contact
        name = st.text_input(
            "Votre nom",
            placeholder="Ex: Kamal HAMIDI",
        )

        # Secteur
        industry = st.selectbox("Secteur d'activité", ENTERPRISE_INDUSTRIES)

        # Taille
        org_size = st.selectbox("Taille de l'organisation", ORG_SIZES)

        # Cas d'usage
        st.markdown("**Cas d'utilisation principaux :**")
        use_cases = []
        uc_cols = st.columns(2)
        for i, uc in enumerate(ENTERPRISE_USE_CASES):
            with uc_cols[i % 2]:
                if st.checkbox(uc, key=f"uc_{i}"):
                    use_cases.append(uc.split(" ", 1)[1])

        # Types de documents
        st.markdown("**Types de documents que vous utiliserez :**")
        doc_types = []
        dt_cols = st.columns(2)
        for i, dt in enumerate(ENTERPRISE_DOC_TYPES):
            with dt_cols[i % 2]:
                if st.checkbox(dt, key=f"dt_{i}"):
                    doc_types.append(dt.split(" ", 1)[1])

        # Confidentialité
        confidentiality = st.radio(
            "Niveau de confidentialité",
            list(CONFIDENTIALITY_LEVELS.keys()),
            horizontal=True,
        )

        # Style de communication
        comm_style = st.radio(
            "Style de communication préféré",
            ["📝 Formel et professionnel", "💬 Conversationnel"],
            horizontal=True,
        )

        # Langue
        language = st.selectbox(
            "Langue préférée",
            list(LANGUAGES.keys()),
        )

        submitted = st.form_submit_button(
            "Continuer →", type="primary", use_container_width=True,
        )

        if submitted:
            data = st.session_state.onboarding_data
            data["name"] = name or "Utilisateur"
            data["role"] = "enterprise_admin"
            data["expertise_level"] = "intermediate"
            data["response_style"] = (
                "detailed" if "Formel" in comm_style else "concise"
            )
            data["language"] = LANGUAGES.get(language, "fr")
            data["goals"] = use_cases
            data["interests"] = doc_types
            data["organization_data"] = {
                "industry": INDUSTRY_VALUES.get(industry, "other"),
                "org_size": org_size,
                "use_cases": use_cases,
                "doc_types": doc_types,
                "confidentiality": CONFIDENTIALITY_LEVELS.get(
                    confidentiality, "standard",
                ),
                "comm_style": comm_style,
            }
            st.session_state.onboarding_step = 3
            st.rerun()

    # Bouton retour
    if st.button("← Retour", key="btn_back_2_ent"):
        st.session_state.onboarding_step = 1
        st.rerun()


# ═══════════════════════════════════════════════════════════
# Étape 3 : Confirmation
# ═══════════════════════════════════════════════════════════

def _step_3_confirmation() -> None:
    """Étape 3 : Récapitulatif et confirmation."""
    data = st.session_state.onboarding_data

    st.subheader("✅ Récapitulatif de votre profil")

    user_type = data.get("user_type", "individual")
    type_label = "👤 Individuel" if user_type == "individual" else "🏢 Entreprise"

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Type :** {type_label}")
        st.markdown(f"**Nom :** {data.get('name', 'Utilisateur')}")
        st.markdown(f"**Rôle :** {data.get('role', '—')}")
        st.markdown(f"**Niveau :** {data.get('expertise_level', 'intermediate')}")

    with col2:
        st.markdown(f"**Style :** {data.get('response_style', 'detailed')}")
        st.markdown(f"**Langue :** {data.get('language', 'fr')}")
        goals = data.get("goals", [])
        if goals:
            st.markdown(f"**Objectifs :** {', '.join(goals[:3])}")
        interests = data.get("interests", [])
        if interests:
            st.markdown(f"**Intérêts :** {', '.join(interests[:5])}")

    if user_type == "enterprise":
        org = data.get("organization_data", {})
        if org:
            st.divider()
            st.markdown("**🏢 Données entreprise :**")
            st.markdown(f"- Secteur : {org.get('industry', '—')}")
            st.markdown(f"- Taille : {org.get('org_size', '—')}")
            st.markdown(
                f"- Confidentialité : {org.get('confidentiality', 'standard')}"
            )

    st.divider()
    st.info(
        "💡 Vous pourrez modifier ces paramètres à tout moment "
        "dans la page **Profil**."
    )

    col_confirm, col_back = st.columns([3, 1])

    with col_confirm:
        if st.button(
            "🚀 Commencer à utiliser CogniAssist !",
            type="primary",
            use_container_width=True,
            key="btn_confirm",
        ):
            _finalize_onboarding()

    with col_back:
        if st.button("← Modifier", key="btn_back_3"):
            st.session_state.onboarding_step = 2
            st.rerun()


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════

def _finalize_onboarding() -> None:
    """Sauvegarde les données d'onboarding et redirige vers l'app."""
    user_id = st.session_state.get("user_id", "default")
    data = st.session_state.onboarding_data

    try:
        from user import get_user_manager
        mgr = get_user_manager(user_id)
        mgr.complete_onboarding(data)

        # Nettoyer l'état d'onboarding
        del st.session_state["onboarding_step"]
        del st.session_state["onboarding_data"]

        st.success("🎉 Profil configuré avec succès !")
        st.cache_data.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Erreur lors de la sauvegarde : {e}")


def _skip_onboarding() -> None:
    """Ignore l'onboarding avec des valeurs par défaut."""
    user_id = st.session_state.get("user_id", "default")
    try:
        from user import get_user_manager
        mgr = get_user_manager(user_id)
        mgr.complete_onboarding({
            "user_type": "individual",
            "name": "Utilisateur",
        })

        if "onboarding_step" in st.session_state:
            del st.session_state["onboarding_step"]
        if "onboarding_data" in st.session_state:
            del st.session_state["onboarding_data"]

        st.rerun()
    except Exception as e:
        st.error(f"Erreur : {e}")


if __name__ == "__main__":
    st.switch_page("app.py")

