"""
pages/profile.py — Page de gestion du profil utilisateur.

Permet à l'utilisateur de consulter et modifier son profil,
ses préférences et de visualiser ses statistiques personnelles.
"""

import streamlit as st


def init_session_state() -> None:
    """Initialise les variables de session Streamlit."""
    defaults: dict = {
        "messages": [],
        "user_id": "default",
        "documents_loaded": False,
        "current_page": "Profil",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_profile_page() -> None:
    """Affiche la page de gestion du profil utilisateur."""
    init_session_state()
    st.header("👤 Profil Utilisateur")
    st.markdown("Gérez votre profil et vos préférences pour personnaliser CogniAssist.")

    user_id = st.session_state.get("user_id", "default")

    # Formulaire de profil
    with st.form("profile_form"):
        st.subheader("📝 Informations personnelles")

        username = st.text_input("Nom d'utilisateur", value="Utilisateur")
        email = st.text_input("Email (optionnel)", value="")

        language = st.selectbox(
            "Langue préférée",
            options=["fr", "en", "ar"],
            format_func=lambda x: {"fr": "🇫🇷 Français", "en": "🇬🇧 English", "ar": "🇲🇦 العربية"}.get(x, x),
        )

        expertise_level = st.select_slider(
            "Niveau d'expertise",
            options=["débutant", "intermédiaire", "expert"],
            value="intermédiaire",
        )

        interests = st.text_area(
            "Centres d'intérêt (un par ligne)",
            placeholder="Intelligence Artificielle\nScience des données\nDéveloppement web",
        )

        submitted = st.form_submit_button("💾 Sauvegarder", type="primary")

        if submitted:
            try:
                from user.db import DatabaseManager
                from user.profile import UserProfile, UserProfileManager

                db = DatabaseManager()
                db.create_tables()
                manager = UserProfileManager(db)

                profile = UserProfile(
                    user_id=user_id,
                    username=username,
                    email=email if email else None,
                    language=language,
                    expertise_level=expertise_level,
                    interests=[i.strip() for i in interests.split("\n") if i.strip()],
                )

                existing = manager.get_profile(user_id)
                if existing:
                    manager.update_profile(
                        user_id,
                        username=username,
                        email=email,
                        language=language,
                        expertise_level=expertise_level,
                        interests=",".join(profile.interests),
                    )
                else:
                    manager.create_profile(profile)

                st.success("✅ Profil sauvegardé avec succès !")

            except Exception as e:
                st.error(f"❌ Erreur lors de la sauvegarde : {str(e)}")

    st.markdown("---")

    # Statistiques utilisateur
    st.subheader("📊 Vos statistiques")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Questions posées", len([m for m in st.session_state.get("messages", []) if m.get("role") == "user"]))
    with col2:
        st.metric("Réponses reçues", len([m for m in st.session_state.get("messages", []) if m.get("role") == "assistant"]))
    with col3:
        st.metric("ID Utilisateur", user_id)


render_profile_page()
