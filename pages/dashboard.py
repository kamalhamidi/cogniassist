"""
pages/dashboard.py — Tableau de bord analytique avec ACPE.

Affiche les KPIs, l'activité récente, les sujets explorés,
les statistiques par document, les recommandations,
le profil de connaissances et les insights comportementaux.
"""

import streamlit as st
from collections import Counter


@st.cache_data(ttl=60)
def _load_dashboard_data(user_id: str) -> dict:
    """Charge toutes les données du tableau de bord (cache 1 min)."""
    from user import get_interaction_history, get_recommender, get_user_manager

    history = get_interaction_history(user_id)
    rec = get_recommender(user_id)
    mgr = get_user_manager(user_id)

    return {
        "stats": history.get_interaction_stats(),
        "progress": rec.get_learning_progress(),
        "docs": mgr.get_user_documents(),
        "topics": history.get_frequent_topics(10),
        "recent": history.get_recent_history(10),
        "profile": mgr.get_profile(),
    }


@st.cache_data(ttl=60)
def _load_acpe_data(user_id: str) -> dict:
    """Charge les données ACPE (cache 1 min)."""
    try:
        from user.profile_evolution import ProfileEvolutionEngine
        evo = ProfileEvolutionEngine(user_id)
        return evo.get_evolution_summary()
    except Exception:
        return {
            "knowledge_profile": [],
            "top_domains": [],
            "weak_domains": [],
            "insights": [],
            "usage_patterns": {},
        }


def show_dashboard_page() -> None:
    """Affiche le tableau de bord."""
    user_id = st.session_state.get("user_id", "default")

    from ui import page_header
    page_header(
        "Tableau de bord",
        "Votre activité et progression sur CogniAssist",
        icon="📊",
    )

    # Charger les données
    try:
        data = _load_dashboard_data(user_id)
    except Exception as e:
        st.error(f"Erreur chargement données : {e}")
        return

    stats = data["stats"]
    progress = data["progress"]
    profile = data["profile"]

    # ═══ Section 1 : KPIs ═══
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📚 Documents", progress["documents_uploaded"])
    c2.metric("💬 Questions posées", stats["total_interactions"])
    c3.metric("⭐ Score de connaissance", f"{progress['knowledge_score']}/100")
    c4.metric("👍 Taux de satisfaction", f"{progress['positive_feedback_rate']:.0f}%")

    st.divider()

    # ═══ Section 2 : Activité + Sujets ═══
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📈 Activité récente")
        recent = data["recent"]
        if recent:
            # Compter les interactions par jour
            day_counts: Counter = Counter()
            for r in recent:
                date_str = r.get("created_at", "")[:10]
                if date_str:
                    day_counts[date_str] += 1

            if day_counts:
                import pandas as pd
                df = pd.DataFrame(
                    list(day_counts.items()), columns=["Date", "Interactions"]
                ).sort_values("Date")
                st.bar_chart(df.set_index("Date"))
            else:
                st.info("Aucune donnée d'activité")
        else:
            st.info("Aucune activité enregistrée")

    with col_right:
        st.subheader("🏷️ Sujets explorés")
        topics = data["topics"]
        if topics:
            for i, topic in enumerate(topics):
                weight = (len(topics) - i) / len(topics)
                st.write(f"**{topic}**")
                st.progress(weight)
        else:
            st.info("Commencez à poser des questions pour voir vos sujets")

    st.divider()

    # ═══ Section 3 : Profil de connaissances ACPE ═══
    st.subheader("🧠 Profil de connaissances")

    acpe_data = _load_acpe_data(user_id)
    knowledge = acpe_data.get("knowledge_profile", [])

    if knowledge:
        col_radar, col_details = st.columns([3, 2])

        with col_radar:
            # Graphique radar Plotly
            try:
                import plotly.graph_objects as go

                domains = [k["domain"] for k in knowledge[:8]]
                scores = [k["mastery_score"] for k in knowledge[:8]]

                fig = go.Figure(data=go.Scatterpolar(
                    r=scores,
                    theta=domains,
                    fill="toself",
                    name="Maîtrise",
                    line_color="#4CAF50",
                    fillcolor="rgba(76, 175, 80, 0.3)",
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=False,
                    margin=dict(l=40, r=40, t=20, b=20),
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                # Fallback si Plotly non disponible
                import pandas as pd
                df_k = pd.DataFrame(knowledge[:8])
                st.bar_chart(df_k.set_index("domain")["mastery_score"])

        with col_details:
            st.markdown("**Domaines les mieux maîtrisés :**")
            top = acpe_data.get("top_domains", [])
            if top:
                for d in top[:5]:
                    kp = next((k for k in knowledge if k["domain"] == d), None)
                    score = kp["mastery_score"] if kp else 0
                    confidence = kp["confidence"] if kp else 0
                    st.markdown(f"🟢 **{d}** — {score}/100")
                    st.caption(f"   Confiance : {confidence}%")
            else:
                st.caption("Posez des questions pour construire votre profil")

            weak = acpe_data.get("weak_domains", [])
            if weak:
                st.markdown("**Axes de progression :**")
                for d in weak[:3]:
                    st.markdown(f"🔴 {d}")

    else:
        st.info(
            "📋 Votre profil de connaissances se construira automatiquement "
            "au fil de vos interactions."
        )

    st.divider()

    # ═══ Section 4 : Insights ACPE ═══
    insights = acpe_data.get("insights", [])
    if insights:
        st.subheader("💡 Insights comportementaux")
        for insight in insights:
            st.markdown(f"• {insight}")
        st.divider()

    # ═══ Section 5 : Documents ═══
    st.subheader("📂 Activité par document")
    docs = data["docs"]
    if docs:
        import pandas as pd
        df = pd.DataFrame([
            {
                "Document": d["file_name"],
                "Chunks": d["chunk_count"],
                "Consultations": d["access_count"],
                "Résumé": "✅" if d["has_summary"] else "❌",
                "Importé le": d["upload_date"][:10] if d["upload_date"] else "—",
            }
            for d in docs
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Aucun document indexé")

    st.divider()

    # ═══ Section 6 : Dernières interactions ═══
    st.subheader("🕐 Dernières interactions")
    recent = data["recent"]
    if recent:
        for interaction in recent[:5]:
            q_preview = interaction["question"][:80]
            date_str = interaction.get("created_at", "")[:10]
            with st.expander(f"❓ {q_preview}... — {date_str}"):
                st.markdown("**Question :**")
                st.write(interaction["question"])
                st.markdown("**Réponse :**")
                answer = interaction["answer"]
                st.write(answer[:500] + "..." if len(answer) > 500 else answer)
                if interaction["sources"]:
                    st.caption(f"Sources : {', '.join(interaction['sources'])}")
                fb = interaction.get("feedback")
                if fb == 1:
                    st.caption("Feedback : 👍")
                elif fb == -1:
                    st.caption("Feedback : 👎")
    else:
        st.info("Aucune interaction enregistrée. Commencez à chatter !")

    st.divider()

    # ═══ Section 7 : Recommandations ═══
    st.subheader("💡 Recommandations")
    try:
        from user import get_recommender
        rec = get_recommender(user_id)
        doc_recs = rec.get_document_recommendations()
        if doc_recs:
            for r in doc_recs:
                st.warning(f"📄 {r['file_name']} — {r['reason']}")
        else:
            st.success("✅ Tous vos documents ont été consultés récemment")
    except Exception as e:
        st.error(f"Erreur recommandations : {e}")

    # ═══ Section 8 : Enterprise insights (conditionnel) ═══
    if profile.get("user_type") == "enterprise":
        st.divider()
        st.subheader("🏢 Aperçu Enterprise")
        org = profile.get("organization_data", {})
        if org:
            col_e1, col_e2, col_e3 = st.columns(3)
            col_e1.metric("Secteur", org.get("industry", "—").title())
            col_e2.metric("Taille", org.get("org_size", "—"))
            col_e3.metric(
                "Confidentialité",
                org.get("confidentiality", "standard").title(),
            )

            use_cases = org.get("use_cases", [])
            if use_cases:
                st.markdown("**Cas d'usage :**")
                for uc in use_cases:
                    st.markdown(f"  • {uc}")


if __name__ == "__main__":
    st.session_state.current_page = "📊 Dashboard"
    st.switch_page("app.py")

