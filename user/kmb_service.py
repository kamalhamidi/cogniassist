"""
user/kmb_service.py — Service de gestion pour "Know Me Better".

Gère le stockage, la mise à jour, la récupération et le calcul de complétion
des données du questionnaire personnalisé "Know Me Better".
"""

import json
import logging
from datetime import datetime

from user.db import uses_db_session
from user.profile import PersonalProfileData

logger = logging.getLogger("cogniassist.user")


class KMBManager:
    """
    Gère le stockage, la récupération et la validation de 'Know Me Better'.
    
    Calcule dynamiquement le taux de complétion en temps réel.
    """

    def __init__(self, user_id: str = "default") -> None:
        """
        Initialise le gestionnaire KMB pour un utilisateur.

        Args:
            user_id: Identifiant de l'utilisateur.
        """
        self.user_id = user_id
        self._ensure_kmb_exists()

    @uses_db_session
    def _ensure_kmb_exists(self, session) -> None:
        """Crée l'enregistrement KMB vide s'il n'existe pas en base."""
        try:
            # S'assurer que l'utilisateur de base existe
            from user.profile import UserProfileManager
            UserProfileManager(self.user_id)

            existing = (
                session.query(PersonalProfileData)
                .filter_by(user_id=self.user_id)
                .first()
            )
            if existing is None:
                kmb = PersonalProfileData(user_id=self.user_id)
                session.add(kmb)
                session.commit()
                logger.info("Enregistrement 'Know Me Better' créé pour '%s'.", self.user_id)
        except Exception as e:
            session.rollback()
            logger.error("Erreur création KMB pour '%s' : %s", self.user_id, e)
            raise

    @uses_db_session
    def get_kmb_data(self, session) -> dict:
        """
        Récupère les données KMB complètes.

        Returns:
            Dictionnaire des réponses et état de complétion.
        """
        kmb = (
            session.query(PersonalProfileData)
            .filter_by(user_id=self.user_id)
            .first()
        )
        if kmb:
            res = kmb.to_dict()
            res["kmb_completed"] = (res["completion_percentage"] == 100)
            return res
        return {}

    @uses_db_session
    def update_kmb_field(self, session, field_name: str, value: any) -> int:
        """
        Met à jour un champ KMB individuel avec recalcul de complétion.

        Args:
            field_name: Le nom de la colonne/champ.
            value: La nouvelle valeur (chaîne, liste ou date).

        Returns:
            Nouveau pourcentage de complétion.
        """
        try:
            kmb = (
                session.query(PersonalProfileData)
                .filter_by(user_id=self.user_id)
                .first()
            )
            if not kmb:
                return 0

            # Champs nécessitant un encodage JSON (uniquement les listes issues des multiselects)
            json_fields = {
                "languages", "hobbies", "content_preferences", "learning_style"
            }

            if field_name in json_fields:
                if isinstance(value, list):
                    value = json.dumps(value, ensure_ascii=False)

            setattr(kmb, field_name, value)
            
            # Recalculer le pourcentage
            kmb.completion_percentage = self._calculate_completion(kmb)
            kmb.updated_at = datetime.utcnow()
            
            session.commit()
            return kmb.completion_percentage
        except Exception as e:
            session.rollback()
            logger.error("Erreur mise à jour KMB champ '%s' : %s", field_name, e)
            raise

    @uses_db_session
    def has_seen_kmb_onboarding(self, session) -> bool:
        """Vérifie si l'onboarding KMB a été vu ou ignoré."""
        kmb = (
            session.query(PersonalProfileData)
            .filter_by(user_id=self.user_id)
            .first()
        )
        return kmb.kmb_onboarding_seen if kmb else False

    @uses_db_session
    def mark_kmb_onboarding_seen(self, session) -> None:
        """Marque l'onboarding KMB comme complété ou vu (pour ne plus l'afficher)."""
        try:
            kmb = (
                session.query(PersonalProfileData)
                .filter_by(user_id=self.user_id)
                .first()
            )
            if kmb:
                kmb.kmb_onboarding_seen = True
                session.commit()
                logger.info("Onboarding KMB marqué comme vu pour '%s'.", self.user_id)
        except Exception as e:
            session.rollback()
            logger.error("Erreur mark_kmb_onboarding_seen : %s", e)

    def _calculate_completion(self, kmb: PersonalProfileData) -> int:
        """Calcule le pourcentage de complétion basé sur les 20 questions."""
        fields_to_check = [
            "date_of_birth", "gender", "country", "languages", "hobbies",
            "interests", "favorite_topics", "content_preferences", "followed_communities",
            "current_skills", "future_skills", "yearly_goals", "learning_style",
            "learning_frequency", "occupation", "industry", "challenges",
            "motivations", "communication_style", "additional_information"
        ]
        
        answered = 0
        for field in fields_to_check:
            val = getattr(kmb, field)
            if val is None:
                continue
            
            # Pour les listes encodées en JSON
            if field in {
                "languages", "hobbies", "content_preferences", "learning_style"
            }:
                try:
                    loaded = json.loads(val)
                    if loaded and len(loaded) > 0:
                        answered += 1
                except Exception:
                    pass
            elif isinstance(val, str):
                if val.strip() and val.strip() != "Prefer not to say" and val.strip() != "[]":
                    answered += 1
                elif val.strip() == "Prefer not to say":
                    # Option explicite de non-réponse considérée comme valide
                    answered += 1
                    
        return int((answered / len(fields_to_check)) * 100)
