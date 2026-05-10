import logging
from typing import Tuple, List

from app.core.schemas import CandidateVector

logger = logging.getLogger(__name__)


async def ml_predict(vector: CandidateVector) -> Tuple[float, List[str]]:
    """
    Вызов ML-модуля для предсказания удержания кандидата.

    Parameters
    ----------
    vector : CandidateVector
        Вектор признаков кандидата, извлеченный AI-модулем.

    Returns
    -------
    Tuple[float, List[str]]
        Вероятность удержания [0.0 - 1.0] и список факторов риска.
    """

    try:
        from app.ml_legacy.predictor import RetentionPredictor

        predictor = RetentionPredictor()
        predictor.load_model()

        if predictor.model is None:
            predictor.train_model()
            predictor.save_model()

        features = {
            "skills_verified_count": vector.skills_verified_count,
            "years_experience": vector.years_experience,
            "commute_time_minutes": vector.commute_time_minutes,
            "shift_preference": vector.shift_preference.value,
            "salary_expectation": vector.salary_expectation,
            "has_certifications": vector.has_certifications,
        }

        prediction = predictor.predict_retention(features)
        risk_factors = predictor.explain_prediction(features)

        return float(prediction["retention_probability"]), risk_factors

    except Exception as e:
        logger.error(f"ML модуль недоступен или ошибка предсказания: {e}")

        score = 0.85
        risks = []

        if vector.commute_time_minutes > 60:
            score -= 0.3
            risks.append("Долгая дорога до работы (>60 мин)")

        if not vector.has_certifications:
            score -= 0.1
            risks.append("Отсутствуют сертификаты")

        if vector.years_experience < 2.0:
            score -= 0.2
            risks.append("Недостаточный опыт (<2 лет)")

        score = max(0.0, min(1.0, score))
        return score, risks
