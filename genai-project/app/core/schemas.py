from typing import List
from pydantic import BaseModel, Field
from app.core.enums import ShiftPreference


class CandidateVector(BaseModel):
    """
    Чистый вектор данных кандидата для ML-модели.

    Это структура данных, которая передается от AI-модуля (экстрактора) в ML-модуль (предиктора).

    Attributes
    ----------
    skills_verified_count : int
        Количество подтвержденных навыков.
    years_experience : float
        Опыт работы в годах.
    commute_time_minutes : int
        Время пути до работы в минутах.
    shift_preference : ShiftPreference
        Предпочитаемый график.
    salary_expectation : int
        Ожидаемая зарплата.
    has_certifications : bool
        Наличие сертификатов/корочек.
    """

    skills_verified_count: int = Field(..., ge=0, description="Количество подтвержденных навыков")
    years_experience: float = Field(..., ge=0.0, description="Опыт работы в годах")
    commute_time_minutes: int = Field(..., ge=0, description="Время пути до работы в минутах")
    shift_preference: ShiftPreference = Field(..., description="Предпочитаемый график")
    salary_expectation: int = Field(..., ge=0, description="Ожидаемая зарплата")
    has_certifications: bool = Field(..., description="Наличие сертификатов/корочек")


class Candidateresult(BaseModel):
    """
    Финальный результат анализа кандидата.

    Объединяет персональные данные, вектор признаков и результаты предсказания ML-модели.
    Структура данных, которая возвращается на frontend.

    Attributes
    ----------
    id : str
        Уникальный идентификатор кандидата (UUID).
    full_name : str
        Полное имя кандидата.
    raw_summary : str
        Краткое резюме, сгенерированное LLM.
    vector : CandidateVector
        Структурированный вектор признаков кандидата, используемый ML-моделью.
    retention_score : float
        Вероятность удержания кандидата в компании. Значение в диапазоне [0.0, 1.0].
    risk_factors : List[str]
        Список текстовых факторов риска, выявленных моделью или логикой анализа.
    """

    id: str = Field(..., description="Уникальный ID кандидата (UUID)")
    full_name: str = Field(..., description="ФИО кандидата")
    raw_summary: str = Field(..., description="Краткое резюме, сгенерированное LLM")

    vector: CandidateVector

    retention_score: float = Field(..., ge=0.0, le=1.0, description="Вероятность удержания [0.0 — 1.0]")
    risk_factors: List[str] = Field(default_factory=list, description="Список текстовых пояснений рисков")
