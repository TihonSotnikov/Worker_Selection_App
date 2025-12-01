from sqlmodel import SQLModel, Field
from datetime import datetime
from typing import Optional
import uuid


class CandidateTable(SQLModel, table=True):
    """
    Плоская таблица для хранения результатов анализа кандидата.

    Используется как финальное хранилище: объединяет персональные данные,
    краткое резюме, предсказание ML-модели и все векторные признаки.
    Формат соответствует ограничениям SQLite (списки сериализуются в JSON).

    Attributes
    ----------
    id : Optional[int]
        Первичный ключ. Генерируется SQLite автоматически.
    created_at : datetime
        Timestamp создания записи.
    full_name : str
        Полное имя кандидата.
    raw_summary : str
        Краткое резюме, сгенерированное LLM.
    retention_score : float
        Предсказанная ML-моделью вероятность удержания кандидата.
    risk_factors : str
        JSON-строка со списком текстовых факторов риска.
        Храним как строку из-за ограничений SQLite.

    vec_skills_count : int
        Количество подтверждённых навыков.
    vec_years_experience : float
        Опыт работы в годах.
    vec_commute_minutes : int
        Время пути до работы.
    vec_shift_preference : int
        Код предпочитаемого графика (enum → int).
    vec_salary_expectation : int
        Ожидаемая зарплата.
    vec_has_certifications : bool
        Наличие сертификатов/корочек.
    """

    __tablename__ = "candidates"

    id: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True,
        index=True,
        nullable=False
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    full_name: str
    raw_summary: str
    retention_score: float

    # SQLite не хранит списки, поэтому храним JSON-строку
    risk_factors: str

    vec_skills_count: int
    vec_years_experience: float
    vec_commute_minutes: int
    vec_shift_preference: int
    vec_salary_expectation: int
    vec_has_certifications: bool
