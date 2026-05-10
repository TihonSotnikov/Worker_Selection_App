import json
from typing import List

from sqlmodel import Session, select

from app.api.models_db import CandidateTable
from app.core.schemas import CandidateResult, CandidateVector
from app.core.enums import ShiftPreference


def save_candidate(
    session: Session,
    full_name: str,
    raw_summary: str,
    retention_score: float,
    risk_factors: List[str],
    vector: CandidateVector,
) -> CandidateTable:
    """
    Сохраняет результаты анализа кандидата в базу данных.

    Parameters
    ----------
    session : Session
        Сессия базы данных.
    full_name : str
        Полное имя кандидата.
    raw_summary : str
        Текстовое резюме от LLM.
    retention_score : float
        Вероятность удержания от ML-модели.
    risk_factors : List[str]
        Список факторов риска.
    vector : CandidateVector
        Вектор признаков кандидата.

    Returns
    -------
    CandidateTable
        Сохраненная запись из базы данных.
    """

    db_candidate = CandidateTable(
        full_name=full_name,
        raw_summary=raw_summary,
        retention_score=retention_score,
        risk_factors=json.dumps(risk_factors, ensure_ascii=False),
        vec_skills_count=vector.skills_verified_count,
        vec_years_experience=vector.years_experience,
        vec_commute_minutes=vector.commute_time_minutes,
        vec_shift_preference=int(vector.shift_preference),
        vec_salary_expectation=vector.salary_expectation,
        vec_has_certifications=vector.has_certifications,
    )

    session.add(db_candidate)
    session.commit()
    session.refresh(db_candidate)

    return db_candidate


def get_all_candidates(session: Session) -> List[CandidateResult]:
    """
    Получает список всех кандидатов из БД.

    Parameters
    ----------
    session : Session
        Сессия БД.

    Returns
    -------
    List[CandidateResult]
        Список всех кандидатов, отсортированных по дате (новые первые).
    """

    candidates = session.exec(
        select(CandidateTable).order_by(CandidateTable.created_at.desc())
    ).all()

    results = []
    for db_candidate in candidates:
        vector = CandidateVector(
            skills_verified_count=db_candidate.vec_skills_count,
            years_experience=db_candidate.vec_years_experience,
            commute_time_minutes=db_candidate.vec_commute_minutes,
            shift_preference=ShiftPreference(db_candidate.vec_shift_preference),
            salary_expectation=db_candidate.vec_salary_expectation,
            has_certifications=db_candidate.vec_has_certifications,
        )

        result = CandidateResult(
            id=str(db_candidate.id),
            full_name=db_candidate.full_name,
            raw_summary=db_candidate.raw_summary,
            vector=vector,
            retention_score=db_candidate.retention_score,
            risk_factors=json.loads(db_candidate.risk_factors),
        )
        results.append(result)

    return results
