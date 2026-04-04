import os
import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

os.environ["TESTING"] = "1"

from main import app
from app.core.schemas import CandidateVector
from app.core.enums import ShiftPreference


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client


def test_get_history(client):
    """
    Тестирует эндпоинт получения истории кандидатов.

    Проверяет статус-код ответа и тип возвращаемых данных.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    response = client.get("/api/history")

    assert response.status_code == 200
    assert isinstance(response.json(), list)


@patch("app.api.services.ml_predict", new_callable=AsyncMock)
@patch("app.api.services.ai_extract", new_callable=AsyncMock)
def test_post_analyze(mock_ai_extract, mock_ml_predict, client):
    """
    Тестирует эндпоинт загрузки и анализа резюме кандидата.

    Использует мокирование тяжелых ML и AI сервисов для изоляции
    проверки транспортного уровня и бизнес-логики.

    Parameters
    ----------
    mock_ai_extract : unittest.mock.AsyncMock
        Мок асинхронной функции AI-экстракции данных из файла.
    mock_ml_predict : unittest.mock.AsyncMock
        Мок асинхронной функции ML-предсказания удержания.

    Returns
    -------
    None
    """
    vector = CandidateVector(
        skills_verified_count=3,
        years_experience=2.5,
        commute_time_minutes=40,
        shift_preference=list(ShiftPreference)[0],
        salary_expectation=60000,
        has_certifications=True,
    )

    mock_ai_extract.return_value = ("Test Candidate", "Test Summary", vector)
    mock_ml_predict.return_value = (0.95, ["No risks"])

    files = {"file": ("resume.txt", b"test file content", "text/plain")}
    response = client.post("/api/analyze", files=files)

    assert response.status_code == 201
    response_data = response.json()
    assert response_data["full_name"] == "Test Candidate"
    assert response_data["raw_summary"] == "Test Summary"
    assert response_data["retention_score"] == 0.95
    assert response_data["risk_factors"] == ["No risks"]
