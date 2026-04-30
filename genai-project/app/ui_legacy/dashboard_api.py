from functools import lru_cache
from pathlib import Path
import random
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, status

from app.core.enums import ShiftPreference
from app.ml_legacy.generator import generate_if_needed
from app.ml_legacy.predictor import RetentionPredictor


router = APIRouter(tags=["dashboard"])


RISK_LEVEL_LABELS = {
    "LOW": "Низкий риск",
    "MEDIUM": "Нужна проверка",
    "HIGH": "Высокий риск",
}


RISK_LEVEL_SHORTS = {
    "LOW": "Можно рассматривать",
    "MEDIUM": "Проверить детали",
    "HIGH": "Не спешить",
}


SHIFT_LABELS = {
    ShiftPreference.DAY_ONLY.value: "Дневная смена",
    ShiftPreference.NIGHT_ONLY.value: "Ночная смена",
    ShiftPreference.ANY.value: "Любая смена",
}


REQUIRED_CANDIDATE_FIELDS = {
    "skills_verified_count",
    "years_experience",
    "age",
    "commute_time_minutes",
    "shift_preference",
    "salary_expectation",
    "has_certifications",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_dataset(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)

    if "age" not in df.columns:
        raise ValueError(
            "В data/train_dataset.csv нет колонки age. "
            "Перегенерируй датасет через app/ml_legacy/generator.py"
        )

    invalid_rows = df[df["years_experience"] > (df["age"] - 18).clip(lower=0)]

    if not invalid_rows.empty:
        raise ValueError(
            f"В датасете {len(invalid_rows)} нереалистичных строк. "
            "Удалите train_dataset.csv, model.pkl и перезапустите генерацию."
        )

    return df


def row_to_candidate(row: pd.Series) -> dict[str, Any]:
    return {
        "skills_verified_count": int(row["skills_verified_count"]),
        "years_experience": float(row["years_experience"]),
        "age": int(row["age"]),
        "commute_time_minutes": int(row["commute_time_minutes"]),
        "shift_preference": int(row["shift_preference"]),
        "salary_expectation": int(row["salary_expectation"]),
        "has_certifications": bool(row["has_certifications"]),
    }


def build_edge_case_candidate() -> dict[str, Any]:
    rng = random.Random(42)

    return {
        "skills_verified_count": rng.randint(3, 8),
        "years_experience": 0.0,
        "age": rng.randint(21, 35),
        "commute_time_minutes": rng.randint(20, 90),
        "shift_preference": rng.choice(
            [
                ShiftPreference.DAY_ONLY.value,
                ShiftPreference.NIGHT_ONLY.value,
                ShiftPreference.ANY.value,
            ]
        ),
        "salary_expectation": rng.randint(50000, 110000),
        "has_certifications": rng.choice([True, False]),
    }


@lru_cache(maxsize=1)
def get_dashboard_runtime():
    root = project_root()

    model_path = root / "app" / "ml_legacy" / "model.pkl"
    data_path = root / "data" / "train_dataset.csv"

    dataset_existed = data_path.exists()

    generate_if_needed()

    dataset = load_dataset(data_path)

    predictor = RetentionPredictor()
    model_loaded = predictor.load_model(model_path)

    model_trained = False

    if not model_loaded:
        predictor.train_model(data_path)
        predictor.save_model(model_path)
        model_loaded = True
        model_trained = True

    boot_info = {
        "dataset_created": not dataset_existed,
        "model_trained": model_trained,
        "model_loaded": model_loaded,
    }

    return predictor, dataset, boot_info


def get_status_payload() -> dict[str, Any]:
    _, dataset, boot_info = get_dashboard_runtime()

    return {
        **boot_info,
        "dataset_rows": int(len(dataset)),
        "available_presets": [
            "ideal",
            "problematic",
            "borderline",
            "edge",
        ],
    }


def sample_candidate(category: str) -> dict[str, Any]:
    predictor, dataset, _ = get_dashboard_runtime()

    if category == "edge":
        return build_edge_case_candidate()

    if category == "ideal":
        subset = dataset[
            (dataset["skills_verified_count"] >= 7)
            & (dataset["years_experience"] >= 5)
            & (dataset["commute_time_minutes"] <= 40)
            & (dataset["has_certifications"] == 1)
            & (dataset["retention"] == 1)
        ]

    elif category == "problematic":
        subset = dataset[
            (
                (dataset["skills_verified_count"] < 3)
                | (dataset["commute_time_minutes"] > 90)
                | (
                    (dataset["shift_preference"] == ShiftPreference.NIGHT_ONLY.value)
                    & (dataset["age"] > 50)
                )
                | (
                    (dataset["years_experience"] < 2)
                    & (dataset["salary_expectation"] > 100000)
                )
            )
            & (dataset["retention"] == 0)
        ]

    elif category == "borderline":
        work_df = dataset.copy()

        if predictor.model is not None:
            work_df["pred_prob"] = predictor.model.predict_proba(
                work_df[predictor.feature_names]
            )[:, 1]

            subset = work_df[work_df["pred_prob"].between(0.45, 0.55)]

            if subset.empty:
                subset = work_df[work_df["pred_prob"].between(0.40, 0.60)]
        else:
            subset = work_df[
                (work_df["skills_verified_count"].between(4, 6))
                & (work_df["years_experience"].between(2, 5))
                & (work_df["commute_time_minutes"].between(50, 90))
            ]

    else:
        raise ValueError(
            "Неизвестный пресет. Доступны: ideal, problematic, borderline, edge"
        )

    if subset.empty:
        subset = dataset

    preset_random_states = {
        "ideal": 101,
        "problematic": 202,
        "borderline": 303,
    }

    row = subset.sample(
        n=1,
        random_state=preset_random_states.get(category, 42),
    ).iloc[0]

    return row_to_candidate(row)


def normalize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    missing = REQUIRED_CANDIDATE_FIELDS - set(candidate.keys())

    if missing:
        raise ValueError(f"Отсутствуют поля кандидата: {sorted(missing)}")

    normalized = {
        "skills_verified_count": int(candidate["skills_verified_count"]),
        "years_experience": float(candidate["years_experience"]),
        "age": int(candidate["age"]),
        "commute_time_minutes": int(candidate["commute_time_minutes"]),
        "shift_preference": int(candidate["shift_preference"]),
        "salary_expectation": int(candidate["salary_expectation"]),
        "has_certifications": bool(candidate["has_certifications"]),
    }

    if normalized["age"] < 18:
        raise ValueError("Возраст кандидата должен быть не меньше 18 лет")

    if normalized["years_experience"] > max(0, normalized["age"] - 18):
        raise ValueError("Опыт работы не может быть больше возраста минус 18 лет")

    if normalized["shift_preference"] not in SHIFT_LABELS:
        raise ValueError("Некорректное значение shift_preference")

    for key in (
        "skills_verified_count",
        "years_experience",
        "commute_time_minutes",
        "salary_expectation",
    ):
        if normalized[key] < 0:
            raise ValueError(f"Поле {key} не может быть отрицательным")

    return normalized


def predict_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    predictor, _, _ = get_dashboard_runtime()

    normalized = normalize_candidate(candidate)

    prediction = predictor.predict_retention(normalized)
    risk_factors = predictor.explain_prediction(normalized)

    display_risk_factors = [
        factor for factor in risk_factors if factor != "Требуется уточнение опыта"
    ]

    risk_level = prediction["risk_level"]
    risk_label = RISK_LEVEL_LABELS.get(risk_level, risk_level)

    if prediction["requires_review"]:
        status_text = "Требуется дополнительная проверка"
        zone_text = "Жёлтая зона: требуется ручная проверка данных"

    elif risk_level == "LOW":
        status_text = "Низкий риск: подходит для дальнейшего рассмотрения"
        zone_text = "Зелёная зона: низкий риск"

    elif risk_level == "MEDIUM":
        status_text = "Средний риск: рекомендуется адаптация и вводное сопровождение"
        zone_text = "Жёлтая зона: средний риск"

    else:
        status_text = "Высокий риск: требуется осторожная оценка и дополнительная проверка"
        zone_text = "Красная зона: высокий риск"

    return {
        "candidate": normalized,
        "retention_probability": float(prediction["retention_probability"]),
        "will_stay": bool(prediction["will_stay"]),
        "risk_level": risk_level,
        "risk_label": risk_label,
        "risk_short": RISK_LEVEL_SHORTS.get(risk_level, "Проверить"),
        "requires_review": bool(prediction["requires_review"]),
        "risk_factors": risk_factors,
        "display_risk_factors": display_risk_factors,
        "status_text": status_text,
        "zone_text": zone_text,
        "shift_label": SHIFT_LABELS[normalized["shift_preference"]],
        "uncertainty_low": float(prediction.get("uncertainty_low", 0.0)),
        "uncertainty_high": float(prediction.get("uncertainty_high", 1.0)),
        "uncertainty_margin": float(prediction.get("uncertainty_margin", 0.08)),
        "uncertainty_note": prediction.get(
            "uncertainty_note",
            "Ориентировочный коридор неопределённости.",
        ),
    }


@router.get("/demo/status")
def demo_status() -> dict[str, Any]:
    try:
        return get_status_payload()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not prepare dashboard status: {str(e)}",
        )


@router.get("/demo/candidate/{category}")
def get_demo_candidate(category: str) -> dict[str, Any]:
    try:
        candidate = sample_candidate(category)

        return {
            "category": category,
            "candidate": candidate,
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not create demo candidate: {str(e)}",
        )


@router.post("/demo/predict")
def predict_demo_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    try:
        return predict_candidate(candidate)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not predict candidate: {str(e)}",
        )