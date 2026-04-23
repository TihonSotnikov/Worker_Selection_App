"""
Streamlit UI
"""

import streamlit as st
import pandas as pd
import sys
import os
import random
from pathlib import Path

# Добавляем путь для импортов
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from app.ml_legacy.predictor import RetentionPredictor
from app.ml_legacy.generator import generate_if_needed
from app.core.enums import ShiftPreference


RISK_THEME = {
    "LOW": {
        "name": "Низкий риск",
        "zone": "Зелёная зона",
        "bg": "#10261c",
        "border": "#1f6f4a",
        "text": "#86efac",
        "bar": "#22c55e",
    },
    "MEDIUM": {
        "name": "Средний риск",
        "zone": "Жёлтая зона",
        "bg": "#2b2412",
        "border": "#8b6f2b",
        "text": "#f6d365",
        "bar": "#eab308",
    },
    "HIGH": {
        "name": "Высокий риск",
        "zone": "Красная зона",
        "bg": "#2a1518",
        "border": "#8f2d38",
        "text": "#fca5a5",
        "bar": "#ef4444",
    },
}


@st.cache_resource
def init_dashboard():
    base_dir = Path(__file__).resolve().parents[2]
    model_path = base_dir / "app" / "ml_legacy" / "model.pkl"
    data_path = base_dir / "data" / "train_dataset.csv"

    dataset_existed = data_path.exists()

    generate_if_needed()
    load_dataset.clear()

    predictor = RetentionPredictor()
    ok = predictor.load_model(model_path)

    model_trained = False
    if not ok:
        predictor.train_model()
        predictor.save_model(model_path)
        model_trained = True

    return predictor, {
        "dataset_created": not dataset_existed,
        "model_trained": model_trained,
        "model_loaded": ok or model_trained,
    }


@st.cache_data
def load_dataset() -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parents[2]
    data_path = base_dir / "data" / "train_dataset.csv"

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


def row_to_candidate(row: pd.Series) -> dict:
    return {
        "skills_verified_count": int(row["skills_verified_count"]),
        "years_experience": float(row["years_experience"]),
        "age": int(row["age"]),
        "commute_time_minutes": int(row["commute_time_minutes"]),
        "shift_preference": int(row["shift_preference"]),
        "salary_expectation": int(row["salary_expectation"]),
        "has_certifications": bool(row["has_certifications"]),
    }


def build_edge_case_candidate() -> dict:
    return {
        "skills_verified_count": random.randint(3, 8),
        "years_experience": 0.0,
        "age": random.randint(21, 35),
        "commute_time_minutes": random.randint(20, 90),
        "shift_preference": random.choice([
            ShiftPreference.DAY_ONLY.value,
            ShiftPreference.NIGHT_ONLY.value,
            ShiftPreference.ANY.value,
        ]),
        "salary_expectation": random.randint(50000, 110000),
        "has_certifications": random.choice([True, False]),
    }


def sample_candidate(
    df: pd.DataFrame, category: str, predictor: RetentionPredictor | None = None
) -> dict:
    if category == "ideal":
        subset = df[
            (df["skills_verified_count"] >= 7)
            & (df["years_experience"] >= 5)
            & (df["commute_time_minutes"] <= 40)
            & (df["has_certifications"] == 1)
            & (df["retention"] == 1)
        ]

    elif category == "problematic":
        subset = df[
            (
                (df["skills_verified_count"] < 3)
                | (df["commute_time_minutes"] > 90)
                | (
                    (df["shift_preference"] == ShiftPreference.NIGHT_ONLY.value)
                    & (df["age"] > 50)
                )
                | ((df["years_experience"] < 2) & (df["salary_expectation"] > 100000))
            )
            & (df["retention"] == 0)
        ]

    elif category == "borderline":
        if predictor is not None and predictor.model is not None:
            work_df = df.copy()
            work_df["pred_prob"] = predictor.model.predict_proba(
                work_df[predictor.feature_names]
            )[:, 1]

            subset = work_df[work_df["pred_prob"].between(0.45, 0.55)]

            if subset.empty:
                subset = work_df[work_df["pred_prob"].between(0.40, 0.60)]
        else:
            subset = df[
                (df["skills_verified_count"].between(4, 6))
                & (df["years_experience"].between(2, 5))
                & (df["commute_time_minutes"].between(50, 90))
            ]

    else:
        subset = df

    if subset.empty:
        subset = df

    row = subset.sample(n=1, random_state=random.randint(0, 1_000_000)).iloc[0]
    return row_to_candidate(row)


def main():
    # Настройка страницы
    st.set_page_config(page_title="Retention AI", layout="wide")

    # Заголовок
    st.title("Система прогнозирования удержания персонала")

    # Инициализация предсказателя
    predictor, boot_info = init_dashboard()

    st.caption("Локальный автономный режим: dashboard работает без main.py и API")

    if boot_info["dataset_created"]:
        st.success("Локально создан train_dataset.csv")

    if boot_info["model_trained"]:
        st.success("Локально обучена и сохранена model.pkl")

    if boot_info["model_loaded"] and not boot_info["model_trained"]:
        st.success("Модель успешно загружена")

    try:
        dataset = load_dataset()
    except Exception as e:
        st.error(f"Не удалось загрузить датасет: {e}")
        st.stop()

    # Боковая панель с кнопками-пресетами (ТЗ требование)
    with st.sidebar:
        st.header("Демонстрация")

        # Кнопка для идеального кандидата
        if st.button("Загрузить идеального кандидата", use_container_width=True):
            st.session_state.current_candidate = sample_candidate(dataset, "ideal", predictor)
            st.session_state.candidate_name = "Идеальный кандидат"

        # Кнопка для проблемного кандидата
        if st.button("Загрузить проблемного кандидата", use_container_width=True):
            st.session_state.current_candidate = sample_candidate(
                dataset, "problematic", predictor
            )
            st.session_state.candidate_name = "Проблемный кандидат"

        if st.button("Загрузить спорного кандидата", use_container_width=True):
            st.session_state.current_candidate = sample_candidate(dataset, "borderline", predictor)
            st.session_state.candidate_name = "Спорный кандидат"

        if st.button("Загрузить Edge Case", use_container_width=True):
            st.session_state.current_candidate = build_edge_case_candidate()
            st.session_state.candidate_name = "Edge Case: требуется уточнение опыта"

    # Основная панель
    if "current_candidate" in st.session_state:
        candidate = st.session_state.current_candidate
        name = st.session_state.candidate_name

        st.subheader(f"Анализ: {name}")

        # Предсказание удержания
        prediction = predictor.predict_retention(candidate)

        theme = RISK_THEME[prediction["risk_level"]]
        zone_text = (
            "Жёлтая зона: требуется ручная проверка данных"
            if prediction["requires_review"]
            else f"{theme['zone']}: {theme['name'].lower()}"
        )

        # Визуализация вероятности удержания (ТЗ: индикаторы риска)
        col1, col2 = st.columns(2)

        with col1:
            # Прогресс-бар с вероятностью
            st.metric(
                "Вероятность удержания", f"{prediction['retention_probability']:.1%}"
            )

            risk_level_labels = {
                "LOW": "Низкий риск",
                "MEDIUM": "Средний риск",
                "HIGH": "Высокий риск",
            }
            st.metric(
                "Уровень риска",
                risk_level_labels.get(
                    prediction["risk_level"], prediction["risk_level"]
                ),
            )

            # Индикатор решения
            if prediction["requires_review"]:
                status_text = "Требуется дополнительная проверка"
                status_bg = "#111827"
                status_border = "#263244"
                status_color = "#e5e7eb"
            elif prediction["risk_level"] == "LOW":
                status_text = "Низкий риск: подходит для дальнейшего рассмотрения"
                status_bg = theme["bg"]
                status_border = theme["border"]
                status_color = theme["text"]
            elif prediction["risk_level"] == "MEDIUM":
                status_text = "Средний риск: рекомендуется адаптация и вводное сопровождение"
                status_bg = theme["bg"]
                status_border = theme["border"]
                status_color = theme["text"]
            else:
                status_text = "Высокий риск: требуется осторожная оценка и дополнительная проверка"
                status_bg = theme["bg"]
                status_border = theme["border"]
                status_color = theme["text"]

            st.markdown(
                f"""
                <div style="
                    margin-top: 18px;
                    padding: 18px;
                    border-radius: 14px;
                    background: {status_bg};
                    border: 1px solid {status_border};
                    color: {status_color};
                    font-weight: 600;
                    font-size: 16px;
                ">
                    {status_text}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            progress_width = int(prediction["retention_probability"] * 100)

            st.markdown(
                f"""
                <div style="margin-top: 8px;">
                    <div style="
                        width: 100%;
                        height: 16px;
                        background: #1f2430;
                        border-radius: 999px;
                        overflow: hidden;
                        border: 1px solid #2b3240;
                    ">
                        <div style="
                            width: {progress_width}%;
                            height: 100%;
                            background: {theme['bar']};
                            border-radius: 999px;
                        "></div>
                    </div>
                    <div style="
                        margin-top: 10px;
                        font-size: 15px;
                        color: #9ca3af;
                    ">
                        Retention Score: {prediction['retention_probability']:.1%}
                    </div>
                    <div style="
                        margin-top: 18px;
                        padding: 16px 18px;
                        border-radius: 14px;
                        background: {theme['bg']};
                        border: 1px solid {theme['border']};
                        color: {theme['text']};
                        font-weight: 600;
                        font-size: 16px;
                    ">
                        {zone_text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Факторы риска (ТЗ: объяснение предсказания)
        st.subheader("Факторы риска")
        risk_factors = predictor.explain_prediction(candidate)

        display_risk_factors = [
            factor for factor in risk_factors
            if factor != "Требуется уточнение опыта"
        ]

        if prediction["requires_review"]:
            st.markdown(
                """
                <div style="
                    margin-bottom: 16px;
                    padding: 16px 18px;
                    border-radius: 14px;
                    background: #111827;
                    border: 1px solid #263244;
                    color: #e5e7eb;
                    line-height: 1.5;
                ">
                    <div style="
                        font-size: 18px;
                        font-weight: 700;
                        margin-bottom: 10px;
                        color: #f3f4f6;
                    ">
                        Ключевой фактор риска
                    </div>
                    <div style="
                        font-size: 16px;
                        color: #d1d5db;
                    ">
                        Отсутствуют или неполны данные об опыте работы.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if display_risk_factors and not prediction["requires_review"]:
            risk_text = "<br>".join([f"• {factor}" for factor in display_risk_factors])
            st.markdown(
                f"""
                <div style="
                    padding:18px;
                    border-radius:14px;
                    background:#111827;
                    border:1px solid #263244;
                    color:#e5e7eb;
                    line-height:1.6;
                ">
                    <div style="
                        font-size:18px;
                        font-weight:700;
                        margin-bottom:12px;
                        color:#f3f4f6;
                    ">
                        Ключевые факторы риска
                    </div>
                    <div style="font-size:16px; color:#d1d5db;">
                        {risk_text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif not prediction["requires_review"]:
            st.info("Значительных факторов риска не обнаружено")

        # Детали кандидата
        st.subheader("Детали кандидата")

        # Таблица с признаками
        details_df = pd.DataFrame([candidate]).T.reset_index()
        details_df.columns = ["Признак", "Значение"]

        # Преобразование кодов смен в понятные названия
        shift_map = {
            ShiftPreference.DAY_ONLY.value: "Дневная смена",
            ShiftPreference.NIGHT_ONLY.value: "Ночная смена",
            ShiftPreference.ANY.value: "Любая смена",
        }

        details_df.loc[details_df["Признак"] == "shift_preference", "Значение"] = (
            shift_map.get(candidate["shift_preference"], "Не указано")
        )

        details_df.loc[details_df["Признак"] == "has_certifications", "Значение"] = (
            "Да" if candidate["has_certifications"] else "Нет"
        )

        details_df["Значение"] = details_df["Значение"].astype(str)

        st.table(details_df)
    else:
        # Инструкция при первом запуске
        st.info("""
        Используйте кнопки в боковой панели для загрузки демо-кандидатов:

        1. **Идеальный кандидат** - соответствует всем правилам удержания
        2. **Проблемный кандидат** - нарушает несколько правил
        3. **Спорный кандидат** — пограничный кейс около порога решения модели
        4. **Edge Case** — неполные данные по опыту, требуется ручная проверка

        Система покажет:
        • Вероятность удержания
        • Уровень риска
        • Факторы риска
        • Рекомендацию по дальнейшему рассмотрению
        """)


if __name__ == "__main__":
    main()
