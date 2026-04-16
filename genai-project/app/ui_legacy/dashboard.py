"""
Streamlit UI
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
import random
from pathlib import Path

# Добавляем путь для импортов
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from app.ml_legacy.predictor import RetentionPredictor
from app.core.enums import ShiftPreference


def init_dashboard():
    """Инициализация дашборда: проверка наличия модели и датасета, загрузка модели."""
    base_dir = Path(__file__).resolve().parents[2]
    model_path = base_dir / "app" / "ml_legacy" / "model.pkl"
    data_path = base_dir / "data" / "train_dataset.csv"

    if not model_path.exists():
        st.error(f" ML модель не найдена по пути: {model_path}")
        st.info("Сначала запустите main.py для генерации данных и обучения модели")
        return None

    # Проверяем наличие данных
    if not data_path.exists():
        st.error(f" Тренировочные данные не найдены: {data_path}")
        return None

    # Пытаемся загрузить модель
    try:
        predictor = RetentionPredictor()
        predictor.load_model(model_path)
        return predictor
    except Exception as e:
        st.error(f" Ошибка загрузки модели: {str(e)}")
        return None


@st.cache_data
def load_dataset() -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parents[2]
    data_path = base_dir / "data" / "train_dataset.csv"
    return pd.read_csv(data_path)


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


def sample_candidate(df: pd.DataFrame, category: str) -> dict:
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
                | (
                    (df["years_experience"] < 2)
                    & (df["salary_expectation"] > 100000)
                )
            )
            & (df["retention"] == 0)
        ]

    elif category == "borderline":
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
    predictor = init_dashboard()

    dataset = load_dataset()

    # Загрузка модели
    if predictor is None:
        st.stop()

    st.success("Модель успешно загружена")

    # Боковая панель с кнопками-пресетами (ТЗ требование)
    with st.sidebar:
        st.header("Демонстрация")

        # Кнопка для идеального кандидата
        if st.button("Загрузить идеального кандидата"):
            st.session_state.current_candidate = sample_candidate(dataset, "ideal")
            st.session_state.candidate_name = "Идеальный кандидат"

        # Кнопка для проблемного кандидата
        if st.button("Загрузить проблемного кандидата"):
            st.session_state.current_candidate = sample_candidate(dataset, "problematic")
            st.session_state.candidate_name = "Проблемный кандидат"

        if st.button("Загрузить спорного кандидата"):
            st.session_state.current_candidate = sample_candidate(dataset, "borderline")
            st.session_state.candidate_name = "Спорный кандидат"

    # Основная панель
    if "current_candidate" in st.session_state:
        candidate = st.session_state.current_candidate
        name = st.session_state.candidate_name

        st.subheader(f"Анализ: {name}")

        # Предсказание удержания
        prediction = predictor.predict_retention(candidate)

        # Визуализация вероятности удержания (ТЗ: индикаторы риска)
        col1, col2 = st.columns(2)

        with col1:
            # Прогресс-бар с вероятностью
            st.metric(
                "Вероятность удержания", f"{prediction['retention_probability']:.1%}"
            )

            # Индикатор решения
            if prediction["will_stay"]:
                st.success(" Рекомендуется к найму")
            else:
                st.error(" Высокий риск увольнения")

        with col2:
            # Круговая диаграмма риска
            fig = go.Figure(
                data=[
                    go.Indicator(
                        mode="gauge+number",
                        value=prediction["retention_probability"] * 100,
                        title={"text": "Вероятность удержания"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "darkblue"},
                            "steps": [
                                {"range": [0, 30], "color": "red"},
                                {"range": [30, 70], "color": "yellow"},
                                {"range": [70, 100], "color": "green"},
                            ],
                        },
                    )
                ]
            )
            st.plotly_chart(fig, use_container_width=True)

        # Факторы риска (ТЗ: объяснение предсказания)
        st.subheader("Факторы риска")
        risk_factors = predictor.explain_prediction(candidate)

        if risk_factors:
            for factor in risk_factors:
                st.warning(f"• {factor}")
        else:
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

        Система покажет:
        • Вероятность удержания
        • Факторы риска
        • Рекомендацию по найму
        """)


if __name__ == "__main__":
    main()
