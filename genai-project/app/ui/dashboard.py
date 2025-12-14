"""
Streamlit UI
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Добавляем путь для импортов
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from app.ml.predictor import RetentionPredictor
from app.core.enums import ShiftPreference


def init_dashboard():
    """Инициализация дашборда - проверка наличия модели"""
    # Проверяем наличие модели
    model_path = "app/ml/model.pkl"

    if not os.path.exists(model_path):
        st.error(f" ML модель не найдена по пути: {model_path}")
        st.info("Сначала запустите main.py для генерации данных и обучения модели")
        return None

    # Проверяем наличие данных
    data_path = "data/train_dataset.csv"
    if not os.path.exists(data_path):
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


def create_demo_vectors():
    """Создание демо-кандидатов для кнопок-пресетов (ТЗ требование)"""
    # Идеальный кандидат (следует всем правилам)
    perfect = {
        "skills_verified_count": 8,
        "years_experience": 10,
        "commute_time_minutes": 20,  # < 90 минут
        "shift_preference": ShiftPreference.DAY_ONLY.value,
        "salary_expectation": 80000,  # Реалистичная ЗП
        "has_certifications": True
    }

    # Проблемный кандидат (нарушает правила)
    problematic = {
        "skills_verified_count": 2,  # < 3 навыков
        "years_experience": 1,  # Мало опыта
        "commute_time_minutes": 120,  # > 90 минут
        "shift_preference": ShiftPreference.NIGHT_ONLY.value,
        "salary_expectation": 120000,  # Высокая ЗП при малом опыте
        "has_certifications": False  # Нет сертификатов
    }

    return perfect, problematic


def main():
    # Настройка страницы
    st.set_page_config(page_title="Retention AI", layout="wide")

    # Заголовок
    st.title("Система прогнозирования удержания персонала")

    # Инициализация предсказателя
    predictor = RetentionPredictor()

    # Загрузка модели
    try:
        predictor.load_model("app/ml/model.pkl")
        st.success("Модель успешно загружена")
    except:
        st.warning("Модель не найдена. Сначала обучите модель.")

    # Боковая панель с кнопками-пресетами (ТЗ требование)
    with st.sidebar:
        st.header("Демонстрация")

        # Кнопка для идеального кандидата
        if st.button("Загрузить идеального кандидата"):
            perfect, _ = create_demo_vectors()
            st.session_state.current_candidate = perfect
            st.session_state.candidate_name = "Идеальный кандидат"

        # Кнопка для проблемного кандидата
        if st.button("Загрузить проблемного кандидата"):
            _, problematic = create_demo_vectors()
            st.session_state.current_candidate = problematic
            st.session_state.candidate_name = "Проблемный кандидат"

    # Основная панель
    if 'current_candidate' in st.session_state:
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
                "Вероятность удержания",
                f"{prediction['retention_probability']:.1%}"
            )

            # Индикатор решения
            if prediction['will_stay']:
                st.success(" Рекомендуется к найму")
            else:
                st.error(" Высокий риск увольнения")

        with col2:
            # Круговая диаграмма риска
            fig = go.Figure(data=[
                go.Indicator(
                    mode="gauge+number",
                    value=prediction['retention_probability'] * 100,
                    title={'text': "Вероятность удержания"},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': "darkblue"},
                           'steps': [
                               {'range': [0, 30], 'color': "red"},
                               {'range': [30, 70], 'color': "yellow"},
                               {'range': [70, 100], 'color': "green"}]}
                )
            ])
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
            ShiftPreference.ANY.value: "Любая смена"
        }

        details_df.loc[details_df['Признак'] == 'shift_preference', 'Значение'] = \
            shift_map.get(candidate['shift_preference'], "Не указано")

        details_df.loc[details_df['Признак'] == 'has_certifications', 'Значение'] = \
            "Да" if candidate['has_certifications'] else "Нет"

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