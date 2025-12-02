"""
ML Model & Interpretability
ТЗ: 
1. Обучить CatBoostClassifier / RandomForest
2. Реализовать функцию explain_prediction(vector), возвращающую топ-факторы риска
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import catboost as cb
import pickle
from typing import Dict, List
from app.core.enums import ShiftPreference


class RetentionPredictor:
    def __init__(self):
        self.model = None
        self.feature_names = None

    def train_model(self, data_path="../data/train_dataset.csv"):
        """Обучение CatBoost модели"""
        # Загрузка сгенерированных данных
        df = pd.read_csv(data_path)

        # Признаки для обучения
        feature_cols = [
            'skills_verified_count',
            'years_experience',
            'commute_time_minutes',
            'shift_preference',
            'salary_expectation',
            'has_certifications'
        ]

        X = df[feature_cols]
        y = df['retention']

        self.feature_names = feature_cols

        # Разделение на тренировочную и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Обучение CatBoost модели
        self.model = cb.CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.1,
            loss_function='Logloss',
            verbose=False,
            random_state=42
        )

        self.model.fit(X_train, y_train, eval_set=(X_test, y_test))

        # Оценка модели
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        print(f"Точность модели: {accuracy:.3f}")
        print(f"ROC-AUC: {roc_auc:.3f}")

        return self.model

    def predict_retention(self, features: Dict):
        """Предсказание вероятности удержания"""
        # Преобразование признаков в формат для модели
        feature_df = pd.DataFrame([features])[self.feature_names]

        # Получение вероятности удержания
        retention_prob = self.model.predict_proba(feature_df)[0, 1]

        return {
            "retention_probability": float(retention_prob),
            "will_stay": retention_prob > 0.5
        }

    def explain_prediction(self, features: Dict) -> List[str]:
        """
        ТЗ: Функция, возвращающая топ-факторы риска
        Анализирует вектор признаков и возвращает список факторов риска
        """
        risk_factors = []

        # Проверка времени в пути
        if features.get('commute_time_minutes', 0) > 90:
            risk_factors.append("Дорога на работу занимает больше 90 минут")

        # Проверка количества навыков
        if features.get('skills_verified_count', 0) < 3:
            risk_factors.append("Мало проверенных навыков (меньше 3)")

        # Проверка смены и возраста
        if (features.get('shift_preference') == ShiftPreference.NIGHT_ONLY.value):
            risk_factors.append("Предпочтение ночной смены")

        # Проверка опыта и зарплатных ожиданий
        if (features.get('years_experience', 0) < 2 and
                features.get('salary_expectation', 0) > 100000):
            risk_factors.append("Мало опыта при высокой зарплатной ожидании")

        # Проверка сертификатов
        if not features.get('has_certifications', False):
            risk_factors.append("Отсутствие подтверждающих сертификатов")

        # Возвращаем топ-3 фактора риска
        return risk_factors[:3]

    def save_model(self, path="../data/model.pkl"):
        """Сохранение обученной модели"""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names
            }, f)

    def load_model(self, path="../data/model.pkl"):
        """Загрузка обученной модели"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_names = data['feature_names']