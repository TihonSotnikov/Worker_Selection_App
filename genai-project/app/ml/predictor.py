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
import os
from typing import Dict, List
from app.core.enums import ShiftPreference

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(CURRENT_DIR, "model.pkl")
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "train_dataset.csv")


class RetentionPredictor:
    def __init__(self):
        self.model = None
        self.feature_names = None

    def train_model(self, data_path=DEFAULT_DATA_PATH):
        
        if not os.path.exists(data_path):
             raise FileNotFoundError(f"Dataset not found at: {data_path}")

        df = pd.read_csv(data_path)

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

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model = cb.CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.1,
            loss_function='Logloss',
            verbose=False,
            random_state=42
        )

        self.model.fit(X_train, y_train, eval_set=(X_test, y_test))

        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        print(f"Точность модели: {accuracy:.3f}")
        print(f"ROC-AUC: {roc_auc:.3f}")

        return self.model

    def predict_retention(self, features: Dict):
        if not self.model:
             raise ValueError("Model is not loaded or trained.")

        feature_df = pd.DataFrame([features])[self.feature_names]

        retention_prob = self.model.predict_proba(feature_df)[0, 1]

        return {
            "retention_probability": float(retention_prob),
            "will_stay": retention_prob > 0.5
        }

    def explain_prediction(self, features: Dict) -> List[str]:
        risk_factors = []

        if features.get('commute_time_minutes', 0) > 90:
            risk_factors.append("Дорога на работу занимает больше 90 минут")

        if features.get('skills_verified_count', 0) < 3:
            risk_factors.append("Мало проверенных навыков (меньше 3)")

        if (features.get('shift_preference') == ShiftPreference.NIGHT_ONLY.value):
            risk_factors.append("Предпочтение ночной смены")

        if (features.get('years_experience', 0) < 2 and
                features.get('salary_expectation', 0) > 100000):
            risk_factors.append("Мало опыта при высокой зарплатной ожидании")

        if not features.get('has_certifications', False):
            risk_factors.append("Отсутствие подтверждающих сертификатов")

        return risk_factors[:3]

    def save_model(self, path=DEFAULT_MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names
            }, f)
        print(f"Model saved explicitly to: {path}")

    def load_model(self, path=DEFAULT_MODEL_PATH):
        """Загрузка обученной модели"""
        if not os.path.exists(path):
            print(f"Model file not found at {path}")
            return
            
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_names = data['feature_names']


def train_if_needed():
    data_path = DEFAULT_MODEL_PATH

    print(f"Checking model at: {data_path}")

    if not os.path.exists(data_path) or os.path.getsize(data_path) == 0:
        print("Training model...")
        model = RetentionPredictor()
        model.train_model() 
        model.save_model(data_path)
        print("Model is trained and saved.")
    else:
        print("Model is trained yet.")
