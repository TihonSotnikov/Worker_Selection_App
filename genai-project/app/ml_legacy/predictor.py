"""
ML Model & Interpretability
ТЗ:
1. Обучить CatBoostClassifier / RandomForest
2. Реализовать функцию explain_prediction(vector), возвращающую топ-факторы риска
"""

import pandas as pd
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

    def _map_risk_level(self, retention_probability: float) -> str:
        if retention_probability >= 0.7:
            return "LOW"
        if retention_probability >= 0.4:
            return "MEDIUM"
        return "HIGH"
    
    def _prepare_feature_df(self, features: Dict) -> pd.DataFrame:
        features = dict(features)
        features.setdefault("age", 30)
        return pd.DataFrame([features])[self.feature_names]

    def _rule_based_weighted_risks(self, features: Dict) -> List[str]:
        features = dict(features)
        features.setdefault("age", 30)

        scored_risks = []

        def add(condition: bool, weight: float, text: str):
            if condition:
                scored_risks.append((weight, text))
        
        add(
            features.get("years_experience", 0) <= 0,
            3.5,
            "Требуется уточнение опыта",
        )

        add(features.get("commute_time_minutes", 0) > 120, 3.0, "Очень длинная дорога до работы")
        add(
            90 < features.get("commute_time_minutes", 0) <= 120,
            2.2,
            "Дорога на работу занимает больше 90 минут",
        )
        add(
            60 < features.get("commute_time_minutes", 0) <= 90,
            1.0,
            "Длительное время в пути до работы",
        )

        add(features.get("skills_verified_count", 0) < 3, 2.6, "Мало проверенных навыков (меньше 3)")
        add(
            3 <= features.get("skills_verified_count", 0) < 5,
            1.1,
            "Недостаточно подтвержденных навыков",
        )

        add(
            features.get("shift_preference") == ShiftPreference.NIGHT_ONLY.value
            and features.get("age", 30) > 50,
            2.0,
            "Возраст 50+ при выборе только ночных смен",
        )

        add(
            features.get("years_experience", 0) < 2
            and features.get("salary_expectation", 0) > 100000,
            1.8,
            "Мало опыта при высоких зарплатных ожиданиях",
        )

        add(
            not features.get("has_certifications", False)
            and features.get("skills_verified_count", 0) > 5,
            1.4,
            "Много навыков без подтверждающих сертификатов",
        )

        scored_risks.sort(key=lambda x: x[0], reverse=True)

        result = []
        seen = set()
        for _, text in scored_risks:
            if text not in seen:
                result.append(text)
                seen.add(text)
            if len(result) == 3:
                break

        return result

    def _format_feature_risk(self, feature_name: str, value, features: Dict) -> str | None:
        if feature_name == "commute_time_minutes":
            if value > 120:
                return "Очень длинная дорога до работы"
            if value > 90:
                return "Дорога на работу занимает больше 90 минут"
            if value > 60:
                return "Длительное время в пути до работы"
            return None

        if feature_name == "skills_verified_count":
            if value < 3:
                return "Мало проверенных навыков (меньше 3)"
            if value < 5:
                return "Недостаточно подтвержденных навыков"
            return None

        if feature_name == "shift_preference":
            if value == ShiftPreference.NIGHT_ONLY.value and features.get("age", 30) > 50:
                return "Возраст 50+ при выборе только ночных смен"
            return None

        if feature_name == "years_experience":
            if value <= 0:
                return "Требуется уточнение опыта"
            if value < 2 and features.get("salary_expectation", 0) > 100000:
                return "Мало опыта при высоких зарплатных ожиданиях"
            if value < 3:
                return "Невысокий релевантный опыт"
            return None

        if feature_name == "salary_expectation":
            if features.get("years_experience", 0) < 2 and value > 100000:
                return "Высокие зарплатные ожидания для текущего опыта"
            return None

        if feature_name == "has_certifications":
            if not bool(value) and features.get("skills_verified_count", 0) > 5:
                return "Много навыков без подтверждающих сертификатов"
            if not bool(value):
                return "Нет подтверждающих сертификатов"
            return None

        if feature_name == "age":
            if features.get("shift_preference") == ShiftPreference.NIGHT_ONLY.value and value > 50:
                return "Возраст усиливает риск при ночном графике"
            return None

        return None

    def train_model(self, data_path=DEFAULT_DATA_PATH):

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at: {data_path}")

        df = pd.read_csv(data_path)

        if "age" not in df.columns:
            df["age"] = 30

        feature_cols = [
            "skills_verified_count",
            "years_experience",
            "age",
            "commute_time_minutes",
            "shift_preference",
            "salary_expectation",
            "has_certifications",
        ]

        X = df[feature_cols]
        y = df["retention"]

        self.feature_names = feature_cols

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model = cb.CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.1,
            loss_function="Logloss",
            verbose=False,
            random_state=42,
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

        feature_df = self._prepare_feature_df(features)
        retention_prob = self.model.predict_proba(feature_df)[0, 1]

        return {
            "retention_probability": float(retention_prob),
            "will_stay": bool(retention_prob > 0.5),
            "risk_level": self._map_risk_level(retention_prob),
        }

    def explain_prediction(self, features: Dict) -> List[str]:
        features = dict(features)
        features.setdefault("age", 30)

        if not self.model or not self.feature_names:
            return self._rule_based_weighted_risks(features)

        try:
            feature_df = self._prepare_feature_df(features)
            pool = cb.Pool(feature_df)

            shap_values = self.model.get_feature_importance(pool, type="ShapValues")
            shap_row = shap_values[0][:-1]  # последний элемент — bias

            contributions = []
            for feature_name, shap_value in zip(self.feature_names, shap_row):
                # Берем только негативные вклады в класс "удержится"
                if shap_value >= -1e-6:
                    continue

                message = self._format_feature_risk(
                    feature_name=feature_name,
                    value=feature_df.iloc[0][feature_name],
                    features=features,
                )

                if message:
                    contributions.append((abs(float(shap_value)), message))

            if contributions:
                contributions.sort(key=lambda x: x[0], reverse=True)

                model_risks = []
                seen = set()
                for _, message in contributions:
                    if message not in seen:
                        model_risks.append(message)
                        seen.add(message)
                    if len(model_risks) == 3:
                        break

                fallback_risks = self._rule_based_weighted_risks(features)

                result = []
                for risk in model_risks + fallback_risks:
                    if risk not in result:
                        result.append(risk)
                    if len(result) == 3:
                        break

                if result:
                    return result
        except Exception:
            pass

        return self._rule_based_weighted_risks(features)

    def save_model(self, path=DEFAULT_MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "feature_names": self.feature_names}, f)
        print(f"Model saved explicitly to: {path}")

    def load_model(self, path=DEFAULT_MODEL_PATH):
        """Загрузка обученной модели"""
        if not os.path.exists(path):
            print(f"Model file not found at {path}")
            return

        with open(path, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.feature_names = data["feature_names"]


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
