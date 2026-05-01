"""
ML Model & Interpretability
ТЗ:
1. Обучить CatBoostClassifier / RandomForest
2. Реализовать функцию explain_prediction(vector), возвращающую топ-факторы риска
"""

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from typing import Dict, List
import catboost as cb
import pandas as pd
import pickle
import os

from app.ml_legacy.feature_contract import FEATURE_COLS, FEATURE_DEFAULTS, FAMILY_WITH_KIDS
from app.core.enums import ShiftPreference

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(CURRENT_DIR, "model.pkl")
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "train_dataset.csv")


class RetentionPredictor:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self._prediction_cache = {}
        self._explain_cache = {}
        self._positive_cache = {}

    def _map_risk_level(self, retention_probability: float) -> str:
        if retention_probability >= 0.7:
            return "LOW"
        if retention_probability >= 0.4:
            return "MEDIUM"
        return "HIGH"
    
    def _detect_requires_review(self, features: Dict) -> bool:
        return features.get("years_experience", 0) <= 0
    
    def _estimate_uncertainty_band(
        self,
        retention_probability: float,
        features: Dict,
        requires_review: bool = False,
    ) -> dict:
        """
        UX-коридор неопределённости.

        Это НЕ строгий статистический доверительный интервал.
        Это честный интерфейсный индикатор: чем ближе кандидат к спорной зоне
        или чем хуже полнота данных, тем шире коридор.
        """

        if requires_review:
            margin = 0.12
        elif 0.4 <= retention_probability < 0.7:
            margin = 0.08
        elif retention_probability < 0.4:
            margin = 0.06
        else:
            margin = 0.05

        return {
            "uncertainty_low": max(0.0, retention_probability - margin),
            "uncertainty_high": min(1.0, retention_probability + margin),
            "uncertainty_margin": margin,
            "uncertainty_note": (
                "Ориентировочный коридор неопределённости. "
                "Это не строгий статистический доверительный интервал, "
                "а UX-подсказка для HR-интерпретации результата."
            ),
        }

    def _prepare_feature_df(self, features: Dict) -> pd.DataFrame:
        features = {
            **FEATURE_DEFAULTS,
            **dict(features),
        }

        required_features = set(self.feature_names or [])
        missing = required_features - set(features.keys())

        if missing:
            raise ValueError(f"Отсутствуют признаки для модели: {sorted(missing)}")

        return pd.DataFrame([features])[self.feature_names]

    def _rule_based_weighted_risks(self, features: Dict) -> List[str]:
        features = {
            **FEATURE_DEFAULTS,
            **dict(features),
        }

        scored_risks = []

        def add(condition: bool, weight: float, text: str):
            if condition:
                scored_risks.append((weight, text))

        add(
            features.get("years_experience", 0) <= 0,
            3.5,
            "Требуется уточнение опыта",
        )

        add(
            features.get("commute_time_minutes", 0) > 120,
            3.0,
            "Очень длинная дорога до работы",
        )
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

        add(
            features.get("skills_verified_count", 0) < 3,
            2.6,
            "Мало проверенных навыков (меньше 3)",
        )
        add(
            3 <= features.get("skills_verified_count", 0) < 5,
            1.1,
            "Недостаточно подтвержденных навыков",
        )

        add(
            features.get("shift_preference") == ShiftPreference.NIGHT_ONLY.value
            and features.get("age", 0) > 50,
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

        add(
            features.get("previous_turnovers", 0) > 3,
            3.2,
            "Частая смена прошлых мест работы",
        )

        add(
            2 <= features.get("previous_turnovers", 0) <= 3,
            1.6,
            "Есть история частой смены работы",
        )

        add(
            features.get("family_status") in FAMILY_WITH_KIDS
            and features.get("shift_preference") == ShiftPreference.NIGHT_ONLY.value,
            2.5,
            "Семейная нагрузка при выборе ночных смен",
        )

        add(
            not features.get("has_transport", False)
            and features.get("commute_time_minutes", 0) > 60,
            2.0,
            "Нет личного транспорта при долгой дороге",
        )

        add(
            features.get("housing_type") == 2,
            1.2,
            "Нестабильные жилищные условия",
        )

        add(
            features.get("education_level") == 0
            and features.get("skills_verified_count", 0) < 3,
            1.0,
            "Низкий уровень образования при малом числе подтверждённых навыков",
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

    def _format_feature_risk(
        self, feature_name: str, value, features: Dict
    ) -> str | None:
        features = {
            **FEATURE_DEFAULTS,
            **dict(features),
        }

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
            if (
                value == ShiftPreference.NIGHT_ONLY.value
                and features.get("age", 0) > 50
            ):
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
            if (
                features.get("shift_preference") == ShiftPreference.NIGHT_ONLY.value
                and value > 50
            ):
                return "Возраст усиливает риск при ночном графике"
            return None

        if feature_name == "previous_turnovers":
            if value > 3:
                return "Частая смена прошлых мест работы"
            if value >= 2:
                return "Есть история частой смены работы"
            return None

        if feature_name == "family_status":
            if (
                value in FAMILY_WITH_KIDS
                and features.get("shift_preference") == ShiftPreference.NIGHT_ONLY.value
            ):
                return "Семейная нагрузка при выборе ночных смен"
            return None

        if feature_name == "housing_type":
            if value == 2:
                return "Нестабильные жилищные условия"
            return None

        if feature_name == "has_transport":
            if (
                not bool(value)
                and features.get("commute_time_minutes", 0) > 60
            ):
                return "Нет личного транспорта при долгой дороге"
            return None

        if feature_name == "education_level":
            if value == 0 and features.get("skills_verified_count", 0) < 3:
                return "Низкий уровень образования при малом числе подтверждённых навыков"
            return None

        return None

    def _feature_cache_key(self, features: Dict) -> tuple:
        prepared = {
            **FEATURE_DEFAULTS,
            **dict(features),
        }

        names = self.feature_names or FEATURE_COLS

        return tuple((name, prepared.get(name)) for name in names)
    
    def train_model(self, data_path=DEFAULT_DATA_PATH):

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at: {data_path}")

        df = pd.read_csv(data_path)

        if "age" not in df.columns:
            raise ValueError("В датасете нет колонки age")

        invalid_rows = df[df["years_experience"] > (df["age"] - 18).clip(lower=0)]

        if not invalid_rows.empty:
            raise ValueError(
                 f"В датасете есть нереалистичные записи: {len(invalid_rows)} строк. "
                 "Перегенерируй train_dataset.csv"
            )

        feature_cols = FEATURE_COLS

        missing_columns = set(feature_cols + ["retention"]) - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"В датасете нет колонок {sorted(missing_columns)}. "
                "Перегенерируй train_dataset.csv"
            )

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
            allow_writing_files=False,
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
        if self.model is None:
            raise ValueError("Model is not loaded or trained.")

        cache_key = self._feature_cache_key(features)

        if cache_key in self._prediction_cache:
            return dict(self._prediction_cache[cache_key])

        feature_df = self._prepare_feature_df(features)
        retention_prob = float(self.model.predict_proba(feature_df)[0, 1])
        requires_review = self._detect_requires_review(features)

        # Edge Case вшит в бизнес-логику:
        # если опыт отсутствует или нулевой, кейс уходит в Yellow Zone
        if requires_review:
            risk_level = "MEDIUM"
        else:
            risk_level = self._map_risk_level(retention_prob)

        uncertainty = self._estimate_uncertainty_band(
            retention_probability=retention_prob,
            features=features,
            requires_review=requires_review,
        )

        result = {
            "retention_probability": retention_prob,
            "will_stay": bool(retention_prob > 0.5),
            "risk_level": risk_level,
            "requires_review": requires_review,
            **uncertainty,
        }

        self._prediction_cache[cache_key] = dict(result)

        return result

    def explain_prediction(self, features: Dict) -> List[str]:
        features = {
            **FEATURE_DEFAULTS,
            **dict(features),
        }

        requires_review = self._detect_requires_review(features)
        cache_key = self._feature_cache_key(features)

        if cache_key in self._explain_cache:
            return list(self._explain_cache[cache_key])

        if self.model is None or not self.feature_names:
            result = self._rule_based_weighted_risks(features)

            if requires_review and "Требуется уточнение опыта" not in result:
                result.insert(0, "Требуется уточнение опыта")

            final_result = result[:3]
            self._explain_cache[cache_key] = list(final_result)
            return final_result

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

                if requires_review and "Требуется уточнение опыта" not in result:
                    result.insert(0, "Требуется уточнение опыта")

                if result:
                    final_result = result[:3]
                    self._explain_cache[cache_key] = list(final_result)
                    return final_result

        # SHAP-объяснение — best effort.
        # Если модельная интерпретация не сработала из-за проблем с признаками
        # или внутренней ошибкой CatBoost, возвращаем устойчивый rule-based fallback.
        except (
            ValueError,
            KeyError,
            TypeError,
            IndexError,
            AttributeError,
            cb.CatBoostError,
        ) as e:
            print(f"Explain fallback activated: {e}")

        result = self._rule_based_weighted_risks(features)

        if requires_review and "Требуется уточнение опыта" not in result:
            result.insert(0, "Требуется уточнение опыта")

        final_result = result[:3]
        self._explain_cache[cache_key] = list(final_result)

        return final_result

    def explain_positive_factors(self, features: Dict) -> List[str]:
        features = {
            **FEATURE_DEFAULTS,
            **dict(features),
        }

        cache_key = self._feature_cache_key(features)

        if cache_key in self._positive_cache:
            return list(self._positive_cache[cache_key])

        positives = []

        if features.get("skills_verified_count", 0) >= 7:
            positives.append("Много подтверждённых навыков")

        if features.get("years_experience", 0) >= 5:
            positives.append("Хороший релевантный опыт")

        if features.get("commute_time_minutes", 999) <= 40:
            positives.append("Короткая дорога до работы")

        if features.get("has_certifications", False):
            positives.append("Есть подтверждающие сертификаты")

        if features.get("has_transport", False):
            positives.append("Есть личный транспорт")

        if features.get("previous_turnovers", 99) <= 1:
            positives.append("Нет частой смены рабочих мест")

        if features.get("housing_type") == 0:
            positives.append("Стабильные жилищные условия")

        result = positives[:3]
        self._positive_cache[cache_key] = list(result)

        return result

    def save_model(self, path=DEFAULT_MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "feature_names": self.feature_names}, f)
        print(f"Model saved explicitly to: {path}")

    def load_model(self, path=DEFAULT_MODEL_PATH):
        """Загрузка обученной модели. Возвращает False, если файл отсутствует или поврежден."""
        if not os.path.exists(path):
            print(f"Model file not found at {path}")
            self.model = None
            self.feature_names = None
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            
            model = data["model"]
            feature_names = data["feature_names"]

            if model is None or not feature_names:
                print(f"Invalid model payload at {path}")
                self.model = None
                self.feature_names = None
                return False
            
            self.model = model
            self.feature_names = feature_names
            return True
        except (OSError, EOFError, pickle.UnpicklingError, KeyError, TypeError) as e:
            print(f"Failed to load model from {path}: {e}")
            self.model = None
            self.feature_names = None
            return False


def train_if_needed():
    model_path = DEFAULT_MODEL_PATH
    data_path = DEFAULT_DATA_PATH

    print(f"Checking model at: {model_path}")

    model = RetentionPredictor()
    model_loaded = model.load_model(model_path)

    needs_retrain = (
        not model_loaded
        or model.feature_names != FEATURE_COLS
    )

    if needs_retrain:
        print("Training model...")
        model.train_model(data_path)
        model.save_model(model_path)
        print("Model is trained and saved.")
    else:
        print("Model is already trained and feature-compatible.")