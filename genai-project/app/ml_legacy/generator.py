"""
Synthetic Data Generator (Rule-Based)
ТЗ: Скрипт генерации train.csv (1000 строк) с жесткими правилами
Пример правила: Если Commute > 90 мин, то Retention = 0
"""

import pandas as pd
import random
import os
import math
from app.core.enums import ShiftPreference

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "train_dataset.csv")


class SyntheticDataGenerator:
    def __init__(self, n_samples=1000, seed=42):
        self.n_samples = n_samples
        self.rng = random.Random(seed)

    def _compute_risk_score(
        self,
        skills_verified_count: int,
        years_experience: float,
        age: int,
        commute_time_minutes: int,
        shift_preference: ShiftPreference,
        salary_expectation: int,
        has_certifications: bool,
    ) -> float:
        score = 0.0

        # Время в пути
        if commute_time_minutes > 120:
            score += 2.6
        elif commute_time_minutes > 90:
            score += 1.8
        elif commute_time_minutes > 60:
            score += 0.8
        else:
            score -= 0.2

        # Навыки
        if skills_verified_count < 3:
            score += 2.2
        elif skills_verified_count < 5:
            score += 0.9
        elif skills_verified_count >= 8:
            score -= 0.5

        # Опыт
        if years_experience < 1:
            score += 1.8
        elif years_experience < 3:
            score += 0.9
        elif years_experience >= 8:
            score -= 0.5

        # Сменность и возраст
        if shift_preference == ShiftPreference.NIGHT_ONLY:
            score += 0.5
            if age > 50:
                score += 1.0
        elif shift_preference == ShiftPreference.ANY:
            score += 0.1

        # Зарплатные ожидания относительно опыта
        if years_experience < 2 and salary_expectation > 100000:
            score += 1.7
        elif years_experience < 4 and salary_expectation > 120000:
            score += 0.9

        # Сертификаты
        if not has_certifications:
            score += 0.4
            if skills_verified_count > 5:
                score += 0.8
        else:
            score -= 0.2

        # Взаимодействия признаков
        if commute_time_minutes > 90 and shift_preference == ShiftPreference.NIGHT_ONLY:
            score += 0.4

        if skills_verified_count < 3 and years_experience < 2:
            score += 0.6

        # Небольшой джиттер вместо грубого flip 5%
        score += self.rng.uniform(-0.25, 0.25)

        return score

    def generate_dataset(self):
        """Генерация датасета с жесткими правилами для удержания"""
        data = []

        for i in range(self.n_samples):
            # Генерация признаков для ML модели
            skills_verified_count = self.rng.randint(0, 10)
            years_experience = self.rng.uniform(0, 30)
            commute_time_minutes = self.rng.randint(10, 180)
            shift_preference = self.rng.choice(list(ShiftPreference))
            salary_expectation = self.rng.randint(30000, 150000)
            has_certifications = self.rng.random() > 0.7
            age = self.rng.randint(20, 60)  # Для дополнительных правил

            risk_score = self._compute_risk_score(
                skills_verified_count=skills_verified_count,
                years_experience=years_experience,
                age=age,
                commute_time_minutes=commute_time_minutes,
                shift_preference=shift_preference,
                salary_expectation=salary_expectation,
                has_certifications=has_certifications,
            )

            # Чем выше risk_score, тем ниже вероятность удержания
            retention_probability = 1.0 / (1.0 + math.exp(risk_score - 2.0))
            retention = 1 if self.rng.random() < retention_probability else 0

            record = {
                "skills_verified_count": skills_verified_count,
                "years_experience": round(years_experience, 1),
                "age": age,
                "commute_time_minutes": commute_time_minutes,
                "shift_preference": shift_preference.value,
                "salary_expectation": salary_expectation,
                "has_certifications": int(has_certifications),
                "retention": retention,
            }
            data.append(record)

        return pd.DataFrame(data)

    def save_to_csv(self, path=DEFAULT_DATA_PATH):
        """Сохранение в CSV файл"""
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)

        df = self.generate_dataset()
        df.to_csv(path, index=False)
        print(f"Данные сохранены в {path}")
        return df


def generate_if_needed():
    """Проверяет наличие датасета и генерирует если нужно"""
    data_path = DEFAULT_DATA_PATH
    if not os.path.exists(data_path) or os.path.getsize(data_path) == 0:
        print("Генерация тренировочных данных...")
        generator = SyntheticDataGenerator(n_samples=1000)
        generator.save_to_csv(data_path)
        print(f"Сгенерировано 1000 записей в {data_path}")
    else:
        print(f"Датасет уже существует: {data_path}")
    return data_path
