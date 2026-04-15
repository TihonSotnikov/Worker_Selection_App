"""
Synthetic Data Generator (Rule-Based)
ТЗ: Скрипт генерации train.csv (1000 строк) с жесткими правилами
Пример правила: Если Commute > 90 мин, то Retention = 0
"""

import pandas as pd
import random
import os
from app.core.enums import ShiftPreference

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "train_dataset.csv")

class SyntheticDataGenerator:
    def __init__(self, n_samples=1000, seed=42):
        self.n_samples = n_samples
        self.rng = random.Random(seed)

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
            age = random.randint(20, 60)  # Для дополнительных правил

            # Бинарная целевая переменная: 1 - останется, 0 - уволится
            retention = 1  # По умолчанию кандидат останется

            # Правило 1: Если дорога > 90 минут → увольняется
            if commute_time_minutes > 90:
                retention = 0

            # Правило 2: Если мало навыков (<3) → увольняется
            elif skills_verified_count < 3:
                retention = 0

            # Правило 3: Ночная смена + возраст > 50 → увольняется
            elif shift_preference == ShiftPreference.NIGHT_ONLY and age > 50:
                retention = 0

            # Правило 4: Мало опыта + высокая зарплата → увольняется
            elif years_experience < 2 and salary_expectation > 100000:
                retention = 0

            # Правило 5: Много навыков без сертификатов → увольняется
            elif not has_certifications and skills_verified_count > 5:
                retention = 0

            # 5% шума для реалистичности данных
            if random.random() < 0.05:
                retention = 1 - retention

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
