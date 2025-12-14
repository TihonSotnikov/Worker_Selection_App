"""
Synthetic Data Generator (Rule-Based)
ТЗ: Скрипт генерации train.csv (1000 строк) с жесткими правилами
Пример правила: Если Commute > 90 мин, то Retention = 0
"""
import pandas as pd
import numpy as np
import random
import os
from app.core.enums import ShiftPreference

class SyntheticDataGenerator:
    def __init__(self, n_samples=1000):
        self.n_samples = n_samples

    def generate_dataset(self):
        """Генерация датасета с жесткими правилами для удержания"""
        data = []

        for i in range(self.n_samples):
            # Генерация признаков для ML модели
            skills_verified_count = random.randint(0, 10)
            years_experience = random.uniform(0, 30)
            commute_time_minutes = random.randint(10, 180)
            shift_preference = random.choice(list(ShiftPreference))
            salary_expectation = random.randint(30000, 150000)
            has_certifications = random.random() > 0.7
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
                "commute_time_minutes": commute_time_minutes,
                "shift_preference": shift_preference.value,
                "salary_expectation": salary_expectation,
                "has_certifications": int(has_certifications),
                "retention": retention,
            }
            data.append(record)

        return pd.DataFrame(data)

    def save_to_csv(self, path="data/train_dataset.csv"):
        """Сохранение в CSV файл"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)

        df = self.generate_dataset()
        df.to_csv(path, index=False)
        print(f"Данные сохранены в {path}")
        return df

def generate_if_needed():
    """Проверяет наличие датасета и генерирует если нужно"""
    data_path = "data/train_dataset.csv"
    if not os.path.exists(data_path) or os.path.getsize(data_path) == 0:
        print("Генерация тренировочных данных...")
        generator = SyntheticDataGenerator(n_samples=1000)
        generator.save_to_csv(data_path)
        print(f"Сгенерировано 1000 записей в {data_path}")
    else:
        print(f"Датасет уже существует: {data_path}")
    return data_path
