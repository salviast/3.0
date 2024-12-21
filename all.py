# cook your dish here
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler # Добавили масштабирование признаков

try:
    # 1. Загрузка данных
    wine = load_wine()
    X, y = wine.data, wine.target

    # 2. Создание выборки (с масштабированием)
    scaler = StandardScaler() # Масштабируем данные для лучшей работы MLPClassifier
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # 30% для теста

    # 3. Обучение модели
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, solver='adam', random_state=42) # Настройка параметров
    mlp.fit(X_train, y_train)

    # 4. Предсказание и оценка
    y_pred = mlp.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Точность модели: {accuracy:.4f}")

except Exception as e:
    print(f"Произошла ошибка: {e}")