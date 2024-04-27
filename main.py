# Импорт необходимых библиотек
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Загрузка данных
df = pd.read_csv("1.csv")

# Предобработка данных
# Разделение данных на признаки (X) и целевую переменную (y)
X = df.drop(columns=['rank', 'query_id'])
y = df['rank']

# Заполнение пропущенных значений средними
X.fillna(X.mean(), inplace=True)

# Масштабирование признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение отмасштабированных данных на обучающий и тестовый наборы
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Тюнинг модели
# Определение сетки параметров для подбора
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Создание модели RandomForestRegressor
rf_model = RandomForestRegressor(random_state=42)

# Подбор оптимальных гиперпараметров с помощью GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
print(1)
grid_search.fit(X_train_scaled, y_train)
print(2)
# Вывод наилучших гиперпараметров
print("Наилучшие гиперпараметры:", grid_search.best_params_)

# Получение наилучшей модели
best_model = grid_search.best_estimator_

# Обучение наилучшей модели на всем обучающем наборе
best_model.fit(X_train_scaled, y_train)

# Предсказание рангов на тестовом наборе
y_pred = best_model.predict(X_test_scaled)

# Вычисление метрики ранжирования NDCG@5
ndcg_5 = ndcg_score(np.expand_dims(y_test, axis=0), np.expand_dims(y_pred, axis=0), k=5)
print("NDCG@5 на тестовом наборе данных с улучшенной моделью:", ndcg_5)