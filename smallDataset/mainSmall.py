import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import StandardScaler

# Загрузка данных
df = pd.read_csv("intern_task_small.csv")

# Предобработка данных
# Разделение данных на признаки (X) и целевую переменную (y)
X = df.drop(columns=['rank', 'query_id'])
y = df['rank']

# Масштабирование признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение отмасштабированных данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Создание и обучение модели RandomForestRegressor
rf_model = RandomForestRegressor(random_state=42)

rf_model.fit(X_train, y_train)

# Предсказание рангов на тестовом наборе
y_pred = rf_model.predict(X_test)

# Вычисление метрики ранжирования NDCG@5
ndcg_5 = ndcg_score(np.expand_dims(y_test, axis=0), np.expand_dims(y_pred, axis=0), k=5)
print("NDCG@5 на тестовом наборе данных:", ndcg_5)
