import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import ndcg_score, average_precision_score

# Загрузка данных из CSV-файла
df = pd.read_csv('smallDataset/intern_task_small.csv')

# Предобработка данных
# Разделение данных на признаки (X) и целевую переменную (y)
X = df.drop(columns=['rank', 'query_id'])
y = df['rank']

# Масштабирование признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Обработка пропущенных значений, если такие есть
df.fillna(df.mean(), inplace=True)

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

y_test_reshape = np.array(y_test).reshape(-1, 1)
y_pred_reshape = np.array(y_pred).reshape(-1, 1)

# Вычисление средней точности (MAP)
map_score = average_precision_score(y_test_reshape, y_pred_reshape)
print(f"Mean Average Precision (MAP): {map_score}")

def mean_reciprocal_rank(y_true, y_pred):
    # Сортируем индексы предсказанных значений по убыванию вероятности
    sorted_indexes = np.argsort(y_pred)[::-1]
    
    for idx in sorted_indexes:
        # Находим первый индекс, для которого y_true равно 1 (соответствует релевантному результату)
        if y_true[idx] == 1:
            # Возвращаем обратную величину ранга
            return 1 / (idx + 1)
    
    # Если ни один релевантный результат не найден
    return 0

# Вычисляем взаимные ранги
reciprocal_ranks = [mean_reciprocal_rank(y_test, y_pred) for y_test, y_pred in zip(y_test_reshape, y_pred_reshape)]

# Фильтруем значения None (если они есть)
valid_reciprocal_ranks = [rr for rr in reciprocal_ranks if rr is not None]

# Вычисляем среднее значение действительных взаимных рангов
mrr_score = np.mean(valid_reciprocal_ranks)
print(f"Mean Reciprocal Rank (MRR): {mrr_score}")
