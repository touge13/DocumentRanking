
## Условие задачи
У нас есть датасет для ранжирования `intern_task.csv`: 

https://drive.google.com/file/d/1viFKqtYTtTiP9_EdBXVpCmWbNmxDiXWG/view?usp=sharing

Скачайте csv файл и разместите в `../DocumentRanking/intern_task.csv`

В нем есть `query_id` - айдишник поисковой сессии, фичи
релевантности документа запросу, `rank` - оценка релевантности.
Наша задача:
- подготовить и проверить датасет.
- натренировать на любом удобном фреймворке модель, которая будет ранжировать документы по их фичам внутри одной сессии (query_id) (по вектору фичей предсказывать ранк документа).
- посчитать метрики ранжирования для своей модели (ndcg_5 как минимум).

---

# Модель ранжирования

Этот проект представляет собой пример работы с данными, подготовкой и обучением модели ранжирования на основе алгоритма `RandomForestRegressor`. В этом руководстве приведены шаги для подготовки данных, обучения модели и оценки ее производительности с использованием различных метрик.

# Что такое smallDataset?

Дело в том, что на данный момент мой компьютер не может позволить полностью обработать файл intern_task.csv, поэтому я был вынужден оставить от всего датасета только первые 1000 строк.

Я создал отдельную папку `smallDataset`, в котором находится:

- файл `solutionSmall.ipynb` абсолютно аналогичный файлу `solution.ipynb` (в ноутбуке solution.ipynb, в отличие от solutionSmall.ipynb, никаких выходных значений нет)
- файл `intern_task_small.csv`, в котором хранятся первые 1000 строк файла `intern_task.csv`
- файл `mainSmall.py`, в котором содержится такое же решение, как и в `solutionSmall.ipynb`, только в обычном формате .py

## Шаги проекта

### 1. Подготовка данных

- **Загрузка данных**: Исходные данные загружаются из файла `intern_task_small.csv`.
- **Изучение данных**: Анализируются основные статистики и распределение значений признаков.
- **Предобработка данных**: Числовые признаки масштабируются.

### 2. Обучение модели

- **Разделение данных**: Данные разделяются на обучающий и тестовый наборы.
- **Обучение модели**: Модель `RandomForestRegressor` обучается на обучающем наборе данных.

### 3. Оценка модели

- **Вычисление метрик**:
  - **NDCG@5 (Normalized Discounted Cumulative Gain)**: Оценка качества ранжирования модели.
  - **MAP (Mean Average Precision)**: Вычисление средней точности модели.
  - **MRR (Mean Reciprocal Rank)**: Расчет среднего взаимного ранга первого верного результата.

## Зависимости

Проект использует следующие библиотеки Python:

- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `scikit-learn`