### Set-up:

* Для выполнения данного задания я использовал **Docker**. Для сборки образа можно воспользоваться следующей командой: `docker build -t core-ml .`
* Также, я использую GPU в контейнере. Для этого необходимо установить дополнительное расширения для Docker. 
* Для запуска контейнера я воспользовался следующей командой: `docker run -it -p 8888:8888 -v $(pwd):/app --gpus all core-ml`.
* Для воспроизведения кода нужно: склонировать репозиторий, создать папку `data`, скачать и разархивировать данные в этой папке.

Выполнил [Елизаров Павел](https://vk.com/epepepepepepepepepepepepepepepep).

---

### Импорт необходимых библиотек


```python
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.sparse as sp
import torch
import scipy
import optuna
from tqdm import tqdm
```

### Предобработка для работы над моделями коллаборативной фильрации:
Здесь сначала преобразуется часть исключительно для коллаборативной части


```python
# Начнем с самого вкусного - с рейтингов
ratings = pd.read_csv('./data/rating.csv')
# Трансформируем в timestamp для более удобной работы
ratings['timestamp'] = pd.to_datetime(ratings['timestamp']).astype('int64') // 10**9
```

Следует проверить, какие фильмы просмотрели редко, а какие часто. Это позволит оставить только те фильмы, которые не будут зашумлять работу коллаборативной фильтрации. 
Для того, чтобы убрать редко-просматриваемые фильмы, воспользуемся правилом "логтя".


```python
counts = ratings.groupby('movieId')['userId'].count()

plt.figure(figsize=(9, 7))
plt.hist(counts[counts < 100], bins=50)
# Число 11 - ложится на логоть
plt.axvline(11, color='red', linestyle='--')
plt.show()
```


    
![png](output_7_0.png)
    


Уберем те фильмы, которые встречаются очень редко, а именно, которые встречаются меньше 11 раз


```python
ratings = ratings[ratings['movieId'].isin(counts[counts >= 11].index)]
```

### Разделение на тренировочную, валидационную и тестовую выборки:

* Для разделения на выборки стоит использовать критерий по времени. То есть, в тестовой и валидационной выборках должны присутствовать взаимодействия, которые произошли после взаимодействий из тренировочной выборки. Выберем 70% данных на тренировочную выборку, 10% на валидационную выборку и 20% на тест. 
* Тренировочная выбора создается для обучения моделей, валидационная - для настройки гиперпараметров, тестовая - для анализа качества модели.
* Все выборки преобразуем к sparse csr матрицам с фиксированным размером всего датасета.


```python
# Переделаем индексы, чтобы было проще работать со sparse-матрицами: создадим энкодеры и декодеры в последовательные id
movie_encoder = {k: v for v, k in enumerate(np.unique(ratings['movieId']))}
user_encoder = {k: v for v, k in enumerate(np.unique(ratings['userId']))}
movie_decoder = {k: v for k, v in enumerate(np.unique(ratings['movieId']))}
user_decoder = {k: v for k, v in enumerate(np.unique(ratings['userId']))}

# Переделаем Id
ratings['movieId'] = ratings['movieId'].apply(lambda x: movie_encoder[x])
ratings['userId'] = ratings['userId'].apply(lambda x: user_encoder[x])



# Выбор трешхолдов для разделения на тренировочную, валидационную и тесовую выборки:
val_threshold, test_threshold = np.percentile(ratings['timestamp'], 70), np.percentile(ratings['timestamp'], 80)

# Разделяем датасеты:
train, train_val, validation, test = ratings[ratings['timestamp'] < val_threshold],\
                                                 ratings[ratings['timestamp'] < test_threshold],\
                                                 ratings[(ratings['timestamp'] >= val_threshold) & (ratings['timestamp'] < test_threshold)],\
                                                 ratings[ratings['timestamp'] >= test_threshold]

# Также сохраним количество уникальных пользователей и количество уникальных фильмов для созранения sparse матриц.
matrix_shape = (len(np.unique(ratings['userId'])), len(np.unique(ratings['movieId'])))

# Преобразование в sparse-матрицы
train, train_val, validation, test = sp.csr_matrix((train['rating'], (train['userId'], train['movieId'])), shape=matrix_shape),\
                                     sp.csr_matrix((train_val['rating'], (train_val['userId'], train_val['movieId'])), shape=matrix_shape),\
                                     sp.csr_matrix((validation['rating'], (validation['userId'], validation['movieId'])), shape=matrix_shape),\
                                     sp.csr_matrix((test['rating'], (test['userId'], test['movieId'])), shape=matrix_shape)
```

---

### Пробуем применить только коллаборативный подход: *MostPopular-модель*, *SVD-like*, *ALS*, и *AutoEncoder*.



---

#### _Немного о метрике качества:_

Для измерения качества, я выбрал метрику ранжирования **nDCG@k**. Это метрика ранжирования, где степень релевантности задается не бинарной величиной. Так как в нашем случае релевантность задается оценкой, степень релевантности - не бинарная. 

---

Также можно использовать **MAP@k**, однако, тогда оценки придется переводить в бинарные. (Например, 1 - если оценка выше или рана 4, 0 - иначе).

#### _Немного о статистической проверке:_
После предсказаний для каждого пользователя, мы можем расчитать для них метрику nDCG@k. Для каждой модели будем проверять, различаются ли метрики, составленные из рекомендаций моделей, от метрик, составленных из рандомных рекомендаций с помошью t-теста. k выберем равным 20.

---

#### _Немного о настройке гиперпараметров:_
Для некоторых моделей я буду настраивать гиперпараметры с помощью Байесовского алгоритма, реализованного в библиотеке *optuna*. Также для настройки гиперпарамтров можно использовать поиск по сетке или случайный поиск. Во время настройки гиперпараметров будем ориентироваться на качество валидационной выборки.

---


```python
# Импортируем модели
from src.models import MostPopular, SVDRecommender, RandomRecommender
from src.torch_models import TorchALS, AutoEncoderRecommender

# Импортируем ALS из библиотеки implicit 
from implicit.als import AlternatingLeastSquares

# Импортируем метрику
from src.utils import ndcg_at_k

device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

Для того, чтобы сэкономить память, проверять качество модели будем для самых активных пользователей, а именно для тех, кто всего посмотрел больше 20 фильмов, а за время тестового периода - больше 5.
То же самое сделаем для валидации.


```python
test_active, train_for_test, train_val_for_test= test[(train.getnnz(1) > 10) & (test.getnnz(1) > 5)], train[(train.getnnz(1) > 10) & (test.getnnz(1) > 5)],\
                                                 train_val[(train.getnnz(1) > 10) & (test.getnnz(1) > 5)]
val_active, train_for_val = validation[(train.getnnz(1) > 10) & (validation.getnnz(1) > 5)], train[(train.getnnz(1) > 10) & (validation.getnnz(1) > 5)]
```

### Most Popular: 
Основана на том, что берет либо самые популярные фильмы, либо лучшие по рейтингу


```python
# Тренируем и смотрим на перформанс MostPopular
most_pop = MostPopular(popularity_type='by_rating')
most_pop.fit(train_val)
print("Predicting stage")
predictions = most_pop.predict(train_for_test, test_active, excluding_predictions=train_val_for_test, batch_size=100, drop_cold_users=True)

print("NDCG calculating stage")
# Считаем метрику
most_pop_ndcg = [ndcg_at_k(predictions, test_active, k=i).mean() for i in (10, 20, 50, 100)]
```

      3%|▎         | 1/30 [00:00<00:03,  8.79it/s]

    Predicting stage


    100%|██████████| 30/30 [00:03<00:00,  8.98it/s]
     20%|██        | 6/30 [00:00<00:00, 51.57it/s]

    NDCG calculating stage


    100%|██████████| 30/30 [00:00<00:00, 52.85it/s]
    100%|██████████| 30/30 [00:00<00:00, 52.81it/s]
    100%|██████████| 30/30 [00:00<00:00, 52.25it/s]
    100%|██████████| 30/30 [00:00<00:00, 51.97it/s]



```python
# тестируем
random_predictor = RandomRecommender()
random_predictor.predict(test_active, excluding_predictions=train_val_for_test, random_state=123)
# random_predictions = np.random.randint(0, train.shape[1], (test_active.shape[0], 20))
random_predictions = random_predictor.predict(test_active, excluding_predictions=train_val_for_test, random_state=123, number_of_predictions=20)
model_prediction = most_pop.predict(train_for_test, test_active, excluding_predictions=train_val_for_test, batch_size=100, drop_cold_users=True, number_of_predictions=20)

random_ndcg = ndcg_at_k(random_predictions, test_active, k=20)
model_ndcg = ndcg_at_k(model_prediction, test_active, k=20)

print(
f"""
P-value: {scipy.stats.ttest_ind(random_ndcg, model_ndcg)[1]}
"""
)

```

    100%|██████████| 30/30 [00:03<00:00,  9.17it/s]
    100%|██████████| 30/30 [00:00<00:00, 54.36it/s]
    100%|██████████| 30/30 [00:00<00:00, 54.36it/s]

    
    P-value: 6.18506653378542e-303
    


    


Средние значения метрик, полученные с помощью рандомных предсказаний и с помощью предсказаний модели, статистически значимо различаются.

### SVD:
Основана на SVD-разложении


```python
# Применим оптимизацию гиперпараметров
val_user_ids = np.arange(validation.shape[0])[(train.getnnz(1) > 10) & (validation.getnnz(1) > 5)]
def objective(trial):
    n_components = trial.suggest_int("n_components", 60, 100, 20)
    n_iter = trial.suggest_int("n_iter", 10, 20, 5)
    svd_rec = SVDRecommender(n_components=n_components, n_iter=n_iter)
    svd_rec.fit(train)
    predictions = svd_rec.predict(train, val_active, excluding_predictions=train, batch_size=100, drop_cold_users=True,
                             user_ids=val_user_ids)
    
    return ndcg_at_k(predictions, val_active, k=20).mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=3)

best_params = study.best_params

svd_rec = SVDRecommender(**best_params)
svd_rec.fit(train_val)
user_ids = np.arange(test.shape[0])[(train.getnnz(1) > 10) & (test.getnnz(1) > 5)]

print("Predicting stage")
predictions = svd_rec.predict(train_val_for_test, test_active, excluding_predictions=train_val, batch_size=100, drop_cold_users=True,
                             user_ids=user_ids)

print("NDCG calculating stage")
# Считаем метрику
svd_ndcg = [ndcg_at_k(predictions, test_active, k=i).mean() for i in (10, 20, 50, 100)]
```

    [32m[I 2021-06-26 20:14:38,390][0m A new study created in memory with name: no-name-08ef43f9-a4f3-4f27-a47f-74c094984c7a[0m
    100%|██████████| 40/40 [00:00<00:00, 54.30it/s]
    [32m[I 2021-06-26 20:15:08,741][0m Trial 0 finished with value: 0.1437793147346196 and parameters: {'n_components': 60, 'n_iter': 20}. Best is trial 0 with value: 0.1437793147346196.[0m
    100%|██████████| 40/40 [00:00<00:00, 53.99it/s]
    [32m[I 2021-06-26 20:15:38,592][0m Trial 1 finished with value: 0.14375745266130424 and parameters: {'n_components': 60, 'n_iter': 20}. Best is trial 0 with value: 0.1437793147346196.[0m
    100%|██████████| 40/40 [00:00<00:00, 54.48it/s]
    [32m[I 2021-06-26 20:16:14,067][0m Trial 2 finished with value: 0.13687632240507933 and parameters: {'n_components': 100, 'n_iter': 15}. Best is trial 0 with value: 0.1437793147346196.[0m


    Predicting stage


     20%|██        | 6/30 [00:00<00:00, 53.54it/s]

    NDCG calculating stage


    100%|██████████| 30/30 [00:00<00:00, 54.15it/s]
    100%|██████████| 30/30 [00:00<00:00, 53.33it/s]
    100%|██████████| 30/30 [00:00<00:00, 53.58it/s]
    100%|██████████| 30/30 [00:00<00:00, 52.69it/s]



```python
# тестируем
random_predictor = RandomRecommender()
random_predictor.predict(test_active, excluding_predictions=train_val_for_test, random_state=123)
model_prediction = svd_rec.predict(train_for_test, test_active, excluding_predictions=train_val, batch_size=100, drop_cold_users=True, number_of_predictions=20)

random_ndcg = ndcg_at_k(random_predictions, test_active, k=20)
model_ndcg = ndcg_at_k(model_prediction, test_active, k=20)

print(
f"""
P-value: {scipy.stats.ttest_ind(random_ndcg, model_ndcg)[1]}
"""
)
```

    100%|██████████| 30/30 [00:00<00:00, 54.15it/s]
    100%|██████████| 30/30 [00:00<00:00, 54.04it/s]

    
    P-value: 2.1889036417709382e-63
    


    


Средние значения метрик, полученные с помощью рандомных предсказаний и с помощью предсказаний модели, статистически значимо различаются.

### ALS from Implicit
Готовая реализация ALS модели из библиотеки `implicit`.


```python
# Применим оптимизацию гиперпараметров
val_user_ids = np.arange(validation.shape[0])[(train.getnnz(1) > 10) & (validation.getnnz(1) > 5)]
def objective(trial):
    factors = trial.suggest_int("factors", 64, 128, 32)
    regularization = trial.suggest_uniform("regularization", 0.01, 0.1)
    implicit_als = AlternatingLeastSquares(factors=factors, iterations=15, regularization=regularization)
    implicit_als.fit(train.transpose(), show_progress=False)
    predictions = [np.array(list(map(lambda x: x[0], implicit_als.recommend(i, train, N=20)))) for i in val_user_ids]
    predictions = np.vstack(predictions)
    
    return ndcg_at_k(predictions, val_active, k=20).mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=3)

best_params = study.best_params

implicit_als = AlternatingLeastSquares(**best_params)

user_ids = np.arange(test.shape[0])[(train.getnnz(1) > 10) & (test.getnnz(1) > 5)]

# Требует матрицу Items X Users, поэтому транспонируем
implicit_als.fit(train_val.transpose(), show_progress=False)

print("Predicting stage")
predictions = [np.array(list(map(lambda x: x[0], implicit_als.recommend(i, train_val, N=100)))) for i in user_ids]
predictions = np.vstack(predictions)

print("NDCG calculating stage")
als_ndcg = [ndcg_at_k(predictions, test_active, k=i).mean() for i in (10, 20, 50, 100)]
```

    [32m[I 2021-06-26 20:16:53,913][0m A new study created in memory with name: no-name-1b562ffa-0d5c-48c7-b186-c8743c24aa04[0m
    WARNING:root:OpenBLAS detected. Its highly recommend to set the environment variable 'export OPENBLAS_NUM_THREADS=1' to disable its internal multithreading
    100%|██████████| 40/40 [00:00<00:00, 52.72it/s]
    [32m[I 2021-06-26 20:21:13,908][0m Trial 0 finished with value: 0.12679326594850088 and parameters: {'factors': 96, 'regularization': 0.0157811260905565}. Best is trial 0 with value: 0.12679326594850088.[0m
    100%|██████████| 40/40 [00:00<00:00, 53.86it/s]
    [32m[I 2021-06-26 20:25:36,893][0m Trial 1 finished with value: 0.1274288682905004 and parameters: {'factors': 96, 'regularization': 0.07161748063785628}. Best is trial 1 with value: 0.1274288682905004.[0m
    100%|██████████| 40/40 [00:00<00:00, 52.81it/s]
    [32m[I 2021-06-26 20:29:58,635][0m Trial 2 finished with value: 0.12740563329061524 and parameters: {'factors': 96, 'regularization': 0.05860324147756538}. Best is trial 1 with value: 0.1274288682905004.[0m


    Predicting stage


     17%|█▋        | 5/30 [00:00<00:00, 45.38it/s]

    NDCG calculating stage


    100%|██████████| 30/30 [00:00<00:00, 52.39it/s]
    100%|██████████| 30/30 [00:00<00:00, 53.81it/s]
    100%|██████████| 30/30 [00:00<00:00, 53.35it/s]
    100%|██████████| 30/30 [00:00<00:00, 52.76it/s]



```python
# тестируем
random_predictor = RandomRecommender()
random_predictor.predict(test_active, excluding_predictions=train_val_for_test, random_state=123)
model_prediction = [np.array(list(map(lambda x: x[0], implicit_als.recommend(i, train_val, N=100)))) for i in user_ids]
model_prediction = np.vstack(model_prediction)

random_ndcg = ndcg_at_k(random_predictions, test_active, k=20)
model_ndcg = ndcg_at_k(model_prediction, test_active, k=20)

print(
f"""
P-value: {scipy.stats.ttest_ind(random_ndcg, model_ndcg)[1]}
"""
)
```

    100%|██████████| 30/30 [00:00<00:00, 52.14it/s]
    100%|██████████| 30/30 [00:00<00:00, 54.10it/s]

    
    P-value: 0.0
    


    


Средние значения метрик, полученные с помощью рандомных предсказаний и с помощью предсказаний модели, статистически значимо различаются.

### Torch ALS
Попробовал реализовать собственную версию ALS на PyTorch. Гиперпараметры оставлю с прошлой модели


```python
torch_als = TorchALS(n_factors=best_params["factors"], regularization=best_params["regularization"], 
                    n_iterations=15, device=device)
user_ids = np.arange(test.shape[0])[(train.getnnz(1) > 10) & (test.getnnz(1) > 5)]
torch_als.fit(train_val)


print("Predicting stage")
predictions = torch_als.predict(train_val_for_test, test_active, excluding_predictions=train_val, batch_size=100, drop_cold_users=True,
                                 user_ids=user_ids)

print("NDCG calculating stage")
torch_als_ndcg = [ndcg_at_k(predictions, test_active, k=i).mean() for i in (10, 20, 50, 100)]
```

    100%|██████████| 15/15 [03:44<00:00, 14.95s/it]


    Predicting stage


     20%|██        | 6/30 [00:00<00:00, 53.93it/s]

    NDCG calculating stage


    100%|██████████| 30/30 [00:00<00:00, 54.58it/s]
    100%|██████████| 30/30 [00:00<00:00, 54.34it/s]
    100%|██████████| 30/30 [00:00<00:00, 53.33it/s]
    100%|██████████| 30/30 [00:00<00:00, 52.86it/s]



```python
# тестируем
random_predictor = RandomRecommender()
random_predictor.predict(test_active, excluding_predictions=train_val_for_test, random_state=123)
model_prediction = torch_als.predict(train_for_test, test_active, excluding_predictions=train_val, batch_size=100, drop_cold_users=True, number_of_predictions=20)

random_ndcg = ndcg_at_k(random_predictions, test_active, k=20)
model_ndcg = ndcg_at_k(model_prediction, test_active, k=20)

print(
f"""
P-value: {scipy.stats.ttest_ind(random_ndcg, model_ndcg)[1]}
"""
)
```

    100%|██████████| 30/30 [00:00<00:00, 54.45it/s]
    100%|██████████| 30/30 [00:00<00:00, 54.52it/s]

    
    P-value: 5.278349129332578e-82
    


    


Средние значения метрик, полученные с помощью рандомных предсказаний и с помощью предсказаний модели, статистически значимо различаются.

### AutoEncoder
Это один из видов нейросетевых рекомендаций. Также постарался реализовать эту модель на pytorch. Здесь также можно затюнить гиперпараметры, однако, это куда затруднительнее с точки зрения вычислений. 


```python
from src.torch_models import AutoEncoderRecommender

val_user_ids = np.arange(validation.shape[0])[(train.getnnz(1) > 10) & (validation.getnnz(1) > 5)]

autoencoder = AutoEncoderRecommender(n_items=train.shape[1],
                                    layers_dims=[1024, 512, 256],
                                    activation='swish',
                                    n_epoch=10,
                                    device=device,
                                    lr=0.003,
                                    optimizer='adam',
                                    batch_size=512,
                                    dropout=0.8,
                                    augmentation_step=1)

autoencoder.to(device)
autoencoder.train()
autoencoder.fit(train, val_active, val_user_ids=val_user_ids)
```

    Epoch 1:   0%|          | 0/197 [00:00<?, ?it/s]/usr/local/lib/python3.9/site-packages/torch/nn/functional.py:1805: UserWarning: nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.
      warnings.warn("nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.")
    Epoch 1: 100%|██████████| 197/197 [00:14<00:00, 13.48it/s, Train Loss: 5135.83301]  
    100%|██████████| 40/40 [00:00<00:00, 53.48it/s]
    Epoch 2:   1%|          | 2/197 [00:00<00:15, 12.75it/s, Train Loss: 6947.78271]

    Validation NDCG: 0.09874638615209687


    Epoch 2: 100%|██████████| 197/197 [00:14<00:00, 13.60it/s, Train Loss: 624184.875]  
    100%|██████████| 40/40 [00:00<00:00, 53.46it/s]
    Epoch 3:   1%|          | 2/197 [00:00<00:15, 12.31it/s, Train Loss: 565326.6875]

    Validation NDCG: 0.07078591786859549


    Epoch 3: 100%|██████████| 197/197 [00:14<00:00, 13.67it/s, Train Loss: 140511.96875]
    100%|██████████| 40/40 [00:00<00:00, 53.28it/s]
    Epoch 4:   1%|          | 2/197 [00:00<00:16, 11.77it/s, Train Loss: 133047.875]

    Validation NDCG: 0.10819827833440061


    Epoch 4: 100%|██████████| 197/197 [00:14<00:00, 13.60it/s, Train Loss: 113157.35938]
    100%|██████████| 40/40 [00:00<00:00, 53.39it/s]
    Epoch 5:   1%|          | 2/197 [00:00<00:16, 11.97it/s, Train Loss: 101389.25781]

    Validation NDCG: 0.1105789439340459


    Epoch 5: 100%|██████████| 197/197 [00:14<00:00, 13.51it/s, Train Loss: 80971.8125]  
    100%|██████████| 40/40 [00:00<00:00, 53.48it/s]
    Epoch 6:   1%|          | 2/197 [00:00<00:16, 11.61it/s, Train Loss: 75954.54688]

    Validation NDCG: 0.11225503075307268


    Epoch 6: 100%|██████████| 197/197 [00:14<00:00, 13.54it/s, Train Loss: 57004.11719]
    100%|██████████| 40/40 [00:00<00:00, 53.37it/s]
    Epoch 7:   1%|          | 2/197 [00:00<00:15, 12.53it/s, Train Loss: 51860.73438]

    Validation NDCG: 0.11383947241372286


    Epoch 7: 100%|██████████| 197/197 [00:14<00:00, 13.58it/s, Train Loss: 43741.11719]
    100%|██████████| 40/40 [00:00<00:00, 53.49it/s]
    Epoch 8:   1%|          | 2/197 [00:00<00:16, 11.95it/s, Train Loss: 44401.25]

    Validation NDCG: 0.11588393716898428


    Epoch 8: 100%|██████████| 197/197 [00:14<00:00, 13.56it/s, Train Loss: 25509.60547]
    100%|██████████| 40/40 [00:00<00:00, 52.90it/s]
    Epoch 9:   1%|          | 2/197 [00:00<00:16, 11.64it/s, Train Loss: 26413.36719]

    Validation NDCG: 0.11790562135312518


    Epoch 9: 100%|██████████| 197/197 [00:14<00:00, 13.49it/s, Train Loss: 16970.66016]
    100%|██████████| 40/40 [00:00<00:00, 53.46it/s]
    Epoch 10:   1%|          | 2/197 [00:00<00:16, 12.08it/s, Train Loss: 16380.78125]

    Validation NDCG: 0.11888835219603894


    Epoch 10: 100%|██████████| 197/197 [00:14<00:00, 13.58it/s, Train Loss: 8715.34766] 
    100%|██████████| 40/40 [00:00<00:00, 53.15it/s]

    Validation NDCG: 0.1186450848815234


    



```python
torch.cuda.empty_cache()
```


```python
autoencoder.eval()
# autoencoder.load_state_dict(autoencoder.best_params)
user_ids = np.arange(test.shape[0])[(train.getnnz(1) > 10) & (test.getnnz(1) > 5)]
predictions = autoencoder.predict(train_val, test_active, excluding_predictions=train_val, batch_size=100, drop_cold_users=True,
                             user_ids=user_ids)

autoencoder_ndcg = [ndcg_at_k(predictions, test_active, k=i).mean() for i in (10, 20, 50, 100)]
```

    100%|██████████| 30/30 [00:00<00:00, 54.76it/s]
    100%|██████████| 30/30 [00:00<00:00, 54.27it/s]
    100%|██████████| 30/30 [00:00<00:00, 53.46it/s]
    100%|██████████| 30/30 [00:00<00:00, 53.09it/s]



```python
# тестируем
random_predictor = RandomRecommender()
random_predictor.predict(test_active, excluding_predictions=train_val_for_test, random_state=123)
model_prediction = autoencoder.predict(train_val, test_active, excluding_predictions=train_val, batch_size=100, drop_cold_users=True,
                             user_ids=user_ids, number_of_predictions=20)

random_ndcg = ndcg_at_k(random_predictions, test_active, k=20)
model_ndcg = ndcg_at_k(model_prediction, test_active, k=20)

print(
f"""
P-value: {scipy.stats.ttest_ind(random_ndcg, model_ndcg)[1]}
"""
)
```

    100%|██████████| 30/30 [00:00<00:00, 54.49it/s]
    100%|██████████| 30/30 [00:00<00:00, 54.08it/s]

    
    P-value: 0.0
    


    


Средние значения метрик, полученные с помощью рандомных предсказаний и с помощью предсказаний модели, статистически значимо различаются.

### Сравнение моделей


```python
fig, ax = plt.subplots(figsize=(9,7))
x = np.arange(0, 4)
ax.plot(x, most_pop_ndcg, marker='^', markersize=20, label='Most Popular')
ax.plot(x, svd_ndcg, marker='^', markersize=20, label='SVD')
ax.plot(x, als_ndcg, marker='^', markersize=20, label='ALS')
ax.plot(x, torch_als_ndcg, marker='^', markersize=20, label='Torch_ALS')#
ax.plot(x, autoencoder_ndcg, marker='^', markersize=20, label='AutoEncoder')
ax.legend()
ax.set_xticks(x)
labels = [f"k={i}" for i in (10, 20, 50, 100)]
ax.set_xticklabels(labels)
ax.set_xlabel("K")
ax.set_ylabel("nDCG@k value")
ax.grid()
plt.show()
```


    
![png](output_41_0.png)
    


Лучше всего себя показали SVD и ALS модели. AutoEncoder не обошел даже MostPopular модель. Однако, так как AutoEncoder имеет в себе много гиперпараметров, которые стоит настроить, а также, требует больше эпох для лучшего схождения, данный алгоритм может быть улучшен.

### Предобработка данных признаков фильмов
Преобразуем данные в матрицу признаков для фильмов.


```python
# Подгрущим данные с признаками о фильмах
items_features = pd.read_csv('./data/genome_scores.csv')

# оставим только фильмы, которые присутсвуют у нас в датасете
items_features = items_features[items_features['movieId'].isin(movie_decoder.values())]

# Сделаем новые MovieID для создания Sparse Матрицы
items_features['movieId'] = items_features['movieId'].apply(lambda x: movie_encoder[x])

# id тегов тоже переделаем в нужный формат
items_features['tagId'] -= 1

# Преобразуем в csr матрицу
items_features = sp.csr_matrix((items_features['relevance'], (items_features['movieId'], items_features['tagId'])))
```

### LightFM


```python
from lightfm import LightFM

lf = LightFM(no_components=30)
lf.fit(train_val, item_features=items_features, epochs=10, num_threads=6)
```




    <lightfm.lightfm.LightFM at 0x7fcb2a4f8a30>




```python
import pickle

with open('lf.pkl', 'wb') as write_file:
    pickle.dump(lf, write_file, pickle.HIGHEST_PROTOCOL)
```


```python
# Set the ids of users to predict
user_ids = np.arange(test.shape[0])[(train.getnnz(1) > 10) & (test.getnnz(1) > 5)]
items_ids = np.arange(0, test.shape[1])

# set empty prediction list
predictions_all = []

# Iterating over batches to make predictions
for start_ind in tqdm(range(0, len(user_ids), 10)):
    batch = user_ids[start_ind: start_ind + 10]
    
    scores = lf.predict(user_ids=np.repeat(batch, train_val.shape[1]), item_ids=np.tile(items_ids, len(batch)), item_features=items_features)
    
    # excluding train predictions
    scores[train_val[batch, :].nonzero()] = -np.inf
    
    # predictions 
    predictions = np.argsort(scores, axis=1)[:, ::-1][:, :100]
    
    # Appending results
    predictions_all.extend(predictions)
    
predictions_all = np.array(predictions_all)
```

    100%|██████████| 295/295 [03:52<00:00,  1.27it/s]



```python
lf_ndcg = [ndcg_at_k(predictions_all, test_active, k=i).mean() for i in (10, 20, 50, 100)]
```

    100%|██████████| 30/30 [00:00<00:00, 51.85it/s]
    100%|██████████| 30/30 [00:00<00:00, 53.90it/s]
    100%|██████████| 30/30 [00:00<00:00, 52.53it/s]
    100%|██████████| 30/30 [00:00<00:00, 51.44it/s]



```python
# тестируем
random_predictor = RandomRecommender()
random_predictor.predict(test_active, excluding_predictions=train_val_for_test, random_state=123)
model_prediction = predictions_all[:, :20]

random_ndcg = ndcg_at_k(random_predictions, test_active, k=20)
model_ndcg = ndcg_at_k(model_prediction, test_active, k=20)

print(
f"""
P-value: {scipy.stats.ttest_ind(random_ndcg, model_ndcg)[1]}
"""
)
```

    100%|██████████| 30/30 [00:00<00:00, 53.99it/s]
    100%|██████████| 30/30 [00:00<00:00, 54.01it/s]

    
    P-value: 0.0
    


    


Средние значения метрик, полученные с помощью рандомных предсказаний и с помощью предсказаний модели, статистически значимо различаются.

### DSSM
Попробовал реализовать DSSM на Pytorch. Гиперпараметры также не вычислялись из-за вычислительных ограничений.


```python
from src.torch_models import DSSM

dssm = DSSM(user_dim=128, item_dim=items_features.shape[1], layers_dims=[1024, 512, 256],
            device=device, n_users=train.shape[0],
           batch_size=512)
dssm.to(device)

val_user_ids = np.arange(validation.shape[0])[(train.getnnz(1) > 10) & (validation.getnnz(1) > 5)]
train_user_ids = np.arange(validation.shape[0])

dssm.fit(train, val_active, item_features=items_features, user_ind_train=train_user_ids, user_ind_val=val_user_ids)
```

    Epoch 1: 100%|██████████| 271/271 [00:37<00:00,  7.15it/s, Train Loss: 6.303806617920702e+20] 
    100%|██████████| 40/40 [00:00<00:00, 52.94it/s]
    Epoch 2:   0%|          | 1/271 [00:00<00:41,  6.54it/s, Train Loss: 8.526408328584312e+20]

    Validation NDCG: 0.11054194120366037


    Epoch 2: 100%|██████████| 271/271 [00:36<00:00,  7.52it/s, Train Loss: 2.9159600763261315e+20]
    100%|██████████| 40/40 [00:00<00:00, 52.90it/s]
    Epoch 3:   0%|          | 1/271 [00:00<00:29,  9.14it/s, Train Loss: 4.0495748604366737e+20]

    Validation NDCG: 0.11028292021262767


    Epoch 3: 100%|██████████| 271/271 [00:37<00:00,  7.26it/s, Train Loss: 1.3600965872463538e+20]
    100%|██████████| 40/40 [00:00<00:00, 51.50it/s]
    Epoch 4:   0%|          | 1/271 [00:00<00:40,  6.74it/s, Train Loss: 1.805556070638871e+20]

    Validation NDCG: 0.11023647724103322


    Epoch 4: 100%|██████████| 271/271 [00:38<00:00,  7.10it/s, Train Loss: 1.0572367251019661e+20]
    100%|██████████| 40/40 [00:00<00:00, 52.70it/s]
    Epoch 5:   0%|          | 1/271 [00:00<00:41,  6.57it/s, Train Loss: 1.0864740586984833e+20]

    Validation NDCG: 0.11001960782771722


    Epoch 5: 100%|██████████| 271/271 [00:37<00:00,  7.14it/s, Train Loss: 7.557454470709038e+19] 
    100%|██████████| 40/40 [00:00<00:00, 52.95it/s]
    Epoch 6:   0%|          | 1/271 [00:00<00:41,  6.53it/s, Train Loss: 5.6629167712137904e+19]

    Validation NDCG: 0.10982049498764887


    Epoch 6: 100%|██████████| 271/271 [00:37<00:00,  7.18it/s, Train Loss: 6.1362173843574096e+19]
    100%|██████████| 40/40 [00:00<00:00, 52.81it/s]
    Epoch 7:   0%|          | 1/271 [00:00<00:41,  6.56it/s, Train Loss: 4.7743125842684805e+19]

    Validation NDCG: 0.1096352125696313


    Epoch 7: 100%|██████████| 271/271 [00:38<00:00,  7.13it/s, Train Loss: 3.703433998498372e+19] 
    100%|██████████| 40/40 [00:00<00:00, 53.04it/s]
    Epoch 8:   0%|          | 1/271 [00:00<00:40,  6.65it/s, Train Loss: 3.3756990491699708e+19]

    Validation NDCG: 0.10940291965249553


    Epoch 8: 100%|██████████| 271/271 [00:38<00:00,  7.12it/s, Train Loss: 3.206853645562177e+19] 
    100%|██████████| 40/40 [00:00<00:00, 52.96it/s]
    Epoch 9:   0%|          | 1/271 [00:00<00:40,  6.68it/s, Train Loss: 2.20439810819322e+19]

    Validation NDCG: 0.10919461152887008


    Epoch 9: 100%|██████████| 271/271 [00:37<00:00,  7.14it/s, Train Loss: 2.2793821623796367e+19]
    100%|██████████| 40/40 [00:00<00:00, 53.10it/s]
    Epoch 10:   0%|          | 1/271 [00:00<00:41,  6.53it/s, Train Loss: 2.267703589674051e+19]

    Validation NDCG: 0.10906961766458684


    Epoch 10: 100%|██████████| 271/271 [00:37<00:00,  7.17it/s, Train Loss: 1.7195504239346975e+19]
    100%|██████████| 40/40 [00:00<00:00, 53.02it/s]

    Validation NDCG: 0.10896395391036418


    



```python
user_ids = np.arange(test.shape[0])[(train.getnnz(1) > 10) & (test.getnnz(1) > 5)]
predictions = dssm.predict(item_features=items_features, number_of_predictions=100, user_ids=user_ids, excluding_predictions=train_val)
```


```python
ndcg_dssm = [ndcg_at_k(predictions, test_active, k=i).mean() for i in (10, 20, 50, 100)]
```

    100%|██████████| 30/30 [00:00<00:00, 52.86it/s]
    100%|██████████| 30/30 [00:00<00:00, 53.78it/s]
    100%|██████████| 30/30 [00:00<00:00, 53.08it/s]
    100%|██████████| 30/30 [00:00<00:00, 52.58it/s]



```python
fig, ax = plt.subplots(figsize=(9,7))
x = np.arange(0, 4)
ax.plot(x, most_pop_ndcg, marker='^', markersize=20, label='Most Popular')
ax.plot(x, svd_ndcg, marker='^', markersize=20, label='SVD')
ax.plot(x, als_ndcg, marker='^', markersize=20, label='ALS')
ax.plot(x, torch_als_ndcg, marker='^', markersize=20, label='Torch_ALS')
ax.plot(x, autoencoder_ndcg, marker='^', markersize=20, label='AutoEncoder')
ax.plot(x, lf_ndcg, marker='^', markersize=20, label='LightFM')
ax.plot(x, ndcg_dssm, marker='^', markersize=20, label='DSSM')
ax.legend()
ax.set_xticks(x)
labels = [f"k={i}" for i in (10, 20, 50, 100)]
ax.set_xticklabels(labels)
ax.set_xlabel("K")
ax.set_ylabel("nDCG@k value")
ax.grid()
plt.show()
```


    
![png](output_56_0.png)
    


DSSM также может быть улучшена путем более тщательной настройки гиперпараметров и обучения на большем количестве эпох.


```python
# тестируем
random_predictor = RandomRecommender()
random_predictor.predict(test_active, excluding_predictions=train_val_for_test, random_state=123)
model_prediction = dssm.predict(item_features=items_features, number_of_predictions=20, user_ids=user_ids, excluding_predictions=train_val,)

random_ndcg = ndcg_at_k(random_predictions, test_active, k=20)
model_ndcg = ndcg_at_k(model_prediction, test_active, k=20)

print(
f"""
P-value: {scipy.stats.ttest_ind(random_ndcg, model_ndcg)[1]}
"""
)
```

    100%|██████████| 30/30 [00:00<00:00, 54.26it/s]
    100%|██████████| 30/30 [00:00<00:00, 53.96it/s]

    
    P-value: 0.0
    


    


Средние значения метрик, полученные с помощью рандомных предсказаний и с помощью предсказаний модели, статистически значимо различаются.

---

### Что можно еще было бы реализовать:
* По моделям, использующим только коллаборативный подход: Использовать модели, основанные на других разложениях (SVD++, Factorization Machines, DeepMF, ...)
* По моделям, испольпользующим колабборативный и контентный подходы: Развивать идею DSSM, использовать другие модели, основанные на параллельных сетях (Например, CoNet)
* По данным: Преобразовать оценки в бинарные признаки, или какие-то другие, где также можно высчитать позитивный негативный фидбэк. Также использовать другие лоссы и метрики.
* По статистической проверке: Можно посчитать доверительные интервалы для метрик nDCG с помощью bootsrap-выборок без возвращения, чтобы посмотреть, как они изменяются при увеличении K (как изменяется размер интервала, повторяет ли он динамику основной величины)
* Также стоит в иных кейсах тюнить гиперпараметры всех моделей. Отдельно обрабатывать холодных пользователей и холодные айтемы.
