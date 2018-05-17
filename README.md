# myGboosting
Простая реализация градиентного бустинга. Работает с задачами регрессии, использует метрику MSE.

## Использование

Утилита предполагает вызов из командной строки. 
В качестве обучающей и тестовой выборки используются стандартные файлы csv. 
Утилита имеет 2 режима работы: обучение (fit) и применение (predict).

### Обучение

Формат запуска:

`myGboosting fit <file path> [optional parameters]`

`<file path>` - путь к csv файлу, содержащему обучающую выборку
#### Параметры:

TODO: написать актуальные параметры

### Применение

Формат запуска:

myGboosting predict \<file path> [optional parameters]

\<file path> - путь к csv файлу, содержащему тестовую выборку
#### Параметры:

TODO: написать актуальные параметры

### Архитектура и фичи

- Модель состоит из двух частей: бинаризатора признаков и набора решающих деревьев
- Для бинаризация континуальных признаков используется гистограммных подход с бинами
- Для бинаризации категориальных признаков используется one hot encoding
- Параллелизация сделана средствами OpenMP
- В качестве Weak Learner использованые Oblivious Decision Trees

### Cравнение с LightGBM:

Параметры запуска:

nice -n 10 time myGboosting fit train.csv --model model.dat --sample_rate 0.5 --learning_rate 0.1 --min_leaf_count 10 --depth 6 --max_bins 256 --iterations 500 --num_threads N

nice -n 10 time lightgbm application=regression_l2 data=train.csv label=31 metric=mse train_metric=true learning_rate=0.1 bagging_fraction=0.5 bagging_freq=1 num_leaves=64 max_bin=256 num_iterations=500 num_threads=N

|      App    |   1   |   2   |   4   |   8   |
|-------------|-------|-------|-------|-------|
| myGboosting | 22.07 | 18.95 | 15.69 | 15.95 |
| LightGBM    | 20.35 | 17.29 | 14.95 | 16.24 |