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

| Параметр       | Описание                                                                            | Значение по умолчанию |
|----------------|-------------------------------------------------------------------------------------|-----------------------|
| column_names   | путь к файлу, содержащему названия столбцов обучающей выборки                       |                       |
| model-path     | имя файла, в который будет сохранена обученная модель                               |                       |
| output-path    | имя файла, в который будут сохранены прогнозы обученной модели на обучающей выборке |                       |
| target         | название колонки, которая содержит значение целевой переменной                      |                       |
| thread-count   | число параллельных потоков, используемых для обучения                               |                       |
| delimiter      | разделитель, используемый в csv-файлах                                              |                       |
| has-header     | имеет ли входной csv-файл заголовок с названиями столбцов                           |                       |
| iterations     | максимальное количество деревьев в модели                                           |        50             |
| learning-rate  | темп обучения модели                                                                |        0.8            |
| l2-leaf-reg    | количество элементов в листовой вершине                                             |                       |
| depth          | глубина решающего дерева                                                            |        6              |
| max_bins       | количество сплитов в гистограмме для числовых признаков (от 1 до 255)               |        10             |
| logging-level  | степень подробности выводимой в консоль информации                                  |                       |
| sample_rate    | вероятность сэмплинга строк для каждого дерева (какую часть датасета использовать)  |        0.66           |
| min_leaf_count | минимальное количество объектов в листовой вершине                                  |        10             |


### Применение

Формат запуска:

myGboosting predict \<file path> [optional parameters]

\<file path> - путь к csv файлу, содержащему тестовую выборку
#### Параметры:

| Параметр       | Описание                                                                            | Значение по умолчанию |
|----------------|-------------------------------------------------------------------------------------|-----------------------|
| column_names   | путь к файлу, содержащему названия столбцов тестовой выборки                        |                       |
| model-path     | имя файла, из которого будет считана обученная модель                               |                       |
| output-path    | имя файла, в который будут сохранены прогнозы модели на тестовой выборке            |                       |
| delimiter      | разделитель, используемый в csv-файлах                                              |                       |
| has-header     | имеет ли входной csv-файл заголовок с названиями столбцов                           |                       |
| logging-level  | степень подробности выводимой в консоль информации                                  |                       |


### Архитектура и фичи

- Модель состоит из двух частей: бинаризатора признаков и набора решающих деревьев
- Бинаризация континуальных признаков используется гистограммных подход с бинами
- Для бинаризации категориальных признаков используется one hot encoding
- Параллелизация при обучении происходит при выборе оптимального сплита
- Данные лежат в памяти cache-friendly образом

### Cделано ###

- Интерфейс командной строки
- Чтение csv файла
- Гистограммы и бинаризация
- Решающее дерево в качестве weak learner'a
- Наборы данных для тестирования

### Осталось сделать

- Реализовать алгоритм градиентного бустинга
- Попробовать применять oblivious decision trees
- Научиться сохранить модель на диск
- Аккуратно реализовать и поддержать все режимы и опции из командной строки

### Используемые библиотеки

1) https://github.com/ben-strasser/fast-cpp-csv-parser
2) https://github.com/Taywee/args

