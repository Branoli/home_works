import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("dark")
plt.rcParams['figure.figsize'] = 16, 12
from tqdm import tqdm_notebook
import pandas as pd
from collections import defaultdict

# поменяйте на свой путь
DS_FILE_NAME = '../data/stackoverflow_sample_125k.tsv'
TAGS_FILE_NAME = '../data/top10_tags.tsv'


top_tags = []
with open(TAGS_FILE_NAME, 'r') as f:
    for line in f:
        top_tags.append(line.strip())
top_tags = set(top_tags)
print(top_tags)


class LogRegressor():
    """Конструктор

    Параметры
    ----------
    tags : list of string, default=top_tags
        список тегов
    """

    def __init__(self, tags=top_tags):
        # словарь который содержит мапинг слов предложений и тегов в индексы (для экономии памяти)
        # пример: self._vocab['exception'] = 17 означает что у слова exception индекс равен 17
        self._vocab = {}

        # параметры модели: веса
        # для каждого класса/тега нам необходимо хранить собственный вектор весов
        # по умолчанию у нас все веса будут равны нулю
        # мы заранее не знаем сколько весов нам понадобится
        # поэтому для каждого класса мы сосздаем словарь изменяемого размера со значением по умолчанию 0
        # пример: self._w['java'][self._vocab['exception']]  содержит вес для слова exception тега java
        self._w = dict([(t, defaultdict(int)) for t in tags])

        # параметры модели: смещения или вес w_0
        self._b = dict([(t, 0) for t in tags])

        self._tags = set(tags)

    """Один прогон по датасету

    Параметры
    ----------
    fname : string, default=DS_FILE_NAME
        имя файла с данными

    top_n_train : int
        первые top_n_train строк будут использоваться для обучения, остальные для тестирования

    total : int, default=10000000
        информация о количестве строк в файле для вывода прогресс бара

    learning_rate : float, default=0.1
        скорость обучения для градиентного спуска

    tolerance : float, default=1e-16
        используем для ограничения значений аргумента логарифмов
    """

    def iterate_file(self,
                     fname=DS_FILE_NAME,
                     top_n_train=100000,
                     total=125000,
                     learning_rate=0.1,
                     tolerance=1e-16,
                     lmbda=0.01):

        self._loss = []
        n = 0
        accurate_sample = []
        # откроем файл
        with open(fname, 'r') as f:

            # прогуляемся по строкам файла
            for line in tqdm_notebook(f, total=total, mininterval=1):
                desired_tags = []
                pair = line.strip().split('\t')
                if len(pair) != 2:
                    continue
                sentence, tags = pair
                # слова вопроса, это как раз признаки x
                sentence = sentence.split(' ')
                # теги вопроса, это y
                tags = set(tags.split(' '))

                # значение функции потерь для текущего примера
                sample_loss = 0

                # прокидываем градиенты для каждого тега
                for tag in self._tags:
                    # целевая переменная равна 1 если текущий тег есть у текущего примера
                    y = int(tag in tags)

                    # расчитываем значение линейной комбинации весов и признаков объекта
                    # инициализируем z
                    # ЗАПОЛНИТЕ ПРОПУСКИ В КОДЕ
                    z = self._b[tag]
                    for word in sentence:
                        # если в режиме тестирования появляется слово которого нет в словаре, то мы его игнорируем
                        if n >= top_n_train and word not in self._vocab:
                            continue
                        if word not in self._vocab:
                            self._vocab[word] = len(self._vocab)
                        z += self._w[tag][self._vocab[word]]
                    # вычисляем вероятность наличия тега
                    # ЗАПОЛНИТЕ ПРОПУСКИ В КОДЕ
                    if z >= 0:
                        sigma = 1 / (1 + np.exp(-z))
                    else:
                        sigma = 1 - 1 / (1 + np.exp(z))

                    # обновляем значение функции потерь для текущего примера
                    # ЗАПОЛНИТЕ ПРОПУСКИ В КОДЕ
                    if y == 1:
                        sample_loss += -y * np.log(np.max([tolerance, sigma]))

                    else:
                        sample_loss += -(1 - y) * np.log(1 - np.min([1 - tolerance, sigma]))

                    # если мы все еще в тренировочной части, то обновим параметры
                    if n < top_n_train:
                        # вычисляем производную логарифмического правдоподобия по весу
                        # ЗАПОЛНИТЕ ПРОПУСКИ В КОДЕ
                        dLdw = y - sigma

                        # делаем градиентный шаг
                        # мы минимизируем отрицательное логарифмическое правдоподобие (второй знак минус)
                        # поэтому мы идем в обратную сторону градиента для минимизации (первый знак минус)
                        for word in sentence:
                            self._w[tag][self._vocab[word]] -= -learning_rate * dLdw
                        self._b[tag] -= -learning_rate * dLdw
                    if sigma > 0.9:
                        desired_tags.append(tag)
                if (n > top_n_train):
                    accurate_sample.append(len(tags.intersection(desired_tags)) / len(tags.union(desired_tags)))
                n += 1

                self._loss.append(sample_loss)
            return (np.mean(accurate_sample))


model = LogRegressor()
model.iterate_file()