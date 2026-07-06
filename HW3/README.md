# Homework 3: Fully Connected Networks Architecture and Regularization

## Project Description
Решение содержит скрипты для анализа влияния глубины, ширины слоев и различных техник регуляризации (Dropout, BatchNorm, L2 Decay) на полносвязных нейронных сетях (MLP) при классификации датасета MNIST.

## Structure
* `homework_depth_experiments.py` — Сравнение сетей разной глубины (от 1 до 7 слоев).
* `homework_width_experiments.py` — Оценка влияния ширины скрытых слоев (от Narrow до Ultra-Wide) и подсчет параметров.
* `homework_regularization_experiments.py` — Исследование комбинаций Dropout, BatchNorm и L2-регуляризации с анализом распределения весов.

---

## Experiment Results

### 1. Depth Experiments (Задание 1)
* **1 Layer**: Final Test Acc: ~0.9230 (линейный классификатор ограничивает точность).
* **2 Layers**: Time: 51.88s | Final Test Acc: 0.9777
* **3 Layers**: Time: 50.89s | Final Test Acc: 0.9764
* **5 Layers**: Time: 55.49s | Final Test Acc: 0.9724 (наблюдается легкое затухание точности без BatchNorm).
* **7 Layers**: Time: 57.49s | Final Test Acc: 0.9799 (наилучший результат при увеличении числа эпох).

### 2. Width Experiments (Задание 2)
* **Narrow** `[64, 32, 16]`: Params: 53,018 | Time: 27.56s
* **Medium** `[256, 128, 64]`: Params: 242,762 | Time: 26.42s
* **Wide** `[1024, 512, 256]`: Params: 1,462,538 | Time: 30.47s
* **Ultra-Wide** `[2048, 1024, 512]`: Params: 4,235,786 | Time: 30.70s

### 3. Regularization Experiments (Задание 3)
* **No Reg**: Test Accuracy: 0.9736
* **Dropout 0.3**: Test Accuracy: 0.9752
* **BatchNorm**: Test Accuracy: 0.9782 (Наилучший показатель стабильности и точности)
* **BN + Dropout 0.3**: Test Accuracy: 0.9763
* **L2 Decay**: Test Accuracy: 0.9760

---

## Generated Artifacts
* `plots/depth_experiments.png` — График динамики accuracy по эпохам для разной глубины.
* `plots/weight_distributions.png` — Гистограмма распределения весов (L2 Decay сильнее всего группирует веса вокруг нуля, снижая дисперсию).