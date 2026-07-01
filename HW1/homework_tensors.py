import torch

### 1.1 Создание тензоров (7 баллов)
# Создайте следующие тензоры:
# - Тензор размером 3x4, заполненный случайными числами от 0 до 1
# - Тензор размером 2x3x4, заполненный нулями
# - Тензор размером 5x5, заполненный единицами
# - Тензор размером 4x4 с числами от 0 до 15 (используйте reshape)

print("\nЗАДАНИЕ 1.1")

tensor1 = torch.rand(3, 4)
print(tensor1)

tensor2 = torch.zeros(2, 3, 4)
print(tensor2)

tensor3 = torch.ones(5, 5)
print(tensor3)

tensor4 = torch.arange(16).reshape(4, 4)
print(tensor4)

### 1.2 Операции с тензорами (6 баллов)
# Дано: тензор A размером 3x4 и тензор B размером 4x3
# Выполните:
# - Транспонирование тензора A
# - Матричное умножение A и B
# - Поэлементное умножение A и транспонированного B
# - Вычислите сумму всех элементов тензора A

print("\nЗАДАНИЕ 1.2")

A = torch.arange(1, 13).reshape(3, 4)
B = torch.arange(1,13).reshape(4, 3)

A_T = A.T
print(A_T)

AxB = A @ B
print(AxB)

B_T = B.T
AxB_T = A * B_T
print(AxB_T)

A_Sum = A.sum().item()
print(A_Sum)

### 1.3 Индексация и срезы (6 баллов)
# Создайте тензор размером 5x5x5
# Извлеките:
# - Первую строку
# - Последний столбец
# - Подматрицу размером 2x2 из центра тензора
# - Все элементы с четными индексами

print("\nЗАДАНИЕ 1.3")

tensor = torch.arange(1, 126).reshape(5,5,5)
print(tensor)

firstRow = tensor[0, 0, :]
print(firstRow)

lastCol = tensor[:, :, -1]
print(lastCol)

slicedMatrix = tensor[2, 2:4, 2:4]
print(slicedMatrix)

evenEls = tensor[::2, ::2, ::2]
print(evenEls)

### 1.4 Работа с формами (6 баллов)
# Создайте тензор размером 24 элемента
# Преобразуйте его в формы:
# - 2x12
# - 3x8
# - 4x6
# - 2x3x4
# - 2x2x2x3

print("\nЗАДАНИЕ 1.4")

tensorRS = torch.arange(1, 25)
print(tensorRS)

print(tensorRS.reshape(2, 12))
print(tensorRS.reshape(3, 8))
print(tensorRS.reshape(4, 6))
print(tensorRS.reshape(2, 3, 4))
print(tensorRS.reshape(2, 2, 2, 3))

