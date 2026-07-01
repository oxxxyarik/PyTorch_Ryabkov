import torch

### 2.1 Простые вычисления с градиентами (8 баллов)
# Создайте тензоры x, y, z с requires_grad=True
# Вычислите функцию: f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y*z
# Найдите градиенты по всем переменным
# Проверьте результат аналитически

print("\nЗАДАНИЕ 2.1")
x = torch.arange(4.).reshape(2, 2).requires_grad_(True)
y = torch.arange(1., 5.).reshape(2, 2).requires_grad_(True)
z = torch.arange(2., 6.).reshape(2, 2).requires_grad_(True)

f = x ** 2 + y ** 2 + z ** 2 + 2 * x * y * z
f = f.sum() #превращаем функцию в скаляр
print(f)

f.backward()

#проведем аналитическую проверку

analyticalGradX = 2 * x + 2 * y * z
analyticalGradY = 2 * y + 2 * x * z
analyticalGradZ = 2 * z + 2 * x * y

print(f"\nсравнение градиентов по Х")

print(x.grad)
print(analyticalGradX)

print(f"\nсравнение градиентов по Y")

print(y.grad)
print(analyticalGradY)

print(f"\nсравнение градиентов по Z")

print(z.grad)
print(analyticalGradZ)

### 2.2 Градиент функции потерь (9 баллов)
# Реализуйте функцию MSE (Mean Squared Error):
# MSE = (1/n) * Σ(y_pred - y_true)^2
# где y_pred = w * x + b (линейная функция)
# Найдите градиенты по w и b

print("\nЗАДАНИЕ 2.2")

x = torch.arange(4.).reshape(2, 2)
w = torch.tensor(4., requires_grad=True)
b = torch.tensor(1., requires_grad=True)
y_true = torch.arange(10., 50., 10.).reshape(2, 2)
y_pred = w * x + b

MSE = ((y_pred - y_true) ** 2).mean()
MSE.backward()
print(w.grad)
print(b.grad)

### 2.3 Цепное правило (8 баллов)
# Реализуйте составную функцию: f(x) = sin(x^2 + 1)
# Найдите градиент df/dx
# Проверьте результат с помощью torch.autograd.grad

print("\nЗАДАНИЕ 2.3")

x = torch.arange(4.).reshape(2, 2).requires_grad_(True)
f = torch.sin(x ** 2 + 1)

gradient = torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f))
analyticalGrad = torch.cos(x ** 2 + 1) * (2 * x)

print("Сравним градиенты")
print(gradient)
print(analyticalGrad)


