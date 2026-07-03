import sys
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, current_dir)
sys.path.insert(0, project_root)

from homework_datasets import CSVDataset
from homework_model_modification import AdvancedLinearRegression, compute_l1_loss, compute_l2_loss

# Генерируем нелинейную зависимость для проверки полиномов
def generate_nonlinear_data(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.random.seed(42)
    x = np.random.uniform(-3, 3, 200)
    # Формула содержит квадрат: y = 2x^2 - x + 1 + шум
    y = 2 * (x ** 2) - x + 1 + np.random.normal(0, 1, 200)
    df = pd.DataFrame({'feature_x': x, 'target_y': y})
    df.to_csv(path, index=False)

# Функция для добавления полиномиальных признаков степени 2
def add_polynomial_features(X_tensor):
    # X_tensor имеет размерность [N, in_features]
    squares = X_tensor ** 2
    # Склеиваем исходные признаки и их квадраты в одну матрицу
    return torch.cat([X_tensor, squares], dim=1)

# Функция для проведения одного эксперимента обучения
def train_experiment(train_loader, val_loader, in_features, lr, alpha2, epochs=50):
    model = AdvancedLinearRegression(in_features)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            # Используем L2 регуляризацию (Ridge)
            loss = criterion(pred, batch_y) + alpha2 * compute_l2_loss(model)
            loss.backward()
            optimizer.step()
            
    # Считаем финальную ошибку на валидации
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            pred = model(batch_X)
            total_val_loss += criterion(pred, batch_y).item()
            
    return total_val_loss / len(val_loader)

if __name__ == '__main__':
    import pandas as pd
    
    csv_path = os.path.join(project_root, 'data', 'nonlinear_reg.csv')
    generate_nonlinear_data(csv_path)
    
    # 1. Тест Feature Engineering (Обычные признаки vs Полиномиальные)
    dataset = CSVDataset(csv_path, target_column='target_y')
    
    # Разделение данных 80/20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
    
    # Эксперимент на базовых признаках
    base_val_loss = train_experiment(train_loader, val_loader, in_features=1, lr=0.01, alpha2=0.0)
    
    # Преобразуем признаки в полиномиальные
    # Пересоздаем DataLoader'ы с измененными тензорами внутри Dataset
    dataset.X = add_polynomial_features(dataset.X)
    train_data_poly, val_data_poly = random_split(dataset, [train_size, val_size])
    
    train_loader_poly = DataLoader(train_data_poly, batch_size=16, shuffle=True)
    val_loader_poly = DataLoader(val_data_poly, batch_size=16, shuffle=False)
    
    # Эксперимент на полиномиальных признаках
    poly_val_loss = train_experiment(train_loader_poly, val_loader_poly, in_features=2, lr=0.01, alpha2=0.0)
    
    print("=== Эксперимент 1: Feature Engineering ===")
    print(f"MSE на базовых признаках (линейная модель): {base_val_loss:.4f}")
    print(f"MSE на полиномиальных признаках (x + x^2):  {poly_val_loss:.4f}")
    print("-" * 60)
    
    # 2. Перебор гиперпараметров для регуляризации
    print("=== Эксперимент 2: Перебор гиперпараметров ===")
    lr_list = [0.001, 0.01, 0.1]
    alpha_list = [0.0, 0.01, 0.1, 1.0]
    
    results = {}
    
    for lr in lr_list:
        results[lr] = []
        for alpha in alpha_list:
            val_loss = train_experiment(train_loader_poly, val_loader_poly, in_features=2, lr=lr, alpha2=alpha)
            results[lr].append(val_loss)
            print(f"Параметры: lr={lr}, alpha2={alpha} -> Val MSE: {val_loss:.4f}")
            
    # Построение графика по результатам перебора
    plt.figure(figsize=(10, 6))
    for lr in lr_list:
        plt.plot(alpha_list, results[lr], marker='o', label=f'lr={lr}')
        
    plt.xscale('symlog', linthresh=0.01)
    plt.xlabel('Alpha2 (L2 Penalty Coefficient)')
    plt.ylabel('Validation MSE')
    plt.title('Hyperparameter Grid Search Analysis')
    plt.legend()
    plt.grid(True)
    
    project_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'HW2')
    plots_dir = os.path.join(project_root, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    graph_path = os.path.join(plots_dir, 'hyperparameter_search.png')
    plt.savefig(graph_path)
    plt.close()
    
    print(f"\nГрафик перебора гиперпараметров сохранен в: {graph_path}")