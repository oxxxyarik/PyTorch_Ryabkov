import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Автоматический выбор устройства
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загрузка данных
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), transforms.Lambda(torch.flatten)])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Конструктор гибкой глубины сети
class DeepMLP(nn.Module):
    def __init__(self, input_dim, num_classes, num_layers, hidden_dim=256, use_bn=False, dropout_rate=0.0):
        super().__init__()
        layers = []
        current_dim = input_dim
        
        # Сборка скрытых слоев
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
            
        # Классификатор
        layers.append(nn.Linear(current_dim, num_classes))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def train_and_evaluate(model, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_acc': [], 'test_acc': []}
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
        train_acc = correct / total
        
        # Тестирование
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        test_acc = correct / total
        
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
    elapsed_time = time.time() - start_time
    return history, elapsed_time

if __name__ == '__main__':
    os.makedirs('plots', exist_ok=True)
    # Список конфигураций
    depths = [1, 2, 3, 5, 7]
    plt.figure(figsize=(10, 6))
    
    for d in depths:
        print(f"Training model with depth: {d} layers...")
        # 1 слой — линейный классификатор, остальные — с внутренними слоями
        model = DeepMLP(input_dim=784, num_classes=10, num_layers=d, use_bn=False, dropout_rate=0.0).to(device)
        history, duration = train_and_evaluate(model)
        print(f"Depth {d} | Time: {duration:.2f}s | Final Test Acc: {history['test_acc'][-1]:.4f}")
        
        plt.plot(history['test_acc'], label=f'{d} Layers (Test)')
        
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Depth Experiment Analysis')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/depth_experiments.png')
    plt.close()
    print("Depth experiment graph saved.")