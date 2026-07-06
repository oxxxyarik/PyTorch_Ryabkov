import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), transforms.Lambda(torch.flatten)])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

class RegularizedMLP(nn.Module):
    def __init__(self, use_bn=False, dropout_rate=0.0):
        super().__init__()
        layers = []
        
        # Слой 1
        layers.append(nn.Linear(784, 256))
        if use_bn: layers.append(nn.BatchNorm1d(256))
        layers.append(nn.ReLU())
        if dropout_rate > 0: layers.append(nn.Dropout(dropout_rate))
            
        # Слой 2
        layers.append(nn.Linear(256, 128))
        if use_bn: layers.append(nn.BatchNorm1d(128))
        layers.append(nn.ReLU())
        if dropout_rate > 0: layers.append(nn.Dropout(dropout_rate))
            
        # Выходной
        layers.append(nn.Linear(128, 10))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def run_experiment(use_bn, dropout, weight_decay=0.0):
    model = RegularizedMLP(use_bn=use_bn, dropout_rate=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Обучение на 3 эпохах для аналитики
    for epoch in range(3):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            
    # Проверка точности
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
    return correct / total, model

if __name__ == '__main__':
    os.makedirs('plots', exist_ok=True)
    
    experiments = {
        'No Reg': (False, 0.0, 0.0),
        'Dropout 0.3': (False, 0.3, 0.0),
        'BatchNorm': (True, 0.0, 0.0),
        'BN + Dropout 0.3': (True, 0.3, 0.0),
        'L2 Decay': (False, 0.0, 1e-4)
    }
    
    plt.figure(figsize=(10, 6))
    
    for name, config in experiments.items():
        acc, trained_model = run_experiment(config[0], config[1], config[2])
        print(f"Method: {name:<16} | Test Accuracy: {acc:.4f}")
        
        # Сбор весов первого слоя для гистограммы распределения
        weights = trained_model.network[0].weight.detach().cpu().numpy().flatten()
        plt.hist(weights, bins=50, alpha=0.5, label=name, histtype='step', linewidth=2)
        
    plt.title('Weight Distributions Across Regularization Techniques')
    plt.xlabel('Weight Value')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig('plots/weight_distributions.png')
    plt.close()
    print("Regularization weights plot saved.")