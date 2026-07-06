import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), transforms.Lambda(torch.flatten)])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

class WidthMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes):
        super().__init__()
        layers = []
        current_dim = input_dim
        # Сборка слоев по переданному списку размерностей
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, num_classes))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    # Конфигурации из задания
    architectures = {
        'Narrow': [64, 32, 16],
        'Medium': [256, 128, 64],
        'Wide': [1024, 512, 256],
        'Ultra-Wide': [2048, 1024, 512]
    }
    
    criterion = nn.CrossEntropyLoss()
    
    for name, dims in architectures.items():
        model = WidthMLP(784, dims, 10).to(device)
        params_count = count_parameters(model)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        start_time = time.time()
        
        # 2 эпохи экспресс-теста для замера скорости
        model.train()
        for epoch in range(2):
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                loss = criterion(model(data), target)
                loss.backward()
                optimizer.step()
                
        duration = time.time() - start_time
        print(f"Config: {name:<10} | Params: {params_count:<9} | Time for 2 epochs: {duration:.2f}s")