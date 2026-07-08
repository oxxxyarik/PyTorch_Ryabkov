import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Архитектуры моделей
class FullyConnectedNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.net(x)

class SimpleCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, num_classes)
        )
    def forward(self, x): return self.classifier(self.features(x))

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class ResNetCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.res_block = ResidualBlock(32)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.res_block(self.init_conv(x)))

# 2. Обучающий цикл
def evaluate_model(model, loader, criterion):
    model.eval()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss += criterion(out, target).item()
            correct += out.argmax(dim=1).eq(target).sum().item()
            total += target.size(0)
    return correct / total, loss / len(loader)

def train_model(model, train_loader, test_loader, epochs=3):
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
            out = model(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            correct += out.argmax(dim=1).eq(target).sum().item()
            total += target.size(0)
            
        test_acc, _ = evaluate_model(model, test_loader, criterion)
        history['train_acc'].append(correct / total)
        history['test_acc'].append(test_acc)
        
    duration = time.time() - start_time
    return history, duration

if __name__ == '__main__':
    os.makedirs('plots', exist_ok=True)
    
    # Сравнение на MNIST
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
    
    models = {
        'FC_Net': FullyConnectedNet(784, 10),
        'Simple_CNN': SimpleCNN(1, 10),
        'ResNet_CNN': ResNetCNN(1, 10)
    }
    
    plt.figure(figsize=(10, 6))
    for name, model in models.items():
        model = model.to(device)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        history, duration = train_model(model, train_loader, test_loader, epochs=2)
        print(f"Model: {name:<12} | Params: {params:<8} | Time: {duration:.2f}s | Test Acc: {history['test_acc'][-1]:.4f}")
        plt.plot(history['test_acc'], label=f'{name} Test Acc')
        
    plt.title('CNN vs FC Performance (MNIST)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('plots/mnist_comparison.png')
    plt.close()