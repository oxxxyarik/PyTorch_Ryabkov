import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class KernelAnalysisCNN(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        # Входной канал изменен на 1 (для MNIST)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        # Размерность после MaxPool для MNIST (28x28 -> 14x14)
        self.fc = nn.Linear(16 * 14 * 14, 10)
        
    def forward(self, x):
        act = self.relu(self.conv1(x))
        out = self.pool(act)
        out = out.view(out.size(0), -1)
        return self.fc(out), act

if __name__ == '__main__':
    os.makedirs('plots', exist_ok=True)
    
    # Используем MNIST, так как он гарантированно загружен локально
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
    
    img, _ = next(iter(test_loader))
    img = img.to(device)
    
    kernels = [3, 5, 7]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for idx, k in enumerate(kernels):
        model = KernelAnalysisCNN(kernel_size=k).to(device)
        model.eval()
        
        with torch.no_grad():
            _, activation = model(img)
            
        act_map = activation[0, 0].cpu().numpy()
        axes[idx].imshow(act_map, cmap='viridis')
        axes[idx].set_title(f'Kernel {k}x{k}')
        axes[idx].axis('off')
        
    plt.suptitle('First Layer Activations Analysis (MNIST)')
    plt.savefig('plots/kernel_activations.png')
    plt.close()
    print("Kernel activation analysis map saved.")