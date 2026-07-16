import os
import time
import psutil
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from homework_custom_transforms import CustomRandomBlur, CustomRandomBrightnessContrast, CustomSaltPepperNoise
from augmentations_basics.datasets import CustomImageDataset

class AugmentationPipeline:
    def __init__(self):
        self.transforms_dict = {}

    def add_augmentation(self, name, aug):
        self.transforms_dict[name] = aug

    def remove_augmentation(self, name):
        if name in self.transforms_dict:
            del self.transforms_dict[name]

    def get_augmentations(self):
        return list(self.transforms_dict.keys())

    def apply(self, image):
        img = image.copy()
        for name, transform in self.transforms_dict.items():
            img = transform(img)
        return img

    def get_compose(self):
        return transforms.Compose(list(self.transforms_dict.values()))

def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    
    train_dir = 'data/train'
    if not os.path.exists(train_dir):
        print(f"Error: Directory '{train_dir}' not found. Please setup the dataset in data/ first.")
        exit(1)

    print("=== Задание 3: Анализ датасета ===")
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    class_counts = {}
    all_sizes = []

    for cls in classes:
        cls_dir = os.path.join(train_dir, cls)
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        class_counts[cls] = len(files)
        for f in files:
            with Image.open(os.path.join(cls_dir, f)) as img:
                all_sizes.append(img.size)

    widths = [s[0] for s in all_sizes]
    heights = [s[1] for s in all_sizes]

    print(f"Всего классов: {len(classes)}")
    print(f"Распределение по классам: {class_counts}")
    print(f"Размеры: Min={min(all_sizes)}, Max={max(all_sizes)}, Avg=({np.mean(widths):.1f}, {np.mean(heights):.1f})")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(class_counts.keys(), class_counts.values(), color='skyblue', edgecolor='black')
    axes[0].set_title('Images per Class')
    axes[0].tick_params(axis='x', rotation=45)
    
    axes[1].scatter(widths, heights, alpha=0.5, color='purple')
    axes[1].set_xlabel('Width')
    axes[1].set_ylabel('Height')
    axes[1].set_title('Image Resolutions Scatter')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('results/dataset_analysis.png')
    plt.close()

    print("\n=== Задания 1, 2, 4: Конфигурации Пайплайна ===")
    pipeline = AugmentationPipeline()
    configs = {
        'light': [
            ('Flip', transforms.RandomHorizontalFlip(p=1.0)),
            ('Rotate', transforms.RandomRotation(degrees=15))
        ],
        'medium': [
            ('Flip', transforms.RandomHorizontalFlip(p=0.5)),
            ('Rotate', transforms.RandomRotation(degrees=30)),
            ('Color', transforms.ColorJitter(brightness=0.2, contrast=0.2)),
            ('CustomBlur', CustomRandomBlur(p=0.5, max_radius=2))
        ],
        'heavy': [
            ('Flip', transforms.RandomHorizontalFlip(p=0.5)),
            ('Rotate', transforms.RandomRotation(degrees=45)),
            ('Color', transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)),
            ('CustomBlur', CustomRandomBlur(p=0.8, max_radius=4)),
            ('CustomNoise', CustomSaltPepperNoise(p=0.8, amount=0.04)),
            ('CustomBC', CustomRandomBrightnessContrast(p=0.8))
        ]
    }

    first_class_dir = os.path.join(train_dir, classes[0])
    sample_img_name = [f for f in os.listdir(first_class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][0]
    orig_img = Image.open(os.path.join(first_class_dir, sample_img_name)).convert('RGB')

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(orig_img)
    axes[0].set_title('Original')
    axes[0].axis('off')

    for idx, (config_name, steps) in enumerate(configs.items(), start=1):
        pipeline.transforms_dict.clear()
        for name, op in steps:
            pipeline.add_augmentation(name, op)
        
        aug_img = pipeline.apply(orig_img)
        axes[idx].imshow(aug_img)
        axes[idx].set_title(f'Config: {config_name.upper()}')
        axes[idx].axis('off')
        
    plt.suptitle('Pipeline Configurations Comparison')
    plt.savefig('results/pipeline_comparison.png')
    plt.close()
    print("Сравнение конфигураций сохранено в results/pipeline_comparison.png")

    print("\n=== Задание 5: Эксперимент с размерами изображений ===")
    sizes = [64, 128, 224, 512]
    times, memories = [], []

    img_paths = []
    for cls in classes:
        cls_dir = os.path.join(train_dir, cls)
        img_paths.extend([os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    test_paths = (img_paths * (100 // len(img_paths) + 1))[:100]

    for sz in sizes:
        pipeline.transforms_dict.clear()
        pipeline.add_augmentation('Resize', transforms.Resize((sz, sz)))
        pipeline.add_augmentation('Flip', transforms.RandomHorizontalFlip(p=0.5))
        pipeline.add_augmentation('ToTensor', transforms.ToTensor())

        torch.cuda.empty_cache()
        mem_start = get_memory_usage_mb()
        start_time = time.time()

        for path in test_paths:
            with Image.open(path).convert('RGB') as img:
                _ = pipeline.apply(img)

        elapsed = time.time() - start_time
        mem_used = get_memory_usage_mb() - mem_start

        times.append(elapsed)
        memories.append(max(0.1, mem_used))
        print(f"Size: {sz:3d}x{sz:3d} | Time: {elapsed:.4f}s | Delta Mem: {mem_used:.2f} MB")

    fig, ax1 = plt.subplots(figsize=(10, 5))
    color = 'tab:red'
    ax1.set_xlabel('Image Resolution (px)')
    ax1.set_ylabel('Time (s) for 100 imgs', color=color)
    ax1.plot(sizes, times, marker='o', color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Process Delta Memory (MB)', color=color)
    ax2.plot(sizes, memories, marker='s', color=color, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Resource Usage vs Image Resolution')
    fig.tight_layout()
    plt.savefig('results/resource_scaling.png')
    plt.close()
    print("График масштабирования сохранен в results/resource_scaling.png")