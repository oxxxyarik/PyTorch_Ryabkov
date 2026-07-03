import torch
import torch.nn as nn
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

device = 'cuda' if torch.cuda.is_available() else 'cpu'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, current_dir)
sys.path.insert(0, project_root)

class AdvancedLinearRegression(nn.Module):
    """
    Расширенная модель линейной регрессии.
    """
    def __init__(self, in_features):
        super().__init__()
        
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)
    
class EarlyStopping:
    '''
    Класс для своевременной остановки обучения. 
    Останавливает обучение если количество эпох 
    без уменьшения ошибки превысило patience.
    '''
    def __init__(self, patience, minDelta):
        self.patience = patience 
        self.minDelta = minDelta #минимальный сдвиг ошибки
        self.counter = 0
        self.bestLoss = float('inf') # минимальная ошибка
    
    def __call__(self, valLoss):
        if valLoss < (self.bestLoss - self.minDelta):
            self.bestLoss = valLoss
            self.counter = 0
            return False
        
        else:
            self.counter += 1
            if (self.counter >= self.patience):
                return True

class MulticlassLogisticRegression(nn.Module):
    """
    Модель логистической регрессии для многоклассовой классификации.
    Принимает на вход in_features признаков и выдает num_classes логитов.
    """
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.linear(x)

def evaluate_multiclass_metrics(y_true, y_pred_logits, num_classes):
    """
    Вычисляет метрики классификации: Precision, Recall, F1, ROC-AUC
    
    Параметры:
        y_true (np.ndarray): Истинные метки классов.
        y_pred_logits (np.ndarray): Сырые выходные предсказания модели.
        num_classes (int): Количество целевых классов.
    """
    # Переводим логиты в вероятности классов с помощью Softmax
    logits_tensor = torch.tensor(y_pred_logits)
    probabilities = torch.softmax(logits_tensor, dim=1).numpy()
    
    # Определяем предсказанный класс как индекс максимальной вероятности
    y_pred_labels = np.argmax(probabilities, axis=1)
    
    # Расчет метрик
    precision = precision_score(y_true, y_pred_labels, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred_labels, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred_labels, average='macro', zero_division=0)
    
    try:
        if num_classes == 2:
            roc_auc = roc_auc_score(y_true, probabilities[:, 1])
        else:
            roc_auc = roc_auc_score(y_true, probabilities, multi_class='ovo', average='macro')
    except ValueError:
        roc_auc = 0.0
        
    return precision, recall, f1, roc_auc, y_pred_labels

def plot_confusion_matrix(y_true, y_pred_labels, save_path):
    """Генерирует и сохраняет график матрицы ошибок."""
    cm = confusion_matrix(y_true, y_pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    plt.title("Confusion Matrix")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def compute_l1_loss(model):
    """Считает сумму абсолютных значений всех весов модели"""
    l1_sum = 0.0
    
    # Перебираем все параметры внутри модели
    for param in model.parameters():
        if param.dim() > 1:
            l1_sum += torch.abs(param).sum()
            
    return l1_sum

def compute_l2_loss(model):
    """Считает сумму квадратов всех весов модели"""
    l2_sum = 0.0
    
    # Перебираем все параметры внутри модели
    for param in model.parameters():
        if param.dim() > 1:
            l2_sum += (param ** 2).sum()
            
    return l2_sum

if __name__ == '__main__':
    from utils import make_regression_data, RegressionDataset, log_epoch, mse, make_classification_data, ClassificationDataset
    from torch.utils.data import DataLoader, random_split

    #Загружаем датасет diabetes
    X, y = make_regression_data(source='diabetes')
    
    #Создаем датасет
    full_dataset = RegressionDataset(X, y)
    
    # Разбиваем датасет на обучающий 80% и валидационный 20%, чтобы early stopping правильно работал.
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Создаем даталоадеры для батчей
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"Обучающих примеров: {len(train_dataset)}, Валидационных: {len(val_dataset)}")
    
    # Инициализируем модель
    # X.shape[1] определяет, сколько признаков у датасета
    in_features = X.shape[1]
    model = AdvancedLinearRegression(in_features=in_features)
    
    # Настраиваем стандартный оптимизатор и базовую функцию потерь
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
    criterion = nn.MSELoss()
    
    # Создаем инструмент ранней остановки
    # Если 5 эпох подряд ошибка на валидации не падает хотя бы на 0.01 — останавливаемся
    early_stopping = EarlyStopping(patience=5, minDelta=1e-2)
    
    # Коэффициенты регуляризации
    alpha1 = 0.01   # Вес для L1 штрафа
    alpha2 = 0.005  # Вес для L2 штрафа
    
    epochs = 1000
    print("\nНачало цикла обучения")
    print("-" * 64)
    
    for epoch in range(1, epochs + 1):
        model.train() # режим обучения
        total_train_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad() # зануляем старые градиенты
            
            y_pred = model(batch_X) # делаем предсказание
            
            # Считаем базовую ошибку
            base_loss = criterion(y_pred, batch_y)
            
            # Считаем кастомные штрафы регуляризации
            l1_penalty = compute_l1_loss(model)
            l2_penalty = compute_l2_loss(model)
            
            # Складываем всё
            total_loss = base_loss + (alpha1 * l1_penalty) + (alpha2 * l2_penalty)
            
            total_loss.backward() # считаем комбинированные градиеты
            optimizer.step()      # обновляем веса w и b
            
            total_train_loss += total_loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        model.eval() # режим оценки
        total_val_loss = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                y_pred = model(batch_X)
                val_base_loss = criterion(y_pred, batch_y)
                total_val_loss += val_base_loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Логируем результаты каждые 5 эпох
        if epoch % 5 == 0 or epoch == 1:
            print(f"Эпоха {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
        # проверка early stopping
        if early_stopping(avg_val_loss):
            print("-" * 50)
            print(f"Обучение остановлено досрочно на эпохе {epoch}")
            print(f"Лучшая ошибка на валидации была: {early_stopping.best_loss:.4f}")
            break
    
    print("\nОбучение Логистической Регрессии")
    X_cls, y_cls = make_classification_data(source='breast_cancer')
    
    # Исходный датасет содержит 2 класса. Наша модель поддерживает N классов.
    num_classes = 2 
    cls_dataset = ClassificationDataset(X_cls, y_cls)
    
    c_train_size = int(0.8 * len(cls_dataset))
    c_val_size = len(cls_dataset) - c_train_size
    cls_train, cls_val = random_split(cls_dataset, [c_train_size, c_val_size])
    
    cls_train_loader = DataLoader(cls_train, batch_size=32, shuffle=True)
    cls_val_loader = DataLoader(cls_val, batch_size=32, shuffle=False)
    
    cls_model = MulticlassLogisticRegression(in_features=X_cls.shape[1], num_classes=num_classes).to(device)
    # Используем CrossEntropyLoss для многоклассового анализа
    cls_criterion = nn.CrossEntropyLoss()
    cls_optimizer = torch.optim.Adam(cls_model.parameters(), lr=0.005)
    
    for epoch in range(1, 51):
        cls_model.train()
        for batch_X, batch_y in cls_train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            cls_optimizer.zero_grad()
            
            logits = cls_model(batch_X)
            loss = cls_criterion(logits, batch_y.squeeze(1).long())
            loss.backward()
            cls_optimizer.step()
            
    # Финальная оценка качества на валидационных данных
    cls_model.eval()
    all_targets = []
    all_logits = []
    
    with torch.no_grad():
        for batch_X, batch_y in cls_val_loader:
            batch_X = batch_X.to(device)
            logits = cls_model(batch_X)
            
            all_targets.append(batch_y.numpy())
            all_logits.append(logits.cpu().numpy())
            
    y_true_all = np.vstack(all_targets).squeeze()
    y_logits_all = np.vstack(all_logits)
    
    # Считаем продвинутые метрики
    prec, rec, f1, auc, y_pred_labels = evaluate_multiclass_metrics(y_true_all, y_logits_all, num_classes)
    
    print(f"Результаты оценки модели:")
    print(f" Precision : {prec:.4f}")
    print(f" Recall    : {rec:.4f}")
    print(f" F1-Score  : {f1:.4f}")
    print(f" ROC-AUC   : {auc:.4f}")
    
    # Отрисовка матрицы ошибок
    project_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'HW2')
    plot_path = os.path.join(project_root, 'plots', 'confusion_matrix.png')
    plot_confusion_matrix(y_true_all, y_pred_labels, plot_path)
    print(f"\nГрафик сохранен по пути: {plot_path}")