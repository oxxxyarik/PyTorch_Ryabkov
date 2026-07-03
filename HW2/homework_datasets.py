import sys
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, current_dir)
sys.path.insert(0, project_root)

project_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'HW2')

class CSVDataset(Dataset):
    def __init__(self, file_path, target_column, is_classification=False):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")
            
        self.is_classification = is_classification
        df = pd.read_csv(file_path)
        
        y_raw = df[target_column].copy()
        X_raw = df.drop(columns=[target_column]).copy()
        
        # One-Hot Encoding для категориальных признаков
        categorical_cols = X_raw.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        if categorical_cols:
            # dtype=float преобразует True/False в 1.0/0.0
            X_raw = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True, dtype=float)
            
        # Мин-Макс нормализация числовых колонок
        numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            min_val = X_raw[col].min()
            max_val = X_raw[col].max()
            if max_val - min_val > 0:
                X_raw[col] = (X_raw[col] - min_val) / (max_val - min_val)
            else:
                X_raw[col] = 0.0

        # Конвертация данных в тензоры PyTorch с явным приведением всех колонок к float
        self.X = torch.tensor(X_raw.values.astype(np.float32), dtype=torch.float32)
        self.y = torch.tensor(y_raw.values.astype(np.float32), dtype=torch.float32).unsqueeze(1)
            
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

if __name__ == '__main__':
    from homework_model_modification import AdvancedLinearRegression, MulticlassLogisticRegression

    os.makedirs(os.path.join(project_root, 'data'), exist_ok=True)
    
    # Регрессионный CSV
    reg_csv_path = os.path.join(project_root, 'data', 'test_regression.csv')
    pd.DataFrame({
        'mileage': [120000, 50000, 200000, 10000],
        'condition': ['bad', 'good', 'bad', 'perfect'],
        'price': [350000, 800000, 150000, 1200000]
    }).to_csv(reg_csv_path, index=False)
    
    # Классификационный CSV
    cls_csv_path = os.path.join(project_root, 'data', 'test_classification.csv')
    pd.DataFrame({
        'age': [23, 45, 12, 67, 34, 56],
        'gender': ['male', 'female', 'male', 'female', 'male', 'female'],
        'clicked': [0, 1, 0, 1, 0, 1]
    }).to_csv(cls_csv_path, index=False)

    print("CSV files created")
    print("-" * 60)

    # Тестирование регрессионного датасета
    print("Testing regression dataset:")
    reg_dataset = CSVDataset(file_path=reg_csv_path, target_column='price', is_classification=False)
    print(f"Loaded rows: {len(reg_dataset)}")
    sample_X, sample_y = reg_dataset[0]
    print(f"X tensor: {sample_X}")
    print(f"y tensor: {sample_y}")
    
    model_reg = AdvancedLinearRegression(in_features=sample_X.shape[0])
    pred = model_reg(sample_X.unsqueeze(0))
    print(f"Model prediction: {pred.item():.4f}")
    
    print("-" * 60)

    # Тестирование классификационного датасета
    print("Testing classification dataset:")
    cls_dataset = CSVDataset(file_path=cls_csv_path, target_column='clicked', is_classification=True)
    print(f"Loaded rows: {len(cls_dataset)}")
    sample_col_X, sample_col_y = cls_dataset[0]
    print(f"X tensor: {sample_col_X}")
    print(f"y tensor: {sample_col_y}")
    
    model_cls = MulticlassLogisticRegression(in_features=sample_col_X.shape[0], num_classes=2)
    pred_logits = model_cls(sample_col_X.unsqueeze(0))
    print(f"Model output: {pred_logits.detach().numpy()}")