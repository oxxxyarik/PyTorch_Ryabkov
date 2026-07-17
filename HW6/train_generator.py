import os
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from model import GeneratorTransformer

class TextDataset(Dataset):
    def __init__(self, text_path, tokenizer_path, max_length=128):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_length = max_length
        
        # Чтение и токенизация всего текста
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Получаем ID токенов
        encoded = self.tokenizer.encode(text)
        self.tokens = encoded.ids
        
        # Сдвигаем окно по тексту на max_length
        self.chunks = []
        for i in range(0, len(self.tokens) - max_length, max_length):
            self.chunks.append(self.tokens[i:i + max_length])

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def generate(model, tokenizer, prompt, device, max_length=128, temperature=1.0, max_out_tokens=100):
    model.eval()
    bos_token_id = tokenizer.token_to_id("<s>") or 1
    eos_token_id = tokenizer.token_to_id("</s>") or 2
    
    # Токенизируем промпт
    input_ids = tokenizer.encode(prompt).ids
    if not input_ids:
        input_ids = [bos_token_id]
        
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_out_tokens):
            # Передаем актуальное окно контекста
            outputs = model(input_ids)
            # Извлекаем логиты для самого последнего токена
            next_token_logits = outputs[0, -1, :] / temperature
            
            # Сэмплирование через Multinomial
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # Конкатенируем предсказание
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            # Сдвигаем контекст на 1 токен влево
            input_ids = generated[:, -max_length:]
            
            if next_token.item() == eos_token_id:
                break
                
    return tokenizer.decode(generated[0].tolist())

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer_path = "transformer_basics/mistral_tokenizer.json"
    text_path = "input.txt"
    checkpoint_path = "checkpoint.pt"
    
    if not os.path.exists(text_path):
        print(f"Пожалуйста, создайте файл '{text_path}' с текстом для обучения!")
        return

    # Параметры модели
    max_length = 128
    batch_size = 4
    epochs = 20
    lr = 1e-4
    
    dataset = TextDataset(text_path, tokenizer_path, max_length=max_length+1)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    vocab_size = dataset.tokenizer.get_vocab_size()
    
    model = GeneratorTransformer(vocab_size=vocab_size, max_len=max_length+100).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Инструмент для Mixed Precision
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            # Обучение с использованием автокаста точности (Mixed Precision Float16)
            with autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16):
                logits = model(x)
                # Изменяем размерность для функции потерь
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
                
        # Каждую эпоху тестируем генерацию для наглядности
        sample = generate(model, dataset.tokenizer, "Привет", device, max_length=max_length)
        print(f"--- Тестовая генерация: {sample}\n")
        
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Модель сохранена в {checkpoint_path}")

# --- 4. Интерактивный Чат-Интерфейс ---
def chat():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer_path = "transformer_basics/mistral_tokenizer.json"
    checkpoint_path = "checkpoint.pt"
    
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    
    model = GeneratorTransformer(vocab_size=vocab_size)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Веса успешно загружены.")
    else:
        print("Файл чекпоинта не найден")
        return
        
    model.to(device)
    model.eval()
    
    print("\nЧат готов. Напишите начало фразы, и модель продолжит её. Наберите 'quit' для выхода.")
    while True:
        user_input = input("\nВы: ")
        if user_input.lower() == 'quit':
            break
        response = generate(model, tokenizer, user_input, device, temperature=0.8, max_out_tokens=80)
        print(f"Бот: {response}")

if __name__ == "__main__":
    # Запускаем сначала обучение, а затем интерактивный тест
    train()
    chat()