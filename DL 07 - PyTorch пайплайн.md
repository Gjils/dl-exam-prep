# DL 07 — PyTorch пайплайн

## 1. Что такое PyTorch?

PyTorch — фреймворк для DL с двумя ключевыми возможностями:
1. **Тензоры на GPU** (numpy-подобные массивы, ускоренные CUDA)
2. **Autograd** — автоматическое дифференцирование (вычислительный граф)

---

## 2. Tensor — фундаментальная структура

**Tensor** = многомерный массив (как numpy ndarray) + поддержка GPU и autograd.

### Создание тензоров:

```python
import torch

# Из списка
x = torch.tensor([[1, 2], [3, 4]])          # shape (2, 2)

# Случайные
x = torch.randn(3, 224, 224)                 # нормальное распределение
x = torch.zeros(64, 10)                      # нули
x = torch.ones(3, 3)                         # единицы

# Из numpy
import numpy as np
x = torch.from_numpy(np.array([1, 2, 3]))

# На GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.randn(3, 224).to(device)
```

### Важные атрибуты:

| Атрибут | Описание | Пример |
|---|---|---|
| `.shape` | Размеры | `torch.Size([32, 3, 224, 224])` |
| `.dtype` | Тип данных | `torch.float32`, `torch.int64` |
| `.device` | CPU/GPU | `'cuda:0'` |
| `.requires_grad` | Нужен ли градиент | `True`/`False` |

### Операции:

```python
a = torch.randn(3, 4)
b = torch.randn(4, 5)

c = a @ b              # матричное умножение
d = a.sum()            # сумма всех элементов
e = a.mean(dim=0)      # среднее по батч-размерности
f = a.view(12)         # изменение формы
g = a.permute(1, 0)    # перестановка размерностей
```

---

## 3. Autograd — автоматический расчёт градиентов

```python
x = torch.randn(3, requires_grad=True)   # включаем отслеживание градиентов
y = x.sum()**2                            # forward pass
y.backward()                              # backward pass — заполняет x.grad
print(x.grad)                             # dy/dx = 2 * x.sum()
```

### Вычислительный граф:
- Строится **динамически** (define-by-run)
- Каждая операция — узел графа
- `.backward()` считает градиенты через chain rule

### Отключение градиентов:

```python
# Для eval / inference
with torch.no_grad():
    y = model(x)          # граф не строится, экономит память

# Для detach (отрезать от графа)
z = x.detach()            # z не требует градиента
```

### Градиенты — накапливаются!

```python
loss.backward()    # grad += d(loss)/dw
loss.backward()    # grad += d(loss)/dw ещё раз!
```

**Важно**: надо обнулять перед каждым шагом:
```python
optimizer.zero_grad()   # или model.zero_grad()
```

---

## 4. nn.Module — строительный блок

Базовый класс для всех слоёв и моделей.

### Определение модели:

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)   # полносвязный слой
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = MyModel()
```

### Ключевые методы:

| Метод | Назначение |
|---|---|
| `model.forward(x)` | Forward pass (вызывается через `model(x)`) |
| `model.parameters()` | Итератор по всем параметрам |
| `model.named_parameters()` | Параметры с именами |
| `model.state_dict()` | Словарь весов (для сохранения/загрузки) |
| `model.train()` | Режим обучения (Dropout, BN) |
| `model.eval()` | Режим оценки (Dropout выключен, BN — running stats) |
| `model.to(device)` | Переместить все параметры на устройство |

### Sequential — для простых цепочек:

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 10)
)
```

---

## 5. Loss Functions (nn.Module)

```python
criterion = nn.CrossEntropyLoss()  # для классификации
# или
criterion = nn.MSELoss()           # для регрессии
```

Использование:
```python
loss = criterion(predictions, targets)   # targets — сырые метки (не one-hot!)
```

CrossEntropyLoss внутри делает **Softmax + CrossEntropy** — не надо softmax отдельно в выходном слое!

---

## 6. Dataset и DataLoader

### Dataset — абстракция данных:

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
```

### DataLoader — батчи, перемешивание, параллелизм:

```python
from torch.utils.data import DataLoader

dataset = MyDataset(X, y)
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,          # перемешиваем каждую эпоху
    num_workers=4,         # параллельная загрузка
    drop_last=True         # отбросить неполный батч
)
```

---

## 7. Optimizer

```python
import torch.optim as optim

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
# или
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

---

## 8. Полный train/eval цикл

### Train:

```python
model.train()                          # включаем Dropout, BN
for batch_idx, (inputs, targets) in enumerate(train_loader):
    inputs, targets = inputs.to(device), targets.to(device)
    
    optimizer.zero_grad()              # обнуляем градиенты
    
    outputs = model(inputs)            # forward
    loss = criterion(outputs, targets) # loss
    
    loss.backward()                    # backward — вычисляем градиенты
    optimizer.step()                   # обновляем веса
```

### Evaluation:

```python
model.eval()                           # выключаем Dropout, BN → running stats
total_loss = 0
correct = 0

with torch.no_grad():                  # не строим граф (экономия памяти)
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
        
        _, predicted = outputs.max(1)  # argmax
        correct += predicted.eq(targets).sum().item()

accuracy = correct / len(val_loader.dataset)
```

### Разница train() и eval():

```python
# train mode — Dropout активен, BatchNorm считает статистику батча
model.train()

# eval mode — Dropout выключен, BatchNorm использует running mean/var
model.eval()
```

**Критически важно** не забыть переключить! Иначе:
- Dropout на eval → случайные обнуления → неправильные предсказания
- BN на eval со статистикой батча → нестабильность

---

## 9. Сохранение и загрузка

```python
# Сохранить
torch.save(model.state_dict(), 'model.pth')

# Загрузить
model.load_state_dict(torch.load('model.pth', map_location=device))

# Сохранить весь чекпоинт
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')
```

---

## 10. Полный пайплайн (шаблон)

```python
# 1. Подготовка данных
dataset = MyDataset(X, y)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 2. Модель + loss + optim
model = MyModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# 3. Цикл обучения
for epoch in range(10):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    # 4. Оценка
    model.eval()
    with torch.no_grad():
        # ... val loop ...
        pass
    
    print(f"Epoch {epoch}: val_acc = {val_acc:.4f}")
```

---

## 11. Резюме

- **Tensor**: многомерный массив на CPU/GPU с autograd
- **Autograd**: динамический вычислительный граф, `.backward()` считает градиенты (см. [[DL 04 - Backpropagation и autodiff]])
- **nn.Module**: базовый класс для слоёв и моделей (включая [[DL 11 - Базовая CNN|свёрточные слои]])
- **Dataset/DataLoader**: абстракция данных + батчи + перемешивание
- **Optimizer**: реализация градиентного спуска ([[DL 05 - Оптимизация|AdamW — стандарт]])
- **train()/eval()**: переключение режимов ([[DL 06 - Стабилизация и регуляризация|Dropout, BatchNorm]])
- **with torch.no_grad()**: отключение графа на inference

---

**Связанные вопросы:**
- [[DL 04 - Backpropagation и autodiff]] — autograd и вычислительный граф
- [[DL 05 - Оптимизация]] — оптимизаторы (SGD, Adam, AdamW)
- [[DL 08 - Диагностика обучения]] — анализ train/val loss, отладка пайплайна
- [[DL 11 - Базовая CNN]] — построение CNN с nn.Conv2d
