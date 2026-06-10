# DL 11 — Базовая CNN для классификации

## 1. Архитектура типичной CNN для классификации

Базовая CNN для классификации изображений состоит из трёх блоков:

```
[Вход] → [Feature Extractor (Conv блоки)] → [Classifier Head (FC)] → [Softmax → класс]
```

---

## 2. [[DL 10 - Операция свертки|Feature Maps]] (карты признаков)

**Feature map** — выход свёрточного слоя: тензор размером $H \times W \times C_{\text{out}}$.

Каждый канал = один **детектор признаков** (одно ядро, одно "умение"):
- Канал 0: детектор горизонтальных рёбер
- Канал 1: детектор вертикальных рёбер
- ...
- Канал K: детектор текстуры "в клеточку"

### Иерархия feature maps по глубине:

```
Слой 1:  H×W × 64    — рёбра, углы, цветовые пятна
Слой 2:  H/2×W/2 × 128 — текстуры, простые формы
Слой 3:  H/4×W/4 × 256 — части объектов (глаза, колёса)
Слой 4:  H/8×W/8 × 512 — объекты целиком
```

**Пространственный размер уменьшается, глубина каналов растёт.**

---

## 3. Блок Conv-BN-ReLU

Стандартный строительный блок:

```python
x → [Conv2d] → [BatchNorm] → [ReLU] → x'
```

Каждый компонент:
- **Conv2d**: извлечение признаков (локальная свёртка)
- **BatchNorm**: стабилизация распределения активаций
- **ReLU**: нелинейность (обнуление отрицательных)

Вместе: **наиболее эффективная комбинация** для CV.

```python
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
```

---

## 4. Pooling (пулинг)

Уменьшает пространственный размер (понижает разрешение).

### Max Pooling:

$$y[i,j] = \max_{m=0}^{k-1} \max_{n=0}^{k-1} x[i\cdot s + m, j\cdot s + n]$$

- Берёт **максимум** в окне $k \times k$
- Обычно $k=2$, $s=2$ → уменьшение в 2 раза
- **Инвариантность к небольшим сдвигам**

### Average Pooling:

$$y[i,j] = \frac{1}{k^2} \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} x[i\cdot s + m, j\cdot s + n]$$

- Среднее значение в окне
- Используется реже (в основном в Global Average Pooling)

### Global Average Pooling (GAP):

```python
x = torch.mean(x, dim=(2, 3))   # → [batch, channels]
```

Усредняет **каждый канал целиком** → вектор размера $C_{\text{out}}$.

**Заменяет FC-слой** перед классификатором, радикально уменьшая число параметров.

### Сравнение:

| Операция | Размер выхода | Параметры | Инвариантность |
|---|---|---|---|
| **MaxPool (2×2, s=2)** | $H/2 \times W/2 \times C$ | 0 | К сдвигу |
| **AvgPool (2×2, s=2)** | $H/2 \times W/2 \times C$ | 0 | Нет |
| **Global AvgPool** | $1 \times 1 \times C$ | 0 | Полная |
| **Strided Conv (3×3, s=2)** | $H/2 \times W/2 \times C$ | $9C^2$ | Обучаемая |

Современная тенденция: **strided convolution вместо pooling** (см. [[DL 12 - Виды сверток]]).

---

## 5. Classifier Head (голова классификатора)

После feature extractor нужно от отображения $H' \times W' \times C'$ получить логиты классов.

### Вариант 1: FC после флаттена

```python
x = x.view(x.size(0), -1)          # flatten: [B, C*H*W] → большое
x = nn.Linear(C*H*W, 1024)(x)
x = nn.ReLU()(x)
x = nn.Dropout(0.5)(x)
x = nn.Linear(1024, num_classes)(x)
```

**Минус**: много параметров, зависимость от входного размера.

### Вариант 2: Global Average Pooling + FC (современный стандарт)

```python
x = torch.mean(x, dim=(2, 3))      # GAP: [B, C, 1, 1] → [B, C]
x = nn.Linear(C, num_classes)(x)   # только C × num_classes параметров
```

**Плюсы**: минимум параметров, устойчивость к размеру входа.

---

## 6. Полная архитектура: LeNet-5 (классический пример)

```
INPUT:    32×32×1   (grayscale)
CONV1:    6 filters 5×5, s=1 → 28×28×6       # 156 params
POOL1:    Avg 2×2, s=2     → 14×14×6
CONV2:    16 filters 5×5   → 10×10×16         # 2,416 params
POOL2:    Avg 2×2          → 5×5×16
FLATTEN:  → 400
FC1:      → 120
FC2:      → 84
OUTPUT:   → 10 (softmax)
```

---

## 7. Современная минимальная CNN

```python
class MiniCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Feature extractor
        self.features = nn.Sequential(
            # Block 1: 3×224×224 → 64×112×112
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Block 2: 64×112×112 → 128×56×56
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Block 3: 128×56×56 → 256×28×28
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # Block 4: 256×28×28 → 512×14×14
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # Block 5: 512×14×14 → 512×7×7
            nn.Conv2d(512, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d(1),   # → 512×1×1
        )
        
        # Classifier head
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)          # → [B, 512]
        x = self.classifier(x)
        return x
```

### Поток данных:

```
Input:  (B, 3, 224, 224)
Block 1: (B, 64, 112, 112)
Block 2: (B, 128, 56, 56)
Block 3: (B, 256, 28, 28)
Block 4: (B, 512, 14, 14)
Block 5: (B, 512, 7, 7)
GAP:     (B, 512, 1, 1)
Flatten: (B, 512)
FC:      (B, num_classes)
```

---

## 8. Схема CNN

```
                   ┌─────────┐
                   │  ВХОД   │  (3, 224, 224)
                   └────┬────┘
                        │
                   ┌────▼────┐
                   │ Conv-BN │  (64, 112, 112)
                   │ -ReLU   │
                   └────┬────┘
                        │
                   ┌────▼────┐
                   │ Conv-BN │  (128, 56, 56)
                   │ -ReLU   │
                   └────┬────┘
                        │   × N раз
                   ┌────▼────┐
                   │ ...     │  уменьшение H×W
                   │         │  увеличение Ch
                   └────┬────┘
                        │
                   ┌────▼────┐
                   │   GAP   │  (C', 1, 1)
                   └────┬────┘
                        │
                   ┌────▼────┐
                   │  Linear │  → num_classes
                   └────┬────┘
                        │
                   ┌────▼────┐
                   │ Softmax │  → вероятности
                   └─────────┘
```

---

## 9. Эволюция: от LeNet до ResNet

| Архитектура | Год | Число слоёв | Ключевая идея |
|---|---|---|---|
| **LeNet-5** | 1998 | 5 | Первая CNN |
| **AlexNet** | 2012 | 8 | ReLU, Dropout, GPU |
| **VGG-16** | 2014 | 16 | Только 3×3, глубоко |
| **GoogLeNet** | 2014 | 22 | Inception модули |
| **ResNet** | 2015 | 50–152 | Skip connections |
| **DenseNet** | 2017 | 121 | Плотные соединения |
| **EfficientNet** | 2019 | — | NAS-оптимизированная |
| **ConvNeXt** | 2022 | — | Современный редизайн (см. [[DL 13 - Классические CNN-архитектуры]]) |

---

## 10. Резюме

- **Feature maps**: выходы свёрточных слоёв (каналы = детекторы признаков).
- **Conv → BN → ReLU**: стандартный строительный блок.
- **Pooling** (MaxPool / Strided Conv): уменьшение пространственного размера.
- **Global Average Pooling**: замена FC-слоя → минимум параметров.
- **Classifier Head**: GAP + Linear → логиты классов.
- **Паттерн**: пространственный размер ↓, глубина каналов ↑.

---

**Связанные вопросы:**
- [[DL 09 - Изображение как тензор, мотивация CNN]] — мотивация свёрток вместо полносвязных слоёв
- [[DL 10 - Операция свертки]] — как работает операция свёртки
- [[DL 12 - Виды сверток]] — 1×1, dilated, group, depthwise separable
- [[DL 13 - Классические CNN-архитектуры]] — эволюция CNN (AlexNet, VGG, GoogLeNet, ResNet)
- [[DL 06 - Стабилизация и регуляризация]] — BatchNorm в Conv-BN-ReLU
