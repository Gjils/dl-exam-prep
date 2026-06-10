# 14. ResNet и современные CNN-backbone

## Skip connections, residual block, bottleneck, EfficientNet, ConvNeXt

---

## 1. Проблема глубины: Degradation

Подробнее об эволюции CNN-архитектур см. [[DL 13 - Классические CNN-архитектуры]].

### Наивное увеличение глубины
Интуиция: чем глубже сеть, тем лучше (более абстрактные признаки). На практике:

- **Vanishing/Exploding gradients** — решается Batch Normalization и нормальной инициализацией
- **Degradation problem**: с ростом глубины **ошибка на обучении растёт** (а не переобучение!)

### Экспериментальное наблюдение
- 20-layer → training error ~8%
- 56-layer → training error ~12%
- Это не overfitting (test error тоже выше), это проблема **оптимизации**

### Гипотеза
Более глубокая сеть должна как минимум не хуже помещать identity mapping дополнительных слоёв. Но оптимизация **не может научиться** отображать слои в identity.

---

## 2. Residual Block

### Идея
Вместо того чтобы учить $H(x)$ напрямую, учим **остаток**:

$$ \mathcal{F}(x) = H(x) - x $$

Где $H(x)$ — желаемое отображение, $x$ — вход блока.

### Residual Block (Basic Block)

$$ y = \mathcal{F}(x, \{W_i\}) + x $$

Если $x$ и $\mathcal{F}$ разной размерности — проекция:

$$ y = \mathcal{F}(x, \{W_i\}) + W_s x $$

### Почему это работает

1. **Identity shortcut**: градиент может течь напрямую через skip connection
   $$ \frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \frac{\partial y}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \left(1 + \frac{\partial \mathcal{F}}{\partial x}\right) $$

2. Даже если $\frac{\partial \mathcal{F}}{\partial x} \to 0$, градиент всё равно течёт через $1$
3. Легко обучить identity mapping: просто обнулить веса свёрток

### Формально
ResNet v1:
$$ y = \sigma(\mathcal{F}(x, W) + x) $$

ResNet v2 (pre-activation):
$$ y = \mathcal{F}(\sigma(x), W) + x $$
- BatchNorm + ReLU **перед** свёрткой
- Identity shortcut без активаций на пути градиента

### Варианты Residual Block

| Тип | Размерность | Параметры | Применение |
|-----|-------------|-----------|------------|
| **Basic** | 2× Conv3×3, 64→64 | ~38K | ResNet-18, ResNet-34 |
| **Bottleneck** | 1×1→c/4, 3×3→c/4, 1×1→c | ~70K | ResNet-50, 101, 152 |
| **Wide Residual** | 2× Conv3×3, wider | больше | WideResNet |

---

## 3. Bottleneck Block

### Структура

```
Input (256 channels)
    │
    ▼
┌─────────────────────┐
│ 1×1 Conv, 64 (↓×4) │ ← reduction
│ BN, ReLU            │
├─────────────────────┤
│ 3×3 Conv, 64        │ ← spatial features
│ BN, ReLU            │
├─────────────────────┤
│ 1×1 Conv, 256 (↑×4) │ ← expansion
│ BN                  │
└─────────────────────┘
    │         ┌───────┐
    └─────────┤  Add  │
              └───┬───┘
                  ▼
                ReLU
```

### Зачем bottleneck?

- **Параметры**: 3×3×256×256 = 589K → 1×1×256×64 + 3×3×64×64 + 1×1×64×256 = **70K** (в 8.4 раза меньше)
- **Глубина**: можно сделать сеть глубже при тех же вычислительных затратах
- **Бутылочное горлышко** сжимает-расширяет, создавая информационное узкое место

### ResNet семейство

| Архитектура | Блоков | Параметры | FLOPS | ImageNet top-1 |
|-------------|--------|-----------|-------|----------------|
| ResNet-18 | Basic ×8 | 11.7M | 1.8G | 70.3% |
| ResNet-34 | Basic ×16 | 21.8M | 3.6G | 73.3% |
| ResNet-50 | Bottleneck ×16 | 25.6M | 4.1G | 76.0% |
| ResNet-101 | Bottleneck ×33 | 44.5M | 7.6G | 77.4% |
| ResNet-152 | Bottleneck ×50 | 60.2M | 11.3G | 78.6% |

### Pre-activation ResNet (v2)

- BN + ReLU **до** свёртки (pre-act)
- Identity shortcut без ReLU
- Лучше градиентный поток → легче обучить 1000+ слоёв
- Лучшая регуляризация (BN до свёртки)

---

## 4. ResNeXt (Grouped Convolutions)

### Идея
**Разделяй и властвуй**: параллельные residual блоки с разными группами каналов.

$$ \text{ResNeXt block: } y = x + \sum_{i=1}^{C} \mathcal{T}_i(x) $$

Где $\mathcal{T}_i$ — преобразование в i-й группе (cardinality $C$).

### 1×1 Conv → Grouped 3×3 → 1×1 Conv + Sum

- **Cardinality** — новый гиперпараметр: количество групп
- Увеличение cardinality эффективнее увеличения глубины или ширины
- 32 группы 4-канальных 3×3 свёрток

---

## 5. DenseNet (Dense Connections)

### Идея
Каждый слой получает на вход **все предыдущие feature maps**:

$$ x_l = H_l([x_0, x_1, \ldots, x_{l-1}]) $$

- **Concat**, не add (в отличие от ResNet)
- Требует **рост каналов** (growth rate $k$, обычно 12-32)
- **Transition layer**: 1×1 Conv + AvgPool каждые N dense blocks

### Преимущества
- Улучшенный градиентный поток
- Feature reuse (легче учить сложные комбинации)
- Меньше параметров (не нужно переучивать уже полученные признаки)

---

## 6. EfficientNet (Compound Scaling)

### Проблема
Как масштабировать сеть? Больше глубины? Больше ширины? Больше разрешения?

### Наивный подход
Увеличивать один параметр — насыщение:

- Только глубина: $d = 2.0$ → diminishing returns
- Только ширина: $w = 2.0$ → diminishing returns
- Только разрешение: $r = 2.0$ → diminishing returns

### Compound Scaling

$$ \text{глубина: } d = \alpha^\phi $$
$$ \text{ширина: } w = \beta^\phi $$
$$ \text{разрешение: } r = \gamma^\phi $$
$$ \text{где } \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2 $$

**Ключ**: масштабировать **все три** измерения одновременно, сбалансированно.

### MBConv (Mobile Inverted Bottleneck)

Основа EfficientNet:
```
Input (c channels)
    │
    ▼
1×1 Expand (×6)
    │
    ▼
3×3 Depthwise Conv (DWConv)
    │
    ▼
SE (Squeeze-and-Excitation)
    │
    ▼
1×1 Project (back to c)
    │
    ▼
+ Skip if same size
```

### Squeeze-and-Excitation (SE)

```
Global AvgPool → FC(c/16) → ReLU → FC(c) → Sigmoid → Scale
```

- **Squeeze**: глобальное усреднение каждого канала
- **Excitation**: два FC → sigmoid-веса для каналов
- **Scale**: умножаем feature map на канальные веса

### EfficientNet результаты

| Модель | Параметры | FLOPS | ImageNet top-1 |
|--------|-----------|-------|----------------|
| B0 | 5.3M | 0.4G | 77.3% |
| B3 | 12M | 1.8G | 81.1% |
| B7 | 66M | 37G | 84.4% |

EfficientNet-B7 при 66M параметров достигает **state-of-the-art** на ImageNet (2019).

---

## 7. ConvNeXt (Modern ConvNet)

### Идея
Взять лучшие идеи Transformer-архитектур и применить к ConvNet. Доказать, что **ConvNet не умер**.

### Изменения ResNet → ConvNeXt

| Компонент | ResNet | ConvNeXt | Зачем |
|-----------|--------|----------|-------|
| **Stage compute ratio** | 3:4:6:3 | 3:3:9:3 | Как Swin-T |
| **Patch stem** | 7×7 conv, stride 2 | 4×4 conv, stride 4 | Non-overlapping |
| **Downsampling** | stride conv | 2×2 conv, stride 2 | Separate downsample |
| **Normalization** | BN | LN (LayerNorm) | Transformer-style |
| **Активация** | ReLU | GELU | Smooth, Swin-style |
| **Activation placement** | после conv | один после первой conv | Fewer activations |
| **Conv kernel** | 3×3 | 7×7 | Больше рецептивное поле |
| **Depthwise** | нет | depthwise conv | Separable, как Swin |
| **Bottleneck ratio** | 1:4 | 1:4 → 1:4×6 | Inverse bottleneck |

### ConvNeXt Block

```
Input
    │
    ▼
7×7 Depthwise Conv
    │
    ▼
LayerNorm
    │
    ▼
1×1 Conv (×4 expansion)
    │
    ▼
GELU activation
    │
    ▼
1×1 Conv (projection)
    │
    ▼
+ Residual
```

### Результаты ConvNeXt

| Модель | Параметры | ImageNet top-1 |
|--------|-----------|----------------|
| ConvNeXt-T | 29M | 82.1% |
| ConvNeXt-S | 50M | 83.1% |
| ConvNeXt-B | 89M | 83.8% |
| ConvNeXt-L | 198M | 84.3% |

Современные ConvNet на уровне Swin Transformer.

---

## 8. Сравнительная таблица backbone

| Архитектура | Год | Параметры | Идея | ImageNet top-1 |
|-------------|-----|-----------|------|----------------|
| ResNet-50 | 2015 | 25.6M | Skip connections | 76.0% |
| ResNeXt-101 | 2017 | 83M | Grouped conv | 78.8% |
| DenseNet-169 | 2017 | 14M | Dense concat | 76.2% |
| EfficientNet-B7 | 2019 | 66M | Compound scaling | 84.4% |
| ConvNeXt-B | 2022 | 89M | Modern ConvNet | 83.8% |

---

## Ключевые выводы к экзамену

1. **ResNet** решила degradation problem через skip connections
2. **Residual block** — $y = \mathcal{F}(x) + x$, градиентный highway
3. **Bottleneck** — 1×1 → 3×3 → 1×1, снижает параметры в 8 раз
4. **EfficientNet** — compound scaling (глубина + ширина + разрешение), MBConv + SE
5. **ConvNeXt** — ConvNet meets Transformer: LN, GELU, DWConv, 7×7 kernel
6. Все backbone используются как feature extractors в [[DL 16 - Семантическая сегментация|сегментации]] и [[DL 20 - Object Detection|детекции]], а также как основа для [[DL 15 - Transfer Learning|transfer learning]].

ResNet-блоки (skip connections) стали прообразом residual-связей в [[DL 33 - Архитектура Transformer|архитектуре Transformer]].

---

**Связанные вопросы:**
- [[DL 13 - Классические CNN-архитектуры]] — эволюция CNN до ResNet
- [[DL 15 - Transfer Learning]] — использование ResNet как backbone
- [[DL 33 - Архитектура Transformer]] — residual connections в Transformer
