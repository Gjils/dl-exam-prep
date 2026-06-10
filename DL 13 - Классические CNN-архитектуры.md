# 13. Классические CNN-архитектуры

## LeNet, AlexNet, VGG, Inception: ключевые идеи, эволюция feature extraction

---

## 1. LeNet-5 (1998, Yann LeCun)

### Ключевая идея
Первая полноценная свёрточная нейронная сеть, применённая к распознаванию рукописных цифр (MNIST). Заложила архитектурный паттерн **Conv → Pooling → Conv → Pooling → FC → FC → Output**.

### Архитектура

| Слой | Детали |
|------|--------|
| Conv1 | 6 фильтров 5×5, stride 1, padding 0 → 28×28×6 |
| Pool1 | Average pooling 2×2, stride 2 → 14×14×6 |
| Conv2 | 16 фильтров 5×5 → 10×10×16 |
| Pool2 | Average pooling 2×2 → 5×5×16 |
| FC3 | 120 нейронов |
| FC4 | 84 нейрона |
| Output | 10 нейронов (Gaussian connections / softmax) |

### Особенности
- **Активация**: Sigmoid / Tanh (ReLU тогда ещё не было)
- **Subsampling**: Average pooling (не Max Pooling)
- **Полносвязная часть**: небольшая (120 → 84 → 10), классификатор — RBF
- **Размер входа**: 32×32 (с padding до входа, на деле MNIST 28×28)

### Эволюционное значение
LeNet доказала, что **обучение признаков через свёртки** эффективнее hand-crafted признаков (SIFT, HOG). Ввела понятие **convolutional feature hierarchy**.

---

## 2. AlexNet (2012, Alex Krizhevsky)

### Ключевая идея
Глубокая CNN, выигравшая ImageNet 2012 с отрывом 10.8% top-5 (против 25.8% у第二名). Сочетание **ReLU, Dropout, Data Augmentation, GPU-параллелизма**.

### Архитектура

| Слой | Детали |
|------|--------|
| Conv1 | 96 фильтров 11×11, stride 4, ReLU, LRN, MaxPool 3×3 stride 2 |
| Conv2 | 256 фильтров 5×5, pad 2, ReLU, LRN, MaxPool 3×3 stride 2 |
| Conv3 | 384 фильтра 3×3, pad 1, ReLU |
| Conv4 | 384 фильтра 3×3, pad 1, ReLU |
| Conv5 | 256 фильтров 3×3, pad 1, ReLU, MaxPool 3×3 stride 2 |
| FC6 | 4096, ReLU, Dropout 0.5 |
| FC7 | 4096, ReLU, Dropout 0.5 |
| FC8 | 1000 (softmax) |

### Ключевые инновации

1. **ReLU** (Rectified Linear Unit)
   $$f(x) = \max(0, x)$$
   - Решает проблему исчезающего градиента Tanh/Sigmoid
   - Обучение в 6 раз быстрее

2. **Dropout** (Hinton, 2012)
   - Случайное отключение нейронов (p=0.5) в FC-слоях
   - Эффект: ансамбль сетей + регуляризация

3. **Local Response Normalization (LRN)**
   $$b_{x,y}^i = \frac{a_{x,y}^i}{\left(k + \alpha \sum_{j=\max(0,i-n/2)}^{\min(N-1,i+n/2)} (a_{x,y}^j)^2\right)^\beta}$$
   - Подавление сильно активных нейронов соседями
   - Позже вытеснена Batch Normalization

4. **Data Augmentation**
   - Горизонтальные отражения, сдвиги, изменение цвета (PCA color augmentation)

5. **Overlap Pooling**
   - MaxPool с stride < kernel size → меньше ошибок

6. **GPU распараллеливание**
   - Сеть разделена на 2 GPU (позднее стало стандартом)

### Результаты
- Top-5 error: 15.3% (ILSVRC 2012)
- Переломный момент: начало эры Deep Learning

---

## 3. VGG (2014, Oxford VGG)

### Ключевая идея
**Глубина важнее размера фильтров.** Стопка 3×3 свёрток вместо больших фильтров. Простая, однородная архитектура.

### Замена больших фильтров

| Один фильтр | Эквивалентная стопка | Параметров | Рецептивное поле |
|-------------|----------------------|------------|------------------|
| 5×5 | 2× 3×3 | 25 → 18 (меньше!) | 5×5 |
| 7×7 | 3× 3×3 | 49 → 27 (в 1.8 раз меньше!) | 7×7 |

$$ \text{ReLU между каждой свёрткой → } \sigma(W_3\sigma(W_2\sigma(W_1 x))) \text{ с } \sigma = \text{ReLU} $$

- Дополнительный ReLU между свёртками даёт **больше нелинейности**
- Меньше параметров → легче обучать, меньше переобучения

### Архитектуры VGG

| Конфиг | VGG-16 | VGG-19 |
|--------|--------|--------|
| Блок 1 | Conv64×2, Pool | Conv64×2, Pool |
| Блок 2 | Conv128×2, Pool | Conv128×2, Pool |
| Блок 3 | Conv256×3, Pool | Conv256×4, Pool |
| Блок 4 | Conv512×3, Pool | Conv512×4, Pool |
| Блок 5 | Conv512×3, Pool | Conv512×4, Pool |
| FC | 4096-4096-1000 | 4096-4096-1000 |
| Всего | **138M параметров** | **144M параметров** |

### Особенности
- Все свёртки: kernel 3×3, stride 1, padding 1
- Все pooling: MaxPool 2×2, stride 2
- Огромное количество параметров (138M) — в основном в FC-слоях

### Эволюционное значение
- Показала: **глубина сильно улучшает качество**
- Стандартизировала архитектурные блоки
- Послужила основой для transfer learning (VGG-Face, VGG-Texture)
- Недостаток: **очень много параметров**, медленно

---

## 4. Inception (GoogLeNet, 2014, Google)

### Ключевая идея
**Разные масштабы признаков — в одном блоке.** Сеть внутри сети (NIN) + параллельные свёртки разных размеров.

### Проблема, которую решает Inception
Какой размер фильтра лучше? 1×1, 3×3, 5×5, 7×7?
**Ответ**: все сразу, пусть сеть выбирает.

### Inception Module (naive)

```
    ┌── 1×1 Conv ──┐
    │              │
Input ── 3×3 Conv ──┤── Concatenate
    │              │
    └── 5×5 Conv ──┘
    └── 3×3 MaxPool ┘
```

### Inception Module (with 1×1 bottleneck)

```
    ┌── 1×1 Conv ── 1×1 Conv ──┐
    │              │            │
Input ── 1×1 Conv ── 3×3 Conv ──┤── Concatenate
    │              │            │
    └── 1×1 Conv ── 5×5 Conv ──┘
    └── 3×3 MaxPool ── 1×1 Conv ┘
```

**1×1 Conv «bottleneck»**: снижает число каналов, уменьшает параметры в 10-50 раз.

### GoogLeNet (Inception v1) — архитектура

- **22 слоя** (27 включая pooling)
- **9 Inception модулей**, сведённых в стек
- **Вспомогательные классификаторы** (auxiliary classifiers) на 1/3 и 2/3 глубины:
  - Для борьбы с vanishing gradient
  - Потери взвешиваются с весом 0.3
  - На inference отключаются

### Эволюция Inception

| Версия | Год | Ключевое изменение |
|--------|-----|-------------------|
| **v1** (GoogLeNet) | 2014 | Inception module, auxiliary classifiers |
| **v2** | 2015 | + Batch Normalization, 5×5 → 2×3×3 |
| **v3** | 2015 | Factorized 7×7, Label Smoothing, RMSProp, 3×3 → 3×1 + 1×3 |
| **v4** | 2016 | + Residual connections (Inception-ResNet) |

### Batch Normalization (Inception v2)

$$ \hat{x}^{(k)} = \frac{x^{(k)} - \mathbb{E}[x^{(k)}]}{\sqrt{\text{Var}[x^{(k)}] + \epsilon}} $$
$$ y^{(k)} = \gamma^{(k)} \hat{x}^{(k)} + \beta^{(k)} $$

- Решает Internal Covariate Shift
- Позволяет использовать **выше learning rate**
- Частичный регуляризатор (уменьшает need for Dropout)

### Label Smoothing (Inception v3)

$$ q'(k) = (1 - \epsilon) \delta_{k,y} + \frac{\epsilon}{K} $$
- Предотвращает overconfidence
- Улучшает обобщение

### Factorized Convolutions (Inception v3)

$$ \text{7×7} \rightarrow 3 \times \text{(1×7 + 7×1)} $$
$$ \text{3×3} \rightarrow \text{3×1 + 1×3} $$

Принцип: асимметричная факторизация уменьшает параметры и добавляет глубину.

---

## 5. Сравнительная таблица классических CNN

| Характеристика | LeNet-5 | AlexNet | VGG-16 | GoogLeNet (Inception v1) |
|----------------|---------|---------|--------|--------------------------|
| **Год** | 1998 | 2012 | 2014 | 2014 |
| **Глубина** | 5 слоёв | 8 слоёв | 16 слоёв | 22 слоя |
| **Параметры** | 60K | 60M | 138M | 7M |
| **Top-5 error** | — | 15.3% | 7.3% | 6.7% |
| **Активация** | Tanh/Sigmoid | ReLU | ReLU | ReLU |
| **Нормализация** | — | LRN | — | Batch Norm (v2+) |
| **Регуляризация** | — | Dropout | Dropout | Dropout + Label Smoothing |
| **DataAug** | — | Translation, flip, color | Scale, flip | Scale, flip, color |
| **Фильтры** | 5×5 | 11×5×3 | 3×3 | 1×1, 3×3, 5×5 |
| **Ключевая идея** | Conv → Pool | Глубина + ReLU | Глубина важнее | Разные масштабы |

---

## 6. Эволюция feature extraction

```
Hand-crafted (SIFT, HOG) → LeNet (end-to-end Conv) → 
AlexNet (ReLU + Dropout → глубже) → 
VGG (3×3 stacking → ещё глубже) → 
Inception (multiscale + 1×1 bottleneck → эффективней)
```

Подробнее об операции свёртки см. [[DL 10 - Операция свертки]], о базовой архитектуре CNN — [[DL 11 - Базовая CNN]].

### Основные тренды эволюции

1. **Глубина**: 5 → 8 → 16 → 22 слоя
2. **Размер фильтров**: 5×5 → 11×11 → 3×3 → 1×1 + 3×3 + 5×5
3. **Параметры**: 60K → 60M → 138M → 7M (Inception резко снизила)
4. **Эффективность**: больше качества за меньше параметров
5. **Аккуратность**: на ILSVRC от 25.8% (2011) до 3.6% (2015, human-level)

### Почему это важно
Эти архитектуры — **backbone-фундамент** для всех последующих задач CV: детекции, сегментации, генерации. Понимание их мотивации необходимо для современных методов ([[DL 14 - ResNet|ResNet]], EfficientNet, [[DL 51 - Vision Transformer|Vision Transformer]]).

---

## Ключевые выводы к экзамену

1. **LeNet** — первая CNN, паттерн Conv-Pool-FC
2. **AlexNet** — прорыв благодаря ReLU, Dropout, GPU
3. **VGG** — глубина и однородность, очень много параметров
4. **Inception** — multiscale + 1×1 bottleneck + BN + LS, эффективная
5. **Эволюция** → глубже, эффективнее, разнообразнее масштабы
6. Все архитектуры — основа pretrained backbones для [[DL 15 - Transfer Learning|transfer learning]]

---

**Связанные вопросы:**
- [[DL 10 - Операция свертки]] — операция свёртки и пулинг
- [[DL 11 - Базовая CNN]] — устройство базовой свёрточной сети
- [[DL 14 - ResNet]] — ResNet и эволюция backbone
- [[DL 51 - Vision Transformer]] — Vision Transformer как альтернатива CNN
