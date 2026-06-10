# 17. U-Net и современные подходы

## Skip connections, upsampling, multiscale features, SAM, Mask2Former, transformer/promptable segmentation

---

## 1. U-Net (Ronneberger et al., 2015)

### Мотивация
- Медицинские изображения: **небольшое количество данных**, но нужна **высокая точность**
- Объекты имеют чёткие границы, которые теряются при downsampling
- Нужна архитектура, которая эффективно использует **контекст** и **точные границы**
- U-Net решает задачу [[DL 16 - Семантическая сегментация|семантической сегментации]] с архитектурой encoder-decoder

### Архитектура

```
Input (H × W × C)
    │
    ▼
┌─────────────────────────────────────────────────────┐
│   ENCODER (contracting path)                        │
│                                                      │
│  Conv3×3, ReLU → Conv3×3, ReLU → MaxPool 2×2        │
│     1 → 64 chs                64 → 64 chs           │
│                                                      │
│  Conv3×3, ReLU → Conv3×3, ReLU → MaxPool 2×2        │
│     64 → 128 chs              128 → 128 chs         │
│                                                      │
│  Conv3×3, ReLU → Conv3×3, ReLU → MaxPool 2×2        │
│     128 → 256 chs             256 → 256 chs         │
│                                                      │
│  Conv3×3, ReLU → Conv3×3, ReLU → MaxPool 2×2        │
│     256 → 512 chs             512 → 512 chs         │
│                                                      │
└──────────────────────────┬───────────────────────────┘
                           │
                    Bottleneck: 1024 chs
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│   DECODER (expanding path)                          │
│                                                      │
│  UpConv 2×2 (↑) → Concat(encoder) → Conv3×3 ×2     │
│     1024 → 512 chs       512+512=1024 → 512 chs    │
│                                                      │
│  UpConv 2×2 (↑) → Concat(encoder) → Conv3×3 ×2     │
│     512 → 256 chs       256+256=512 → 256 chs      │
│                                                      │
│  UpConv 2×2 (↑) → Concat(encoder) → Conv3×3 ×2     │
│     256 → 128 chs       128+128=256 → 128 chs      │
│                                                      │
│  UpConv 2×2 (↑) → Concat(encoder) → Conv3×3 ×2     │
│     128 → 64 chs        64+64=128 → 64 chs          │
│                                                      │
└─────────────────────────────────────────────────────┘
    │
    ▼
1×1 Conv (64 → K)
    │
    ▼
Segmentation Map (H × W × K)
```

### Ключевые компоненты

#### A. Contracting path (encoder)
- 4 блока **Conv 3×3 → BN → ReLU → Conv 3×3 → BN → ReLU → MaxPool 2×2**
- Каналы: 64 → 128 → 256 → 512
- Размер: H → H/2 → H/4 → H/8 → H/16 (128×128 → 8×8 для входа 256×256)

#### B. Bottleneck
- 2× Conv 3×3, 512 → 1024 каналов (наименьшее пространственное разрешение, наибольшая семантика)

#### C. Expanding path (decoder)
- **Up-convolution** 2×2 (transposed conv): уменьшает каналы вдвое, удваивает размер
- **Skip connection**: concatenation с соответствующим encoder feature map
- 2× Conv 3×3: обработка конкатенированных признаков
- Каналы: 1024 → 512 → 256 → 128 → 64

#### D. Output
- 1×1 Conv: 64 → K (количество классов)
- H × W выход (соответствует входному)

### Skip connections в U-Net

**Отличие от FCN**: в U-Net используется **concatenation** (не sum), и передаются **все уровни encoder**.

**Почему concat, а не add?**
- Encoder признаки содержат **детали границ** (high-res)
- Decoder признаки содержат **семантику** (low-res)
- Concat сохраняет **оба типа информации**
- Сеть сама учится комбинировать

### Upsampling методы в U-Net

| Метод | Параметры | Артефакты | Использование |
|-------|-----------|-----------|---------------|
| **Transposed Conv** | Обучаемые | Шахматный паттерн | Оригинальный U-Net |
| **Bilinear up + Conv** | Минимум | Нет | Современные варианты |
| **Nearest up + Conv** | Минимум | Блочность | Скоростные варианты |
| **PixelShuffle** | Обучаемые | Нет | Efficient variants |

### Аугментация в U-Net
Критична для малых датасетов:
- Elastic deformations (медицинская специфика)
- Random rotation, scaling, flipping
- Intensity shift (для микроскопии)
- CutMix, MixUp

### Результаты U-Net
- ISBI Cell Tracking Challenge 2015 — **победа**
- Обучается на **~30 изображениях**
- IoU на клетках: ~0.92

---

## 2. U-Net++ (Nested U-Net)

### Идея
Добавить **плотные skip connections** между уровнями encoder и decoder.

### Архитектура
```
Encoder:    E1 ──── E2 ──── E3 ──── E4 ──── E5
               ╲  ╱  ╲  ╱  ╲  ╱  ╲  ╱
Decoder:        X11 ── X12 ── X13 ── X14
                 ╲    ╱  ╲    ╱  ╲
                  X21 ── X22 ── X23
                   ╲    ╱    ╲
                    X31 ── X32
                     ╲    ╱
                      X41
```

- Каждый X(i,j) получает вход от предыдущего уровня decoder (upsampled) + encoder (skip)
- Уменьшает семантический разрыв между encoder и decoder
- Nested dense connections → лучше градиентный поток

### Deep Supervision
- Каждый выход decoder уровня может быть использован для loss
- Общий loss — взвешенная сумма всех выходов
- На inference: только финальный выход (глубочайший уровень)

---

## 3. Attention U-Net

### Идея
Добавить **attention gates** перед skip connections, чтобы decoder фокусировался на релевантных регионах.

### Attention Gate

$$ \alpha = \sigma(\psi^T(\sigma(W_x^T x_i + W_g^T g_i + b_g)) + b_\psi) $$

Где:
- $x_i$ — encoder feature map (skip connection)
- $g_i$ — gating signal из decoder (грубый контекст)
- $\alpha$ — attention coefficient (0-1)

**Эффект**: подавление нерелевантных областей (фон, шум) в skip connection.

---

## 4. DeepLab серия (Atrous/ASPP)

### Atrous (Dilated) Convolution

$$ y[i] = \sum_{k} x[i + r \cdot k] \cdot w[k] $$

- $r$ — rate (шаг между пикселями ядра)
- **Плюс**: рецептивное поле растёт без потери разрешения
- **Минус**: grid artifacts при больших rate

### ASPP (Atrous Spatial Pyramid Pooling)

```
Input feature map
    │
    ├── 1×1 Conv
    ├── 3×3 Conv, rate=6
    ├── 3×3 Conv, rate=12
    ├── 3×3 Conv, rate=18
    └── Image Pooling (GAP → 1×1 → upsample)
    │
    ▼
    Concatenation → 1×1 Conv → Output
```

- **Multiscale features** через разные dilation rates
- Image Pooling — глобальный контекст

### DeepLab v3+ (2018)

Лучшая версия: encoder-decoder + ASPP:
- **Encoder**: backbone (ResNet) + ASPP (multiscale)
- **Decoder**: простой upsampling + skip connections (encoder features до ASPP)
- Ключ: **ASPP** + **Encoder-Decoder**

---

## 5. Transformer-based segmentation

### SETR (SEgmentation TRansformer, 2021)

**Идея**: заменить CNN backbone на ViT (Vision Transformer):
- Изображение → patch embedding (16×16)
- Transformer encoder (12-24 layers)
- Progressive upsampling decoder

**Минус**: нет индуктивного смещения (locality), нужно много данных.

### Segmenter (2021)

- ViT encoder + Mask Transformer decoder
- Class embedding + patch embeddings → mask prediction
- Mask Transformer: cross-attention между class tokens и patch tokens

### Mask2Former (2022)

**Универсальный фреймворк** для семантической, instance и panoptic сегментации.

### Ключевые компоненты Mask2Former

1. **Pixel Decoder**: backbone + multiscale features (FPN-like)
2. **Transformer Decoder with Masked Attention**:
   - Каждый query attends **только в пределах предсказанной маски** (masked attention)
   - Значительно эффективнее обычного cross-attention
3. **Декодирование маски**:
   - Каждый query → mask embedding + class prediction
   - Mask: dot product mask embedding × pixel features → sigmoid → mask

### Masked Attention

$$ \text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}} + M\right)V $$

Где $M$ — маска внимания (0 в границах предсказанной маски, $-\infty$ вне).

**Преимущество**: линейная сложность по пространственному разрешению (вместо квадратичной).

---

## 6. SAM (Segment Anything Model, Meta 2023)

### Идея
**Промпт-управляемая сегментация**: можно указать точку, bounding box, или грубую маску — модель сегментирует объект.

### Архитектура SAM

```
┌──────────┐    ┌──────────┐    ┌──────────┐
│ Image    │───▶│ Image    │───▶│ Mask     │
│ Encoder  │    │ Encoder  │    │ Decoder  │
│          │    │          │    │          │
│ (ViT-H)  │    │ (ViT)   │    │ (light)  │
└──────────┘    └──────────┘    └──────────┘
                     ▲               │
                     │               ▼
                ┌──────────┐    ┌──────────┐
                │ Prompt   │    │ Valid    │
                │ Encoder  │    │ Masks    │
                └──────────┘    └──────────┘
```

### Компоненты

1. **Image Encoder**: ViT-H (массивный, 632M параметров)
2. **Prompt Encoder**: sparse (точки, bbox → positional encoding) + dense (маска → conv)
3. **Mask Decoder**: лёгкий transformer decoder
   - Token → cross-attention с image features
   - Декодирование нескольких масок (по количеству объектов)

### Тренировка на SA-1B
- Датасет: **1 миллиард масок** на 11 миллионах изображений
- Полуавтоматический сбор: SAM сам себе генерирует маски + human validation
- **Data engine**: 3 цикла (manual → semi-automatic → fully automatic)

### Возможности SAM

| Промпт | Пример | Применение |
|--------|--------|------------|
| Point | Клик по объекту | Interative segmentation |
| Box | Bounding box | Аккуратная сегментация |
| Coarse mask | Грубая маска | Refinement |
| Auto (grid) | Регулярная сетка | All objects |
| Text (SAM 2, Grounding SAM) | "dog" | Open-vocabulary |

### SAM vs U-Net

| Характеристика | U-Net | SAM |
|----------------|-------|-----|
| **Data** | 30-100 изображений | 11M изображений |
| **Параметры** | 7M-30M | 632M (ViT-H) |
| **Инференс** | Быстро (любой GPU) | Тяжело (необходим GPU 16GB+) |
| **Гибкость** | Один датасет | Zero-shot, любой объект |
| **Промпты** | Нет | Point, box, mask, auto |
| **Контролируемость** | Обучается под задачу | Не обучается под задачу |

---

## 7. Promptable segmentation

### Концепция
Пользователь задаёт **промпт** (точка, бокс, текст, полоска), модель сегментирует объект.

### Методы

| Метод | Промпт | Архитектура |
|-------|--------|-------------|
| **SAM** | Точка, бокс, маска | ViT + prompt encoder |
| **SEEM** (2023) | Текст, аудио, изображение | Unified prompt encoder |
| **Grounding DINO + SAM** | Текст | Detection → SAM |
| **PerSAM** | 1-shot reference | Personalize SAM |

### Promptable vs Interactive

- **Promptable**: однократный запрос → маска
- **Interactive**: серия уточнений (click correct → refine)

---

## 8. Сравнительная таблица

| Модель | Год | Backbone | Upsampling | Skip | Multiscale | Данные | Параметры |
|--------|-----|----------|-----------|------|-----------|--------|-----------|
| **U-Net** | 2015 | CNN | Transposed conv | Concat all | — | Мало | 30M |
| **U-Net++** | 2018 | CNN | Transposed conv | Nested | — | Мало | 36M |
| **DeepLab v3+** | 2018 | ResNet/EffNet | Bilinear + conv | ASPP | ASPP | Много | 60M |
| **SETR** | 2021 | ViT | Progressive up | — | — | Много | 300M |
| **Mask2Former** | 2022 | Swin | Pixel decoder | FPN | Masked attn | Много | 200M |
| **SAM** | 2023 | ViT-H | Transformer decoder | — | — | 1B masks | 632M |

---

## Ключевые выводы к экзамену

1. **U-Net**: encoder-decoder + concat skip connections. Золотой стандарт для малых датасетов.
2. **Skip connections**: concat в U-Net (vs add в ResNet), передача деталей границ
3. **Upsampling**: transposed conv (оригинал), современные — bilinear + conv
4. **Multiscale**: ASPP (DeepLab), FPN, Masked Attention (Mask2Former)
5. **SAM**: ViT-based promptable segmentation. Zero-shot. SA-1B — 1B масок
6. **Mask2Former**: универсальная архитектура (semantic + instance + panoptic), masked attention
7. **Тренд**: от U-Net к transformers и promptable моделям, но U-Net остаётся стандартом для малых данных

U-Net также используется как основа для диффузионных моделей — [[DL 58 - Диффузионные модели]].

---

**Связанные вопросы:**
- [[DL 16 - Семантическая сегментация]] — постановка задачи сегментации
- [[DL 18 - Instance, Panoptic, Pose Estimation]] — instance-сегментация и Mask R-CNN
- [[DL 58 - Диффузионные модели]] — U-Net как backbone в диффузии
