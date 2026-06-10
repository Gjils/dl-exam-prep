# 22. Two-stage Object Detection

## Sliding window, R-CNN, Fast R-CNN, Faster R-CNN, Region Proposal Network, Cascade R-CNN

---

## 1. Общая идея Two-stage детекции

### Концепция
**Сначала — регионы интереса → потом — классификация и уточнение box.**

Постановка задачи детекции — в [[DL 20 - Object Detection]]. Базовые метрики и NMS — в [[DL 21 - Метрики и NMS в детекции]].
Отличие от одностадийных подходов — в [[DL 23 - One-stage detection]].

```
Stage 1:           Stage 2:
Region Proposals → RoI Classification + BBox Regression
(RPN/Selective    (класс объекта)
 Search)           (уточнение координат)
```

### Преимущества

| Аспект | Two-stage | Комментарий |
|--------|-----------|-------------|
| **Точность** | ✅ Выше | Два этапа уточнения |
| **Высокое качество локализации** | ✅ | RoI pooling + refine box |
| **Обработка масштабов** | ✅ | RoI на разных уровнях |
| **Скорость** | ❌ Медленнее | Два прохода |
| **End-to-end** | 🟡 | RPN + Fast R-CNN (Faster) |

### Эволюция

```
Sliding Window (медленно, неэффективно)
    → R-CNN (Selective Search + CNN, 2014)
    → Fast R-CNN (RoIPool + shared conv, 2015)
    → Faster R-CNN (RPN + end-to-end, 2015)
    → Cascade R-CNN (multi-stage refinement, 2018)
    → Mask R-CNN (instance seg + det, 2017)
```

---

## 2. Sliding Window Approach

### Идея
Перемещать окно фиксированного размера по изображению с шагом. Для каждого окна — классификация.

### Проблемы

1. **Огромная вычислительная сложность**: для изображения 1000×1000 и окна 64×64 с шагом 4 → 56k окон
2. **Разные масштабы**: нужно повторять для каждого размера окна (пирамида масштабов)
3. **Каждое окно → отдельный CNN forward pass** → невозможно в реальном времени
4. **Нет точной локализации**: шаг окна = квантование позиции

### Почему не используется
- На практике все современные методы используют **регионы** (selective search) или **anchors**

---

## 3. R-CNN (Region-based CNN, Girshick et al., 2014)

### Идея
1. **Selective Search** → ~2000 region proposals
2. **Warp** каждый регион к 227×227
3. **CNN** (AlexNet) → feature vector (4096)
4. **SVM** для каждого класса → классификация
5. **BBox regression** → уточнение box

### Pipeline

```
Input Image
    │
    ▼
Selective Search (~2000 regions)
    │
    ▼
Warp each region to 227×227
    │
    ▼
CNN (AlexNet) forward pass × 2000
    │
    ▼
SVM classifier (per-class) + BBox regressor
    │
    ▼
NMS → Final detections
```

### Selective Search

Алгоритм генерации регионов:
1. **Superpixel segmentation** (Felzenszwalb)
2. **Иерархическая группировка** похожих superpixels
3. Разные стратегии подобия: цвет, текстура, размер
4. Выход: ~2000 кандидатов

### Проблемы R-CNN

| Проблема | Причина | Следствие |
|----------|---------|-----------|
| **Медленно** | 2000 CNN forward passes | ~47 секунд на изображение |
| **Много диска** | Сохранённые фичи | ~200GB для PASCAL VOC |
| **Multi-stage** | CNN + SVM + BBox reg | Сложное обучение |
| **Warping** | Изменение пропорций | Искажение объектов |
| **Не энд-ту-энд** | Отдельные компоненты | Нет сквозной оптимизации |

---

## 4. Fast R-CNN (Girshick, 2015)

### Ключевая идея
**Один CNN forward pass** на всё изображение + **RoIPool** для извлечения фич для каждого региона.

### Архитектура

```
Input Image
    │
    ▼
CNN (VGG-16) ← один forward pass через всё изображение
    │
    ▼
Feature Map (H/16 × W/16 × 512)
    │         ▲
    │         │
    ▼         │
RoI Project   │ проекция Selective Search bbox на feature map
    │         │
    ▼         │
RoI Pooling (7×7) ← max pooling каждого региона в фиксированный размер
    │
    ▼
FC layers
    │
    └────────────┬────────────┐
                 ▼            ▼
      Softmax (K+1)      BBox Regressor (4×(K+1))
       class probs         box deltas
```

### RoI Pooling

1. Region proposal (например, [100, 50, 300, 400]) → проекция на feature map (÷16)
2. Деление проекции на H×W ячеек (например, 7×7)
3. Max pooling в каждой ячейке
4. Выход: фиксированный размер (7×7×512) — независимо от размера региона

**Проблема**: квантование при делении (округление) → потеря точности.

### Multi-task loss

$$ \mathcal{L} = \underbrace{\mathcal{L}_{cls}(p, u)}_{\text{log loss}} + \lambda \underbrace{[u \geq 1] \mathcal{L}_{loc}(t^u, v)}_{\text{smooth L1}} $$

Где:
- $p$ — предсказанные вероятности классов
- $u$ — GT класс
- $t^u$ — предсказанные deltas для класса u
- $v$ — GT deltas
- $\lambda$ = 1 (баланс задач)

### Smooth L1 Loss

$$ \text{smooth}_{L1}(x) = \begin{cases} 0.5x^2 & \text{if } |x| < 1 \\ |x| - 0.5 & \text{otherwise} \end{cases} $$

- Менее чувствительна к выбросам, чем L2
- Более гладкая, чем L1

### BBox Regression (преобразование)

$$ t_x = (G_x - P_x) / P_w, \quad t_y = (G_y - P_y) / P_h $$
$$ t_w = \log(G_w / P_w), \quad t_h = \log(G_h / P_h) $$

Где $P$ — proposal box, $G$ — ground truth box.

### Улучшения vs R-CNN

| Характеристика | R-CNN | Fast R-CNN |
|----------------|-------|------------|
| **Время train** | 84 часа | 9.5 часов |
| **Время test** | 47 сек/изображение | 0.3 сек/изображение |
| **mAP (VOC2007)** | 66.0% | 66.9% |
| **Диск** | 200GB | Меньше (нет сохранения фич) |
| **Обучение** | Multi-stage | Single-stage (но SS отдельно) |

---

## 5. Faster R-CNN (Ren et al., 2015)

### Ключевая идея
**Заменить Selective Search на Region Proposal Network (RPN)**. Всё стало end-to-end.

### Архитектура

```
Input Image
    │
    ▼
Backbone CNN (VGG/ResNet)
    │
    ▼
Feature Map (H/16 × W/16 × 512)
    │
    ┌────────────────────────────────────┐
    │                                    │
    ▼                                    ▼
RPN (Region Proposal Network)        Fast R-CNN Head
    │                                    │
    ├── Anchors (9 per location)         │
    ├── Objectness score (fg/bg)         │
    └── BBox refinement (Δx, Δy, Δw, Δh)│
    │                                    │
    ▼ (proposals)                        │
    RoI Pooling ◄────────────────────────┘
    │
    ▼
FC → Class + BBox
```

### Region Proposal Network (RPN)

#### Anchors

В каждой точке feature map:
- **3 масштаба**: 128², 256², 512²
- **3 соотношения**: 1:1, 1:2, 2:1
- **9 anchors** на точку

$$ \text{Всего anchors: } \frac{H}{16} \times \frac{W}{16} \times 9 $$

Для изображения 1000×600: ~20k anchors.

#### RPN Head

```
Feature Map (H/16 × W/16 × 512)
    │
    ▼
3×3 Conv, 512
    │
    ┌─────────────┬─────────────┐
    │             │             │
    ▼             ▼             ▼
1×1 Conv, 18   1×1 Conv, 36   2×(1×1 Conv, 18+36)
(cls scores)   (reg deltas)
(sigmoid)      (linear)
```

- 18 = 2 (object/not) × 9 anchors
- 36 = 4 (Δ) × 9 anchors

#### Anchor Assignment

| IoU с GT | Label | Действие |
|----------|-------|----------|
| > 0.7 | Positive | Обучаем как объект |
| < 0.3 | Negative | Обучаем как фон |
| 0.3-0.7 | Ignore | Не участвует в loss |
| Highest IoU (каждый GT) | Positive | Как минимум один anchor на объект |

#### RPN Loss

$$ \mathcal{L}_{RPN} = \frac{1}{N_{cls}}\sum_i \mathcal{L}_{cls}(p_i, p_i^*) + \lambda \frac{1}{N_{reg}}\sum_i p_i^* \cdot \mathcal{L}_{reg}(t_i, t_i^*) $$

- $\lambda = 10$ (баланс cls/reg)
- $p_i^* = 1$ для positive anchor, 0 для negative
- Только positive anchors участвуют в reg loss

#### NMS в RPN
После RPN:
1. Отсортировать по objectness score
2. NMS (IoU threshold = 0.7)
3. Взять top-N (обычно 2000 train, 300 test)

### 4-Step Alternating Training

1. Train RPN (ImageNet pretrained)
2. Train Fast R-CNN (используя RPN proposals)
3. Fine-tune RPN (используя Fast R-CNN веса)
4. Fine-tune Fast R-CNN (общие свёртки)

**Современный подход**: совместное обучение (joint training) с fixed anchors.

### Результаты

| Модель | mAP (VOC2007) | FPS |
|--------|--------------|-----|
| R-CNN | 66.0% | 0.02 |
| Fast R-CNN | 66.9% | 0.5 |
| Faster R-CNN (VGG) | 69.9% | 7 |
| Faster R-CNN (ZF) | 62.1% | 18 |

---

## 6. Cascade R-CNN (Cai & Vasconcelos, 2018)

### Проблема Faster R-CNN
**Overfitting на пороге IoU**:
- Если RPN обучается с IoU=0.5 → для качественных proposals (IoU > 0.7) детектор работает хуже
- Невозможно обучить детектор на высоком пороге IoU — остаётся слишком мало positive

### Идея
**Каскад детекторов** с последовательно возрастающим порогом IoU:

```
Stage 1: IoU=0.5        Stage 2: IoU=0.6        Stage 3: IoU=0.7
Proposals ───► Detector ───► Refined ───► Detector ───► Refined ───► ...
                  │                       │
             R-CNN Head              R-CNN Head
```

### Архитектура

```
Input → Backbone → Feature Map
    │
    ├── RPN → Proposals
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Cascade Stages                                         │
│                                                          │
│  Stage 1: RoI Align → Head (IoU=0.5) → outputs
│       └───────────────────────────────────┘              │
│  Stage 2: RoI Align → Head (IoU=0.6) → outputs
│       └───────────────────────────────────┘              │
│  Stage 3: RoI Align → Head (IoU=0.7) → outputs
│       └───────────────────────────────────┘              │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Final detections (из Stage 3)
```

### Каскадная архитектура

На каждом этапе:
- **RoI Align** взятие признаков по refined box
- **Head**: FC → classification + bbox regression
- **Loss**: своя для каждого этапа

**Важно**: каждый этап — свой классификатор и регрессор (не shared веса).

### Распределение positive anchors

```
Один детектор (IoU=0.5):
  Positives: хорошо для IoU[0.5, 0.6], плохо для IoU[0.7+]
  
Каскад:
  Stage 1 (IoU=0.5): много positives, но грубые
  Stage 2 (IoU=0.6): вход = качественные proposals → positives
  Stage 3 (IoU=0.7): вход = ещё более качественные → positives
```

### Результаты (на COCO)

| Модель | AP | AP@0.5 | AP@0.75 |
|--------|-----|--------|---------|
| Faster R-CNN (Res-50) | 36.4 | 58.4 | 39.1 |
| Cascade R-CNN (Res-50) | **40.3** | **58.6** | **44.4** |
| Faster R-CNN (Res-101) | 38.4 | 60.0 | 41.6 |
| Cascade R-CNN (Res-101) | **42.0** | **60.2** | **46.0** |

---

## 7. Сравнительная таблица Two-stage

| Модель | Год | Region Gen | RoI Method | End-to-end | mAP (VOC) | Speed |
|--------|-----|-----------|------------|------------|-----------|-------|
| **R-CNN** | 2014 | Selective Search | Warp + CNN | ❌ | 66.0% | ❌ 47s |
| **SPPNet** | 2014 | SS | Spatial Pyramid | ❌ | — | 🟡 |
| **Fast R-CNN** | 2015 | SS | RoI Pooling | ❌ (SS) | 66.9% | ✅ 0.3s |
| **Faster R-CNN** | 2015 | RPN | RoI Pooling | ✅ | 69.9% | ✅ 0.14s |
| **Cascade R-CNN** | 2018 | RPN | RoI Align | ✅ | 73.8%* | 🟡 |
| **Mask R-CNN** | 2017 | RPN | RoI Align | ✅ | 73.3%* | ✅ |

*на COCO, остальные на PASCAL VOC

---

## 8. Компоненты современного Two-stage детектора

### Backbone
- ResNet-50/101 (классика)
- ResNeXt (grouped convolutions)
- Swin Transformer (SOTA)
- ConvNeXt (modern ConvNet)

### Neck
- **FPN** (Feature Pyramid Network): multiscale features
- **PAFPN** (Path Aggregation FPN): bottom-up + top-down

### RPN improvements
- **GA-RPN**: Guided Anchoring (learning anchor shapes)
- **Sparse R-CNN**: learned proposals instead of RPN

### RoI extraction
- **RoI Pooling**: Fast R-CNN (квантование)
- **RoI Align**: Mask R-CNN (билинейная интерполяция)
- **PrRoI Pooling**: точное интегрирование

---

## Ключевые выводы к экзамену

1. **Двухстадийная детекция**: сначала регионы → потом классификация и регрессия
2. **R-CNN**: Selective Search + отдельный CNN для каждого региона (2000×). Медленно
3. **Fast R-CNN**: один CNN → RoI Pooling → FC head. Совместное обучение cls + reg
4. **Faster R-CNN**: **RPN** (Region Proposal Network) заменила Selective Search. 9 anchors на точку. End-to-end
5. **Cascade R-CNN**: каскад детекторов с возрастающим порогом IoU (0.5 → 0.6 → 0.7). Каждый этап — свой head
6. **Эволюция**: SS → RPN, отдельные CNN → shared conv, multi-stage → end-to-end

---

**Связанные вопросы:**
- [[DL 20 - Object Detection]] — постановка задачи детекции
- [[DL 21 - Метрики и NMS в детекции]] — метрики и пост-обработка
- [[DL 23 - One-stage detection]] — одностадийные детекторы (YOLO, SSD, RetinaNet)
