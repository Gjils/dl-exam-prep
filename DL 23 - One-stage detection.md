# 23. One-stage Object Detection

## YOLO, SSD, RetinaNet, anchors vs anchor-free, trade-off скорости и качества

---

## 1. Концепция One-stage детекции

### Идея
**Один прямой проход сети → сразу классы и bbox.** Без отдельного этапа генерации регионов.

Постановка задачи детекции — в [[DL 20 - Object Detection]]. Базовые метрики и NMS — в [[DL 21 - Метрики и NMS в детекции]].

### Отличие от Two-stage

| Аспект | Two-stage | One-stage |
|--------|-----------|-----------|
| **Pipeline** | RPN → RoI → Head | Конечная сеть → предсказания |
| **RoI / proposals** | ✅ Есть | ❌ Нет (или встроены) |
| **RoI Pooling** | ✅ | ❌ |
| **Скорость** | 🟡 10-20 FPS | 🚀 30-150+ FPS |
| **Точность** | ✅ Высокая | 🟡 Ранее ниже, сейчас ≈ |
| **Дисбаланс** | ✅ RoI сэмплинг (меньше негативов) | ❌ Все anchors (классовый дисбаланс) |

---

## 2. YOLO (You Only Look Once, Redmon et al., 2016)

### Ключевая идея
Детекция как **единая задача регрессии**: изображение → сетка + bbox + confidence + класс.

### Архитектура YOLOv1

```
Input (448×448)
    │
    ▼
GoogleNet-inspired CNN (24 conv + 2 FC)
    │
    ▼
Output: S × S × (B×5 + C)  tensor
Где:
  S = 7 (grid size)
  B = 2 (bounding boxes per cell)
  C = 20 (PASCAL classes)
  = 7×7×(2×5+20) = 7×7×30
```

### Как работает YOLOv1

1. **Изображение** делится на сетку $S \times S$ (7×7)
2. **Каждая ячейка** предсказывает:
   - $B$ bbox $(x, y, w, h, confidence)$
   - Распределение классов $p(c)$ для ячейки
3. **Условие**: если центр объекта попадает в ячейку — она отвечает за него

### Loss YOLOv1

$$ \mathcal{L} = \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 $$
$$ + \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2 $$
$$ + \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} (C_i - \hat{C}_i)^2 $$
$$ + \lambda_{noobj} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{noobj} (C_i - \hat{C}_i)^2 $$
$$ + \sum_{i=0}^{S^2} \mathbb{1}_{i}^{obj} \sum_{c \in classes} (p_i(c) - \hat{p}_i(c))^2 $$

Где $\lambda_{coord} = 5$, $\lambda_{noobj} = 0.5$

### Проблемы YOLOv1

| Проблема | Причина |
|----------|---------|
| **Грубая сетка** | 7×7 → маленькие объекты не видны |
| **2 bbox на ячейку** | Не хватает для перекрывающихся |
| **Квадратная ошибка** | IoU не оптимизируется напрямую |
| **Только 20 классов** | PASCAL-специфична |

### YOLO эволюция

| Версия | Год | Ключевые изменения |
|--------|-----|-------------------|
| **v1** | 2016 | Grid + regression |
| **v2** (YOLO9000) | 2017 | Anchor boxes, batch norm, joint COCO+ImageNet |
| **v3** | 2018 | FPN (Darknet-53), logistic cls, multi-label |
| **v4** | 2020 | CSPNet, MISH, BoF, BoS |
| **v5** | 2020 | PyTorch, удобство, auto-anchor |
| **v8** | 2023 | SOTA, anchor-free head |

---

## 3. SSD (Single Shot MultiBox Detector, Liu et al., 2016)

### Ключевая идея
**Мультимасштабные feature maps** для детекции объектов разного размера. Использование **default boxes** (anchors).

### Архитектура

```
Input (300×300)
    │
    ▼
VGG-16 (truncated, conv5 → 38×38×512)
    │
    ▼
Extra Feature Layers:
    ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐
    │Conv6  │   │Conv7  │   │Conv8  │   │Conv9  │   │Conv10 │
    │19×19× │   │10×10× │   │5×5×   │   │3×3×   │   │1×1×   │
    │1024   │   │512    │   │256    │   │256    │   │256    │
    └───┬───┘   └───┬───┘   └───┬───┘   └───┬───┘   └───┬───┘
        │           │           │           │           │
        ▼           ▼           ▼           ▼           ▼
    Detections  Detections  Detections  Detections  Detections
    (19×19)     (10×10)     (5×5)       (3×3)       (1×1)
```

### Default Boxes на каждом уровне

| Feature Map     | Размер | Шкала | **boxes/ячейка** | Всего boxes |
| --------------- | ------ | ----- | ---------------- | ----------- |
| conv4_3 (38×38) | 38×38  | 0.1   | 4                | 5776        |
| conv7 (19×19)   | 19×19  | 0.2   | 6                | 2166        |
| conv8_2 (10×10) | 10×10  | 0.375 | 6                | 600         |
| conv9_2 (5×5)   | 5×5    | 0.55  | 6                | 150         |
| conv10_2 (3×3)  | 3×3    | 0.725 | 4                | 36          |
| conv11_2 (1×1)  | 1×1    | 0.9   | 4                | 4           |
| **Всего**       |        |       |                  | **8732**    |

### Мультимасштабная пирамида

- **Ранние слои** (38×38): мелкие объекты
- **Поздние слои** (1×1): крупные объекты
- Каждый уровень отвечает за свой диапазон размеров

### Потеря SSD

$$ \mathcal{L} = \frac{1}{N} (\mathcal{L}_{conf} + \alpha \mathcal{L}_{loc}) $$

Где:
- $\mathcal{L}_{conf}$ — softmax cross-entropy (K+1 классов)
- $\mathcal{L}_{loc}$ — smooth L1 между default box и GT
- $\alpha = 1$
- $N$ — количество positive matches

### Hard Negative Mining
- Соотношение negative:positive = 3:1
- Выбираем negative с **наибольшим confidence loss**

### SSD vs YOLOv1

| Аспект | YOLOv1 | SSD |
|--------|--------|-----|
| **Anchors** | Нет (прямая регрессия) | ✅ Default boxes |
| **Multiscale** | Нет | ✅ Несколько feature maps |
| **Основа** | GoogleNet | VGG-16 |
| **Small objects** | ❌ Плохо | ✅ Лучше (38×38 map) |
| **mAP (VOC2007)** | 63.4 | 74.3 (SSD300) |

---

## 4. RetinaNet (Lin et al., 2017)

### Ключевая идея
**Focal Loss** решает проблему классового дисбаланса между foreground и background в one-stage детекции.

### Проблема: классовый дисбаланс

```
One-stage детектор (SSD, YOLO):
  ~100k anchors на изображение
  ~10-100 positive (объекты)
  ~99.9k negative (фон)
  
CE Loss: доминирование negative → модель учится "всё фон"
```

### Focal Loss (напоминание)

$$ \mathcal{L}_{focal} = -\alpha_t (1 - p_t)^\gamma \log(p_t) $$

Где:
- $p_t$ — вероятность правильного класса
- $\gamma = 2$ (focusing parameter)
- $\alpha_t = 0.25$ (class weight)

### Архитектура RetinaNet

```
Input
    │
    ▼
Backbone (ResNet-FPN)
    │
    ┌────────────────────────────────────────┐
    │   FPN: P3 ── P4 ── P5 ── P6 ── P7    │
    │   (stride: 8, 16, 32, 64, 128)       │
    └────────────────┬───────────────────────┘
                     │
    ┌────────────────┴────────────────┐
    ▼                                 ▼
┌──────────┐                    ┌──────────┐
│  Subnet  │                    │  Subnet  │
│  Class   │                    │  Box     │
│  Head    │                    │  Head    │
│          │                    │          │
│ 4× Conv  │                    │ 4× Conv  │
│ + ReLU   │                    │ + ReLU   │
│          │                    │          │
│ 1×1, K×A │                    │ 1×1, 4×A │
└──────────┘                    └──────────┘
```

**Особенности**:
- Субсеть классификации и регрессии — **раздельные**
- Каждая — 4 conv 3×3 + 1×1 output
- Только FPN (нет extra layers)

### Anchor параметры

- **3 scales**: $2^0, 2^{1/3}, 2^{2/3}$
- **3 aspect ratios**: 1:2, 1:1, 2:1
- **9 anchors per pixel**
- Каждый FPN уровень: своя базовая шкала

### Результаты RetinaNet

| Модель | AP (COCO) | AP@0.5 | AP@0.75 |
|--------|-----------|--------|---------|
| RetinaNet-50 | 36.0 | 54.6 | 37.8 |
| RetinaNet-101 | 39.1 | 59.0 | 42.3 |
| RetinaNet-101-800 | **40.8** | **61.7** | **44.5** |

**One-stage впервые догнал по точности two-stage!**

---

## 5. Сравнение One-stage детекторов

### Таблица

| Модель | Год | Backbone | Anchors | Multiscale | Потеря | AP (COCO) | FPS |
|--------|-----|----------|---------|------------|--------|-----------|-----|
| **YOLOv1** | 2016 | GoogLeNet | Нет | Нет | SE | — | 45 |
| **SSD300** | 2016 | VGG-16 | Default boxes | ✅ Extra layers | CE | 25.1 | 46 |
| **YOLOv3** | 2018 | Darknet-53 | ✅ | ✅ FPN | BCE | 33.0 | 78 |
| **RetinaNet** | 2017 | ResNet-50-FPN | ✅ | ✅ FPN | Focal | 36.0 | 16 |
| **YOLOv5** | 2020 | CSPDarknet | ✅ | ✅ | BCE+IoU | 37.4 | 140 |
| **FCOS** | 2019 | ResNet | ❌ AF | ✅ FPN | CE + centerness | 38.5 | 12 |
| **YOLOv8** | 2023 | CSPDarknet | ❌ AF | ✅ | DFL + CIoU | 44.7 | 280 |

### Trade-off скорости и качества

```
AP (COCO)
 45 │                        YOLOv8
 40 │                  FCOS      YOLOv5
 35 │       RetinaNet        YOLOv3
 30 │                    SSD
 25 │
 20 │   YOLOv1
 15 │
    └────────────────────────────── FPS
              50   100   150   200   250   300
```

### Рекомендации по выбору

| Сценарий | Рекомендация | Причина |
|----------|--------------|---------|
| **Реал-тайм (30+ FPS)** | YOLOv8-Nano/Small | Скорость |
| **Высокая точность** | YOLOv8-XL / RetinaNet | Точность |
| **Мобильные устройства** | YOLOv8-Nano / MobileNet-SSD | Ограниченные ресурсы |
| **Мелкие объекты** | RetinaNet / FCOS с FPN | Multiscale + no quant |
| **Edge device (Jetson)** | YOLOv5s | Оптимизация под CUDA |

---

## 6. Anchors vs Anchor-Free

### Что такое anchors?

Набор предопределённых bbox (размер + форма), размещённых в каждой точке feature map.

### Anchor-based детекция

**Преимущества:**
- Предопределённые размеры → легче регрессировать
- Проверено временем (YOLOv3-v5, SSD, RetinaNet, Faster R-CNN)
- Хорошо работает с FPN

**Недостатки:**
- **Hyperparameter tuning**: размеры anchors под датасет (K-means на GT)
- **Много anchors** → дисбаланс (99% negative)
- **Сложность**: anchor assignment, IoU computation
- **Негибкость**: плохо для нестандартных объектов

### Anchor-free детекция

**Подходы:**

1. **Keypoint-based** (CenterNet, CornerNet):
   - Детекция центров/углов объектов
   - Группировка в bbox

2. **Pixel-based** (FCOS, YOLOv1, YOLOv8):
   - Каждый пиксель → (class, bbox) напрямую
   - Никаких anchors

### FCOS (Fully Convolutional One-Stage, 2019)

$$ \text{prediction per pixel: } (c, t, l, b, r, \text{centerness}) $$

Где $(l, t, r, b)$ — расстояния от пикселя до четырёх сторон GT box.

**Centerness** (качество):
$$ \text{centerness} = \sqrt{\frac{\min(l,r)}{\max(l,r)} \cdot \frac{\min(t,b)}{\max(t,b)}} $$

Умножается на classification score → уменьшает вес периферийных пикселей.

### Сравнение

| Аспект | Anchor-based | Anchor-free |
|--------|-------------|-------------|
| **Параметры удалён** | Design anchors | NMS threshold |
| **Гиперпараметры** | Много (scale, ratio, IoU thresholds) | Мало |
| **Дисбаланс** | Сильный (все anchors) | Меньше (только пиксели внутри GT) |
| **Скорость** | Медленнее (больше предсказаний) | Быстрее |
| **Точность** | Baseline | ✅  (FCOS ~38.5 AP) |
| **Маленькие объекты** | Хорошо (много anchors) | Сложнее |
| **Надёжность** | Проверенная | Новее, но быстро догоняет |

**Современные тренды**:
- YOLOv8 → anchor-free
- DETR → end-to-end, без anchors и NMS
- Тренд: **меньше гиперпараметров, проще архитектура**

---

## 7. Современные one-stage детекторы

### YOLOv8 (Ultralytics, 2023)

- Anchor-free
- **Task-aligned head**: совместная классификация + регрессия
- **DFL (Distribution Focal Loss)**: предсказание распределения координат
- **CIoU loss**: IoU + расстояние центра + aspect ratio

### DETR (Detection Transformer, Carion et al., 2020)

- Трансформер без anchors/NMS
- **Object queries**: 100 learnable queries → class + bbox
- **Hungarian matching**: бипартитный матчинг предсказаний и GT
- **Проблемы**: медленная сходимость, плохо с мелкими объектами

### Deformable DETR

- Deformable attention (фокус на релевантных точках)
- Мультимасштабные features
- Быстрая сходимость (10× быстрее DETR)

---

## Ключевые выводы к экзамену

1. **One-stage**: один проход → класс + bbox. Быстрее, но было менее точным
2. **YOLOv1**: сетка 7×7 → классификация + регрессия. Пионер one-stage
3. **SSD**: мультимасштабные feature maps (multiscale) + default boxes (anchors)
4. **RetinaNet**: FPN + Focal Loss. Впервые one-stage догнал two-stage по точности
5. **Anchors vs Anchor-free**: anchors — гиперпараметры (discrete), anchor-free — проще (но нужно centerness)
6. **Trade-off**: YOLOv8 — скорость, RetinaNet — точность, FCOS — баланс
7. **Современный тренд**: anchor-free, transformer-based (DETR), меньше гиперпараметров

---

**Связанные вопросы:**
- [[DL 20 - Object Detection]] — постановка задачи детекции
- [[DL 21 - Метрики и NMS в детекции]] — метрики и пост-обработка
- [[DL 22 - Two-stage detection]] — двухстадийные детекторы
