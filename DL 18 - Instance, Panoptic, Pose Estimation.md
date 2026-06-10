# 18. Instance, Panoptic Segmentation и Human Pose Estimation

## Отличия от semantic, Mask R-CNN, pose estimation (2D/3D)

---

## 1. Семантическая сегментация (напоминание)

### Определение
Каждому пикселю — метка класса:

$$ \mathcal{S}(i,j) \in \{1,\ldots,K\} $$

- Все пиксели **одного класса** — неразличимы
- Две машины рядом → одна маска

---

## 2. Instance Segmentation

### Определение
Каждому пикселю — **метка класса + номер инстанса**:

$$ \mathcal{I}(i,j) \in \{1,\ldots,K\} \times \mathbb{Z}^+ $$

- Если два объекта одного класса — **разные маски**
- Выход: набор масок $\{m_1, \ldots, m_N\}$, каждая со своим классом

### Пример
```
Изображение: две машины + пешеход
Semantic:    [car_mask, pedestrian_mask]
Instance:    [car_1_mask, car_2_mask, pedestrian_mask]
```

### Подходы

1. **Detection-based (top-down)**:
   - Сначала детекция bounding boxes → потом сегментация внутри box
   - Mask R-CNN, YOLACT, SOLO

2. **Segmentation-based (bottom-up)**:
   - Сначала попиксельный embedding → clustering инстансов
   - Associative Embedding, Deep Watershed Transform

3. **Transformer-based**:
   - Объектные запросы → маски напрямую
   - Mask2Former, DETR-based

---

## 3. Panoptic Segmentation

### Определение (Kirillov et al., 2019)
Объединение **semantic** и **instance** segmentation:

| Категория | Пример | Задача |
|-----------|--------|--------|
| **Stuff** | Небо, дорога, стена | Semantic (нет инстансов) |
| **Things** | Люди, машины, животные | Instance (есть инстансы) |

Каждый пиксель получает:
- Класс
- ID инстанса (для things) или 0 (для stuff)

### Формальное определение

$$ \mathcal{P}(i,j) \in \{ (k, id) \mid k \in \{1,\ldots,K\}, id \in \{0\} \cup \mathbb{Z}^+ \} $$

- Stuff: $id = 0$
- Things: $id > 0$, уникальный для каждого инстанса данного класса

### Отличия от semantic + instance

| Задача | Stuff | Things |
|--------|-------|--------|
| Semantic | ✅ Маска класса | ✅ Общая маска класса |
| Instance | ❌ Игнорируется | ✅ Отдельные маски |
| Semantic + Instance | ✅ + ❌ (нет связи) | ✅ |
| **Panoptic** | ✅ | ✅ + общая карта |

### Метрика: PQ (Panoptic Quality)

$$ \text{PQ} = \underbrace{\frac{|TP|}{|TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN|}}_{\text{Detection Quality}} \times \underbrace{\frac{\sum_{(p,g)} \text{IoU}(p,g)}{|TP|}}_{\text{Segmentation Quality}} $$

- TP: IoU > 0.5
- FP: лишняя маска
- FN: пропущенная маска

---

## 4. Сравнение задач

| Характеристика | Semantic Seg | Instance Seg | Panoptic Seg |
|----------------|-------------|--------------|--------------|
| **Выход** | Маска класса | Набор масок | Карта (класс+id) |
| **Разделение инстансов** | ❌ | ✅ (things) | ✅ (things) |
| **Stuff** | ✅ | ❌ | ✅ |
| **Метрика** | mIoU | AP | PQ |
| **Сложность** | Средняя | Высокая | Высокая |

---

## 5. Mask R-CNN (He et al., 2017)

### Идея
**Faster R-CNN + FCN** в одной сети. Добавляем **третью голову** (mask branch) к Faster R-CNN (детекция + классификация).

### Архитектура

```
Input Image
    │
    ▼
Backbone (ResNet-FPN)
    │
    ▼
RPN (Region Proposal Network) ── RoI features (H×W×256)
    │
    ▼
RoI Align (исправляет квантование RoIPool)
    │
    └────────────┬────────────┐
                 ▼            ▼
           Box Head       Mask Head
        (class + bbox)   (FCN per RoI)
              │              │
              ▼              ▼
         [K×4 boxes]    [K×28×28 masks]
```

### RoI Align (ключевое нововведение)

**Проблема RoIPool**: квантование координат (округление → потеря информации).

**RoI Align**:
- Билинейная интерполяция вместо округления
- Каждая ячейка RoI: 4 точки выборки
- Значение = среднее 4 точек
- Без квантования → маски на пиксельном уровне точности

$$ \text{RoIAlign}(f, x) = \sum_{i,j} f(i,j) \cdot \max(0, 1 - |x - i|) \cdot \max(0, 1 - |y - j|) $$

### Mask Head

```
RoI features (14×14×256)
    │
    ▼
4× Conv 3×3, 256, ReLU
    │
    ▼
ConvTranspose 2×2, 256
    │
    ▼
1×1 Conv, K (sigmoid)
    │
    ▼
K masks (28×28), каждая бинарная
```

**Важно**:
- Mask head не конкурирует за классы (class-agnostic до вывода)
- Каждый RoI → K масок (по одной на класс)
- Loss считается **только** для ground truth класса
- Inference: class prediction → берём маску соответствующего класса

### Multi-task loss

$$ \mathcal{L} = \mathcal{L}_{RPN} + \lambda_1 \mathcal{L}_{cls} + \lambda_2 \mathcal{L}_{box} + \lambda_3 \mathcal{L}_{mask} $$

$$\mathcal{L}_{mask} = \frac{1}{m^2} \sum_{i,j} \text{BCE}(y_{i,j}, \hat{y}_{i,j})$$

- $\lambda_1 = \lambda_2 = 1$, $\lambda_3 = 1$ (обычно)
- Маска — **бинарная кросс-энтропия**, не softmax (каждая маска независима)

### Feature Pyramid Network (FPN) в Mask R-CNN

```
      P2 ──── P3 ──── P4 ──── P5 ──── P6
      │      │      │      │      │
      │      │      │      │      │
    C2 ◄── C3 ◄── C4 ◄── C5 ◄──
```


- $C_i$ — выходы ResNet stage
- $P_i$ — пирамидальные признаки (1×1 Conv + up + lateral)
- Разные RoI привязаны к разным уровням:
  $$ k = \left\lfloor k_0 + \log_2\left(\frac{\sqrt{wh}}{224}\right) \right\rfloor $$

### Результаты Mask R-CNN

| Датасет | Box AP | Mask AP |
|---------|--------|---------|
| COCO (Res-50-FPN) | 38.2 | 35.4 |
| COCO (Res-101-FPN) | 40.0 | 36.1 |
| COCO (ResNeXt-101) | 42.0 | 37.9 |

---

## 6. Эволюция Instance Segmentation

| Модель | Год | Идея |
|--------|-----|------|
| **FCIS** | 2017 | Instance-sensitive feature maps |
| **Mask R-CNN** | 2017 | RoIAlign + FPN + mask head |
| **YOLACT** | 2019 | Real-time, prototype masks |
| **SOLO v2** | 2020 | Fully convolutional, no RoI |
| **Mask2Former** | 2022 | Universal, transformer |
| **SAM** | 2023 | Promptable, anything |

---

## 7. Human Pose Estimation

### Постановка задачи
**Локализация keypoints (суставов) человеческого тела**.

- **2D pose**: $(x,y)$ координаты ключевых точек на изображении
- **3D pose**: $(x,y,z)$ в трёхмерном пространстве

### Стандартные скелеты

| Датасет | Количество ключевых точек |
|---------|--------------------------|
| **COCO** | 17 (мультиперсонный) |
| **MPII** | 16 (одиночный/мульти) |
| **Body 25** (OpenPose) | 25 (руки+ноги+лицо) |
| **Human3.6M** | 17 (3D) |

### COCO 17 keypoints:
```
Нос, ЛевыйГлаз, ПравыйГлаз, ЛевоеУхо, ПравоеУхо,
ЛевоПлечо, ПравоПлечо, ЛевЛокоть, ПравЛокоть,
ЛевЗапястье, ПравЗапястье, ЛевБедро, ПравБедро,
ЛевКолено, ПравКолено, ЛевЛодыжка, ПравЛодыжка
```

---

## 8. 2D Pose Estimation

### Подходы

#### A. Top-down (детекция → ключевые точки)
1. Детекция человека (bbox)
2. Crop человека
3. Предсказание keypoints в bbox
4. Scale back к изображению

**Модели**: SimpleBaseline, HRNet, ViTPose

#### B. Bottom-up (ключевые точки → группировка в людей)
1. Предсказать **все** keypoints на изображении
2. **Associative Embedding**: каждый keypoint + embedding
3. Группировка keypoints по близости embedding

**Модели**: OpenPose, Associative Embedding, HigherHRNet

### Сравнение

| Характеристика | Top-down | Bottom-up |
|----------------|----------|-----------|
| **Точность** | ✅ Выше | Ниже |
| **Скорость** | Медленнее (N×) | ✅ O(1) |
| **Зависимость от детекции** | ✅ Да (бутылочное горлышко) | Нет |
| **Толпа/перекрытие** | Лучше (crop) | Хуже |
| **Масштабирование на N людей** | $O(N)$ | $O(1)$ |

### HRNet (High-Resolution Network)

**Ключевая идея**: сохранять **высокое разрешение** на всём протяжении сети.

```
┌──────────────────┐
│  Stem            │
└────┬─────────────┘
     │
    Parallel multi-resolution convolutions
     │
┌────▼────┐  ┌────▼────┐  ┌────▼────┐
│ Stream  │──│ Stream  │──│ Stream  │  ...
│ ×1      │  │ ×1/2    │  │ ×1/4    │
└─────────┘  └─────────┘  └─────────┘
     │  ▲         │  ▲          │
     └──┼─────────┘  │          │      (exchange blocks)
        └────────────┼──────────┘
                     │
Heatmaps: (H×W×K)
```

- 4 параллельных разрешения (×1, ×2, ×4, ×8 down)
- **Exchange blocks**: обмен информацией между разрешениями
- Выход: heatmaps для каждой keypoint

#### Loss в 2D pose

$$ \mathcal{L} = \sum_{k=1}^{K} \sum_{i,j} \left\| H_k(i,j) - G_k(i,j) \right\|^2 $$

Где $G_k$ — Gaussian heatmap (ground truth keypoint ± σ)

#### Метрики

| Метрика | Формула | Описание |
|---------|---------|----------|
| **PCK** | $PCK@\alpha = \frac{1}{N}\sum [\|p_i - \hat{p}_i\| < \alpha \cdot d]$ | % correct keypoints (PCKh — по голове) |
| **OKS** | $OKS = \frac{\sum_i \exp(-d_i^2 / 2s^2 k_i^2)}{\sum_i 1}$ | Object Keypoint Similarity |
| **AP** | Average Precision по OKS | COCO standard |

---

## 9. 3D Pose Estimation

### Постановка
По изображению предсказать $(x,y,z)$ координаты суставов в 3D.

### Сложность
- **Depth ambiguity**: одна и та же 2D проекция → бесконечно много 3D поз
- **Occlusion**: закрытые суставы
- **Любая деформация тела**, одежда

### Подходы

#### A. Direct 3D regression (изображение → 3D поза)
- 3D heatmaps (объёмные)
- Volumetric (3D CNN) — очень много памяти

#### B. 2D → 3D lifting (2D keypoints → 3D)
- Предсказать 2D keypoints (top-down pose)
- Lifting network: 2D → 3D
- Можно использовать временные последовательности

#### C. Parametric body model (SMPL)
- SMPL: 10 body shape parameters + 72 pose parameters
- $M(\beta, \theta) \in \mathbb{R}^{6890 \times 3}$ — mesh

### Методы

| Метод | Год | Идея |
|-------|-----|------|
| **Simple Baseline 3D** | 2017 | ResNet → 3D heatmaps |
| **VideoPose3D** | 2019 | 2D→3D lifting с temporal conv |
| **METRO** | 2021 | Transformer, mesh regression |
| **HMR** | 2018 | SMPL regression |
| **CLIFF** | 2022 | SMPL + camera-aware |

---

## 10. Современные unified подходы

### Mask2Former (Universal)
Одна архитектура для semantic + instance + panoptic.

### ViTPose
- ViT backbone
- Top-down, но encoder-only (без декодера)
- Просто: ViT → heatmaps → argmax

### DETR-based
- Object queries → class + bbox + mask + keypoints
- End-to-end (без RPN)
- Стандарт для современных моделей

---

## Ключевые выводы к экзамену

1. **Instance vs Semantic**: semantic — общая маска, instance — отдельные маски для каждого объекта
2. **Panoptic**: semantic (stuff) + instance (things) в одной карте. Метрика — PQ
3. **Mask R-CNN**: Faster R-CNN + mask head. RoIAlign (билинейная интерполяция вместо квантования)
4. **Mask head**: K×28×28 masks, binary CE loss (не softmax)
5. **2D Pose**: top-down (выше точность) vs bottom-up (быстрее)
6. **HRNet**: параллельные stream разного разрешения, exchange blocks
7. **3D Pose**: depth ambiguity — главная проблема. 2D→3D lifting, SMPL-параметрические модели

Mask R-CNN использует [[DL 20 - Object Detection|детекцию]] как первый этап (top-down подход).

---

**Связанные вопросы:**
- [[DL 16 - Семантическая сегментация]] — базовая задача семантической сегментации
- [[DL 17 - U-Net]] — архитектура для сегментации
- [[DL 20 - Object Detection]] — детекция как основа Mask R-CNN
