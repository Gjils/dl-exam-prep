# 15. Transfer Learning в Computer Vision

## Feature extraction, fine-tuning, заморозка слоёв, выбор backbone

---

## 1. Что такое Transfer Learning?

### Определение
Использование знаний, полученных при решении **одной задачи**, для решения **другой, связанной задачи**.

### Мотивация
- Обучать CNN с нуля на маленьком датасете → **переобучение**
- ImageNet содержит **14M изображений, 21K классов**
- Нижние слои изучют **общие признаки**: края, текстуры, паттерны (см. [[DL 11 - Базовая CNN]])
- Верхние слои — **специфичные** для задачи признаки

### Постановка
- **Source domain**: ImageNet (огромный, разнообразный)
- **Target domain**: ваш датасет (маленький, специфичный)
- **Source task**: классификация на 1000 классов
- **Target task**: ваша задача (классификация, детекция, сегментация)

---

## 2. Feature Extraction (признаки без обучения)

### Идея
Берём предобученный backbone, **замораживаем** все его веса. Используем как feature extractor. Обучаем только **классификатор** (fully connected head).

### Процесс

```
Input Image → [Frozen Backbone] → Feature Vector → [Trainable Classifier] → Prediction
                 fixed weights                            learned from scratch
```

### Когда использовать

| Условие | Feature Extraction |
|---------|-------------------|
| Мало данных | ✅ Лучший выбор |
| Данные похожи на ImageNet | ✅ |
| Быстро нужно | ✅ |
| Много данных, своя специфика | ❌ Нужен fine-tune |

### Как получить фичи:

1. **Offline feature extraction**:
   - Прогнать весь датасет через frozen backbone
   - Сохранить feature векторы на диск (как embedding)
   - Обучить классификатор на них (SVM, логистическая регрессия, MLP)
   - ⚡ Очень быстро, можно на CPU

2. **Online (на лету)**:
   - Backbone заморожен в сети
   - Градиент не течёт в backbone
   - Классификатор обучается стохастически
   - Один backward pass за шаг

### Где брать features
- После Global Average Pooling (GAP)
- После последнего conv слоя
- Из intermediate слоёв (если нужен дескриптор)

---

## 3. Fine-tuning (тонкая настройка)

### Идея
Берём предобученный backbone, **размораживаем** часть (или все) слои, дообучаем на target dataset с **маленьким learning rate**.

### Стратегии

#### A. Полный fine-tune
```
[All Layers Unfrozen] → training on target dataset
LR: 1e-5 to 1e-4
```
- Для больших датасетов (>10K images)
- Когда target domain сильно отличается от source

#### B. Частичный fine-tune
```
Last N layers:           unfrozen, LR=1e-5
First layers (shallow):  frozen
Classifier:              unfrozen, LR=1e-3
```

**Почему?**
- Нижние слои: **общие** фильтры (края, углы) — не меняем
- Верхние слои: **специфичные** для задачи — дообучаем
- Меньше шанс переобучения

#### C. Discriminative fine-tuning
Каждому слою — свой learning rate:

$$ \text{LR}_l = \text{LR}_{\text{base}} \cdot \alpha^{L-l} $$

- $\alpha > 1$: верхние слои учатся быстрее
- Нижние слои: маленький LR (почти frozen)
- ULMFiT подход, показало отличные результаты

### Learning rate стратегии

| Параметр | Значение | Комментарий |
|----------|----------|-------------|
| LR для backbone | 1e-5 — 1e-4 | В 10 раз меньше, чем при обучении с нуля |
| LR для head | 1e-3 — 1e-2 | Можно больше (у классификатора нет предобучения) |
| Оптимизатор | Adam, SGD | SGD даёт лучшую сходимость (с warmup) |
| Scheduler | ReduceLROnPlateau, Cosine | Cosine annealing — лучшая практика |
| Warmup | 1-5 эпох | Линейный рост LR для стабилизации |

---

## 4. Заморозка слоёв (Layer Freezing)

### Техника

```python
# PyTorch example
for param in backbone.parameters():
    param.requires_grad = False  # freeze

# Разморозить последний блок
for param in backbone.layer4.parameters():
    param.requires_grad = True

# Классификатор — обучаемый
model.fc = nn.Linear(2048, num_classes)
```

### Прогрессивная разморозка (progressive unfreezing)

1. **Эпоха 1**: frozen backbone, обучается только head
2. **Эпоха 2**: разморозить последний блок, head
3. **Эпоха 3**: разморозить два последних блока
4. **Эпоха N**: вся сеть

**Зачем**: постепенное адаптирование, предотвращение «забывания» предобученных знаний.

### BatchNorm при fine-tuning

**Важно!** BatchNorm содержит running_mean, running_var.
- При fine-tuning **ОБЯЗАТЕЛЬНО** пересчитать statistics на новом датасете
- Если frozen → может дать плохие результаты из-за domain shift
- Лучше: не замораживать BN, или переключиться на GroupNorm

### Влияние размера датасета

| Размер датасета | Стратегия | Заморозка |
|-----------------|-----------|-----------|
| <100 | Feature extraction | Все слои |
| 100-1000 | Частичный fine-tune | 50-80% слоёв |
| 1k-10k | Fine-tune | 30-50% слоёв |
| >10k | Полный fine-tune | Нет |

---

## 5. Выбор backbone

### Критерии выбора

1. **Размер датасета**
   - Маленький → ResNet-18/34 (меньше параметров → меньше переобучения)
   - Большой → EfficientNet-B7 (максимальное качество)

2. **Скорость / latency**
   - Реал-тайм → MobileNet, EfficientNet-B0
   - Сервер → ResNet-152, ConvNeXt-B

3. **Тип задачи**
   - Классификация → любой backbone + GAP + FC
   - Детекция → backbone с FPN (ResNet+FPN)
   - Сегментация → backbone с multiscale (ResNet, EfficientNet)

4. **Domain shift**
   - Похож на ImageNet (объекты, сцены) → любой backbone
   - Медицинские изображения → специальные pre-trained (BiomedCLIP, RadImageNet)
   - Сателлиты → self-supervised маски

### Популярные backbone для transfer learning

| Backbone | Параметры | ImageNet | Скорость | Когда брать |
|----------|-----------|----------|----------|-------------|
| [[DL 14 - ResNet|ResNet-18]] | 11.7M | 70.3% | 🚀 | Мало данных |
| [[DL 14 - ResNet|ResNet-50]] | 25.6M | 76.0% | ✅ | Стандартный выбор |
| Efficient-B0 | 5.3M | 77.3% | 🚀 | Мобильные/лёгкие |
| Efficient-B3 | 12M | 81.1% | ✅ | Баланс |
| ConvNeXt-T | 29M | 82.1% | ✅ | Когда нужно SOTA |
| Swin-T | 29M | 81.3% | ✅ | Transformer backbone |

### Варианты предобученных весов

| Источник | Описание | Когда использовать |
|----------|----------|-------------------|
| ImageNet (supervised) | Классический стандарт | Базовая задача |
| ImageNet (self-supervised) | DINO, MAE, SimCLR | Мало размеченных данных |
| CLIP | Text-image pretraining | Zero-shot, open-vocab |
| Medical | RadImageNet, ChestX-ray | Медицина |
| Places365 | Сцены | Scene classification |

---

## 6. Domain Shift

### Проблема
ImageNet содержит **объектные** изображения (центрированные, качественные). Ваш датасет может быть другим.

### Типы domain shift

| Тип сдвига | Пример | Решение |
|------------|--------|---------|
| **Цвет** | Медицинские рентгены | Normalize по своему датасету |
| **Текстура** | Сателлитные снимки | Fine-tune на ImageNet + target |
| **Разрешение** | Низкокачественные камеры | Super-resolution augment |
| **Ракурс** | Видеонаблюдение | Вращения, аугментация |
| **Фон** | Продукты на конвейере | Random erasing, CutMix |

### Адаптация к domain shift

1. **Domain Adaptation** — дообучение на target domain (un/semi supervised)
2. **Color jitter** — аугментация цветов
3. **Style transfer** — привести source к target стилю
4. **Self-supervised pre-task** — обучить маскирование/реконструкцию на target

---

## 7. Практические рекомендации

### Пошаговый протокол

1. **Выбрать backbone** (ResNet-50 как начальный стандарт)
2. **Заменить классификатор** (num_classes = ваш)
3. **Загрузить ImageNet веса**
4. **Заморозить backbone**, train head (3-5 эпох)
5. **Разморозить часть backbone**, fine-tune (5-10 эпох, LR в 10 раз меньше)
6. **ReduceLROnPlateau** или Cosine annealing
7. **Early stopping** по validation

### Common pitfalls

| Проблема | Причина | Решение |
|----------|---------|---------|
| Overfitting | Мало данных, агрессивный fine-tune | Feature extraction, больше регуляризации |
| Underfitting | Слишком сильная заморозка | Разморозить больше слоёв |
| Loss не сходится | Слишком большой LR | Warmup, smaller LR |
| BatchNorm шум | Frozen BN неправильные stats | Unfreeze BN, пересчитать stats |
| Катастрофическое забывание | Полный fine-tune с большим LR | Progressive unfreezing, L2 в loss |

---

## Ключевые выводы к экзамену

1. **Transfer Learning** — использование предобученных моделей для новой задачи
2. **Feature extraction** — frozen backbone → только классификатор
3. **Fine-tuning** — разморозка + дообучение с маленьким LR
4. **Заморозка слоёв** — нижние frozen (общие признаки), верхние дообучаются
5. **Выбор backbone** — баланс размера датасета, скорости, специфики задачи
6. **Domain shift** — разница между source и target требует адаптации
7. **Практика**: сначала frozen, потом progressive unfreezing

Аналогичный подход fine-tuning применяется к языковым моделям — см. [[DL 38 - Дообучение LLM]].

---

**Связанные вопросы:**
- [[DL 11 - Базовая CNN]] — устройство свёрточной сети
- [[DL 14 - ResNet]] — популярный backbone для transfer learning
- [[DL 38 - Дообучение LLM]] — fine-tuning языковых моделей
