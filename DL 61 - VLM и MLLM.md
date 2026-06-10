# 61. Vision-Language Models и MLLM

## 1. Введение: Vision-Language Models (VLM)

**Vision-Language Models (VLM)** — модели, способные понимать и генерировать как визуальную, так и текстовую информацию. Базовые компоненты: [[DL 51 - Vision Transformer|визуальный энкодер]], [[DL 60 - CLIP|контрастивное предобучение]], [[DL 33 - Архитектура Transformer|Transformer]], [[DL 36 - GPT|LLM]].

### Эволюция VLM:

```
2019: VQA, простые captioning
2020: UNITER, OSCAR (Transformer fusion)
2021: CLIP, ALIGN (contrastive pretraining)
2022: Flamingo, BLIP-2 (LLM-based)
2023: LLaVA, InstructBLIP, GPT-4V (MLLM)
2024+: GPT-4o, Gemini, Llama-3-Vision
```

### Ключевые задачи VLM:

| Задача | Вход | Выход |
|---|---|---|
| **Image Captioning** | Изображение | Описание текстом |
| **VQA (Visual Question Answering)** | Изображение + вопрос | Ответ |
| **OCR** | Изображение с текстом | Распознанный текст |
| **Visual Dialogue** | Изображение + история диалога | Ответ |
| **Referring Expression** | Изображение + описание | Bounding box |
| **Document Understanding** | Документ (схема) | Извлечённые данные |

## 2. BLIP / BLIP-2

### BLIP (Bootstrapping Language-Image Pre-training)

**BLIP** (Li et al., 2022) — предобучение с тремя задачами и фильтрацией шумных данных.

### Модель:

```
                          ┌─ Image-Text Contrastive (ITC)
Image Encoder (ViT) ──→  ├─ Image-Text Matching (ITM)
                          └─ Language Modeling (LM) ← Text Decoder
```

**Три функции потерь:**

1. **ITC (Contrastive)**: сходство image-text (как CLIP)
2. **ITM (Matching)**: бинарная классификация (пара подходит/не подходит)
3. **LM (Language Modeling)**: генерация описания (causal LM)

### CapFilt (Captioning and Filtering):

BLIP использует синтез пар (image, caption) из шумных веб-данных:
- **Captioner**: генерирует описания (обучен на чистых данных)
- **Filter**: отбирает хорошие описания (обучен на ITM)

### BLIP-2 (Li et al., 2023)

BLIP-2 решает ключевую проблему: **стоимость обучения**. Вместо end-to-end обучения, замораживает предобученные Image Encoder (EVA-ViT) и LLM (OPT/FlanT5), обучая только **Q-Former** (Querying Transformer).

### Архитектура BLIP-2:

```
Image → Image Encoder (frozen ViT) → visual features
                                           ↓
Text (optional) → Q-Former (learnable) → query embeddings → LLM (frozen) → output
```

### Q-Former (Querying Transformer)

**Q-Former** — лёгкий Transformer, который "извлекает" информацию из visual features.

Структура:
```
Learned Queries [q1, q2, ..., q32] → cross-attention → Self-attention → LLM input
                                          ↑
                                Image features (frozen)
```

**Компоненты Q-Former:**
- **Learned Queries**: 32 обучаемых токена-запроса
- **Self-attention**: взаимодействие между queries
- **Cross-attention**: queries → image features
- **Три этапа обучения**:

| Этап | Задача | Что обучается |
|---|---|---|
| 1 | ITC + ITM + ITG | Q-Former (alignment image-text) |
| 2 | LM (через fully connected) | Q-Former → LLM проекция |
| 3 | End-to-end fine-tuning | Q-Former + LLM (LoRA) |

### Преимущества BLIP-2:
- **Масштабируемость**: работает с любым LLM (OPT, FlanT5, LLaMA)
- **Эффективность**: всего ~200M параметров Q-Former
- **Гибкость**: можно менять LLM без переобучения Q-Former

## 3. LLaVA (Large Language and Vision Assistant)

**LLaVA** (Liu et al., 2023) — простой и эффективный подход: **visual encoder + projector + LLM**.

### Архитектура:

```
Image → CLIP ViT-L/14 (frozen) → visual tokens
                                       ↓
                              MLP Projector (trainable)
                                       ↓
                              LLM (Vicuna/LLaMA, frozen → fine-tuned)
                                       ↓
                              Output text
```

**Проектор:** простой MLP (2 слоя) — преобразует визуальные эмбеддинги в пространство LLM.

### Формат входа LLM:

Входные токены:
```
[INST] <image> What's in this picture? [/INST]
```

Где `<image>` заменяется на 256 визуальных токенов, проецированных через MLP.

### Обучение LLaVA:

| Этап | Данные | Что обучается |
|---|---|---|
| **Stage 1: Alignment** | CC3M (595K пар) | MLP projector (LLM frozen) |
| **Stage 2: End-to-end** | 158K инструкций (LLaVA-Instruct-150K) | MLP + LLM (LoRA/full) |

### LLaVA-Instruct-150K:

Синтетический датасет инструкций:
- **Conversation**: вопросы о картинке
- **Detailed description**: подробное описание
- **Complex reasoning**: многошаговые рассуждения

Создан через GPT-4: (изображение, bounding boxes → GPT-4 генерирует вопросы+ответы)

### Преимущества LLaVA:
- **Простота** (проектор вместо Q-Former)
- **Сильное языковое понимание** (LLM делал всё языковое)
- **Качество на VQA задачах**

## 4. Архитектуры VLM

### Сравнение архитектур:

| Модель | Visual Encoder | Fusion | LLM | Метод соединения |
|---|---|---|---|---|
| **BLIP-2** | EVA-ViT | Q-Former | OPT/FlanT5 | Cross-attention |
| **LLaVA** | CLIP ViT-L | MLP | Vicuna ([[DL 38 - Дообучение LLM|fine-tuned]]) | Projection |
| **InstructBLIP** | EVA-ViT | Q-Former (instruct) | FlanT5 | Instruction-aware |
| **Flamingo** | NFNet + Perceiver | Gated cross-attn | Chinchilla | Промежуточные слои |
| **Qwen-VL** | ViT-bigG | Resampler | Qwen | Cross-attention |
| **CogVLM** | ViT | Attention | LLaMA | Deep fusion |

### Flamingo (DeepMind, 2022):

**Ключевая инновация:** gated cross-attention слои, вставленные между существующими слоями LLM.

```
Text → LLM Layer → Gated Cross-Attn → LLM Layer → ... → output
                        ↑
Visual → Perceiver Resampler → visual tokens
```

- **Perceiver Resampler**: сжатие визуальных признаков из переменного числа кадров/изображений в фиксированное число токенов
- **Gated cross-attention**: контролируемый поток визуальной информации в LLM

### Итоговая классификация подходов:

| Тип | Примеры | Описание |
|---|---|---|
| **Contrastive (dual encoder)** | CLIP, ALIGN | Раздельные энкодеры + contrastive loss |
| **Encoder-decoder** | BLIP, OFA | Совместный энкодер + декодер |
| **Q-Former (query-based)** | BLIP-2, InstructBLIP | Обучаемые запросы к визуальным признакам |
| **Projector (token-based)** | LLaVA, Qwen-VL | Проекция визуальных признаков → LLM |
| **Cross-attention (deep fusion)** | Flamingo, CogVLM | Визуальные токены через cross-attention в LLM |

## 5. Visual Encoder + Projector + LLM

### Visual Encoder:

| Энкодер | Размерность | Источник | Примечания |
|---|---|---|---|
| CLIP ViT-L/14 | 1024 | OpenCLIP | LLaVA |
| EVA-ViT-G | 1408 | EVA-CLIP | BLIP-2, InstructBLIP |
| SigLIP ViT | 1152 | Google | PaliGemma |
| InternViT-6B | ~3200 | InternLM | InternVL |

**Общие черты:** замороженный ViT, предобученный на миллиардах пар (image, text).

### Projector:

| Тип | Параметры | Модели |
|---|---|---|
| **MLP (2 layer)** | ~20M | LLaVA |
| **Q-Former** | ~200M | BLIP-2, InstructBLIP |
| **Perceiver Resampler** | ~200M | Flamingo |
| **Resampler** | ~80M | Qwen-VL |

Функция проектора: отобразить визуальные признаки в пространство текстовых эмбеддингов LLM.

### LLM:

| LLM | Параметры | Модели |
|---|---|---|
| Vicuna-7B/13B | 7B, 13B | LLaVA-1.5 |
| FlanT5-XL/XXL | 3B, 11B | BLIP-2, InstructBLIP |
| LLaMA-2/3 | 7B-70B | LLaVA-NeXT, Llama-3-Vision |
| Qwen | 7B-72B | Qwen-VL |
| InternLM2 | 7B-20B | InternVL |

## 6. Visual Tokens

### Количество визуальных токенов:

| Модель | Число токенов | На одно изображение |
|---|---|---|
| LLaVA | 256 | одно |
| BLIP-2 | 32 | одно |
| Flamingo | 64 (perceiver) | переменное |
| LLaVA-NeXT | 2880 (4× с grid) | однажды |

### Проблемы с visual tokens:

1. **Информационная ёмкость:** 256 токенов × 4096d ≈ 1M чисел — много, но меньше, чем пикселей
2. **Позиционная информация:** ViT уже имеет positional encoding, но позиции могут теряться при проекции
3. **Множественные изображения:** как представить 2+ изображения? (LLaVA-NeXT: concat tokens with separator)
4. **Dynamic resolution:** изображения разного размера → разное число токенов → паддинг

### Dynamic Resolution (LLaVA-NeXT):

- Изображение разбивается на grid (например, 2×2 = 4 патча)
- Каждый патч кодируется отдельно
- Все токены конкатенируются
- Добавляется токен-разделитель между патчами

Подробнее об оценке VLM см. [[DL 62 - Оценка и ограничения мультимодальных моделей]].

## 7. Типичные сценарии VLM

| Сценарий | Возможности |
|---|---|
| **Free-form VQA** | "What's unusual about this image?" |
| **Chain-of-thought** | "Let's think step by step: ..." |
| **OCR** | Распознавание текста в изображении |
| **Document parsing** | Таблицы, формы, графики |
| **Image comparison** | "Which image is brighter?" |
| **Multi-turn dialogue** | Контекстный диалог об изображении |
| **Video understanding** | Несколько кадров → понимание видео |

---

**Связанные вопросы:** [[DL 51 - Vision Transformer]], [[DL 60 - CLIP]], [[DL 62 - Оценка и ограничения мультимодальных моделей]], [[DL 36 - GPT]], [[DL 33 - Архитектура Transformer]], [[DL 38 - Дообучение LLM]]
