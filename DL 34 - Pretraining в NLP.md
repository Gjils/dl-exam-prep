# 34. Pretraining в NLP

## 1. Self-Supervised Pretraining

### 1.1 Определение

**Self-supervised learning (SSL):** обучение на неразмеченных данных с автоматически генерируемыми "псевдо-метками".

**В NLP:** текст сам по себе — последовательность токенов. Метки можно получить из текста без разметки:
- Предсказать пропущенное слово.
- Предсказать следующее слово.
- Определить, является ли предложение следующим.

### 1.2 Почему это работает?

- **Огромные данные:** интернет, книги, Wikipedia — терабайты текста без разметки.
- **Языковая структура:** сам текст содержит информацию (грамматика, семантика, факты).
- **Transfer learning:** предобученные модели дообучаются на специфических задачах с малым количеством данных.

### 1.3 Зачем pretrain?

| Без pretraining | С pretraining |
|---|---|
| Нужно много размеченных данных | Можно мало размеченных (fine-tuning) |
| Модель учится с нуля | Модель уже знает язык |
| Переобучение на малых данных | Лучшая обобщаемость |
| Медленная сходимость | Быстрая сходимость |

---

## 2. Self-Supervised Objectives (цели предобучения)

### 2.1 Next Token Prediction (NTP)

**Суть:** предсказать следующий токен по предыдущим.

$$
\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t | x_{<t})
$$

**Где используется:** GPT, все decoder-only модели.

**Плюсы:** авторегрессивная генерация совпадает с задачей обучения.

**Минусы:** не использует контекст справа.

### 2.2 Masked Language Modeling (MLM)

**Суть:** замаскировать часть токенов → предсказать их по двустороннему контексту.

$$
\mathcal{L} = -\sum_{i \in \text{masked}} \log P(x_i | x_{\text{context}})
$$

**Где используется:** BERT, RoBERTa, ALBERT.

**Плюсы:** двусторонний контекст, богатые представления.

**Минусы:** не совпадает с генерацией (токены маскируются только на обучении).

### 2.3 Permutation Language Modeling (PLM)

**Суть:** предсказывать токены в случайном порядке (XLNet).

$$
\mathcal{L} = -\sum_{t=1}^{T} \log P(x_{z_t} | x_{z_{<t}})
$$

где $z$ — случайная перестановка.

**Плюсы:** двусторонний контекст + авторегрессивность.

**Минусы:** вычислительно дорого.

### 2.4 Denoising Autoencoding (Span Corruption)

**Суть:** зашумлять текст (маска, удаление, перестановка) → восстановить.

**Где используется:** T5, BART.

**Форма T5 (Span Corruption):**
- Выбрать 15% токенов.
- Сформировать непрерывные spans (участки) замаскированных токенов.
- Цель: предсказать spans по порядку.

**Пример:**
```
Input:  Я <X> кошек и <Y> → target: <X> люблю <Y> собак
```
где `<X>`, `<Y>` — sentinel tokens.

### 2.5 Contrastive Learning

**Суть:** учить представления, где похожие примеры близки, разные — далеко.

**SimCSE (Gao et al., 2021):**
- Положительная пара: одно и то же предложение, но с разным dropout.
- Отрицательные пары: разные предложения в батче.

**Формула:**
$$
\mathcal{L} = -\log \frac{e^{\text{sim}(h_i, h_i^+) / \tau}}{\sum_{j=1}^{N} e^{\text{sim}(h_i, h_j) / \tau}}
$$

---

## 3. Парадигмы архитектур

### 3.1 Encoder-only

**Идея:** двусторонний контекст, цель — понимание текста (не генерация).

**Архитектура:** Transformer Encoder (self-attention без маски).

**Примеры:** BERT, RoBERTa, ALBERT, DeBERTa, ELECTRA.

**Лучше всего для:**
- Классификация текста.
- NER.
- QA (extractive).
- Semantic similarity (STS).
- Sequence labeling.

**Обучение:** MLM, NSP, SOP.

**Итоговое представление:** CLS-токен или среднее/макс пулинг токенов.

### 3.2 Decoder-only

**Идея:** авторегрессивная генерация слева направо.

**Архитектура:** Transformer Decoder (causal mask).

**Примеры:** GPT-1/2/3/4, LLaMA, Mistral, Gemini.

**Лучше всего для:**
- Текстовая генерация.
- Chat/диалог.
- In-context learning.
- Code generation.

**Обучение:** Next Token Prediction (NTP).

**Итоговое представление:** последний токен для генерации, любой токен для представления.

### 3.3 Encoder-Decoder

**Идея:** энкодер читает вход (двусторонний), декодер генерирует выход (авторегрессивно).

**Архитектура:** Полный Transformer (Encoder + Decoder).

**Примеры:** T5, BART, Pegasus, mT5.

**Лучше всего для:**
- Машинный перевод.
- Суммаризация.
- Question generation.
- Text-to-SQL.
- Любые seq2seq задачи.

**Обучение:** Span Corruption / Denoising.

**Итоговое представление:** энкодер для понимания, декодер для генерации.

### 3.4 Сравнение парадигм

| Характеристика | Encoder-only | Decoder-only | Encoder-Decoder |
|---|---|---|---|
| **Архитектура** | Encoder | Decoder | Encoder + Decoder |
| **Attention** | Bidirectional (no mask) | Causal (masked) | Bidirectional (enc) + Causal (dec) |
| **Цель обучения** | MLM / PLM | NTP | Span corruption / Denoising |
| **Генерация** | Нет (только понимание) | Да (авторегрессивная) | Да (с cross-attention) |
| **Параметры** | Меньше (нет cross-attn) | Средне | Больше всего |
| **Инференс** | Быстрый (один проход) | Последовательный | Последовательный (decoder) |
| **Примеры** | BERT, RoBERTa | GPT, LLaMA | T5, BART |

---

## 4. Масштабы pretraining

| Модель | Параметры | Данные | Objective | Цена обучения |
|---|---|---|---|---|
| BERT-base | 110M | 3.3B токенов (Books + Wikipedia) | MLM + NSP | ~4 дня на 4 TPUv2 |
| BERT-large | 340M | 3.3B токенов | MLM + NSP | ~4 дня на 16 TPUv2 |
| GPT-3 | 175B | 570GB текста (Common Crawl + книги) | NTP | ~$4.6M, 355 лет GPU |
| T5-11B | 11B | 750GB (C4) | Span corruption | ~$1.3M |
| LLaMA-65B | 65B | 1.4T токенов | NTP | ~$6M |

---

## 5. Transfer Learning Pipeline

### 5.1 Pretraining → Fine-tuning

1. **Pretraining:** большая модель на огромных неразмеченных данных (self-supervised).
2. **Fine-tuning:** дообучение на специфической задаче с размеченными данными.

**Варианты fine-tuning:**
- **Full fine-tuning:** обновляем все веса (лучше качество, дорого).
- **Adapter-based:** вставляем маленькие слои (храним один предобученный веса + адаптеры).
- **Prompt tuning:** обучаем только soft prompts (входные векторы).
- **LoRA:** низкоранговые матрицы для аппроксимации обновлений.

### 5.2 Zero-shot / Few-shot (без fine-tuning)

Decoder-only модели (GPT-3+) могут решать задачи без дообучения — через **in-context learning** (см. вопрос 36).

---

## 6. Эволюция парадигм

```
2017: Transformer (Enc-Dec) – supervised MT
2018: GPT (Dec-only) – pretraining + fine-tuning
2018: BERT (Enc-only) – bidirectional pretraining
2019: T5 (Enc-Dec) – text-to-text framework
2019: GPT-2 (Dec-only) – zero-shot transfer
2020: GPT-3 (Dec-only) – in-context learning
2022+: LLaMA, Mistral, GPT-4 – decoder-only доминирует
```

**Современный тренд:** decoder-only модели доминируют, т.к. они:
- Проще (одна архитектура).
- Масштабируются лучше.
- In-context learning + fine-tuning + generation — всё в одной модели.
- Эффективнее использование параметров (нет cross-attention).

---

## 7. Итог

- **Self-supervised pretraining:** обучение на неразмеченных текстах с автоматическими метками (NTP, MLM, Span Corruption).
- **Encoder-only (BERT):** понимание текста — классификация, NER, QA.
- **Decoder-only (GPT):** генерация, chat, in-context learning — доминирующая парадигма.
- **Encoder-Decoder (T5):** seq2seq задачи — перевод, суммаризация.
- **Transfer learning:** pretrain на больших данных → fine-tune на малых (см. [[DL 38 - Дообучение LLM]]).

---

**Связанные вопросы:**
- [[DL 33 - Архитектура Transformer]] — архитектурная основа всех pretrained моделей
- [[DL 35 - BERT и T5]] — реализации encoder-only и encoder-decoder pretraining
- [[DL 36 - GPT]] — decoder-only pretraining (NTP)
- [[DL 38 - Дообучение LLM]] — transfer learning после pretraining
