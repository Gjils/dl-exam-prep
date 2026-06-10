# 35. BERT- и T5-подобные модели

## 1. BERT (Bidirectional Encoder Representations from Transformers, Devlin et al., 2019)

### 1.1 Архитектура

**Тип:** Encoder-only Transformer.

**Variantы:**

| Модель | Слои (L) | Hidden (H) | Heads (A) | Параметры |
|---|---|---|---|---|
| BERT-base | 12 | 768 | 12 | 110M |
| BERT-large | 24 | 1024 | 16 | 340M |

**Входные представления:**
- Token embeddings (WordPiece, 30k словарь).
- Segment embeddings (Sentence A/Sentence B).
- Position embeddings (learned, max 512 позиций).

**Специальные токены:**
- `[CLS]` — первый токен, используется для классификации.
- `[SEP]` — разделитель предложений.
- `[MASK]` — маскированный токен (на обучении).
- `[PAD]` — padding.

### 1.2 Pretraining Objectives

#### MLM (Masked Language Model)

**Идея:** замаскировать 15% токенов → предсказать их.

**Детали маскировки (из 15% выбранных токенов):**
- 80% → заменить на `[MASK]`.
- 10% → заменить на случайный токен.
- 10% → оставить без изменений.

**Зачем такая хитрость:** на fine-tuning токенов `[MASK]` нет. Модель должна научиться работать с любыми токенами.

**Функция потерь:**
$$
\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log P(x_i | x_{\setminus \mathcal{M}})
$$

где $\mathcal{M}$ — множество маскированных позиций.

#### NSP (Next Sentence Prediction, original BERT)

**Суть:** определить, является ли предложение B следующим за A.

**Вход:** `[CLS] A [SEP] B [SEP]`.
**Цель:** IsNext / NotNext (50%/50%).

**Проблема NSP:**
- RoBERTa (Liu et al., 2019) показала, что NSP не нужен.
- Losing NSP + dynamic masking = better performance.
- Современные модели (DeBERTa) не используют NSP.

### 1.3 Fine-tuning BERT

**Классификация:**
- Использовать `[CLS]` + классификационная головка (линейный слой).

**NER/Sequence Labeling:**
- Для каждого токена — классификационная головка.

**QA (Extractive):**
- Предсказать стартовую и конечную позиции ответа в тексте.
- Два дополнительных линейных слоя (start/end).

**Natural Language Inference (NLI):**
- `[CLS]` → 3 класса (entailment, contradiction, neutral).

### 1.4 Варианты BERT

| Модель | Отличие |
|---|---|
| **RoBERTa** | Больше данных, dynamic masking, нет NSP |
| **ALBERT** | Factorized embedding, cross-layer sharing |
| **DistilBERT** | Knowledge distillation (60% скорости, 97% качества) |
| **DeBERTa** | Disentangled attention + enhanced mask decoder |
| **ELECTRA** | Discriminator (замена токенов) вместо MLM |
| **SpanBERT** | Span-level masking (не отдельные токены) |

---

## 2. T5 (Text-to-Text Transfer Transformer, Raffel et al., 2020)

### 2.1 Философия Text-to-Text

**Ключевая идея:** все NLP задачи → задача "текст на вход → текст на выход".

**Примеры:**
```
translate English to German: That is good → target: Das ist gut
cola sentence: The course is jumping well → target: not acceptable
stsb sentence1: ..., sentence2: ... → target: 3.8
summarize: ... → target: краткое содержание
```

**Преимущества:**
- Единый фреймворк для всех задач.
- Одна архитектура, один pretraining objective.
- Легко добавлять новые задачи.

### 2.2 Архитектура

**Тип:** Encoder-Decoder Transformer.

**Размеры:**

| Модель | Параметры | d_model | Layers (Enc/Dec) | Heads |
|---|---|---|---|---|
| T5-small | 60M | 512 | 6/6 | 8 |
| T5-base | 220M | 768 | 12/12 | 12 |
| T5-large | 770M | 1024 | 24/24 | 16 |
| T5-3B | 3B | 1024 | 24/24 | 32 |
| T5-11B | 11B | 1024 | 24/24 | 128 |

**Отличия от оригинального Transformer:**
- **LayerNorm:** Pre-LN (нормализация до sublayer, а не после).
- **Activation:** ReLU → "искусственная" ReLU в T5.1.1 — GEGLU.
- **Pos encoding:** Relative bias (обучаемая матрица размера $n_{\text{buckets}}$).
- **No bias в LayerNorm** (в T5.1.1).

### 2.3 Pretraining: Span Corruption

**Алгоритм:**
1. Выбрать 15% токенов.
2. Сгруппировать их в spans (непрерывные участки) — длина span'а выбирается из распределения (длина по Пуассону, $\lambda = 3$).
3. Каждый span заменить на уникальный sentinel token `<X>`, `<Y>`, `<Z>`.
4. **Input:** исходный текст с заменёнными spans на sentinel.
5. **Target:** последовательность sentinel-токенов и соответствующих оригинальных spans.

**Пример:**
```
Original: "Я люблю кошек и собак"
Masked:    "Я <X> и <Y>"
Target:    "<X> люблю кошек <Y> собак"
```

**Зачем spans, а не отдельные токены?**
- Учит предсказывать фразы, а не изолированные слова.
- Лучше для генерации текста.
- Меньше вычислительных затрат (15% токенов → несколько spans).

### 2.4 C4 Dataset (Colossal Clean Crawled Corpus)

- 750GB текста (после очистки Common Crawl).
- Дедлисты:
  - Только строки, заканчивающиеся знаком препинания.
  - Удалить страницы с "lorem ipsum", "javascript", "copyright" и т.д.
  - Языковая фильтрация (только английский).

### 2.5 Fine-tuning T5

- Все задачи формулируются как text-to-text.
- Добавляем префикс к входу: `"summarize: ..."`, `"translate: ..."`.
- Гиперпараметры: learning rate $10^{-3}$, batch размер, dropout.

**Advantage:** можно натренировать на 100+ задачах в одном мультизадачном обучении.

### 2.6 mT5 (Multilingual T5, Xue et al., 2020)

- 101 язык.
- Словарь: SentencePiece (250k токенов).
- mC4: 6.3TB текста.
- Те же типы моделей (small — XL).

---

## 3. BERT vs T5

| Характеристика | BERT | T5 |
|---|---|---|
| **Архитектура** | Encoder-only | Encoder-Decoder |
| **Парадигма** | Embedding-based | Text-to-Text |
| **Objective** | MLM (+NSP) | Span Corruption |
| **Генерация** | Нет (только понимание) | Да |
| **Токенизация** | WordPiece (30k) | SentencePiece (32k, потом 250k) |
| **Transfer** | Fine-tuning головок | Text prefix |
| **Лучше для** | Классификация, NER, extractive QA | Суммаризация, перевод, generative QA |
| **Большая модель** | BERT-large (340M) | T5-11B (11B) |
| **Данные** | 3.3B токенов | 750GB (C4) |
| **Скорость инференса** | Быстрее | Медленнее (2 прохода) |

---

## 4. Другие encoder-decoder модели

### 4.1 BART (Lewis et al., 2020)

**Архитектура:** BERT (encoder) + GPT (decoder) → BART.

**Denoising objectives:**
- **Token masking:** как BERT MLM.
- **Token deletion:** удалить токены, предсказать их.
- **Text infilling:** как span corruption T5.
- **Sentence permutation:** перемешать предложения.
- **Document rotation:** сдвинуть начало.

**Лучше всего для:** суммаризация, генерация.

### 4.2 PEGASUS (Zhang et al., 2020)

**Objective:** GSG (Gap Sentence Generation) — маскировать целые предложения (не токены/spans).

**Идея:** маскировать предложения, наиболее похожие на суммаризацию (важные предложения).

**Лучше для:** суммаризация (state-of-the-art на своё время).

### 4.3 Тенденции

- Современные LLM (GPT-4, LLaMA, Claude) — **decoder-only**.
- Encoder-Decoder сохраняется для специализированных seq2seq задач.
- BERT-подходы — для встраиваний (embeddings), ранжирования, понимания.

---

## 5. Итог

- **BERT:** encoder-only, MLM, двусторонний контекст — понимание.
- **NSP:** признан ненужным (RoBERTa показала).
- **T5:** text-to-text, encoder-decoder, span corruption — генерация + понимание.
- **C4:** большой чистый датасет для pretraining.
- **Fine-tuning:** BERT — головки; T5 — текстовые префиксы.
- BERT — для классификации, T5 — для seq2seq.
- Decoder-only сейчас доминирует (см. [[DL 36 - GPT]]), но encoder-only и encoder-decoder нишево полезны.

---

**Связанные вопросы:**
- [[DL 33 - Архитектура Transformer]] — архитектурная основа BERT и T5
- [[DL 34 - Pretraining в NLP]] — парадигмы pretraining (MLM, Span Corruption)
- [[DL 36 - GPT]] — decoder-only альтернатива
- [[DL 38 - Дообучение LLM]] — fine-tuning BERT/T5 под конкретные задачи
