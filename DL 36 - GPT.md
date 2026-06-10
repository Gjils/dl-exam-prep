# 36. GPT-подобные модели

## 1. Autoregressive Generation (авторегрессивная генерация)

### 1.1 Определение

**Авторегрессивная модель:** предсказывает следующий токен, основываясь на всех предыдущих.

$$
P(x_1, x_2, \dots, x_T) = \prod_{t=1}^{T} P(x_t | x_{<t})
$$

### 1.2 Генерация (Inference)

1. Начать с prompt (контекста).
2. Вычислить $P(x_{t+1} | x_1, \dots, x_t)$.
3. Выбрать следующий токен (разные стратегии).
4. Добавить к последовательности.
5. Повторять пока не сгенерирован `[EOS]` или не достигнута макс. длина.

**Стратегии декодирования:**

| Стратегия | Описание | Плюсы | Минусы |
|---|---|---|---|
| **Greedy** | $\arg\max P(x_t \| x_{<t})$ | Быстро | Повторения, неоптимально |
| **Beam Search** | $k$ гипотез одновременно | Лучше качество | Дорого |
| **Top-k sampling** | Выбираем из top-k токенов | Разнообразие | k — гиперпараметр |
| **Top-p (nucleus)** | Выбираем из токенов с кумулятивной вероятностью $p$ | Адаптивно | Медленнее |
| **Temperature** | $P \propto \exp(\logits / T)$ | Контроль случайности | T=0 → greedy |
| **Contrastive search** | Штраф за повторение (SimCTG) | Меньше повторений | Сложнее |

---

## 2. Next-Token Prediction (NTP)

### 2.1 Objective

**Функция потерь:** cross-entropy между предсказанным распределением и истинным токеном.

$$
\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \log P(x_t | x_{<t}; \theta)
$$

где $x_t$ — истинный токен, $P(x_t | x_{<t})$ — предсказание модели.

### 2.2 Свойства

- **Причинность (causality):** каждый токен видит только прошлые (causal mask).
- **Совпадение обучения и инференса:** на инференсе делаем то же самое, что на обучении.
- **Token-level loss:** каждый токен даёт сигнал.
- **Эффективное использование данных:** каждый токен — обучающий пример.

### 2.3 Проблемы NTP

- **Локальный контекст:** модель может переобучаться на поверхностные паттерны.
- **Дисбаланс частых/редких токенов.**
- **Long-tail:** редкие слова предсказываются плохо.

---

## 3. Decoder-only Architecture

### 3.1 Структура

**Трансформер-декодер** без энкодера.

**Компоненты:**
1. Token embeddings (+ positional embeddings — learned, RoPE или ALiBi).
2. Stack of Transformer decoder layers.
3. LM head (линейный слой + softmax).

**Слой декодера:**
```
Masked Self-Attention → Add & LN → FFN → Add & LN
```

**Нет cross-attention** (нет энкодера — не с чем делать cross-attention).

### 3.2 Парадигма

$$ \text{Decoder-only: текст} \rightarrow \text{текст} $$

**Весь текст — одна последовательность.**
- **Prompt:** начало последовательности (инструкция, контекст).
- **Completion:** продолжение.

### 3.3 Отличия от encoder-decoder

| Характеристика | Decoder-only | Encoder-Decoder |
|---|---|---|
| **Сложность** | Меньше (нет cross-attn) | Больше |
| **Параметры** | Эффективнее | Требует cross-attn |
| **In-context learning** | Естественный | Менее естественный |
| **Генерация** | Autoregressive | Autoregressive |
| **Понимание** | Только слева направо | Двустороннее (encoder) |
| **Примеры** | GPT, LLaMA | T5, BART |

### 3.4 Преимущества decoder-only

- **Простота:** одна архитектура для всего.
- **Масштабирование:** проще распределять на GPU (нет cross-attn).
- **In-context learning:** prompt + completion — естественная форма.
- **Fluent generation:** лучше качество генерации.

---

## 4. In-Context Learning (ICL)

### 4.1 Zero-shot

**Задача решается без примеров.**

**Формат:** `"Translate to French: I love cats →"`

Модель генерирует: `"J'aime les chats"`

**Зачем работает:** модель видела много переводов в pretraining данных.

### 4.2 Few-shot (in-context)

**Формат:** демонстрация $k$ примеров в prompt.

```
Translate English to French:
sea otter → loutre de mer
cheese → fromage
I love cats →
```

Модель генерирует: `"J'aime les chats"`

**Ключевые наблюдения:**
- **K = 0, 1, 2, ...** — чем больше примеров, тем лучше.
- **Качество prompt** (order, formatting) критически важен.
- **Нет обновления весов:** всё через контекст.
- **Работает для** GPT-3+; для маленьких моделей (GPT-1/2) почти не работает.

### 4.3 Почему работает ICL? (Теории)

1. **Bayesian inference:** модель выводит скрытую концепцию из примеров (pattern recognition).
2. **Meta-learning:** pretraining на большом количестве задач — научился учиться.
3. **Механическая имитация:** модель просто продолжает паттерн из prompt.
4. **Induction heads** (Olsson et al., 2022): внутренние механизмы attention, копирующие паттерны.

### 4.4 Chain-of-Thought (CoT)

**Расширение ICL:** демонстрация шагов рассуждения.

**Без CoT:** `"Q: 24 * 7 = ? A: 168"`
**С CoT:** `"Q: 24 * 7 = ? A: 24 * 7 = (20 * 7) + (4 * 7) = 140 + 28 = 168"`

**Зачем:** улучшает рассуждение на сложных задачах (математика, логика).

---

## 5. Эволюция от GPT-1 до GPT-4

### 5.1 GPT-1 (Radford et al., 2018)

**Ключевые моменты:**
- 12-layer Transformer decoder, 117M параметров.
- BooksCorpus (7000 книг, ~1B слов).
- NTP objective.
- **Новизна:** первый, кто показал, что decoder-only + NTP + fine-tuning работает.
- Результаты: SOTA на 9/12 NLP задач.

### 5.2 GPT-2 (Radford et al., 2019)

**Ключевые моменты:**
- 1.5B параметров (48 слоёв).
- WebText (40GB текста из Reddit).
- **Новизна:** zero-shot transfer — без fine-tuning решает задачи.
- Показал, что масштаб важен: большие модели → лучший zero-shot.
- **"Unsupervised multitask learner"** — модель неявно выучила много задач.

**Размеры:**

| Модель | Слои | d_model | Параметры |
|---|---|---|---|
| Small | 12 | 768 | 124M |
| Medium | 24 | 1024 | 355M |
| Large | 36 | 1280 | 774M |
| XL | 48 | 1600 | 1.5B |

### 5.3 GPT-3 (Brown et al., 2020)

**Ключевые моменты:**
- 175B параметров.
- 570GB текста (Common Crawl, WebText2, Books, Wikipedia).
- **Новизна:** in-context learning (zero/few-shot без fine-tuning).
- Показал, что ICL — emergent property (возникает при достаточном масштабе).
- **Scaling laws:** производительность растёт степенно с числом параметров.

**Размеры:**

| Модель | Параметры | d_model | Layers | Heads |
|---|---|---|---|---|
| Ada | 2.7B | — | — | — |
| Babbage | 6.7B | — | — | — |
| Curie | 13B | — | — | — |
| Davinci | 175B | 12288 | 96 | 96 |

### 5.4 GPT-3.5 / InstructGPT / ChatGPT

**GPT-3.5:** улучшения GPT-3.
- **Codex:** обучение на коде.
- **InstructGPT (Ouyang et al., 2022):** RLHF — Reinforcement Learning from Human Feedback.

**RLHF Pipeline:**
1. **SFT:** Supervised Fine-tuning на размеченных инструкциях.
2. **RM:** Обучить Reward Model — предсказывать, какой ответ лучше.
3. **PPO:** Оптимизировать GPT с помощью reward model.

**ChatGPT (Nov 2022):** GPT-3.5 + RLHF + диалоговый формат.

### 5.5 GPT-4 (OpenAI, 2023)

**Ключевые моменты:**
- Размер неизвестен (предположительно 8x220B MoE, 1.7T параметров).
- **Multimodal:** текст + изображения (GPT-4V/Vision).
- **Safer:** RLHF + rule-based rewards.
- **Longer context:** 8k → 32k → 128k токенов (GPT-4 Turbo).
- **SOTA:** reasoning, coding, multilingual.
- **Hallucinations:** всё ещё есть.

---

## 6. Эволюция (таблица)

| Свойство | GPT-1 | GPT-2 | GPT-3 | GPT-4 |
|---|---|---|---|---|
| **Год** | 2018 | 2019 | 2020 | 2023 |
| **Параметры** | 117M | 1.5B | 175B | ~1.7T* (8×220B MoE) |
| **Слои** | 12 | 48 | 96 | ~120 |
| **Данные** | BooksCorpus | WebText | Common Crawl+ | Internet+ |
| **Objective** | NTP | NTP | NTP | NTP + RLHF |
| **Zero-shot** | Нет | Да (скромно) | Да (хорошо) | Да (отлично) |
| **Few-shot ICL** | Нет | Слабо | Да | Да |
| **RLHF** | Нет | Нет | Нет | Да (Instruct) |
| **Vision** | Нет | Нет | Нет | Да |

---

## 7. Итог

- **Autoregressive generation:** $P(x) = \prod P(x_t | x_{<t})$ — основной принцип.
- **Decoder-only:** Transformer без энкодера — простая и эффективная архитектура (см. [[DL 33 - Архитектура Transformer]]).
- **In-context learning:** способность решать задачи через примеры в prompt (zero/few-shot). Подробнее — [[DL 40 - ICL и Prompting]].
- **Эволюция:** GPT-1 (117M, fine-tuning) → GPT-2 (1.5B, zero-shot) → GPT-3 (175B, ICL) → GPT-4 (MoE, multimodal).
- **RLHF:** ключевая техника для выравнивания (alignment) с человеческими предпочтениями — см. [[DL 39 - Preference Tuning и RLHF]].
- **Chain-of-Thought:** рассуждение через шаги.

---

**Связанные вопросы:**
- [[DL 33 - Архитектура Transformer]] — архитектурная основа GPT (decoder-only)
- [[DL 34 - Pretraining в NLP]] — NTP как objective pretraining
- [[DL 37 - Масштабирование и MoE]] — scaling laws и MoE (GPT-4)
- [[DL 38 - Дообучение LLM]] — fine-tuning GPT
- [[DL 39 - Preference Tuning и RLHF]] — RLHF для выравнивания
- [[DL 40 - ICL и Prompting]] — in-context learning
- [[DL 42 - Инференс LLM]] — инференс decoder-only моделей
