# DL 43 — Ускорение и сжатие LLM

> **Efficient attention (FlashAttention, sparse, MQA/GQA), квантизация (INT8/INT4, GPTQ, QLoRA), trade-off памяти/скорости/качества.
> Смежные темы: [[DL 42 - Инференс LLM]], [[DL 44 - Декодирование LLM]], [[DL 24 - Knowledge Distillation]], [[DL 37 - Масштабирование и MoE]].**

---

## 1. Efficient Attention

### 1.1. Проблема стандартного Attention

Стандартный attention (Vaswani et al., 2017):

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

**Вычислительная сложность:** $O(n^2 \cdot d)$ — квадратичная по длине последовательности $n$.

**Проблемы:**
- $n = 2048$: $QK^T$ размером $2048 \times 2048$ — нормально
- $n = 128K$: $128K \times 128K \approx 16 \times 10^9$ значений — **нереально**
- Дорогое чтение/запись из HBM (High Bandwidth Memory)

### 1.2. FlashAttention — Dao et al., 2022

#### Идея

Стандартный attention делает несколько round-trips между SRAM (быстрая, ~20 MB) и HBM (медленная, ~80 GB). FlashAttention **всё вычисляет в SRAM** за один проход.

**Ключевой трюк:** tiling — разбиение $Q$, $K$, $V$ на блоки, которые помещаются в SRAM.

#### Алгоритм (упрощённо)

```
for each block Q_i, K_j, V_j:
    1. Load Q_i, K_j, V_j → SRAM
    2. Compute S_ij = Q_i × K_j^T  в SRAM
    3. Compute softmax(S_ij) с online rescaling
    4. Accumulate partial output O_i

Output: O (финальный attention)
```

#### Сложность

- **HBM access:** $O(n^2 \cdot d^2 / M)$ где $M$ — размер SRAM
- Против $O(n \cdot d + n^2)$ для стандартного attention
- На практике: **2–4× быстрее** стандартного, **10–20× меньше памяти**

#### FlashAttention-2

- Перераспределение работы между блоками
- Увеличение параллелизма
- Ещё 2× ускорение над FlashAttention-1

### 1.3. Sparse Attention

**Идея:** Вычислять attention только для подмножества пар токенов.

#### Типы разреженности

| Тип | Описание | Сложность |
|---|---|---|
| **Sliding window** | Внимание только к окрестности $w$ | $O(n \cdot w)$ |
| **Dilated sliding** | Окно с шагом $d$ для покрытия | $O(n \cdot w)$ |
| **Global tokens** | Несколько токенов видят всё | $O(n \cdot g)$ |
| **Random attention** | Случайные пары | $O(n \cdot r)$ |
| **Strided attention** | Периодический паттерн | $O(n \cdot s)$ |

#### BigBird / Longformer

Комбинация: sliding window + global tokens + random:

$$
\text{Attention}_{\text{sparse}} = \text{Window}(n, w) + \text{Global}(n, g) + \text{Random}(n, r)
$$

Сложность: $O(n \cdot (w + g + r))$ вместо $O(n^2)$

#### Сравнение

| Модель | Сложность | Качество | Применение |
|---|---|---|---|
| Full Attention | $O(n^2)$ | Эталон | Длина ≤ 4096 |
| FlashAttention | $O(n^2)$ (но быстрее) | Идентичное | До 128K |
| Longformer | $O(n \cdot w)$ | 95–99% | Документы, геномы |
| BigBird | $O(n \cdot (w+g+r))$ | 97–99% | Длина до 4096 |
| Reformer (LSH) | $O(n \log n)$ | 90–95% | Очень длинные |

---

### 1.4. Multi-Query Attention (MQA) — Shazeer, 2019

**Идея:** Все query-головы **разделяют** одну пару K,V.

#### MHA (стандарт)

$$
Q \in \mathbb{R}^{n \times d}, \quad K \in \mathbb{R}^{n \times d}, \quad V \in \mathbb{R}^{n \times d}
$$

Каждая из $H$ голов:
- $Q_h \in \mathbb{R}^{n \times d_h}$, $K_h \in \mathbb{R}^{n \times d_h}$, $V_h \in \mathbb{R}^{n \times d_h}$

**Всего:** $3 \cdot H \cdot n \cdot d_h$ параметров K,V на шаг.

#### MQA

- Одна голова $K \in \mathbb{R}^{n \times d_h}$ и одна $V \in \mathbb{R}^{n \times d_h}$ для всех голов
- $Q$ — обычные $H$ голов

**KV-cache MQA:** $2 \cdot L \cdot d_h \cdot T$ — в $H$ раз меньше!

**Эффект:**
- **KV-cache:** в $H$ раз меньше
- **Throughput:** +30–50% за счёт меньшего memory bandwidth
- **Качество:** почти идентичное (в больших моделях)

### 1.5. Grouped-Query Attention (GQA) — Ainslie et al., 2023

**Компромисс MHA ↔ MQA:** $G$ групп KV для $H$ голов внимания.

$$
H_Q = H, \quad H_K = G, \quad \text{где } G < H
$$

**Пример:**
- LLaMA-2-70B: $H = 64, G = 8$
- LLaMA-3-8B: $H = 32, G = 8$
- PaLM: $H = 48, G = 6$

**Сравнение:**

| Тип | $H_K$ | KV-cache размер | Качество | Latency |
|---|---|---|---|---|
| MHA | $H$ | $2 \cdot L \cdot H \cdot d_h \cdot T$ | Эталон | Базовая |
| GQA | $G$ ($1 < G < H$) | $2 \cdot L \cdot G \cdot d_h \cdot T$ | ~99.5% | Быстрее |
| MQA | 1 | $2 \cdot L \cdot 1 \cdot d_h \cdot T$ | ~99% | Самая быстрая |

---

## 2. Квантизация (Quantization)

### 2.1. Основы

**Идея:** Хранить веса и/или активации в формате с меньшей разрядностью.

#### Типы квантизации

| Тип | Разрядность | Память (7B модель) | Качество |
|---|---|---|---|
| fp32 | 32 бит | 28 GB | Эталон |
| fp16 / bf16 | 16 бит | 14 GB | Идентичное |
| INT8 | 8 бит | 7 GB | 99.9% |
| INT4 | 4 бит | 3.5 GB | 97–99% |
| NF4 (QLoRA) | 4 бит normal float | 3.5 GB | 99% |
| INT3 | 3 бит | 2.6 GB | 95–97% |
| INT2 | 2 бит | 1.75 GB | 85–90% |

#### Формула квантизации

**Uniform quantization:**

$$
x_q = \text{round}\left(\frac{x - \text{min}}{\Delta}\right), \quad \Delta = \frac{\text{max} - \text{min}}{2^b - 1}
$$

**Обратное преобразование:**

$$
x_{\text{deq}} = x_q \cdot \Delta + \text{min}
$$

#### Symmetric vs Asymmetric

| Тип | min/max | Пример |
|---|---|---|
| Symmetric | $-\max(|x|), \max(|x|)$ | INT8: [-127, 127] |
| Asymmetric | $\min(x), \max(x)$ | INT8: [0, 255] |

#### Per-tensor vs Per-channel vs Per-group

| Уровень | Гранулярность | Качество |
|---|---|---|
| Per-tensor | 1 scale для всех весов | Низкое (если разброс большой) |
| Per-channel | 1 scale на канал/нейрон | Среднее |
| Per-group | 1 scale на группу 32–128 весов | Хорошее |

---

### 2.2. PTQ (Post-Training Quantization)

**Без дообучения:** просто квантизуем веса обученной модели.

**Проблема:** Outliers в активациях LLM — некоторые значения на порядки больше остальных.

**Решение:**
1. SmoothQuant — сглаживание активаций перед квантизацией
2. AWQ (Activation-aware Weight Quantization) — учёт важности по активациям

---

### 2.3. GPTQ — Frantar et al., 2023

**Основа:** Optimal Brain Quantization (OBQ) — итеративная квантизация с оптимизацией оставшихся весов.

#### Алгоритм (упрощённо)

1. Берём калибровочный датасет (128–1024 примеров)
2. Для каждого слоя:
   - Вычисляем Hessian $H = 2XX^T$ (где $X$ — активации)
   - Итеративно (одна строка матрицы весов за раз):
     - Квантизуем один вес $w_q = \text{quant}(w)$
     - Вычисляем ошибку: $\delta = w_q - w$
     - Обновляем остальные веса, чтобы компенсировать: $\Delta w = -\delta \cdot H^{-1}_{:,q} / H^{-1}_{q,q}$

#### Сложность

- $O(d_{\text{row}} \cdot d_{\text{col}}^2)$ на строку матрицы
- На практике: ~4 часа для LLaMA-65B на A100
- Лавинообразный? Нет — GPU-friendly

#### Результаты

| Метод | PPL (Wikitext2) | Размер | Потеря |
|---|---|---|---|
| fp16 | 5.04 | 13.5 GB | — |
| GPTQ (INT4-g128) | 5.05 | 3.9 GB | ~0.01 |
| GPTQ (INT3-g128) | 5.36 | 3.0 GB | ~0.32 |

---

### 2.4. QLoRA — Dettmers et al., 2023

**Идея:** LoRA + 4-bit NormalFloat (NF4) квантизация base модели + paged attention.

#### NF4 (NormalFloat4)

Специальный 4-битный формат для нормально распределённых весов:

- Разбиение распределения $\mathcal{N}(0, \sigma)$ на $2^4 = 16$ интервалов равной площади
- Больше точности в плотных областях (около 0)
- Меньше — в хвостах

#### Двойная квантизация (Double Quantization)

1. Квантизация весов в NF4
2. Квантизация **scaling factors** в INT8

**Результат:** 4-битная модель занимает в 2× меньше места, чем INT8

#### Пайплайн QLoRA

```
Base model (NF4) ← frozen
LoRA adapters (fp16) ← trainable

Forward pass:
  1. Dequantize NF4 → fp16 (on-the-fly)
  2. Forward through dequantized weights + LoRA
  
Backward pass (gradients):
  3. Градиенты только через LoRA адаптеры
  4. Веса base model не обновляются
```

#### Результаты QLoRA

- LLaMA-65B: 4-bit → 48 GB → **помещается на 1 A100 (80GB)**
- Качество: ~99.5% от fp16 LoRA
- Время обучения: ~24h на A100 для 65B модели

---

## 3. Trade-off памяти/скорости/качества

### Диаграмма выбора

```
                    Высокое качество
                         │
            fp16 MHA     │     fp16 GQA
                         │
      INT8 MHA ──────────┼───────── INT8 GQA
                         │
      INT4 MHA ──────────┼───────── INT4 GQA
                         │
            Низкая ──────┼───── Высокая скорость
            память       │
```

### Количественное сравнение (LLaMA-7B, T=2048)

| Конфиг | Размер | Latency (мс) | Качество (PPL↓) | Память GPU |
|---|---|---|---|---|
| fp16, MHA | 14 GB | 1.0× | 5.68 (1.0×) | ~28 GB |
| fp16, GQA (8) | 14 GB | 0.8× | 5.70 (1.003×) | ~26 GB |
| fp16, MQA | 14 GB | 0.7× | 5.72 (1.007×) | ~24 GB |
| INT8, MHA | 7 GB | 1.1× (dequant) | 5.70 (1.003×) | ~13 GB |
| INT4, MHA | 3.5 GB | 1.0× | 5.75 (1.01×) | ~8 GB |
| INT4, GQA | 3.5 GB | 0.9× | 5.77 (1.02×) | ~7 GB |
| INT3, MHA | 2.6 GB | 1.2× | 6.10 (1.07×) | ~6 GB |

### Практические рекомендации

| Сценарий | Рекомендация |
|---|---|
| Максимум качества, 1+ H100 | fp16 + FlashAttention + GQA |
| Production на A100 | INT8 + GQA |
| Один GPU 24 GB | INT4 (GPTQ) |
| CPU inference | INT4 (GGUF/llama.cpp) |
| Fine-tuning на одном GPU | QLoRA (NF4) |
| Очень длинный контекст (64K+) | FlashAttention-2 + INT4 KV-cache |
| Edge/mobile | INT4 + MQA |

---

## 4. Дополнительные техники сжатия

### Pruning (прореживание)

- **Unstructured:** удаление отдельных весов (SparseGPT)
- **Structured:** удаление целых нейронов/голов/слоёв
- SparseGPT (2023): 50% разреженность с минимальной потерей

### Distillation (дистилляция)

- [[DL 24 - Knowledge Distillation|Маленькая модель (student)]] учится имитировать большую (teacher)
- Orca, Phi, TinyLLaMA
- Loss: $L_{\text{KD}} = \text{KL}(P_{\text{teacher}} \| P_{\text{student}})$

### [[DL 42 - Инференс LLM|Speculative Decoding]]

- Маленькая "drafter" модель генерирует черновик
- Большая модель проверяет/подтверждает
- Ускорение 2–3× без потери качества

---

## 5. Ключевые выводы

1. **Efficient attention** решает $O(n^2)$ проблему: FlashAttention (без потерь, через SRAM tiling) и MQA/GQA (сжатие KV-cache)
2. **Квантизация** — главный способ сжатия: INT8 (почти без потерь), INT4/GPTQ (минимальные потери), INT2/NF4 (для LoRA)
3. **QLoRA** сочетает NF4 квантизацию + LoRA — позволяет fine-tune 65B на одном A100
4. Ключевой **trade-off**: INT4 теряет ~1% качества, но в 4× меньше памяти
5. **GQA** — современный стандарт (LLaMA-3, Gemma, Mistral): компромисс между качеством MHA и скоростью MQA

---

**Связанные вопросы:**
- [[DL 42 - Инференс LLM]] — практические метрики и проблемы, которые решает сжатие (KV-cache, latency)
- [[DL 44 - Декодирование LLM]] — влияние стратегий декодирования на скорость и качество
- [[DL 24 - Knowledge Distillation]] — альтернативный метод сжатия через teacher-student
- [[DL 37 - Масштабирование и MoE]] — масштабирование моделей, MoE как способ сбалансировать качество и скорость
