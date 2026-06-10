# 33. Архитектура Transformer

## 1. Общая архитектура (Vaswani et al., 2017)

**Ключевая идея:** "Attention is All You Need" — убираем RNN/CNN полностью. Вся обработка — через механизм внимания.

### 1.1 Структура

Transformer = **Encoder** (слева) + **Decoder** (справа).

**Encoder:**
- $N$ одинаковых слоёв (оригинал: $N=6$).
- Каждый слой: Self-Attention → Add & Norm → FFN → Add & Norm.

**Decoder:**
- $N$ одинаковых слоёв (оригинал: $N=6$).
- Каждый слой: Masked Self-Attention → Add & Norm → Cross-Attention → Add & Norm → FFN → Add & Norm.

### 1.2 Residual Connections (остаточная связь)

$$
\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))
$$

**Зачем:**
- Решает vanishing gradients (градиент течёт напрямую).
- Позволяет строить глубокие сети (до 100+ слоёв).
- Улучшает сходимость.

**Два варианта:**

| Вариант | Формула | Используется в |
|---|---|---|
| **Post-LN (original)** | $\text{LayerNorm}(x + \text{Sublayer}(x))$ | Оригинальный Transformer |
| **Pre-LN** | $x + \text{Sublayer}(\text{LayerNorm}(x))$ | GPT-2/3, современные модели |

Pre-LN более стабилен (меньше проблем с exploding activations).

### 1.3 Layer Normalization (LayerNorm)

$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
$$

где $\mu = \frac{1}{d}\sum_{i=1}^{d} x_i$, $\sigma^2 = \frac{1}{d}\sum_{i=1}^{d} (x_i - \mu)^2$.

**Отличие от BatchNorm:**
- LayerNorm: нормализация по признакам (по dimension) — одинаково для всех элементов батча.
- BatchNorm: нормализация по батчу (по батч-измерению).

**Зачем LayerNorm в Transformer:**
- Независимость от длины последовательности (разные предложения — разная длина).
- Работает с batch_size = 1 (текст часто разной длины).
- Стабильность при обучении.

---

## 2. Self-Attention (самовнимание)

### 2.1 Определение

Каждый элемент последовательности "смотрит" на все остальные элементы (включая себя).

**Формально:** $Q, K, V$ — линейные проекции одного и того же входа $X$:
$$
Q = X \cdot W^Q,\quad K = X \cdot W^K,\quad V = X \cdot W^V
$$

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

### 2.2 Свойства

- **Пермутационно-симметрично:** если переставить входы, выход переставится так же.
- **Глобальное взаимодействие:** каждый токен видит все остальные за один слой.
- **Параллелизуемо:** все взаимодействия считаются одной матричной операцией.

### 2.3 Convolution vs Self-Attention vs RNN

| Свойство | CNN | RNN | Self-Attention |
|---|---|---|---|
| **Дальность взаимодействия** | Ограничена ядром | Теоретически вся последовательность | Вся последовательность |
| **Параллелизация** | Да (по пространству) | Нет (последовательная) | Да |
| **Путь информации** | $O(\log_k T)$ | $O(T)$ | $O(1)$ |
| **Локальная информация** | Хорошо (receptive field) | Хорошо (шаг за шагом) | Нет приоритета локальности |
| **Сложность** | $O(T)$ | $O(T)$ | $O(T^2)$ |

---

## 3. Masked Self-Attention

**Где:** в декодере.

**Зачем:** предотвращает "подглядывание" в будущие токены при авторегрессивной генерации.

**Механизм:** к матрице scores $S = QK^T$ добавляется маска:

$$
S_{ij} = \begin{cases}
q_i \cdot k_j & \text{если } j \le i \text{ (позиция уже сгенерирована)} \\
-\infty & \text{если } j > i \text{ (будущее — запрещено)}
\end{cases}
$$

После softmax: $\exp(-\infty) = 0$ → будущие токены не влияют.

**Альтернативное название:** causal masking (причинная маска).

---

## 4. Cross-Attention

**Где:** в декодере (между self-attention и FFN).

**Формула:** $Q$ — из декодера (предыдущего слоя), $K, V$ — из энкодера (выход последнего слоя энкодера).

**Назначение:** декодер "смотрит" на входную последовательность при генерации каждого токена.

**Это аналог attention из seq2seq (Bahdanau/Luong), но:**
- Используется scaled dot-product (не additive).
- Multi-head.

---

## 5. Feed-Forward Network (FFN)

### 5.1 Структура

Два полносвязных слоя с функцией активации:

$$
\text{FFN}(x) = W_2 \cdot \text{act}(W_1 x + b_1) + b_2
$$

### 5.2 Варианты активаций

| Модель | Активация | Формула |
|---|---|---|
| Transformer original | ReLU | $\max(0, x)$ |
| GPT-2, BERT | GELU | $x \cdot \Phi(x)$ (Gaussian Error Linear Unit) |
| LLaMA, PaLM | SwiGLU | $\text{swish}(xW_1) \otimes (xW_3) \cdot W_2$ |
| T5 | ReLU | $\max(0, x)$ |

**GELU:** $\text{GELU}(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}(x/\sqrt{2})\right] \approx 0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$

**SwiGLU:** $\text{SwiGLU}(x) = (\text{Swish}(xW_1) \odot xW_3)W_2$ — умножает два преобразования, требует 3 матрицы вместо 2.

### 5.3 Расширение (Expansion factor)

$$
d_{\text{ff}} = 4 \cdot d_{\text{model}} \quad (\text{оригинал: } 2048 = 4 \cdot 512)
$$

**Смысл:** FFN расширяет размерность в 4 раза (больше ёмкости), потом сжимает обратно.

### 5.4 Роль FFN (из исследований)

- **Первые слои:** FFN кодирует синтаксические шаблоны.
- **Средние слои:** семантическая информация.
- **Последние слои:** специфические для задачи.
- Memory retrieval: FFN ≈ key-value memory (Geva et al., 2021).

---

## 6. Positional Encoding

### 6.1 Проблема

Self-attention пермутационно-симметричен — не различает порядок слов.
"Я люблю кошек" = "Кошек люблю я" (с точки зрения attention).

**Решение:** добавить информацию о позиции.

### 6.2 Sinusoidal Positional Encoding (original Transformer)

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$
$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

где $pos$ — позиция токена, $i$ — размерность эмбеддинга.

**Свойства:**
- **Детерминированное:** не обучается.
- **Относительные позиции:** $PE_{pos+k}$ можно выразить как линейную функцию от $PE_{pos}$:
  $$
  \begin{bmatrix}
  \sin(pos + k) \\
  \cos(pos + k)
  \end{bmatrix}
  =
  \begin{bmatrix}
  \cos(k) & \sin(k) \\
  -\sin(k) & \cos(k)
  \end{bmatrix}
  \cdot
  \begin{bmatrix}
  \sin(pos) \\
  \cos(pos)
  \end{bmatrix}
  $$
- **Экстраполяция:** может работать с длинами, не виденными на обучении.
- **Периодичность:** разные частоты на разных размерностях (низкие частоты — долгие зависимости, высокие — короткие).

### 6.3 Learned Positional Encoding

**Просто:** обучаемая матрица $P \in \mathbb{R}^{\text{max\_len} \times d_{\text{model}}}$.

$$
x_i' = x_i + P_i
$$

**Где используется:** BERT (learned), GPT-2 (learned).

**Недостаток:** не экстраполирует — максимум на макс. длину обучения.

**Преимущество:** может выучить оптимальное представление позиции (лучше синусоиды на задачах с фиксированной длиной).

### 6.4 RoPE (Rotary Position Embedding, Su et al., 2021)

**Идея:** вращение query/key векторов на угол, пропорциональный позиции.

**Формула:**
$$
q_i' = \begin{bmatrix}
\cos(i\theta_0) & -\sin(i\theta_0) \\
\sin(i\theta_0) & \cos(i\theta_0)
\end{bmatrix}
\cdot
\begin{bmatrix}
q_{2i} \\
q_{2i+1}
\end{bmatrix}
$$

Аналогично для key-векторов.

**Свойства:**
- **Относительное позиционирование:** attention score $q_i \cdot k_j$ зависит от $i-j$ (а не от $i$ и $j$ отдельно).
- **Убывание веса:** с ростом расстояния вес убывает (soft locality bias).
- **Экстраполяция:** работает на длинах, не виденных при обучении.
- **Современный стандарт:** LLaMA, Mistral, GPT-4, PaLM, Gemma.

### 6.5 Сравнение методов

| Метод | Обучается | Относительный | Экстраполяция | Использование |
|---|---|---|---|---|
| Sinusoidal | Нет | Да ($PE_{pos+k}$ линейна) | Да | Transformer original |
| Learned | Да | Нет | Нет | BERT, GPT-2 |
| RoPE | Нет | Да (явно) | Да (лучшая) | LLaMA, Mistral, GPT-4 |
| ALiBi | Нет | Да (линейное смещение) | Да (отличная) | BLOOM, MPT |

**ALiBi (Press et al., 2022):** добавляет линейный штраф к attention scores: $q_i \cdot k_j - m|i-j|$.

---

## 7. Полная архитектура (в одной схеме)

```
Encoder (N layers)          Decoder (N layers)
     │                           │
  ┌──┴──┐                     ┌──┴──┐
  │Input│                     │Output│
  │Emb  │                     │Emb   │
  └──┬──┘                     └──┬──┘
     │+PE                        │+PE
  ┌──┴──┐                     ┌──┴──┐
  │MHA │                      │Mask │
  │Self│                      │Self │
  │Attn│                      │Attn │
  └──┬──┘                     └──┬──┘
     │+ & LN                     │+ & LN
  ┌──┴──┐                     ┌──┴──┐
  │FFN  │                      │Cross│
  │(2MLP)│                     │Attn │
  └──┬──┘                     └──┬──┘
     │+ & LN                     │+ & LN
     │                        ┌──┴──┐
     │                        │FFN  │
     │                        └──┬──┘
     │                           │+ & LN
  ───┤                           ├───
     └──────► Cross-Attn ◄───────┘
                Q ← Dec
                K,V ← Enc
```

---

## 8. Итог

| Компонент | Формула | Назначение |
|---|---|---|
| Self-Attention | $\text{softmax}(QK^T/\sqrt{d_k})V$ | Глобальное взаимодействие токенов |
| Masked Self-Attention | Causal mask | Предотвращение подглядывания в будущее |
| Cross-Attention | Q от декодера, K,V от энкодера | Связь энкодера и декодера |
| FFN | $W_2 \cdot \text{act}(W_1 x + b_1) + b_2$ | Нелинейное преобразование, "память" |
| Residual | $x + \text{Sublayer}(x)$ | Обход vanishing gradients |
| LayerNorm | $\frac{x-\mu}{\sigma} \cdot \gamma + \beta$ | Стабилизация обучения |
| Pos Encoding | Sinusoidal/Learned/RoPE | Информация о порядке токенов |
| Multi-Head | Concat($H$ heads) | Разные типы отношений |

Подробнее о механизме внимания — см. [[DL 32 - Механизм внимания]].

---

**Связанные вопросы:**
- [[DL 32 - Механизм внимания]] — детальная теория QKV и multi-head attention
- [[DL 34 - Pretraining в NLP]] — как Transformer предобучается на больших данных
- [[DL 35 - BERT и T5]] — encoder-only (BERT) и encoder-decoder (T5) на базе Transformer
- [[DL 36 - GPT]] — decoder-only (GPT) на базе Transformer
- [[DL 51 - Vision Transformer]] — ViT адаптирует Transformer для изображений
