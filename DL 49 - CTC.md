# DL 49 — CTC

> **Blank token, collapse operation, CTC loss, greedy/beam decoding, ограничения.
> Смежные темы: [[DL 48 - ASR]], [[DL 50 - Архитектуры ASR и TTS]].**

---

## 1. CTC (Connectionist Temporal Classification) — Graves et al., 2006

### Мотивация

В [[DL 48 - ASR|ASR]] аудио последовательность $X$ (длины $T$) длиннее текстовой последовательности $Y$ (длины $U$). При этом **alignment** между $X$ и $Y$ неизвестен. CTC решает проблему, вводя **латентную переменную alignment** и суммируя по всем возможным alignments.

### Основная идея

CTC вводит дополнительный **blank token $\epsilon$** (пустой символ), который разрешает модели "не говорить" на протяжении нескольких аудио-фреймов.

На каждом временном шаге $t$ модель предсказывает распределение над расширенным алфавитом:

$$
P(a_t \mid X), \quad a_t \in \mathcal{A}' = \mathcal{A} \cup \{\epsilon\}
$$

где $\mathcal{A}$ — алфавит токенов (символы, фонемы, subwords), $\epsilon$ — blank.

### Размерности

- Вход: $X \in \mathbb{R}^{T \times d}$ (T фреймов, d — размерность признаков)
- Выход CTC: $a \in \mathcal{A}'^T$ (последовательность alignments длины T)
- Транскрипция: $Y \in \mathcal{A}^U$ (U токенов, где $U \leq T$)

---

## 2. Blank Token

### Назначение

1. **Разделение повторяющихся токенов** — в CTC одинаковые токены рядом collapse в один, но blank между ними сохраняет дублирование
2. **Неактивные фреймы** — модель может "молчать" (blank) в промежутках между произносимыми звуками

### Роль blank в collapse

| Alignment (с blank) | Collapsed | Пояснение |
|---|---|---|
| h h _ e e _ l _ l o o | hello | Blanks удаляются |
| h _ h e l l o | hhello → hlo | Без blank — collapse дублей |
| h _ h e l _ l o | hhlo | Blank не влияет |

### Пример с дублями

```
"hello" → как CTC различает "l" (две l) от одной?

С blank:
  T: h _ e l _ l o   → collapse: hello ✅
Без blank:
  T: h e l l o       → collapse: helo ❌ (два раза l → одна)

С blank между одинаковыми:
  T: h e l _ l o     → collapse: hello ✅ (blank разделяет)
```

---

## 3. Collapse Operation ($\mathcal{B}$)

**Collapse** — отображение из alignment $a$ в транскрипцию $y$:

$$
\mathcal{B}(a) = y = (y_1, \dots, y_U)
$$

**Правила collaps'а:**
1. Удалить все blank'и ($\epsilon$)
2. Удалить **последовательные дубликаты** одинаковых токенов

### Формально

$$
\mathcal{B}(a) = \text{unique}\left(\text{remove}(a, \epsilon)\right)
$$

где `unique` — удаление последовательных повторений.

### Примеры

| Alignment $a$ | $\mathcal{B}(a)$ |
|---|---|
| h h _ e e l l _ o o | helo |
| h _ e l _ l o | hello |
| _ _ h e l l o _ _ | helo |
| h h h e e l l o | helo |

---

## 4. CTC Loss

### Постановка

Нам нужно вычислить вероятность транскрипции $Y$ при данной последовательности аудио $X$:

$$
P(Y \mid X) = \sum_{a \in \mathcal{B}^{-1}(Y)} P(a \mid X)
$$

где $\mathcal{B}^{-1}(Y)$ — множество всех alignments, которые collapse в $Y$.

### Вычисление вероятности alignment

Принимая **conditional independence assumption** (наивное предположение):

$$
P(a \mid X) = \prod_{t=1}^T P(a_t \mid X)
$$

**Важно:** это предположение означает, что ASC-модель не моделирует зависимости между соседними токенами alignment. На практике это приводит к тому, что CTC не учит языковую модель (нет контекста между токенами).

### CTC Loss (Negative Log-Likelihood)

$$
\mathcal{L}_{\text{CTC}} = -\log P(Y \mid X) = -\log \sum_{a \in \mathcal{B}^{-1}(Y)} \prod_{t=1}^T P(a_t \mid X)
$$

### Вычисление через Forward-Backward

Прямое суммирование по всем $\mathcal{B}^{-1}(Y)$ имеет экспоненциальную сложность $O(|V|^T)$.

**Решение:** динамическое программирование (Forward-Backward algorithm) за $O(T \cdot U)$.

#### Forward переменные

Пусть $\widetilde{Y}$ — расширенная последовательность с blank между токенами:

$$
\widetilde{Y} = [\epsilon, y_1, \epsilon, y_2, \epsilon, \dots, \epsilon, y_U, \epsilon]
$$

Длина $\widetilde{Y}$: $2U + 1$.

**Forward variable** $\alpha[t][s]$ — вероятность того, что после обработки $t$ фреймов мы находимся в состоянии $s$ расширенной последовательности:

$$
\alpha[t][s] = P(a_{1:t} \mid X) \quad \text{где} \quad \mathcal{B}(a_{1:t}) \text{ оканчивается на } \widetilde{Y}_s
$$

**Рекурсия:**

$$
\alpha[t][s] = P(a_t = \widetilde{Y}_s \mid X) \cdot
\begin{cases}
\alpha[t-1][s] + \alpha[t-1][s-1] & \text{если } \widetilde{Y}_s = \epsilon \\
\alpha[t-1][s] + \alpha[t-1][s-1] + \alpha[t-1][s-2] & \text{если } \widetilde{Y}_s = \widetilde{Y}_{s-2}
\end{cases}
$$

**Backward variable** $\beta[t][s]$ — симметрично с конца.

**Итоговая вероятность:**

$$
P(Y \mid X) = \alpha[T][2U+1] + \alpha[T][2U]
$$

(в конце мы можем быть на blank или на последнем токене)

---

## 5. Greedy Decoding (для CTC)

### Алгоритм

1. Для каждого фрейма $t$ выбираем $a_t^* = \arg\max P(a_t \mid X)$
2. Применяем collapse $\mathcal{B}(a^*)$

```
T = 10
Предсказания: [h, h, _, e, h, l, l, _, l, o]
Greedy argmax: h h _ e h l l _ l o
Collapse: helo   ❌ (пропущена вторая l)
```

### Простота

- $O(T)$ — очень быстро
- Детерминированный
- Часто достаточен для чистого аудио

### Недостатки

- Не учитывает неопределённость модели (argmax слишком жёсткий)
- Может пропускать редкие, но правильные слова
- Нет возможности исправить ошибки

---

## 6. CTC Beam Search

### Идея

Хранить $B$ лучших префиксов (частичных транскрипций) с накопленной вероятностью.

### Алгоритм (упрощённо)

```
Initialize: beams = [{prefix: "", prob: 1.0}]

for t = 1 to T:
    new_beams = []
    for each beam in beams:
        for each token v in V':
            prob = beam.prob × P(a_t = v | X)
            new_prefix = extend(beam.prefix, v, collapse)
            new_beams.append({prefix: new_prefix, prob})

    Keep top-B beams by probability
```

### Учёт Language Model

Часто CTC beam search интегрирует **внешний языковую модель (LM)**:

$$
\text{score}(y) = \log P_{\text{CTC}}(y \mid X) + \lambda \log P_{\text{LM}}(y) + \beta \cdot |y|
$$

где:
- $P_{\text{CTC}}$ — вероятность от акустической модели
- $P_{\text{LM}}$ — языковая модель (n-gram, Transformer LM)
- $\lambda$ — weight LM (типично 0.1–1.0)
- $\beta$ — поощрение длины (length reward)

### Prefix Beam Search

Более эффективная версия: группировка префиксов, которые collapse в одно и то же:

```
"h e" и "h h h e" → collapse: "he" → объединяем вероятности
```

Сложность: $O(T \cdot B \cdot |V|)$ — всё ещё дорого для больших словарей.

### Сравнение

| Метод | Качество | Скорость | Использование |
|---|---|---|---|
| Greedy | Худшее | $O(T)$ | Быстрый прототип |
| Beam (без LM) | Среднее | $O(T \cdot B)$ | Разговорные |
| Beam (с LM) | Лучшее | $O(T \cdot B \cdot \log|V|)$ | Производство |

---

## 7. Ограничения CTC

### 1. Conditional Independence Assumption

$$
P(a_t \mid X) \perp P(a_{t'} \mid X) \text{ для } t \neq t'
$$

**Следствие:** CTC не учит контекст между токенами — каждое предсказание делается независимо. Модель не может "исправить" предыдущую ошибку на основе следующего токена.

### 2. Blank token overhead

- Модель должна явно научиться предсказывать blank на тишине
- Для быстрой речи может не хватать blank'ов
- Сложность: нельзя просто сказать "тишина" — нужно пустой токен

### 3. Проблема с повторяющимися символами

```
"catty" (с двумя t подряд) → collapse может съесть одну t
"mississippi" → 2 s → 2 p → нужно уметь ставить blank
```

### 4. Alignment path explosion

Даже с forward-backward, CTC суммирует по всем alignments. Для $T = 100$, $U = 10$ количество alignments: $\binom{T+U}{T} \approx 10^{30}$.

### 5. Отсутствие языковой модели

- CTC сам по себе не учит контекст
- Для хорошего качества нужен внешний LM (n-gram, Transformer LM)
- Это увеличивает сложность и latency

### 6. Длина последовательности

- **Требование:** $T \geq U$ (фреймов >= токенов)
- Проблема для очень длинных текстов при коротком аудио (не бывает в ASR)
- Reverse: много blank'ов → разреженные градиенты

### CTC vs RNN-T

| Критерий | CTC | [[DL 50 - Архитектуры ASR и TTS|RNN-T]] |
|---|---|---|
| Independence assumption | ✅ (conditional) | ❌ (учёт контекста) |
| Blank token | Один | Один |
| Prediction network | Нет | Есть (текстовый контекст) |
| LM integration | Внешний | Встроенный |
| Streaming | Да | Да |
| Качество | Хорошее | Отличное |
| Сложность | Низкая | Средняя |

---

## 8. Когда CTC — хороший выбор?

| Сценарий | Рекомендация |
|---|---|
| Маленькая модель (edge device) | CTC (эффективен) |
| Быстрый прототип | CTC |
| Высокое качество | [[DL 50 - Архитектуры ASR и TTS|RNN-T]] или LAS |
| Streaming (real-time) | CTC или RNN-T |
| Многоязычный | CTC с BPE |
| Есть ресурсы (GPU) | RNN-T / Conformer |

---

## 9. Ключевые выводы

1. **CTC** решает проблему alignment через введение **blank token** и **collapse operation**: $\mathcal{B}(a)$ — удаление blank'ов и дубликатов
2. **CTC Loss** $= -\log P(Y \mid X)$ — суммирование по всем alignment через forward-backward алгоритм за $O(T \cdot U)$
3. **Greedy decoding** — простой argmax + collapse, но suboptimal
4. **Beam search** с внешней LM даёт лучшее качество
5. **Главное ограничение:** conditional independence assumption — модель не видит контекст между токенами
6. **CTC > [[DL 50 - Архитектуры ASR и TTS|RNN-T]]** по простоте, **RNN-T > CTC** по качеству за счёт prediction network

---

**Связанные вопросы:**
- [[DL 48 - ASR]] — постановка задачи ASR, метрики WER/CER, проблема выравнивания
- [[DL 50 - Архитектуры ASR и TTS]] — RNN-T, LAS, сравнение с CTC, TTS pipeline
