# 29. Проблемы RNN и Gated-архитектуры

## 1. Vanishing and Exploding Gradients

### 1.1 Причина

При BPTT градиент через $T$ шагов:

$$
\frac{\partial h_T}{\partial h_1} = \prod_{t=2}^{T} \frac{\partial h_t}{\partial h_{t-1}} = \prod_{t=2}^{T} W_{hh}^T \cdot \text{diag}(f'(h_{t-1}))
$$

Для $\tanh$: $f'(x) \in (0, 1)$, максимальное значение 1 в 0.
Для sigmoid: $f'(x) \in (0, 0.25]$.

**Произведение якобианов:** если $\|W_{hh}\| < 1$, градиент → 0 (vanishing). Если $\|W_{hh}\| > 1$, градиент → ∞ (exploding).

### 1.2 Последствия Vanishing

- **Долгосрочные зависимости не выучиваются.**
- Параметры дальних шагов почти не обновляются.
- RNN "забывает" первые части последовательности.

### 1.3 Последствия Exploding

- **NaN** в параметрах.
- Нестабильное обучение.
- Решение: **gradient clipping**.

### 1.4 Gradient Clipping

**Нормирование градиента:**
$$
g \leftarrow \begin{cases}
\frac{\text{threshold}}{\|g\|} \cdot g & \text{если } \|g\| > \text{threshold} \\
g & \text{иначе}
\end{cases}
$$

- Типичный порог: 5.0–10.0.
- Не решает vanishing, только exploding.

### 1.5 Неэффективность ReLU

ReLU имеет градиент 0 для отрицательных входов (dead neurons). Для RNN это ещё хуже: если $h_t$ попадает в отрицательную область, информация теряется.

---

## 2. LSTM (Long Short-Term Memory, Hochreiter & Schmidhuber, 1997)

### 2.1 Идея

**Ключевое нововведение:** отдельная **cell state** ($C_t$) — конвейерная лента, через которую информация течёт почти без изменений (линейные преобразования, меньше умножений → меньше затухание).

### 2.2 Архитектура LSTM

LSTM управляет информацией через **gates** (вентили):

1. **Forget gate** — что забыть из cell state.
2. **Input gate** — что записать в cell state.
3. **Output gate** — что выдать как скрытое состояние.

### 2.3 Формулы

**Forget gate:**
$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

**Input gate:**
$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
$$

**Candidate cell state:**
$$
\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
$$

**Cell state update:**
$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

**Output gate:**
$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
$$

**Hidden state:**
$$
h_t = o_t \odot \tanh(C_t)
$$

### 2.3 Объяснение gates

| Gate | Сигнал | Назначение |
|---|---|---|
| $f_t$ | sigmoid (0,1) | 0 = забыть полностью, 1 = сохранить всё |
| $i_t$ | sigmoid (0,1) | 0 = не писать, 1 = писать всё |
| $\tilde{C}_t$ | tanh (-1,1) | Кандидат на добавление (новые факты) |
| $o_t$ | sigmoid (0,1) | 0 = скрыть состояние, 1 = показать всё |

**Поэлементное умножение** $\odot$ позволяет избирательно пропускать/блокировать информацию.

### 2.4 Почему LSTM решает vanishing gradient?

$$
\frac{\partial C_t}{\partial C_{t-1}} = f_t
$$

- Если $f_t \approx 1$ (забывать не нужно), градиент ≈ 1 — не затухает.
- Если $f_t \approx 0$, градиент ≈ 0 — информация забыта (это и нужно).

**Cвязь $C_t$ и $C_{t-1}$** — **линейная** (а через $f_t$ управляемая), что принципиально лучше нелинейного $\tanh(W_{hh} h_{t-1})$ в простой RNN.

---

## 3. GRU (Gated Recurrent Unit, Cho et al., 2014)

### 3.1 Упрощение LSTM

GRU объединяет forget и input gates в **update gate**, убирает отдельный cell state.

### 3.2 Формулы GRU

**Reset gate:** насколько использовать прошлое состояние.
$$
r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)
$$

**Update gate:** сколько взять от нового кандидата vs старого состояния.
$$
z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)
$$

**Candidate hidden state:**
$$
\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h)
$$

**Hidden state:**
$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

### 3.3 LSTM vs GRU

| Характеристика | LSTM | GRU |
|---|---|---|
| **Количество gate** | 3 (forget, input, output) | 2 (reset, update) |
| **Cell state** | Есть ($C_t$ и $h_t$) | Нет (только $h_t$) |
| **Параметры** | Больше (4 матрицы на слой) | Меньше (3 матрицы на слой) |
| **Скорость обучения** | Медленнее | Быстрее |
| **Качество** | Немного лучше на очень длинных последовательностях | Сопоставимо, иногда лучше |
| **Память** | Больше | Меньше |

**Практика:** оба работают хорошо. GRU чуть быстрее, LSTM чуть мощнее для очень долгих зависимостей. Часто GRU предпочтительнее как baseline.

---

## 4. Teacher Forcing

### 4.1 Определение

При обучении генеративных моделей (seq2seq) на шаге $t$ декодер получает **истинный** токен $y_{t-1}$ (а не собственное предсказание) в качестве входа.

**Без teacher forcing:**
$$
\hat{y}_t = \text{decoder}(x_t, \hat{y}_{t-1})
$$

**С teacher forcing:**
$$
\hat{y}_t = \text{decoder}(x_t, y_{t-1}^{\text{true}})
$$

### 4.2 Почему это нужно?

- **Сходимость:** модель не накапливает ошибки на этапе обучения.
- **Скорость:** параллельные вычисления (все истинные токены известны сразу).
- **Стабильность:** не застревает в "плохом" режиме.

### 4.3 Проблема teacher forcing

**Exposure bias:** модель на тесте (inference) никогда не видит свои собственные предсказания в качестве входа → накопление ошибок (error propagation).

**Демонстрация:** если модель ошиблась на шаге $t$, все следующие шаги искажены.

---

## 5. Scheduled Sampling (Bengio et al., 2015)

### 5.1 Идея

Постепенный переход от teacher forcing к полной автономной генерации.

### 5.2 Алгоритм

На каждом шаге с вероятностью $\epsilon_t$ подаём предсказание модели, с вероятностью $1-\epsilon_t$ — истинный токен.

**Расписание $\epsilon_t$ (decay schedule):**
- **Linear:** $\epsilon_t = \min(1, \alpha t)$ — растёт линейно.
- **Exponential:** $\epsilon_t = 1 - e^{-\alpha t}$.
- **Inverse sigmoid:** $\epsilon_t = \frac{k}{k + e^{t/k}}$.

### 5.3 Варианты

- **Soft scheduled sampling:** взвешенная смесь всех кандидатов (не только argmax).
- **Curriculum learning:** начинаем с teacher forcing, переходим к свободной генерации.

### 5.4 Недостатки

- Дополнительный гиперпараметр (схема расписания).
- Можно "переучить" модель на своих ошибках.
- Не всегда стабильно.

---

## 6. Практические рекомендации

| Проблема | Решение |
|---|---|
| Vanishing gradients | LSTM/GRU, identity initialization, residual connections |
| Exploding gradients | Gradient clipping (norm threshold 5-10) |
| Долгосрочные зависимости | LSTM > GRU > vanilla RNN |
| Exposure bias | Teacher forcing → Scheduled Sampling |
| Медленное обучение | GRU, Truncated BPTT |
| Переобучение | Dropout (только на non-recurrent связях), регуляризация |

---

## 7. Итог

- **Vanishing/Exploding gradients:** фундаментальная проблема простых RNN из-за произведения якобианов.
- **LSTM:** cell state + 3 gates — конвейер для долгосрочной памяти.
- **GRU:** упрощение LSTM (2 gates, нет cell state) — сопоставимое качество, меньше параметров.
- **Teacher forcing:** обучение на истинных токенах — быстро, но exposure bias.
- **Scheduled Sampling:** плавный переход от teacher forcing к свободной генерации.

---

**Связанные вопросы:**
- [[DL 28 - RNN]] — оригинальная рекуррентная архитектура
- [[DL 30 - Seq2seq и attention]] — seq2seq, где используются LSTM/GRU
- [[DL 41 - Catastrophic Forgetting]] — забывание при дообучении, смежная проблема с forgetting в RNN
