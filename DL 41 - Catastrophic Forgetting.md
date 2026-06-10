# DL 41 — Catastrophic Forgetting

> **Причины, опасность, способы смягчения (Replay, EWC, regularization, KL в RLHF).
> Смежные темы: [[DL 38 - Дообучение LLM]], [[DL 39 - Preference Tuning и RLHF]], [[DL 24 - Knowledge Distillation]].**

---

## 1. Определение

**Catastrophic Forgetting (катастрофическое забывание)** — явление, при котором нейронная сеть полностью теряет способность выполнять ранее выученные задачи после обучения на новой задаче.

### Формально

Пусть модель обучена на задаче $A$ с параметрами $\theta_A$. После обучения на задаче $B$:

$$
\theta_B \leftarrow \theta_A - \eta \nabla_\theta \mathcal{L}_B(\theta_A)
$$

Производительность на задаче $A$ резко падает:

$$
\text{Perf}_A(\theta_B) \ll \text{Perf}_A(\theta_A)
$$

---

## 2. Причины катастрофического забывания

### 2.1. Пересечение параметров

- Все задачи используют **одни и те же параметры** $\theta$
- Градиенты по задаче $B$ могут быть перпендикулярны/противоположны градиентам задачи $A$

### 2.2. Сдвиг распределения (distribution shift)

$$
P_{\text{train}_A}(x, y) \neq P_{\text{train}_B}(x, y)
$$

- Обучение на $B$ настраивает веса под новое распределение
- Старые веса "забываются", т.к. градиенты по $B$ не учитывают $P_A$

### 2.3. Внутренний ковариатный сдвиг

- В глубоких сетях изменение весов в ранних слоях меняет распределения активаций для всех последующих задач

### 2.4. Пластичность vs Стабильность (plasticity-stability dilemma)

> **Stability-plasticity dilemma**: сеть должна быть достаточно пластичной, чтобы учить новое, и достаточно стабильной, чтобы не забывать старое.

### 2.5. Переобучение под новый датасет

- Если датасет $B$ мал, модель быстро запоминает именно его особенности
- При обучении с нулевой инициализацией (fine-tuning) — веса сильно меняются

---

## 3. Опасность для LLM

### Сценарии проявления

| Сценарий | Что забывается | Последствия |
|---|---|---|
| [[DL 38 - Дообучение LLM|SFT]] на новом датасете | Общие знания (world knowledge) | Модель глупеет |
| Instruction Tuning | Базовая генерация языка | Грамматические ошибки |
| [[DL 39 - Preference Tuning и RLHF|RLHF]] (PPO) | Разнообразие ответов | Ответы становятся однотипными |
| Domain adaptation (медицина→юриспруденция) | Предыдущая область | Не может отвечать в старой области |
| Multi-turn fine-tuning | Первые шаги обучения | Исчезают ранние навыки |

### Почему LLM особенно уязвимы

- **Гигантские модели** — миллиарды параметров, адаптированных под огромный корпус
- **Маленький датасет SFT** — может быть в 10⁶ раз меньше pre-training data
- **High learning rate** — быстрое изменение весов
- **Single pass** — мало эпох (обычно 1–3)

---

## 4. Способы смягчения

### 4.1. Experience Replay (Rehearsal)

**Идея:** Хранить примеры из старых задач и добавлять их в обучение.

#### Варианты

1. **Буфер памяти**: хранить subset старых данных
   - Resevoir sampling для репрезентативной выборки
   - Fixed budget (например, 1% от старых данных)

2. **Generative Replay**: обучить генеративную модель старых данных
   - Генерировать старые данные на лету
   - Используется в Continual Learning

3. **Pseudo-rehearsal**: использовать текущую модель для генерации старых "воспоминаний"

#### Loss с Replay

$$
\mathcal{L}(\theta) = \mathcal{L}_B(\theta) + \lambda \cdot \mathcal{L}_A(\theta; \mathcal{D}_{\text{replay}})
$$

где $\mathcal{D}_{\text{replay}}$ — примеры из старой задачи $A$.

#### Для LLM

- **Mix-in datasets:** при SFT добавляем примеры из pre-training data или общего датасета
- **Пропорция:** 80% целевой датасет + 20% общий датасет
- **Пример:** LLaMA-2 SFT использовал смесь из instruct + general data

---

### 4.2. Elastic Weight Consolidation (EWC) — Kirkpatrick et al., 2017

**Идея:** Замедлять изменение **важных** параметров для старых задач.

#### Алгоритм

1. После обучения на задаче $A$ вычисляем **Fisher Information Matrix** $F$:

$$
F_i = \mathbb{E}_{x \sim \mathcal{D}_A} \left[ \left( \frac{\partial}{\partial \theta_i} \log P(y \mid x; \theta) \right)^2 \right]
$$

$F_i$ — важность параметра $\theta_i$ для задачи $A$.

2. При обучении на задаче $B$ добавляем регуляризацию:

$$
\mathcal{L}(\theta) = \mathcal{L}_B(\theta) + \frac{\lambda}{2} \sum_i F_i \cdot (\theta_i - \theta_i^*)^2
$$

где $\theta_i^*$ — оптимальные значения для задачи $A$.

#### Интуиция

- Параметры с высоким $F_i$ — важные для задачи $A$ (сильно меняют output)
- Штраф $\propto F_i$ — важные параметры меняются медленно
- Неважные параметры ($F_i \approx 0$) могут свободно адаптироваться

#### Для Transformer

EWC применяется ко всем параметрам модели, но на практике — только к attention weights.

#### Проблемы EWC

1. **Вычислительно дорого:** Fisher Information Matrix размера $|\theta| \times |\theta|$
   - На практике: диагональное приближение (diagonal Fisher)
2. **Зависимость от границ задачи:** трудно определить, где заканчивается задача $A$ и начинается $B$
3. **Несколько задач:** накопление регуляризационных членов от всех предыдущих задач

---

### 4.3. Regularization (L2, weight decay, dropout)

#### L2-Regularization к исходным весам

$$
\mathcal{L}(\theta) = \mathcal{L}_B(\theta) + \frac{\lambda}{2} \|\theta - \theta_0\|^2
$$

где $\theta_0$ — веса исходной (pre-trained) модели.

- Простейший способ предотвратить forgetting
- Эквивалентен L2 weight decay с центром в $\theta_0$
- Параметр $\lambda$ контролирует "силу" привязки

#### Weight Decay (стандартный)

$$
\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L} - \eta \lambda \theta
$$

- Штрафует большие веса → более устойчивые представления
- Косвенно помогает от forgetting

#### Dropout

- Dropout **на время fine-tuning** улучшает обобщение на старые задачи
- Эффективен как implicit regularization

#### Label Smoothing

$$
\mathcal{L}_{\text{LS}} = -(1-\epsilon) \log P(y^*) - \epsilon \sum_{j \neq y^*} \frac{1}{C-1} \log P(j)
$$

- Предотвращает overconfidence → лучшее обобщение

---

### 4.4. KL-регуляризация в RLHF

Специфический для RLHF-этапа механизм.

#### Формулировка

При PPO обучении добавляем KL-штраф:

$$
\mathcal{L}_{\text{PPO}}(\theta) = \mathbb{E}_{y \sim \pi_\theta} \left[ r_\phi(x, y) - \beta \cdot \text{KL}(\pi_\theta(\cdot \mid x) \| \pi_{\text{ref}}(\cdot \mid x)) \right]
$$

#### Per-token KL:

$$
\text{KL}_{\text{token}} = \sum_{t} \pi_\theta(y_t \mid y_{<t}) \log \frac{\pi_\theta(y_t \mid y_{<t})}{\pi_{\text{ref}}(y_t \mid y_{<t})}
$$

#### Почему это предотвращает forgetting

- $\pi_{\text{ref}}$ сохраняет знания от SFT / pre-training
- KL-штраф не даёт $\pi_\theta$ слишком отклоняться
- Модель не может "разучиться" языку ради высокой награды

#### Эффект $\beta$

| $\beta$ | Забывание | Reward | Качество языка |
|---|---|---|---|
| $10^{-2}$ | Сильное | Высокий | Плохое (collapse) |
| $10^{-1}$ | Умеренное | Хороший | Хорошее |
| $10^0$ | Минимальное | Умеренный | Отличное |

---

### 4.5. Progressive Prompts / Adapter-based подходы

#### Progressive Prompts

Каждый новый навык = новый набор soft prompts. Старые не трогаются.

#### Adapter-based Continual Learning

- Каждая задача = новый adapter/LoRA
- На инференсе — выбираем adapter по детектору задачи
- Полная защита от forgetting (старые веса не меняются)

#### AdapterFusion (Pfeiffer et al., 2021)

- N адаптеров для N задач
- Обучается Fusion Layer для комбинирования

### 4.6. Distillation (дистилляция) как защита от забывания

[[DL 24 - Knowledge Distillation]] также может смягчать catastrophic forgetting: student-модель обучается имитировать teacher (исходную модель) на старых задачах, сохраняя знания через мягкие метки (soft targets).

---

## 5. Сравнительная таблица методов

| Метод | Сложность | Память | Защита от forgetting | Нагрузка на обучение | Для LLM |
|---|---|---|---|---|---|
| Replay | Средняя | Средняя (buffer) | Высокая | Умеренная | Да (mix-in) |
| EWC | Высокая | Средняя (Fisher) | Средняя | Средняя | Да, но дорого |
| L2-regularization | Низкая | Нет | Низкая | Нет | Да |
| KL in RLHF | Средняя | 1× модель (ref) | Средняя | Умеренная | Стандарт |
| Dropout | Низкая | Нет | Низкая | Нет | Нечасто |
| Adapter/LoRA (per-task) | Средняя | Много адаптеров | Полная | Низкая | Идеально |
| Progressive Prompts | Средняя | Много промптов | Высокая | Низкая | Да |

---

## 6. Практические рекомендации

### Минимизация forgetting при SFT

1. **Mix-in general data:** 10–30% общего / pre-training data
2. **Low learning rate:** $\eta \leq 10^{-5}$ для full FT
3. **Few epochs:** 1–3 эпохи
4. **Warmup + cosine schedule:** плавное обучение
5. **L2-regularization к pre-trained весам:** $\lambda \approx 10^{-4}-10^{-3}$
6. **PEFT (LoRA):** существенно снижает forgetting

### Минимизация forgetting при RLHF

1. **KL-регуляризация:** $\beta \geq 0.1$
2. **Reference model:** копия SFT модели (не обновляется)
3. **Trust region (PPO clip):** $\epsilon \approx 0.2$
4. **Reward normalization:** $\mu=0, \sigma=1$

### Оценка forgetting

- **Perplexity на pre-training / general corpus** — не должна резко расти
- **Benchmarks (MMLU, HellaSwag, etc.)** — до и после fine-tuning
- **Human evaluation** — развёрнутые ответы до/после

---

## 7. Ключевые выводы

1. **Catastrophic forgetting** — острая проблема при дообучении LLM, вызванная пересечением параметров и сдвигом распределения
2. **Основные методы защиты:**
   - Replay (mix-in data) — наиболее практичный для LLM
   - EWC — теоретически обоснованный, но дорогой
   - Регуляризация (L2, KL) — простые и эффективные
   - PEFT (LoRA) — лучшая защита за счёт изоляции параметров
3. **В RLHF** KL-регуляризация играет ключевую роль в предотвращении collapse
4. **Практическое правило:** всегда оценивать forgetting на бенчмарках
5. **Лучшая защита для LLM:** LoRA + mix-in data + умеренная регуляризация
6. [[DL 24 - Knowledge Distillation]] может использоваться как дополнительный механизм сохранения знаний

---

**Связанные вопросы:**
- [[DL 38 - Дообучение LLM]] — SFT, основная причина катастрофического забывания
- [[DL 39 - Preference Tuning и RLHF]] — RLHF/PPO также вызывает forgetting, KL-регуляризация смягчает
- [[DL 24 - Knowledge Distillation]] — teacher-student подход как метод сохранения знаний
