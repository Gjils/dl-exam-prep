# DL 39 — Preference Tuning и RLHF

> **Reward model, 3 этапа (SFT → RM → PPO), отличия RLHF от SFT, роль KL-регуляризации, DPO**

---

## 1. Мотивация

SFT учит модель имитировать **человеческие ответы**, но не учит **предпочтениям** — какой ответ лучше. SFT-модель может:
- Давать фактические ответы, но грубые или небезопасные
- Генерировать многословные и нерелевантные ответы
- Не улавливать нюансы "хорошего" ответа

RLHF (Reinforcement Learning from Human Feedback) решает эту проблему, обучая модель нравиться человеку.

---

## 2. Три этапа RLHF (InstructGPT / ChatGPT pipeline)

### Этап 1: Supervised Fine-Tuning (SFT)

**Цель:** Адаптировать предобученную LLM к формату диалога.

- Собирается датасет $\mathcal{D}_{\text{SFT}} = \{(x_i, y_i)\}$.
- Асессоры пишут ответы на промпты.
- Обучается teacher forcing: $\max_\theta \log P_\theta(y \mid x)$.

Получаем модель $\pi_{\text{SFT}}$.

### Этап 2: Reward Model (RM) — обучение модели награды

**Цель:** Обучить модель, предсказывающую, какой ответ лучше с точки зрения человека.

#### Сбор данных

Для каждого промпта $x$ генерируется $K$ ответов ($K=4$–9). Асессор сортирует их от лучшего к худшему.

#### Формализация

Используем pairwise предпочтения: $(x, y_w, y_l)$, где $y_w$ — предпочтительный ответ (win), $y_l$ — менее предпочтительный (lose).

#### Bradly-Terry модель

Предполагаем, что предпочтения следуют распределению Bradly-Terry:

$$
P(y_w \succ y_l \mid x) = \frac{\exp(r_\phi(x, y_w))}{\exp(r_\phi(x, y_w)) + \exp(r_\phi(x, y_l))} = \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))
$$

где $r_\phi(x, y)$ — reward model (скейлярный выход), $\sigma$ — сигмоида.

#### Loss-функция

$$
\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{\text{RM}}} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]
$$

#### Архитектура RM

Обычно та же модель, что и $\pi_{\text{SFT}}$, но с удалённым LM head и добавленным linear projection до 1 (score).

#### Нормировка

Reward обычно нормируется: $\mathbb{E}[r_\phi(x, y)] = 0$, $\text{Var}[r_\phi(x, y)] \approx 1$.

### Этап 3: PPO (Proximal Policy Optimization)

**Цель:** Оптимизировать политику $\pi_\theta$ так, чтобы максимизировать ожидаемую награду, но не уходить слишком далеко от $\pi_{\text{SFT}}$.

#### Общая формулировка

$$
\max_\theta \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot \mid x)} \left[ r_\phi(x, y) - \beta \cdot \text{KL}(\pi_\theta(\cdot \mid x) \| \pi_{\text{ref}}(\cdot \mid x)) \right]
$$

где:
- $r_\phi(x, y)$ — награда от RM
- $\beta$ — коэффициент KL-регуляризации
- $\pi_{\text{ref}}$ — обычно $\pi_{\text{SFT}}$ или копия $\pi_\theta$ (frozen)
- $\text{KL}(\cdot \| \cdot)$ — KL-дивергенция

#### PPO loss (подробно)

Для каждого токена $t$:

$$
\mathcal{L}_{\text{PPO}}(\theta) = \mathbb{E} \left[ \min\left( \frac{\pi_\theta(y_t \mid s_t)}{\pi_{\text{old}}(y_t \mid s_t)} \hat{A}_t, \text{clip}\left(\frac{\pi_\theta(y_t \mid s_t)}{\pi_{\text{old}}(y_t \mid s_t)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_t \right) \right]
$$

где $\hat{A}_t$ — advantage (часто используется reward-to-go или GAE).

#### Псевдокод PPO для LLM

```
for each iteration:
    for each batch (x):
        # 1. Generate responses
        y ~ π_θ(·|x)
        
        # 2. Compute rewards
        r = r_ϕ(x, y)
        
        # 3. Compute advantages (with KL penalty per token)
        KL_t = KL(π_θ(y_t) || π_ref(y_t))
        reward_t = -β * KL_t      # per-token KL penalty
        reward_T = r               # final token gets RM score
        A_t = GAE(reward_t)        # advantages
        
        # 4. PPO update
        θ ← PPO_step(θ, A_t, π_θ(y_t), π_old(y_t))
        
        # 5. Optionally update reference periodically
```

---

## 3. Роль KL-регуляризации

### Зачем она нужна?

Без KL-регуляризации модель будет:
- Находить "adversarial" ответы, которые дают высокую награду, но бессмысленны
- Rapidly collapse — выдавать один и тот же "безопасный" ответ
- Терять способность к генерации (catastrophic forgetting языка)

### Формально

KL-дивергенция между политикой и reference model:

$$
\text{KL}(\pi_\theta \| \pi_{\text{ref}}) = \mathbb{E}_{y \sim \pi_\theta} \left[ \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right]
$$

Это наказание за отклонение от reference — модель может улучшать свои ответы, но не может "ломать" язык.

### Эффект $\beta$

| $\beta$ | Эффект |
|---|---|
| Очень высокое | Модель почти не меняется (≈ SFT) |
| Умеренное | Постепенное улучшение без collapse |
| Низкое | Быстрое улучшение, но риск collapse |
| Нулевое | Reward hacking, поломка модели |

### Per-token vs per-sequence KL

- **Per-token KL** (классический PPO): штраф за каждый токен — лучше, стабильнее
- **Per-sequence KL**: штраф за всю последовательность — грубее

---

## 4. Отличия RLHF от SFT

| Аспект | SFT | RLHF |
|---|---|---|
| Цель | Имитация ответов | Удовлетворение предпочтений |
| Данные | Правильные ответы | Pairwise сравнения |
| Loss | Cross-entropy | PPO / REINFORCE |
| Поощрение | Точность токенов | Человеческие предпочтения |
| Риск | Exposure bias | Reward hacking |
| Качество ответов | Среднее (но точное) | Лучше (но креативнее) |
| Сложность | Низкая | Высокая (3 этапа, RM) |
| Стабильность | Высокая | Низкая (гиперпараметры PPO) |

### Почему RLHF лучше SFT?

SFT учит "как люди пишут ответы", но не может улучшить ответ за пределы человеческих демонстраций. RLHF может:
- Выучить, что **короткий и точный ответ** лучше длинного
- Учесть **неочевидные предпочтения** (вежливость, безопасность)
- Генерализовать за пределы демонстраций

---

## 5. DPO (Direct Preference Optimization) — Rafailov et al., 2023

### Идея

Вместо трёх этапов (SFT → RM → PPO), DPO объединяет два последних: reward model обучается **неявно** через политику.

### Формальный вывод

Из RLHF: оптимальная политика при KL-регуляризации имеет вид:

$$
\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)
$$

Отсюда выразим reward:

$$
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
$$

Подставим в Bradley-Terry loss:

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]
$$

### Упрощение

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left( \beta \underbrace{\left( \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right)}_{\text{implicit reward margin}} \right) \right]
$$

### Сравнение DPO vs PPO

| Аспект | PPO | DPO |
|---|---|---|
| Этапов | 3 (SFT + RM + PPO) | 2 (SFT + DPO) |
| Нужен RM | Да | Нет (неявный) |
| Сложность | Высокая (PPO sensitive) | Низкая (как SFT) |
| Стабильность | Низкая | Высокая |
| Generations on-policy | Да (sample from π_θ) | Нет (offline data) |
| Качество | Высокое | Сопоставимое |
| Память | Больше | Меньше |

### Достоинства DPO

- Не нужен reward model (экономия ресурсов)
- Стабильное обучение без PPO-гиперпараметров
- Не нужно генерировать on-policy сэмплы
- Масштабируется на большие модели

### Недостатки DPO

- Подвержен **distribution mismatch** (обучение на статических данных)
- Может переобучаться под специфичные предпочтения датасета
- Требует правильного β (аналогично KL weight)

---

## 6. Другие методы Preference Tuning

### RRHF (Rank Responses to align Human Feedback)

- Использует ranking loss (ранжирование ответов)
- Не требует PPO, проще DPO

### KTO (Kahneman-Tversky Optimization)

- Использует **prospect theory** — не только pairwise, но и "хороший/плохой" ответ
- Учитывает асимметрию потерь: lose > win (люди больше чувствуют потери)

### SLiC (Sequence-Level Calibration)

- Calibration loss + ranking loss
- Фокус на калибровке вероятностей

---

## 7. Практические аспекты

### Выбор β

Обычно $\beta \in [0.01, 0.5]$:
- Слишком мало: модель расходится
- Слишком много: модель не меняется

### Размер RM датасета

- Необходимо минимум 10K–50K pairwise примеров
- Больше данных → лучше RM → лучше финальная модель

### Reward normalization

- RM score нормируется к $\mu=0, \sigma=1$
- Важно для стабильного PPO

### Reward scaling

- PPO чувствителен к масштабу награды
- Clip reward (например, $[-5, 5]$) для стабильности

---

## 8. Ключевые выводы

1. **RLHF** — стандартный метод выравнивания LLM под человеческие предпочтения
2. **Три этапа**: SFT → Reward Model → PPO — каждый решает свою задачу
3. **KL-регуляризация** критически важна для предотвращения collapse и [[DL 41 - Catastrophic Forgetting|катастрофического забывания]]
4. **DPO** — альтернатива без RM и PPO, проще и стабильнее, но с offline данными
5. **RLHF > SFT** — позволяет улучшать ответы за пределы демонстраций

---

**Связанные вопросы:**
- [[DL 38 - Дообучение LLM]] — SFT, первый этап пайплайна RLHF
- [[DL 40 - ICL и Prompting]] — альтернативный подход к управлению поведением модели (без обучения)
- [[DL 41 - Catastrophic Forgetting]] — проблема, которую KL-регуляризация в RLHF помогает смягчить
- [[DL 42 - Инференс LLM]] — как обученная модель применяется на практике (TTFT, TPOT, KV-cache)
