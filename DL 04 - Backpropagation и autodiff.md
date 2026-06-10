# DL 04 — Backpropagation и Autodiff

## 1. Зачем нужен backpropagation?

Чтобы обучить нейронную сеть градиентным спуском, нужно вычислить градиент функции потерь по всем параметрам:

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}}, \quad \frac{\partial \mathcal{L}}{\partial b^{(l)}} \quad \forall l = 1 \dots L$$

Прямое аналитическое дифференцирование сети с миллионами параметров невозможно. **Backpropagation** — эффективный алгоритм вычисления этих градиентов через [[DL 03 - Обучение нейронной сети|chain rule]] на **вычислительном графе**.

---

## 2. Вычислительный граф (Computational Graph)

**Вычислительный граф** — направленный ациклический граф (DAG), где:
- **Узлы**: переменные (тензоры) и операции
- **Рёбра**: поток данных

### Пример: $f(x, y) = (x + y) \cdot y$

```
    x ──┐
        ├── (+) ── a ──┐
    y ──┤               ├── (×) ── f
        └───────────────┘
```

### В DL:

```python
# PyTorch — вычислительный граф строится автоматически
z = torch.matmul(W, x) + b     # линейный слой
a = torch.relu(z)              # активация
loss = F.cross_entropy(a, y)   # loss
loss.backward()                # backpropagation
```

**Граф для двуслойной сети:**

```
x → [W1x + b1] → z1 → [ReLU] → a1 → [W2a1 + b2] → z2 → [Softmax+CE] → loss
```

---

## 3. Chain Rule (правило цепочки)

### Скалярный случай:

$$y = f(g(x)) \quad\Rightarrow\quad \frac{dy}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$$

### Многомерный случай:

Для $z = f(y)$, $y = g(x)$:

$$\frac{\partial z}{\partial x_i} = \sum_j \frac{\partial z}{\partial y_j} \cdot \frac{\partial y_j}{\partial x_i}$$

В матричной форме:

$$\nabla_x z = J_g(x)^T \cdot \nabla_y z$$

где $J_g$ — матрица Якоби функции $g$.

### Backpropagation = многократное применение chain rule от выхода к входу.

---

## 4. Алгоритм Backpropagation (детально)

Для сети с $L$ слоями:

### Forward pass:
1. $h^{(0)} = x$
2. Для $l = 1 \dots L$: $z^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)}$, $h^{(l)} = \sigma(z^{(l)})$
3. $\hat{y} = h^{(L)}$
4. $\mathcal{L} = \text{Loss}(\hat{y}, y)$

### Backward pass:

Начинаем с градиента loss по выходу:

$$\delta^{(L)} = \frac{\partial \mathcal{L}}{\partial z^{(L)}} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \odot \sigma'(z^{(L)})$$

Рекуррентно для $l = L-1, \dots, 1$:

$$\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot \sigma'(z^{(l)})$$

Градиенты по параметрам:

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} (h^{(l-1)})^T$$

$$\frac{\partial \mathcal{L}}{\partial b^{(l)}} = \delta^{(l)}$$

### Ключевое наблюдение:
Вычисление градиента **линейно по глубине** ($O(L)$) — один forward + один backward pass.

---

## 5. Forward-mode vs Reverse-mode Autodiff

### Automatic Differentiation (Autodiff) — не численное дифференцирование!

| Метод | Численное (finite differences) | Symbolic | Autodiff |
|---|---|---|---|
| Точность | Приближённая | Точная | Точная |
| Сложность | $O(n)$ на градиент | Взрыв формул | $O(n)$ на градиент |
| Используется | Тестирование | Редко | **Везде** |

### Два режима autodiff:

#### Forward-mode (прямой режим)

- Вычисляем производную одновременно с forward pass.
- Для каждой входной переменной $x_i$ несём "дуал-число": $(x_i, \dot{x_i})$.
- Все операции работают с дуал-числами.

$$\text{Если } v = f(u), \text{ то } \dot{v} = f'(u) \cdot \dot{u}$$

**Сложность**: $O(n)$ на градиент для $n$ входов.

**Когда хорош**: когда входов мало, выходов много ($\mathbb{R}^n \to \mathbb{R}^m$, $n \ll m$).

#### Reverse-mode (обратный режим) = Backpropagation

- **Forward pass**: вычисляем значения.
- **Backward pass**: распространяем градиенты от выхода к входу.

**Сложность**: $O(m)$ на градиент для $m$ выходов.

**Когда хорош**: когда входов много, выход мало ($\mathbb{R}^n \to \mathbb{R}$, $n \gg m$).

**Именно это нужно для DL**: $n$ — миллионы параметров, $m=1$ (скалярный loss).

### Сравнение:

| Характеристика | Forward-mode | Reverse-mode (backprop) |
|---|---|---|
| Проходов | 1 (вперёд) | 2 (вперёд + назад) |
| Память | Мало (только дуал-числа) | Много (храним все промежуточные значения) |
| Скорость для $f: \mathbb{R}^n \to \mathbb{R}$ | $O(n)$ | $O(1)$ (относительно $n$) |
| Использование в DL | Нет | **Да** |
| Когда выгоден | $n \ll m$ | $n \gg m$ |

### Память — узкое место backprop:
Надо хранить все $h^{(l)}$, $z^{(l)}$ для backward pass. Для очень глубоких сетей это миллиарды чисел.
**Решение**: [[DL 06 - Стабилизация и регуляризация|Gradient Checkpointing]] (пересчитываем часть на лету за счёт времени).

---

## 6. Backpropagation через не-differentiable операции

Некоторые операции не везде дифференцируемы (ReLU в 0):

$$\frac{d}{dx}\text{ReLU}(x) = \begin{cases} 1 & x > 0 \\ 0 & x < 0 \end{cases}$$

В $x=0$ выбираем **subgradient** (например, 0). На практике:
```python
# ReLU backward в PyTorch/TensorFlow:
grad_input = grad_output * (input > 0).float()  # 0 или 1
```

---

## 7. Пример ручного backprop

Дано: $f(x, y) = (x + y) \cdot y$

Пусть $x = 2, y = 3$.

### Forward:
$$a = x + y = 5 \quad\Rightarrow\quad f = a \cdot y = 5 \cdot 3 = 15$$

### Backward:

$$\frac{\partial f}{\partial a} = y = 3, \quad \frac{\partial f}{\partial y} = a = 5$$

$$\frac{\partial a}{\partial x} = 1, \quad \frac{\partial a}{\partial y} = 1$$

$$\frac{\partial f}{\partial x} = \frac{\partial f}{\partial a} \cdot \frac{\partial a}{\partial x} = 3 \cdot 1 = 3$$

$$\frac{\partial f}{\partial y} = \frac{\partial f}{\partial a} \cdot \frac{\partial a}{\partial y} + \frac{\partial f}{\partial y} = 3 \cdot 1 + 5 = 8$$

---

## 8. Резюме

- **Backpropagation** = chain rule на вычислительном графе.
- **Вычислительный граф**: DAG узлов-операций, строится автоматически фреймворками ([[DL 07 - PyTorch пайплайн|PyTorch]], TensorFlow).
- **Forward pass**: вычисляем значения.
- **Backward pass**: распространяем градиенты от loss к параметрам.
- **Reverse-mode autodiff** — оптимален для DL ($10^6$ параметров → 1 loss).
- **Forward-mode** выгоден при малом числе входов.
- Узкое место: **память** для хранения промежуточных значений при обратном проходе.

---

**Связанные вопросы:**
- [[DL 03 - Обучение нейронной сети]] — loss функции и алгоритм обучения
- [[DL 05 - Оптимизация]] — градиентный спуск, momentum, AdamW
- [[DL 07 - PyTorch пайплайн]] — autograd, автоматическое построение графа
- [[DL 06 - Стабилизация и регуляризация]] — gradient clipping, инициализация весов
