# 53. Варианты и стабилизация GAN

Основы GAN см. [[DL 52 - GAN]]. Альтернативные генеративные подходы: [[DL 54 - Autoencoders и VAE|VAE]], [[DL 58 - Диффузионные модели|диффузионные модели]].

## 1. Mode Collapse — подробный анализ

**Mode collapse** — ситуация, когда Generator $G$ отображает разные шумовые векторы $z$ в один и тот же выход (или небольшое подмножество выходов), тем самым покрывая лишь часть истинного распределения $p_{data}$.

### Почему возникает?

В оригинальной GAN Generator минимизирует $-\log D(G(z))$. Если $D$ легко обманывается одним типом образцов, $G$ нет стимула генерировать разнообразие.

**Формально:** для фиксированного $D$, $G$ может найти $z^*$, дающую максимальное $D(G(z))$, и использовать только её.

### Методы борьбы с mode collapse:

| Метод | Идея | Пример |
|---|---|---|
| **Mini-batch discrimination** | $D$ видит целый батч и может детектировать однообразие | Salimans et al., 2016 |
| **Unrolled GAN** | $G$ учитывает несколько шагов оптимизации $D$ вперёд | Metz et al., 2017 |
| **Spectral normalization** | Контроль Lipschitz-константы $D$ | Miyato et al., 2018 |
| **Experience replay** | Смешивание старых и новых сэмплов | — |
| **Packing** | $G$ генерирует набор образцов, $D$ оценивает разнообразие | PacGAN |

## 2. DCGAN (Deep Convolutional GAN)

**DCGAN** (Radford et al., 2015) — первая стабильная архитектура GAN для изображений, использующая свёртки.

### Ключевые принципы:

1. **Отсутствие fully-connected слоёв** — только свёрточные/транспонированно-свёрточные
2. **Batch normalization** — во всех слоях, кроме выхода $G$ и входа $D$
3. **Архитектура $G$** — транспонированные свёртки с upsample (4×4 → 8×8 → 16×16 → 32×32)
4. **Архитектура $D$** — обычные свёртки с stride 2 (downsample)
5. **Функции активации:**
   - $G$: ReLU (кроме последнего слоя — Tanh)
   - $D$: Leaky ReLU (slope 0.2)

### Архитектура Generator:

```
z (100-dim) → FC → BN → ReLU → reshape → Deconv → BN → ReLU → Deconv → BN → ReLU → Deconv → Tanh → 64×64×3
```

### Архитектура Discriminator:

```
64×64×3 → Conv (stride 2) → LeakyReLU → Conv (stride 2) → BN → LeakyReLU → Conv (stride 2) → BN → LeakyReLU → FC → Sigmoid
```

### Рекомендации DCGAN:
- Adam: lr = 0.0002, β₁ = 0.5
- Веса инициализировать N(0, 0.02)
- No max-pooling (только strided convolutions)
- No fully connected hidden layers

## 3. Conditional GAN (cGAN)

**cGAN** (Mirza & Osindero, 2014) — GAN, в которой и Generator, и Discriminator получают дополнительную информацию $y$ (класс, текст, изображение).

### Архитектура:

$$G(z, y) \to x_{fake}$$
$$D(x, y) \to [0, 1]$$

Условие $y$ подаётся:
- В Generator: конкатенируется с шумом $z$ на входе
- В Discriminator: конкатенируется с входным изображением на первом или промежуточном слое

### Loss для cGAN:

$$\min_G \max_D V(D, G) = \mathbb{E}_{x,y \sim p_{data}} [\log D(x, y)] + \mathbb{E}_{z \sim p_z, y \sim p_y} [\log (1 - D(G(z, y), y))]$$

### Применения:
- **Класс-условная генерация** — генерация изображений конкретного класса
- **Text-to-image** — StackGAN, AttnGAN
- **Image-to-image translation** — pix2pix (условие = входное изображение)

## 4. Pix2pix

**Pix2pix** (Isola et al., 2017) — условная GAN для image-to-image translation. Условие $y$ — входное изображение (например, карта сегментации → фото).

### Архитектура:
- **Generator**: U-Net (skip connections между encoder и decoder)
- **Discriminator**: PatchGAN — классифицирует каждый патч 70×70 как реальный/фейк

### Loss:

$$L_{cGAN}(G, D) = \mathbb{E}_{x,y} [\log D(x, y)] + \mathbb{E}_{x,z} [\log(1 - D(x, G(x, z)))]$$

В комбинации с L1-loss (для сохранения структуры):

$$G^* = \arg\min_G \max_D L_{cGAN}(G, D) + \lambda L_{L1}(G)$$

где $L_{L1}(G) = \mathbb{E}_{x,y,z} [\|y - G(x, z)\|_1]$

### Почему PatchGAN?
- Обычный $D$ даёт только одно число на всё изображение — теряется локальная информация
- PatchGAN оценивает $N \times N$ патчей, усредняя результат
- Имеет меньше параметров, быстрее
- Заставляет $G$ сохранять высокие частоты (детали)

| Loss | Эффект |
|---|---|
| L1 | Сохраняет low-frequency структуру (размыто, но правильно) |
| cGAN | Добавляет high-frequency детали (резкость, текстуры) |

## 5. CycleGAN

**CycleGAN** (Zhu et al., 2017) — для image-to-image translation без парных данных (unpaired).

### Идея:
Два домена $X$ и $Y$. Два Generator'а и два Discriminator'а:
- $G: X \to Y$, $F: Y \to X$
- $D_X$: отличает реальные $X$ от $F(Y)$
- $D_Y$: отличает реальные $Y$ от $G(X)$

### Cycle consistency loss:

$$L_{cyc}(G, F) = \mathbb{E}_{x \sim p_{data}(x)} [\|F(G(x)) - x\|_1] + \mathbb{E}_{y \sim p_{data}(y)} [\|G(F(y)) - y\|_1]$$

### Полный loss:

$$L(G, F, D_X, D_Y) = L_{GAN}(G, D_Y) + L_{GAN}(F, D_X) + \lambda L_{cyc}(G, F)$$

### Дополнительные потери:
- **Identity loss**: $L_{identity} = \|G(y) - y\|_1 + \|F(x) - x\|_1$ (если на вход подать из целевого домена, генератор не должен менять картинку)

### Сравнение pix2pix vs CycleGAN:

| Характеристика | Pix2pix | CycleGAN |
|---|---|---|
| **Данные** | Требует парные (paired) | Работает с unpaired |
| **Количество генераторов** | 1 | 2 |
| **Сохранение структуры** | L1 loss | Cycle consistency |
| **Качество** | Выше (более точное) | Ниже (менее точное) |
| **Применение** | Карты сегментации → фото | Конь → зебра, день → ночь |

## 6. WGAN / WGAN-GP

### WGAN (Wasserstein GAN)

**Проблема оригинальной GAN:** JSD не даёт полезного градиента, когда распределения не пересекаются.

**Решение WGAN** (Arjovsky et al., 2017): использовать **Wasserstein-1 distance** (Earth Mover's distance):

$$W(p_{data}, p_g) = \inf_{\gamma \in \Pi(p_{data}, p_g)} \mathbb{E}_{(x, y) \sim \gamma} [\|x - y\|]$$

Практически, через Kantorovich-Rubinstein duality:

$$W(p_{data}, p_g) = \sup_{\|f\|_L \leq K} \mathbb{E}_{x \sim p_{data}} [f(x)] - \mathbb{E}_{x \sim p_g} [f(x)]$$

где $f$ — **critic** (не discriminator!) с K-Lipschitz-ограничением.

### WGAN Loss:

$$L = -\mathbb{E}_{x \sim p_{data}} [f(x)] + \mathbb{E}_{z \sim p_z} [f(G(z))]$$

- Generator минимизирует: $-\mathbb{E}_{z} [f(G(z))]$
- Critic максимизирует: $\mathbb{E}_{x} [f(x)] - \mathbb{E}_{z} [f(G(z))]$

### Lipschitz-ограничение в WGAN:

В оригинальном WGAN — **weight clipping**:
$$w \leftarrow \text{clip}(w, -c, c)$$

**Недостатки weight clipping:**
- Слишком маленький $c$ → затухающие градиенты
- Слишком большой $c$ → взрывные градиенты
- Неоптимальное использование ёмкости модели

### WGAN-GP (Gradient Penalty)

**WGAN-GP** (Gulrajani et al., 2017): замена weight clipping на градиентный штраф:

$$L_{GP} = \lambda \mathbb{E}_{\hat{x} \sim P_{\hat{x}}} [(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]$$

где $\hat{x}$ — равномерная интерполяция между реальными и фейковыми точками:
$$\hat{x} = \epsilon x_{real} + (1 - \epsilon) x_{fake}, \quad \epsilon \sim U[0, 1]$$

### Сравнение GAN vs WGAN vs WGAN-GP:

| Характеристика | GAN | WGAN | WGAN-GP |
|---|---|---|---|
| **Функция расстояния** | JSD | Wasserstein-1 | Wasserstein-1 |
| **Discriminator/Critic** | Вероятность (sigmoid) | Счёт (без sigmoid) | Счёт (без sigmoid) |
| **Градиенты** | Исчезают при сильном D | Стабильные | Стабильные |
| **Mode collapse** | Часто | Реже | Редко |
| **Lipschitz constraint** | Нет | Weight clipping | Gradient penalty |
| **Сложность** | Базовая | Средняя | Выше |
| **Качество** | Зависит | Умеренное | ✓ Лучшее |

---

**Связанные вопросы:** [[DL 52 - GAN]], [[DL 54 - Autoencoders и VAE]], [[DL 58 - Диффузионные модели]]
