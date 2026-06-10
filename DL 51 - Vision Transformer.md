# 51. Vision Transformer (ViT)

## 1. Общая идея Vision Transformer

**Vision Transformer (ViT)** — первая архитектура, которая применила чистый [[DL 33 - Архитектура Transformer|Transformer]] (без свёрток) к задаче классификации изображений. Основная идея: разбить изображение на патчи, превратить каждый патч в токен (как слово в NLP), подать последовательность токенов в стандартный Transformer Encoder.

Ключевое отличие от [[DL 13 - Классические CNN-архитектуры|CNN]]: ViT не имеет встроенного inductive bias о локальности и трансляционной эквивариантности — всё учится из данных.

## 2. Patch Embedding

Изображение $x \in \mathbb{R}^{H \times W \times C}$ разбивается на непересекающиеся квадратные патчи размера $P \times P$.

Число патчей:
$$N = \frac{HW}{P^2}$$

Каждый патч "выпрямляется" в вектор и проецируется через обучаемую линейную проекцию (patch embedding):
$$\mathbf{z}_p^{(i)} = \mathbf{E} \cdot \text{flatten}(x^{(i)}) \in \mathbb{R}^D, \quad \mathbf{E} \in \mathbb{R}^{D \times (P^2 C)}$$

- $D$ — размерность скрытого представления (embedding dimension)
- $x^{(i)}$ — i-й патч
- Все патчи обрабатываются одной и той же матрицей $\mathbf{E}$

Изображение $x \in \mathbb{R}^{H \times W \times C}$ — это [[DL 09 - Изображение как тензор, мотивация CNN|трёхмерный тензор]]. Patch embedding можно реализовать как [[DL 10 - Операция свертки|свёртку]] с ядром $P \times P$, stride $P$, количеством каналов $D$.

## 3. Positional Encoding

Поскольку Transformer инвариантен к порядку токенов, необходимо добавить позиционную информацию. ViT использует **обучаемые 1D positional embeddings**:

$$\mathbf{z}_0 = [\mathbf{z}_{cls}; \mathbf{z}_p^{(1)}; \dots; \mathbf{z}_p^{(N)}] + \mathbf{E}_{pos}$$

- $\mathbf{z}_{cls} \in \mathbb{R}^D$ — специальный CLS-токен (как в BERT)
- $\mathbf{E}_{pos} \in \mathbb{R}^{(N+1) \times D}$ — обучаемая матрица позиций
- CLS-токен добавляется в начало последовательности; его финальное представление используется для классификации

Альтернативные варианты позиционного кодирования:
- **2D positional embeddings** — учитывают пространственную структуру (отдельное обучение для строк и столбцов)
- **Sinusoidal** — фиксированные синусоидальные сигналы (как в оригинальном Transformer)

## 4. Self-Attention вместо свёрток

ViT использует стандартный Multi-Head Self-Attention (MSA) из Transformer:

Для одного патча вычисляются Query, Key, Value:
$$\mathbf{Q} = \mathbf{X} \mathbf{W}^Q, \quad \mathbf{K} = \mathbf{X} \mathbf{W}^K, \quad \mathbf{V} = \mathbf{X} \mathbf{W}^V$$

Score (attention):
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}}\right) \mathbf{V}$$

**Multi-Head Attention** — h голов параллельно, результаты конкатенируются:
$$\text{MSA}(\mathbf{X}) = \text{Concat}[\text{head}_1, \dots, \text{head}_h] \mathbf{W}^O$$

где $\text{head}_i = \text{Attention}(\mathbf{X} \mathbf{W}_i^Q, \mathbf{X} \mathbf{W}_i^K, \mathbf{X} \mathbf{W}_i^V)$.

Архитектура блока ViT:
1. LayerNorm → MSA → Residual
2. LayerNorm → MLP (два линейных слоя с GELU) → Residual

$$\mathbf{z}'_l = \text{MSA}(\text{LN}(\mathbf{z}_{l-1})) + \mathbf{z}_{l-1}$$
$$\mathbf{z}_l = \text{MLP}(\text{LN}(\mathbf{z}'_l)) + \mathbf{z}'_l$$

Выходной CLS-токен проходит через голову классификации (MLP + softmax).

## 5. Сравнение ViT и CNN

| Характеристика | CNN | ViT |
|---|---|---|
| **Inductive bias** | Сильный: локальность, трансляционная эквивариантность, иерархичность | Минимальный: только patch embedding (неявная локальность) |
| **Receptive field** | Растёт иерархически (локальный → глобальный) | Глобальный с первого слоя (self-attention) |
| **Data scaling** | Насыщается при большом количестве данных | Продолжает улучшаться с ростом данных (scaling law) |
| **Параметры** | Эффективнее при малых данных | Требует больше данных для обучения |
| **Вычислительная сложность** | $O(HW)$ (линейная по пикселям) | $O(N^2 D)$ (квадратичная по числу патчей) |
| **Трансляционная эквивариантность** | Встроена (stride, padding) | Нет; учится из данных |
| **Глобальные зависимости** | Требуют глубоких сетей | Естественны с первого слоя |

### Inductive Bias

**CNN:**
- Локальность: ядра свёртки малого размера (3×3, 5×5)
- Трансляционная эквивариантность: сдвиг входа → сдвиг выхода
- Иерархичность: pooling/stride создают пирамиду признаков
- Параметр efficiency: меньше данных для обучения

**ViT:**
- Self-attention не имеет bias — может учить любые зависимости
- Patch embedding вносит минимальную локальность
- Positional encoding — единственная пространственная информация

### Data Scaling

ViT демонстрирует **scaling laws**: с ростом объёма данных (ImageNet-21k, JFT-300M) ViT догоняет и перегоняет CNN. При малых данных CNN выигрывают.

| Объём данных | CNN | ViT |
|---|---|---|
| Маленький (~1M) | ✓ Лучше | ✗ Underfitting |
| Средний (~10M) | ∼ Равно | ∼ Равно |
| Большой (>100M) | Насыщение | ✓ Продолжает рост |

**Практический вывод:** ViT требует предобучения на гигантских датасетах. Для небольших датасетов лучше использовать DeiT (Data-efficient Image Transformer) с knowledge distillation от CNN.

## 6. Дополнительные детали

### Индуктивные biases в современных ViT

Последующие работы добавили inductive bias в ViT:
- **Swin Transformer** — оконное attention + иерархия (shifted windows)
- **ViT-AugReg** — сильная регуляризация для работы с меньшими данными
- **CvT** — свёрточные проекции в Transformer
- **PVT** —金字塔 архитектура с multi-scale feature maps

### Квадратичная сложность self-attention

Для изображения 224×224 с патчем 16×16: $N = 196$, что приемлемо.
Для изображения 1024×1024: $N = 4096$, уже тяжело ($O(N^2) \approx 16M$).

Решения: локальное attention (Swin), линейное attention (Performer, Linformer).

### CLIP и ViT

ViT используется как Image Encoder в модели [[DL 60 - CLIP|CLIP]], где он обучается контрастивно с текстовым Transformer. Благодаря ViT, CLIP получает богатые визуальные представления, которые можно использовать zero-shot для классификации.

**Связь с ResNet:** До ViT, CLIP использовал [[DL 14 - ResNet|ResNet]] как image encoder. Переход на ViT дал значительный прирост в качестве (особенно при масштабировании).

### ViT как backbone для downstream задач

- **Segmentation**: SETR (ViT как encoder для сегментации)
- **Detection**: DETR (Transformer для object detection)
- **Video**: TimeSformer (ViT + temporal attention)
- **Multimodal**: Image Encoder в [[DL 61 - VLM и MLLM|VLM]] (CLIP, LLaVA, BLIP-2)

---

**Связанные вопросы:** [[DL 09 - Изображение как тензор, мотивация CNN]], [[DL 10 - Операция свертки]], [[DL 13 - Классические CNN-архитектуры]], [[DL 14 - ResNet]], [[DL 33 - Архитектура Transformer]], [[DL 60 - CLIP]], [[DL 61 - VLM и MLLM]]
