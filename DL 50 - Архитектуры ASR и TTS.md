# DL 50 — Архитектуры ASR и TTS

> **RNN-T/LAS vs CTC; TTS pipeline: text normalization, acoustic model, vocoder; end-to-end TTS.
> Смежные темы: [[DL 48 - ASR]], [[DL 49 - CTC]], [[DL 33 - Архитектура Transformer]], [[DL 58 - Диффузионные модели]].**

---

## 1. ASR Архитектуры

### 1.1. CTC (Connectionist Temporal Classification)

(Подробно — [[DL 49 - CTC|вопрос 49]])

**Кратко:**
- Encoder → CTC head (linear + softmax)
- Blank token + collapse operation
- Conditional independence: $P(a_t \mid X) \perp P(a_{t'} \mid X)$
- Loss: Forward-Backward суммирование по alignments
- Декодирование: greedy или beam search + внешний LM

**Плюсы:** Простой, быстрый, эффективный.
**Минусы:** Нет контекстной зависимости между токенами.

---

### 1.2. RNN-T (Recurrent Neural Network Transducer) — Google, 2012

#### Архитектура

```
                    Text tokens Y
                        ▲
                        │
                  ┌─────┴─────┐
                  │   Joiner  │
                  └──┬────┬───┘
                     │    │
              ┌──────┘    └──────┐
              │                  │
        ┌─────┴─────┐     ┌─────┴─────┐
        │  Encoder  │     │  Predict  │
        │ (Acoustic)│     │  Network  │
        └─────┬─────┘     └─────┬─────┘
              │                  │
           Audio X              Text Y_{<u}
```

**Три компонента:**

1. **Encoder (Acoustic model):** $h_t^{\text{enc}} = f_{\text{enc}}(x_1, \dots, x_t)$
   - Аудио вход, обычно Conformer или Transformer
2. **Prediction Network (Language model):** $h_u^{\text{pred}} = f_{\text{pred}}(y_1, \dots, y_{u-1})$
   - Текстовый контекст, обычно LSTM или Transformer
3. **Joiner (Combiner):** $P(a \mid t, u) = \text{softmax}(\text{Linear}(h_t^{\text{enc}} + h_u^{\text{pred}}))$
   - Сумма/конкатенация + linear + softmax

#### RNN-T Loss

Суммирование по всем путям (path), где на каждом шаге либо emit токен, либо blank:

$$
\mathcal{L}_{\text{RNN-T}} = -\log P(Y \mid X) = -\log\left(\sum_{\text{paths}} \prod_{i} P(a_i \mid t_i, u_i)\right)
$$

Вычисляется также через forward-backward за $O(T \cdot U)$.

#### RNN-T vs CTC

| Критерий | CTC | RNN-T |
|---|---|---|
| Prediction Network | Нет | Есть (текстовый контекст) |
| LM встроенная | Нет | Да |
| Conditional independence | Да (токены независимы) | Нет ($h_u^{\text{pred}}$ — контекст) |
| Качество | Хорошее | Отличное |
| Streaming | Да | Да |
| Latency | Низкая | Средняя |
| Память | Мало | Больше (prediction net) |
| Размер модели | Меньше | Больше |

---

### 1.3. LAS (Listen, Attend and Spell) — Chan et al., 2016

#### Архитектура

```
                  Text tokens Y
                      ▲
                      │
             ┌────────┴────────┐
             │  Spell (Decoder)│
             │  (autoregressive)│
             └────────┬────────┘
                      │
               Cross-Attention
                      │
             ┌────────┴────────┐
             │  Listen (Encoder)│
             │  (pBLSTM / Pyramidal)  │
             └────────┬────────┘
                      │
                   Audio X
```

**Listen (Encoder):**
- **Pyramidal BiLSTM (pBLSTM)** — билатеральная LSTM с сжатием времени
- Каждый слой уменьшает время в 2×: $T \rightarrow T/2 \rightarrow T/4$
- $T = 100 \rightarrow T' = 25$ (до decoder)

**Attend (Attention):**
- Cross-attention между hidden states decoder и encoder
- Content-based: $e_{t,u} = v^T \tanh(W h_u + V s_t)$

**Spell (Decoder):**
- Autoregressive LSTM/Transformer
- На каждом шаге предсказывает следующий токен:

$$
P(y_u \mid y_{<u}, X) = \text{Decoder}(s_{u-1}, c_u, y_{u-1})
$$

где $c_u$ — context vector из attention.

#### LAS vs CTC vs RNNT

| Критерий | CTC | RNN-T | LAS |
|---|---|---|---|
| Alignment | Явный (blank) | Явный (blank+pred) | Неявный (attention) |
| Monotonic | ✅ | ✅ | ❌ (может переставлять) |
| Autoregressive | ❌ | ✅ (prediction net) | ✅ (decoder) |
| Full context | ✅ (bidirectional) | ✅ (bidirectional) | ✅ (full encoder) |
| Streaming | ✅ | ✅ | ❌ (full attention) |
| Качество | Хорошее | Отличное | Отличное |
| LM встроенная | Нет | Да | Да |
| Сложность | Низкая | Средняя | Высокая |

#### Почему LAS реже используется

- Attention не монотонна — может "читать" аудио не по порядку
- Плохо подходит для streaming (нужно всё аудио сразу)
- Зависимость от всей последовательности → latency

---

### 1.4. Современные конфигурации

| Модель | Encoder | Decoder | Декодирование | Streaming |
|---|---|---|---|---|
| [[DL 48 - ASR|Whisper]] (OpenAI) | [[DL 33 - Архитектура Transformer|Transformer Encoder]] | Transformer Decoder | Autoregressive | ❌ |
| [[DL 48 - ASR|Wav2Vec2]] (Meta) | Transformer Encoder | [[DL 49 - CTC|CTC head]] | CTC beam search | ❌ |
| **USM** (Google) | Conformer | RNN-T | RNN-T | ✅ |
| **FastConformer** (NVIDIA) | Conformer (reduced) | CTC / RNN-T | Beam search | ✅ |
| **Branchformer** | Branch (local + global) | CTC / RNN-T | — | ❌ |

---

## 2. TTS (Text-to-Speech) — общая схема

### Классический TTS Pipeline (3 этапа)

```
Text → Text Normalization → Acoustic Model → Vocoder → Audio
```

### 2.1. Text Normalization

**Задача:** Преобразование "сырого" текста в лингвистическое представление.

**Этапы:**
1. **Tokenization:** разбиение на слова, знаки препинания
2. **Normalization:** 
   - Числа: "123" → "one hundred twenty-three" / "сто двадцать три"
   - Аббревиатуры: "Dr." → "doctor"
   - Даты, время, валюты: "$5.99" → "five dollars ninety nine cents"
3. **Phonemization (опционально):** "hello" → [h, ə, l, oʊ]
4. **Prosody marking:** ударения, паузы, интонационные контуры

**Сложность:** Много исключений, зависит от языка.

### 2.2. Acoustic Model (акустическая модель)

**Вход:** Лингвистические признаки (текст/фонемы + prosody)
**Выход:** Акустические признаки (mel-spectrogram, линейный спектр, vocoder parameters)

**Исторические модели:**

| Модель | Тип | Особенности |
|---|---|---|
| **Tacotron 1/2** (Google) | Seq2seq + attention | Mel-spectrogram из текста |
| **Tacotron 2** | Tacotron + WaveNet vocoder | Лучшее качество |
| **FastSpeech 1/2** (Microsoft) | Non-autoregressive | Fast + robust |
| **VITS** | End-to-end VAE + flow | Лучшее качество на одном этапе |
| **YourTTS** | VITS + speaker embedding | Многоязычный, voice cloning |

#### Tacotron 2 (подробно)

```
Text → Character Embedding → CBHG / LSTM Encoder → Attention → Decoder (prenet + LSTM) → Linear projection → Mel-spectrogram
```

- Autoregressive decoder (медленно)
- Teacher forcing на обучении
- Проблема: ошибки накапливаются, "attention collapse"

#### FastSpeech 2 (неавторегрессионный)

```
Text → Phoneme Embedding → FFT Blocks (Transformer) → Variance Adaptor (duration, pitch, energy) → FFT Blocks → Mel-spectrogram
```

**Ключевые компоненты:**
- **Duration predictor** — сколько фреймов на каждый фонем
- **Pitch predictor** — высота тона (F0)
- **Energy predictor** — громкость
- **Non-autoregressive:** все фреймы генерируются параллельно

#### Сравнение Tacotron 2 vs FastSpeech 2

| Аспект | Tacotron 2 | FastSpeech 2 |
|---|---|---|
| Autoregressive | Да (медленно) | Нет (быстро) |
| Robustness | Низкая (attention collapse) | Высокая |
| Скорость | Медленно (реальное время) | Реальное (в 100× быстрее) |
| Качество | Высокое | Высокое |
| Управление | Трудно | Легко (pitch, speed) |

### 2.3. Vocoder (вокодер)

**Задача:** Преобразование акустических признаков (mel-спектрограмма) в waveform.

#### Типы вокодеров

| Вокодер | Тип | Качество | Скорость |
|---|---|---|---|
| **Griffin-Lim** | Фазовое восстановление | Среднее | ⚡ |
| **WaveNet** (DeepMind) | Autoregressive CNN | Отличное | 🐢 |
| **WaveGlow** (NVIDIA) | Normalizing flow | Отличное | ⚡ |
| **HiFi-GAN** (Jungil et al.) | GAN | Отличное | ⚡ |
| **MelGAN** (Kumar et al.) | GAN | Хорошее | ⚡ |
| **LPCNet** | RNN | Среднее | ⚡ |

#### Griffin-Lim

Алгоритмический (не ML): восстановление фазы из magnitude спектра через итерации DFT/IDFT.

- **Плюсы:** Нет обучения
- **Минусы:** Артефакты, низкое качество

#### [[DL 58 - Диффузионные модели|WaveNet]]

- Dilated causal convolutions
- Autoregressive: $P(x_t \mid x_{<t}, c)$ где $c$ — mel-spectrogram
- 16 kHz → 16,000 шагов/сек → медленно
- **Качество:** Лучшее на момент выхода (2016)

#### HiFi-GAN (State-of-the-art, 2021)

- Generator: transposed CNN (upsampling mel → waveform)
- Discriminator: Multi-scale + Multi-period
- **Качество:** Сопоставимо с WaveNet
- **Скорость:** В 1000× быстрее (реальное время на CPU)

---

## 3. End-to-End TTS

### Идея

Одна модель, которая переходит от **текста напрямую к waveform**, минуя промежуточные признаки.

### VITS (Conditional VAE + Flow) — Kim et al., 2021

**Архитектура:**

```
Text → Posterior Encoder (mel) → Flow → Prior Encoder → Decoder (HiFi-GAN) → Waveform
          ▲                                    │
          └─────── mel-spectrogram ────────────┘
```

- **VAE framework:** текст → prior; mel-спектрограмма → posterior
- **Normalizing flow:** связь между prior и posterior
- **Monotonic alignment** (MAS — Monotonic Alignment Search) для выравнивания
- **Vocoder:** встроенный HiFi-GAN

**Результаты:**
- Одна модель вместо трёх
- Качество: MOS ~4.4 (сопоставимо с Tacotron 2 + HiFi-GAN)
- Быстрее: end-to-end оптимизация

### End-to-End vs Pipeline

| Критерий | Pipeline (3 этапа) | End-to-End |
|---|---|---|
| Сложность обучения | Три отдельных модели | Одна модель |
| Качество | Высокое (может превзойти) | Высокое (растёт с данными) |
| Данные | Много (каждый этап) | Очень много |
| Гибкость | Можно менять компоненты | Всё или ничего |
| Управление | Полный контроль | Меньше контроля |
| Латенси | Сумма этапов | Меньше |

---

## 4. Полное сравнение ASR и TTS

| Аспект | [[DL 48 - ASR|ASR]] | TTS |
|---|---|---|
| Направление | Audio → Text | Text → Audio |
| Выход | Текст (дискретный) | Waveform (непрерывный) |
| Главная метрика | WER / CER | MOS (Mean Opinion Score) |
| Encoder | Полный (аудио) | Полный (текст) |
| Decoder | [[DL 49 - CTC|CTC]] / RNN-T head | Autoregressive / Vocoder |
| Alignment | Проблема (blank/attention) | Проблема (продолжительности) |
| Real-time | Да (streaming) | Да (FastSpeech 2) |
| Токены | BPE / Character / Phonemes | Phonemes / Characters |

---

## 5. Современные тенденции

### ASR

- **Self-supervised pre-training** (Wav2Vec2, HuBERT, WavLM) — pre-training на неразмеченных аудио
- **Large-scale multilingual** (Whisper, USM) — одна модель на 100+ языков
- **Efficient inference** (Streaming, FastConformer, гибридные модели)

### TTS

- **Prompt-based TTS** (NaturalSpeech3, Voicebox) — контроль стиля/эмоции
- **Zero-shot voice cloning** — синтез новым голосом без адаптации
- **Expressive TTS** — эмоции, интонации, невербальные звуки (смех, паузы)

### Audio понимание в целом

- **Massively multilingual models** (MMS, Whisper, SeamlessM4T)
- **Audio Foundation Models** — одна модель для ASR + TTS + classification (AudioLM, GSLM)

---

## 6. Ключевые выводы

1. **ASR архитектуры:**
   - **[[DL 49 - CTC|CTC]]** — простой, но без контекстной зависимости (conditional independence)
   - **RNN-T** — CTC + prediction network для текстового контекста, стандарт для streaming
   - **LAS** — encoder-decoder с attention, лучшее качество, но не для streaming

2. **TTS pipeline (классический):**
   - **Text Normalization** → **Acoustic Model** (mel-спектрограмма) → **Vocoder** (waveform)
   - Tacotron 2: авторегрессионный, высокое качество, медленный
   - FastSpeech 2: неавторегрессионный, быстрый, robust

3. **Vocoders:** HiFi-GAN (GAN, быстрый, высокое качество) — современный стандарт; [[DL 58 - Диффузионные модели|WaveNet]] — первый высококачественный нейросетевой vocoder

4. **End-to-End TTS (VITS):** одна модель текст→waveform через VAE+flow+HiFi-GAN

5. **КТ:** RNN-T > CTC для качества; FastSpeech2 > Tacotron2 для скорости/robustness

---

**Связанные вопросы:**
- [[DL 48 - ASR]] — постановка задачи, WER/CER, уровни токенизации
- [[DL 49 - CTC]] — Connectionist Temporal Classification, blank token, loss
- [[DL 33 - Архитектура Transformer]] — основа архитектур encoder-decoder в ASR/TTS
- [[DL 58 - Диффузионные модели]] — WaveNet как первый высококачественный нейровокодер
