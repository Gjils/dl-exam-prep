# DL 48 — ASR

> **Постановка STT, уровни токенизации, WER/CER, alignment problem.
> Смежные темы: [[DL 46 - Звук как сигнал]], [[DL 47 - Частотное представление звука]], [[DL 49 - CTC]], [[DL 50 - Архитектуры ASR и TTS]].**

---

## 1. ASR / STT — постановка задачи

**Automatic Speech Recognition (ASR)** или **Speech-to-Text (STT)**: преобразование речевого сигнала в текстовую транскрипцию.

### Формальная постановка

Дано:
- Аудиосигнал $X = [x_1, x_2, \dots, x_T]$ — последовательность акустических векторов ([[DL 47 - Частотное представление звука|log-mel]], MFCC, raw audio)
- Транскрипция $Y = [y_1, y_2, \dots, y_U]$ — последовательность текстовых токенов

Задача: найти наиболее вероятную последовательность токенов:

$$
\hat{Y} = \arg\max_Y P(Y \mid X)
$$

### Ключевые свойства задачи

| Свойство | Описание |
|---|---|
| **Variable-length input/output** | $T \gg U$ (аудио длиннее текста) |
| **Monotonic alignment** | Прогресс в тексте только вперёд, без перестановок |
| **Noise robustness** | Шум, реверберация, разные дикторы |
| **Real-time** | Меньше задержки — лучше UX |

### Архитектуры ASR (исторически)

| Поколение | Подход | Примеры |
|---|---|---|
| 1 (1980–2010) | GMM-HMM | Устаревшее |
| 2 (2012–2018) | DNN-HMM | DeepSpeech, Kaldi |
| 3 (2018–2022) | End-to-end (CTC, RNNT, LAS) | Jasper, QuartzNet, RNN-T |
| 4 (2022+) | Transformer/Encoder-only | Whisper, Wav2Vec2, HuBERT |

---

## 2. Уровни токенизации

### Проблема

В отличие от NLP, аудиосигнал не имеет естественного разбиения на "слова" — нужно решить, что такое "единицы" транскрипции.

### Уровни

| Уровень | Пример | Размер словаря | Плюсы | Минусы |
|---|---|---|---|---|
| **Character (символы)** | "hello" → h,e,l,l,o | 30–50 (рус/англ) | Простой | Длинные последовательности |
| **Grapheme (буквы)** | "привет" → п,р,и,в,е,т | ~33–50 | Нет OOV | — |
| **Phoneme (фонемы)** | "hello" → həloʊ | 40–60 | Лингвистичен | Нужен lexicon |
| **Subword (BPE)** | "speaking" → speak,ing | 2K–8K | Компромисс | Более сложный |
| **Word (слова)** | "hello world" | 100K+ | Семантичен | OOV, редкие слова |

### Сравнение для ASR

| Критерий | Character | Subword (BPE) | Word |
|---|---|---|---|
| OOV (out-of-vocabulary) | Нет | Почти нет | Да |
| Длина output | Длинная | Средняя | Короткая |
| Зависимости между токенами | Нет (ортография) | Лингвистические | Лингвистические |
| Сложность decoding | Низкая | Средняя | Высокая |
| Современные системы | Whisper (multilingual) | Wav2Vec2, Conformer | Нет (только lexicon) |

### Tokenization in Whisper

- **Vocabulary:** ~50,000 BPE tokens (многоязычный)
- **Special tokens:** `<|startoftranscript|>`, `<|en|>`, `<|notimestamps|>`
- **Language token** указывает язык
- **Timestamp tokens:** `<|0.00|>`, `<|0.04|>`, etc., для сегментации

---

## 3. WER (Word Error Rate)

### Определение

Метрика качества ASR — процент слов, распознанных с ошибкой.

**Формула:**

$$
\text{WER} = \frac{S + D + I}{N} = \frac{\text{Substitutions} + \text{Deletions} + \text{Insertions}}{\text{Total words in reference}}
$$

где:
- $S$ — замены (substitutions)
- $D$ — удаления (deletions)
- $I$ — вставки (insertions)
- $N$ — количество слов в эталонной транскрипции (reference)

### Levenshtein distance на уровне слов

WER = **словесное расстояние Левенштейна**, нормализованное на длину reference.

### Пример

```
Reference:  "I HAVE A CAT"
Hypothesis: "I HATE DOG" 

Alignment:
I    HAVE     A    CAT
I    HATE    ???   DOG
=    S        D     S

WER = (2 + 1 + 0) / 4 = 0.75 = 75%
```

### CER (Character Error Rate)

Аналог WER, но на уровне символов:

$$
\text{CER} = \frac{S_c + D_c + I_c}{N_c}
$$

где $N_c$ — количество символов в reference.

### Сравнение WER и CER

| Метрика | Чувствительность | Когда важнее |
|---|---|---|
| WER | Смысловые ошибки | Разговорные, диалоговые системы |
| CER | Орфографические ошибки | Детальная транскрипция, имена |

### Практические значения

| Качество | WER | Пример |
|---|---|---|
| Идеальное | 0% | Безошибочно |
| Отличное | < 5% | Диктант |
| Хорошее | 5–10% | Разговор в тишине |
| Приемлемое | 10–20% | Шумная обстановка |
| Плохое | > 20% | Нужны улучшения |

### Современные benchmark'и

| Датасет | WER (человек) | WER (лучшая модель) |
|---|---|---|
| LibriSpeech test-clean | ~2.5% | ~1.4% (HuBERT) |
| LibriSpeech test-other | ~5% | ~2.6% |
| Common Voice English | — | ~3% (Whisper) |
| Switchboard (telephone) | ~5% | ~4% (RNN-T) |

---

## 4. Alignment Problem (проблема выравнивания)

### Что это?

В ASR аудиофреймы $[x_1, \dots, x_T]$ и текстовые токены $[y_1, \dots, y_U]$ имеют **разную длину** ($T \gg U$) и в общем случае **неизвестное соответствие**.

Нужно найти **alignment** — какой фрагмент аудио соответствует какому токену.

### Трудности

1. **Разная длина:** $T$ (10 мс фреймы) обычно в 5–10× больше $U$ (токенов)
2. **Monotonic but not strictly monotonic:** прогресс только вперёд, но возможны паузы (silence) между словами
3. **Неоднозначность границ:** где заканчивается "s" и начинается "t" в "street"?
4. **Co-articulation:** звуки влияют друг на друга — границы размыты

### Подходы к решению

| Метод | Выравнивание | Пример |
|---|---|---|
| **[[DL 49 - CTC|CTC]]** | Монотонное, с blank | Jasper, QuartzNet |
| **[[DL 50 - Архитектуры ASR и TTS|RNN-T]]** | Монотонное, с предсказанием | Google RNN-T |
| **Attention (LAS)** | Неявное (soft alignment) | Listen Attend and Spell |
| **External aligner** | Montreal Forced Aligner | GMM-HMM alignment |
| **Monotonic attention** | Жёсткое монотонное | Monotonic Chunkwise Attention |

### [[DL 49 - CTC|CTC]] vs [[DL 50 - Архитектуры ASR и TTS|RNN-T]] vs Attention

| Аспект | CTC | RNN-T | LAS (Attention) |
|---|---|---|---|
| Алгоритм alignment | Blank-based | Prediction network + joiner | Cross-attention |
| Monotonic | ✅ | ✅ | Нет (может переставлять) |
| Длина выхода | ≤ длина входа | ≤ длина входа | Любая |
| Real-time ASR | Да | Да | Нет (полный контекст) |
| Сложность | Низкая | Средняя | Высокая |
| Качество | Хорошее | Отличное | Отличное |

### Почему alignment важен?

1. **Обучение:** без выравнивания нельзя посчитать loss ($P(Y|X)$ — нужно суммировать по всем alignments)
2. **Инференс:** alignment определяет, когда предсказывать следующий токен
3. **Timestamp prediction:** для сегментации (когда произнесено слово)
4. **Streaming:** частичное выравнивание до конца фразы

---

## 5. ASR Pipeline (классический End-to-End)

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ Audio    │──▶│ Feature  │──▶│ Encoder  │──▶│ Decoder  │──▶ Text
│ waveform │   │ Extract  │   │ (Acoustic│   │ (Text    │
│ 16 кГц   │   │ (log-mel)│   │  model)  │   │ model)   │
└──────────┘   └──────────┘   └──────────┘   └──────────┘
                     │              │              │
                  80 ch.          T×d           U chars
                  25ms/10ms
```

### Компоненты

**Encoder (акустическая модель):**
- Вход: [[DL 47 - Частотное представление звука|log-mel спектрограмма]] $X \in \mathbb{R}^{T \times 80}$
- Выход: скрытые представления $H \in \mathbb{R}^{T' \times d}$
- Архитектура: CNN (Jasper, QuartzNet), Transformer, Conformer

**Decoder (текстовая модель):**
- Вход: скрытые представления $H$
- Выход: последовательность токенов $Y$
- Типы: [[DL 49 - CTC|CTC head]], RNN-T joiner, Transformer decoder

---

## 6. Типичные конфигурации ASR

### Whisper (OpenAI)

| Параметр | Значение |
|---|---|
| Модель | Transformer Encoder-Decoder |
| Вход | 80-channel log-mel, 25ms, 10ms hop |
| Токенизация | BPE (50K+ multilingual) |
| Декодирование | Autoregressive (cross-attention) |
| Языки | ~100 |
| Масштабы | tiny (39M) — large (1.5B) |

### Wav2Vec2 (Meta)

| Параметр | Значение |
|---|---|
| Модель | Transformer encoder + CTC |
| Вход | Raw waveform (через CNN feature encoder) |
| Pre-training | Self-supervised (masked prediction) |
| Fine-tuning | Supervised on transcribed audio |
| Декодирование | CTC beam search |

### Conformer (Google)

| Параметр | Значение |
|---|---|
| Модель | Convolution + Transformer encoder |
| Вход | Log-mel + spec augmentation |
| Декодирование | RNN-T или CTC |
| Особенность | Локальные + глобальные контексты |

---

## 7. Ключевые выводы

1. **ASR (STT)** — задача преобразования речи в текст, $P(Y \mid X)$
2. **Уровни токенизации:** character (простота) → subword/BPE (компромисс) → word (редко для ASR)
3. **WER** — основная метрика: $WER = (S + D + I) / N$ (чем меньше, тем лучше, 0% = идеал)
4. **CER** — метрика на уровне символов, дополняет WER
5. **Alignment problem** — центральная трудность ASR: аудио и текст имеют разную длину без прямого соответствия
6. Основные подходы к выравниванию: [[DL 49 - CTC|CTC]] (blank + collapse), [[DL 50 - Архитектуры ASR и TTS|RNN-T]] (prediction + joiner), **attention** (мягкий alignment)
7. Современные системы: Whisper (encoder-decoder), Wav2Vec2 (self-supervised + CTC), Conformer (CNN + Transformer + RNNT)

---

**Связанные вопросы:**
- [[DL 46 - Звук как сигнал]] — дискретизация звука, PCM, теорема Найквиста
- [[DL 47 - Частотное представление звука]] — спектрограммы, mel-шкала, log-mel признаки
- [[DL 49 - CTC]] — Connectionist Temporal Classification для выравнивания аудио и текста
- [[DL 50 - Архитектуры ASR и TTS]] — RNN-T, LAS, TTS pipeline, end-to-end модели
