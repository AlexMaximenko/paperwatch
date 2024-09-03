# SALMONN: TOWARDS GENERIC HEARING ABILITIES FOR LARGE LANGUAGE MODELS

[https://openreview.net/pdf?id=14rn7HpKVk](https://openreview.net/pdf?id=14rn7HpKVk)

![Screenshot 2024-07-05 at 12.48.15.png](SALMONN%20TOWARDS%20GENERIC%20HEARING%20ABILITIES%20FOR%20LARGE%20LANGUAGE%20MODELS/Screenshot_2024-07-05_at_12.48.15.png)

Авторы представляют очередную SpeechLLM — SALMONN. Проводят небольшое, но интересное исследование способоностей модели к обобщению на задачи, которых не было в SFT. Модель и “код для обучения” - в [опенсорсе](https://github.com/bytedance/SALMONN).

## Архитектура

![Untitled](SALMONN%20TOWARDS%20GENERIC%20HEARING%20ABILITIES%20FOR%20LARGE%20LANGUAGE%20MODELS/Untitled.png)

Берем 2 энкодера, соединяем их выходы, прогоняем через qformer и подаем в замороженную LLM-ку с размороженными LoRA-адаптерами.

### Энкодеры

Первый энкодер - энкодер от Whisper large, который отвечает за speech-аудио.

Второй - BEATs audio encoder, который отвечает за Non-speech audio.

У обоих одинаковый framerate - 50**Hz,** поэтому без дополнительного ресэмплинга конкатим фичи и кайф, объединяем speech и non-speech.

### Q-former

![Untitled](SALMONN%20TOWARDS%20GENERIC%20HEARING%20ABILITIES%20FOR%20LARGE%20LANGUAGE%20MODELS/Untitled%201.png)

За месяц до выхода salmonn их стажер [исследовал](https://arxiv.org/pdf/2309.13963)  разные адапетры для SpeechLLM и получилось, что qformer работает лучше всех. Коротко - слой трансформера, в который добавляют фиксированное количество обучаемых векторов. Эти вектора собирают информацию из аудио-модальности с помощью cross-attention, где в качестве query выступают эти обучаемые вектора, а потом шарят эту информацию между собой с помощью self-attention. На выходе получаем фиксированное количество векторов. 

В рамках данной статьи возможность сжимать вход изменяемой длины в выход фиксированной длины не нужна т.к. аутпуты энкодеров делятся на чанки размера **L** и к каждому чанку применяется q-former.

### LLM

Берут Vicuna и навешивают LoRA на **query** и **value** матрицы self-attention-а модели.

## Обучение

Обучают в 2 с половиной этапа: претрейн, инструктивный SFT и activation tuning.

### Претрейн

Размораживают q-former с LoRA и еа большом объеме простых данных (asr + audio captioning) обучают их, делая alignment между textual и audio-информацией. Пишут, что эти две задачи содержат ключевую информацию о speech и non-speech контенту, при этом не требуя сложного reasoning-а и understaining-а.

В качестве данных берут LibriSpech 960h + GigaSpeech M-set1000h для ASR и 2800h WavCaps + AudioCaps + Clotho для audio captioning.

### Инструктивный SFT

Берут следующие задачи:

- ASR
- Automatic Speech Translations
- Automatic Audio Captioning
- Phone Recognition
- Emotion Recognition
- Music Captioning
- Overlapped Speech Recognition
- Spearker Verification
- Gender Recognition
- Speech Question Answering
- Audio Caption Answering
- Music Question Answering

Для Question Answering-тасок генерят вопросы с помощью ChatGPT по текстовому описанию из сэмлпа.

Табличка с данными:

![Screenshot 2024-07-05 at 14.08.46.png](SALMONN%20TOWARDS%20GENERIC%20HEARING%20ABILITIES%20FOR%20LARGE%20LANGUAGE%20MODELS/Screenshot_2024-07-05_at_14.08.46.png)

Формат инструкций для обучения:

![Screenshot 2024-07-05 at 15.00.27.png](SALMONN%20TOWARDS%20GENERIC%20HEARING%20ABILITIES%20FOR%20LARGE%20LANGUAGE%20MODELS/Screenshot_2024-07-05_at_15.00.27.png)

Также в статье предоставляют промпты для работы с ChatGPT: как для генерации инструктивных данных, так и для автоматической оценки ответом.

Делят все таски на 3 уровня:

- **Level-1:** таски, которые были в инструктивном SFT и поэтому простые для модели.
- **Level-2: speech-based NLP** таски, которых не было в SFT:
    - Keyword extracting (KE)
    - Spoken-query-based question answering
    - Speech-based slot filling (SF)
    - AST на языки, которых не было в SFT
    
    Со всеми этими тасками может справиться система Vicuna+ASR, поэтому можем явно сравнить полученную модель с этим бейзлайном
    
- **Level-3:** самые сложные таски:
    - Audio-based storytelling (Story)
    - Speech-Audio co-reasoning: подаем сэмпл аудио (музыка / sound-events) и speech-вопрос по этому аудио. Хотим чтоб модель на это ответила.

### Activation tuning

Инновационный подход к обучению, который позволяет решать проблему task-overfitting (когда модель может забивать на инструкцию к новой задаче и делать, например, просто ASR).

Для начала авторы анализируют причины, по которым возникает такое явление:

- Очень простые и не особо разнообразные инструкции при multimodal-файнтюне в сравнении с text-only sft
- Более конкретные и определенные таргеты на наиболее общих задачах ASR и AAC в срванении с, например, speech/audio answer questioning

Из-за этого условная вероятность правильного ответа модели на тесте при не очень знакомой инструкции получается очень маленькой и модель съезжает в ASR / AAC.

Пусть **Y** - таргет, **X** - входные данные и I - промпт. Тогда:

$$
P(Y | X, I) = \frac{P(Y | X) \cdot P(I | Y, X)}{P(I | X)}
$$

Из-за описанных выше проблем у нас вырождается $P(Y|X)$. 

Как это можно починить?

Регуляризовать $P(Y|X)$. 

Как - можно дообучить на тасках, распределение $P(Y|X)$ которых жестко отличается от ASR / AAC. В качестве таких тасок авторы выбирают задачи **level-3.** Ответы к ним генерятся людьми / lm-кой, основываясь на аудио + тексту к нему (ASR или AAC).

Авторы нашли эффективный способ генерации таких ответов - берут свою обученную модель и уменьшают scale-фактор LoRA. Таким образом тоже происходит регуляризация распределения. Такой трюк сильно ухудшает метрики модели на базовых тасках, поэтому такую урезанную модель используют для генерации лейблов на **level-3** задачках, а затем доучивают полноценную модель на этих лейблах. 

## Сетап и результаты экспериментов

Для извлечения аудио-фичей используют окна по 17 токенов (примерно 0.33 секунды), для каждого из которых q-former выдает 1 вектор → получаем 88 токенов для 30 секунд аудио.

В качестве модельки берут Vicuna-13B.

LoRA: scaling_factor = 4.0, rank=8. Суммарное число обучаемых параметров 33M.

В табличке ниже датасеты и метрики для всех оцениваемых задач.

![Screenshot 2024-07-05 at 14.56.11.png](SALMONN%20TOWARDS%20GENERIC%20HEARING%20ABILITIES%20FOR%20LARGE%20LANGUAGE%20MODELS/Screenshot_2024-07-05_at_14.56.11.png)

Результаты на всех тасках из обучения представлены в табличке ниже. В скобках на тяжелых задачах замеряют Following Rate, который показывает долю сэмплов, на которых модель следовала инструкции, а не делала ASR / AAC

![Screenshot 2024-07-05 at 14.54.34.png](SALMONN%20TOWARDS%20GENERIC%20HEARING%20ABILITIES%20FOR%20LARGE%20LANGUAGE%20MODELS/Screenshot_2024-07-05_at_14.54.34.png)

Также приводят графики качества моделек при уменьшении lora scaling / увеличении числа шагов обучения с помощью activation tuning:

![Screenshot 2024-07-05 at 15.32.22.png](SALMONN%20TOWARDS%20GENERIC%20HEARING%20ABILITIES%20FOR%20LARGE%20LANGUAGE%20MODELS/Screenshot_2024-07-05_at_15.32.22.png)

Интересно, что для шатания модельки в сторону менее тривиальных задачек нужно относительно небольшое число шагов.

И последний график - график изменения перплексии  $P(Y|X)$ и  $P(Y|X, I)$, на регуляризацию которых направлен activation tuning.

![Screenshot 2024-07-05 at 15.37.12.png](SALMONN%20TOWARDS%20GENERIC%20HEARING%20ABILITIES%20FOR%20LARGE%20LANGUAGE%20MODELS/Screenshot_2024-07-05_at_15.37.12.png)

Полученные графики полностью соответствуют решаемой авторами задаче - распределение регуляризуется и сдвигается в нужную сторону.