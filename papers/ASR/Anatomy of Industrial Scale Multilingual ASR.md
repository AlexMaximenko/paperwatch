# Anatomy of Industrial Scale Multilingual ASR

[https://arxiv.org/pdf/2404.09841](https://arxiv.org/pdf/2404.09841)

![Screenshot 2024-05-16 at 18.13.45.png](Anatomy%20of%20Industrial%20Scale%20Multilingual%20ASR/Screenshot_2024-05-16_at_18.13.45.png)

Практикующие датасаентисты из Assembly AI рассказывают, как делали свой SOTA-multilang ASR, бьющий Виспер, ASR-api крупных компаний и имеющий много полезных свойств.

Рассмотрим каждую стадию / аспект обучения модели, чтоб понять как делают лучший ASR “там у них в большой компании”.

## Data

В отличие от Meta и OpenAI, которые своими моделями хотели решить задачу ASR для всех языков мира, авторы решили ограничиться “high-resource” языками: английский, французский, немецкий и испанский. Русский оставили нам.

Табличка с данными ниже:

![Screenshot 2024-05-16 at 18.41.54.png](Anatomy%20of%20Industrial%20Scale%20Multilingual%20ASR/Screenshot_2024-05-16_at_18.41.54.png)

### Unsupervised data

Неразмеченные данные брали из “publicy available sources as well as our partners”. Чтобы убедиться, что сэмплы данных содержат достаточно речи, пробегались по ним SILERO-вадом и отбрасывали все файлы, где доля речи меньше 70%. 

Использовали файлы с длительностью от 8 до 64 секунд, а во время претрейна обрезали / паддили сэмплы до 32 секунд.

### Supervised data

Покупали / размечали / брали из открытых источников с подходящей лицензией и у партнеров. Т.к. хотели supervised-данные максимально чистые, проводили их фильтрацию: сравнивали транскрипции с предсказаниями уже имеющихся у них ASR-моделей. Для фильтрации применяли различные метрики, включая WER, ошибки последовательных удалений и точность определения языка в сэмпле.

### Pseudo-labeled data

Для псевдо-разметки брали две ASR-модели и отбрасывали сэмплы, для которых WER между гипотезами этих моделей превышает 20%. Все получившиеся сэмплы резали на куски длиной не более 30 секунд.

Не могу не вставить картинку из их статьи Conformer-1 для SOTA-распознавания английского языка:

![Screenshot 2024-05-16 at 19.14.18.png](Anatomy%20of%20Industrial%20Scale%20Multilingual%20ASR/Screenshot_2024-05-16_at_19.14.18.png)

Здесь они учат английскую модель без претрейна и показывают эффект от добавления псевдолейблов в разном количестве. Видно, что к значительному количеству размеченных данных (57К часов) они добавляют еще более значительное количество псевдолейблов (520К часов) и это помогает. 

## Model architecture

Архитектура энкодера - 600М конформер (как в USM) c chunk-wise attention с размером чанка 8 секунд. Декодер: RNN-T для основных экспериментов, CTC-для некоторых дополнительных.

![Screenshot 2024-05-16 at 23.03.31.png](Anatomy%20of%20Industrial%20Scale%20Multilingual%20ASR/Screenshot_2024-05-16_at_23.03.31.png)

## Training Method

Обучали с помощью Jax + Flax на Google TPU, также использовали FSDP.

Первая стадия обучения — BEST-RQ  претрейн на 12.5M часов данных.

Вторая - файнтюн на 570K supervised данных + 1M pseudo-labels. Соотношение псевдолейблов и размеченных данных в батче было 2:3, файнтюн делали в fp32 т.к. bf16 вызывал loss spikes.

![Screenshot 2024-05-16 at 23.04.33.png](Anatomy%20of%20Industrial%20Scale%20Multilingual%20ASR/Screenshot_2024-05-16_at_23.04.33.png)

## Inference

Как делают инференс: бьют длинную вавку на сегменты, декодируют каждый сегмент и получают транскрипции + word-level timestamps, затем мерджат их.

Сегментируют с VAD-ом, следят чтоб концы кусков попадали на No-Speech отрезки. В качестве VAD используют WebRTC.

После мерджинга делают постпроцессинг: денормализация + timestamps offset.

## Results

### English ASR

![Screenshot 2024-05-16 at 23.27.45.png](Anatomy%20of%20Industrial%20Scale%20Multilingual%20ASR/Screenshot_2024-05-16_at_23.27.45.png)

### Multilingual ASR

![Screenshot 2024-05-16 at 23.28.05.png](Anatomy%20of%20Industrial%20Scale%20Multilingual%20ASR/Screenshot_2024-05-16_at_23.28.05.png)

### Code-switching

Авторы обнаружили, что их модель естественным образом умеет обобщатся на code-switching speech, т.е. на сэмплы, где спикер меняет языки во время разговора.

Чтобы замерить это в числах, сделали code-switching датасет, для этого конкатили английские сэмплы из LibriSpeech test-clean и Multilingual Librispeech.

Делали это следующим образом:

1. Randomly determine a target duration from the range between 30 and 180 seconds.
2. Randomly select a file from either MLS or LibriSpeech test-clean datasets.
3. Continue selecting files from these two datasets in an alternating fashion until the total audio
duration exceeds the target length.
4. Concatenate all the selected audio files to generate a single audio file. The reference
transcription is also created by concatenating all the corresponding reference transcriptions.

Сравнивались с Whisper-ом и Canary. Большой плюс их модели - не надо указывать токен языка.

![Screenshot 2024-05-16 at 23.35.09.png](Anatomy%20of%20Industrial%20Scale%20Multilingual%20ASR/Screenshot_2024-05-16_at_23.35.09.png)

### Hallucination analysis

Также авторы решили показать, что их модель меньше подвержена галлюцинациям чем Whisper и Canary. Для этого считают следующие метрики:

- Fabrication rate (FRN ): number of N or more consecutive insertion or substitution errors
observed per hour.
- Omission rate (ORN ): number of N or more consecutive deletion errors observed per hour.
- Hallucination rate (HRN ): number of N or more consecutive insertion, substitution or
deletions errors observed per hour.
    
    ![Screenshot 2024-05-16 at 23.41.28.png](Anatomy%20of%20Industrial%20Scale%20Multilingual%20ASR/Screenshot_2024-05-16_at_23.41.28.png)
    
    ![Screenshot 2024-05-16 at 23.43.01.png](Anatomy%20of%20Industrial%20Scale%20Multilingual%20ASR/Screenshot_2024-05-16_at_23.43.01.png)
    
    ### Timestamp estimation
    
    Также авторы сравнивают точность предсказания word timestamps своей модели и Whisper/Canary.
    
    ![Screenshot 2024-05-16 at 23.46.00.png](Anatomy%20of%20Industrial%20Scale%20Multilingual%20ASR/Screenshot_2024-05-16_at_23.46.00.png)
    
    Здесь по оси X они варьируют трешхолд $t$: если модель предсказывает таймстемп с ошибкой не больше $t$ - считают предсказание правильным.
    
    Отмечают (но не показывают на графиках), что стриминговая модель работает значительно хуже, чем bidirectional и часто выдает таймстемпы с запозданием. Также говорят, что для получения точных таймстемпов лучше использовать гриди-декодинг т.к. при бимсерче становится хуже.
    
    ![Screenshot 2024-05-16 at 23.49.35.png](Anatomy%20of%20Industrial%20Scale%20Multilingual%20ASR/Screenshot_2024-05-16_at_23.49.35.png)
    
    ### Impact of Pre-training
    
    ![Screenshot 2024-05-16 at 23.50.30.png](Anatomy%20of%20Industrial%20Scale%20Multilingual%20ASR/Screenshot_2024-05-16_at_23.50.30.png)