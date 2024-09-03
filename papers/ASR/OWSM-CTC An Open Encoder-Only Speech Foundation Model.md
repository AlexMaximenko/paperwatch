# OWSM-CTC: An Open Encoder-Only Speech Foundation Model for Speech Recognition, Translation, and Language Identification

[https://arxiv.org/pdf/2402.12654.pdf](https://arxiv.org/pdf/2402.12654.pdf)

![Screenshot 2024-02-23 at 14.57.59.png](OWSM-CTC%20An%20Open%20Encoder-Only%20Speech%20Foundation%20Model/Screenshot_2024-02-23_at_14.57.59.png)

В конце прошлого года группа авторов решила сделать Whisper-Like модель на октрытых данных (с multitask + multilang). В этой статье исследователи предлагают обучить в таком же сетапе и на этих же данных Encoder-only CTC-модель. Их мотивация в том, что такая модель работает намного быстрее на инференсе + не страдает от пробемы галлюцинаций.

![Screenshot 2024-02-29 at 18.22.46.png](OWSM-CTC%20An%20Open%20Encoder-Only%20Speech%20Foundation%20Model/Screenshot_2024-02-29_at_18.22.46.png)

## Архитектура

Свою модель они назвали OWSM-СTC, модель энкодер-декодер, с которой они сравнивались, называется OWSM v3.1 series.

Модель encoder-only, в качетсве энкодера они выбрали [E-Branchformer](https://arxiv.org/pdf/2210.00077.pdf) (как и OWSM v3.1).

Transformer-like архитектура, глубже не изучал, ниже метрики из статьи. Разница с конформером незначительная.

![Screenshot 2024-02-29 at 18.30.32.png](OWSM-CTC%20An%20Open%20Encoder-Only%20Speech%20Foundation%20Model/Screenshot_2024-02-29_at_18.30.32.png)

Может возникнуть вопрос: “А как мы собираемся учить CTC модель в multitask + multilang сетапе”?

Так же, как это делается в энкодер-декодер моделях - с помощью служебных токенов.

Их мы подаем на вход модели, конкатенируя с аудио-токенами (см. картинку ниже). Также их добавляем в словарь модели, чтоб она могла их предсказывать.

![Screenshot 2024-02-29 at 18.32.28.png](OWSM-CTC%20An%20Open%20Encoder-Only%20Speech%20Foundation%20Model/Screenshot_2024-02-29_at_18.32.28.png)

Глядя на картинку, может возникнуть 2 вопроса:

- Что за фигня слева? (Prompt Encoder)
- Что за фигня справа? (Linear &Add и дополнительные лоссы)

Ответы:

- Prompt Encoder - трансформерный энкодер, который нужен для использования контекстов. На вход он получает текст-контекст, прогоняет через себя и выдает эмбеддинги.
    
    Дальше эти эмбеддинги используются энкодером с помощью Cross-attention-а. Cross-attention модули добавляются в каждый третий слой
    
- Справа - подход из [Relaxing the Conditional Independence Assumption of CTC-based ASR by Conditioning on Intermediate Predictions](https://arxiv.org/pdf/2104.02724.pdf). Почитать можно [тут](Relaxing%20the%20Conditional%20Independence%20Assumption.md).
    
    Отличие от оригинальной статьи только в том, какие таргеты мы подаем. Кажется логичным в каждом промежуточном слое использовать тот же таргет, что и на выходе модели: таргет-перевод при переводе и таргет-транскрипцию при транскрибации. Но, как пишут авторы, в таком сетапе модель не сходится. Они провели ablation на модельке и сете поменьше и предложили в первую половину промежуточных CTC-голов подавать всегда таргет-транскрибацию, имитируя таким образом каскадную систему: сначала модель учится транскрибировать речь, а затем переводить. И все это в рамках одной туши. Табличка из соответствующего ablation-а ниже.
    
    ![Screenshot 2024-02-29 at 18.50.46.png](OWSM-CTC%20An%20Open%20Encoder-Only%20Speech%20Foundation%20Model/Screenshot_2024-02-29_at_18.50.46.png)
    

Размер модели быд подобран таким образом, чтоб вместе с промп-энкодером получилось столько же параметров, сколько и в OWSM v3.1:

- 27 слоев E-Branchformer с размерностью 1024 и 16-ю головами внимания
- 4 промежуточных слоя (6, 12, 15 и 21) содержат в себе CTC-головы. Первые три обучаются на ASR, последняя — на конкретную задачу, как и голова над самим энкодером
- Prompt-encoder есть четырехслойный трансформер с внутренней размерностью 512 и 8 внимательными головами. Как писал раньше, его выходы используются в Cross-Attention аудио-энкодера на каждом третьем слое

Обучается все это совместно. Если в датасете сегментированные длинные записи - в качестве контекста берут транскрипцию предыдущего сегмента с вероятностью 50% и с такой же вероятностью берут <na> токен. Для датасетов с короткими сэмплами без контекста используют всегда <na> токен.

Также стоит упомянуть Language Identification: во время обучения в половине случаев в модель подается токен <unk_lang> вместо токена языка. Так модель учится определять язык.

## Данные

Используют 180К часов открытых данных, рецепт брали из OWSM v3.1 для честного сравнения: 25 открытых сетов, покрывающих 151 язык с различными направлениями для перевода (не только X-En, как в Whisper, но и En-X).

Входные данные, как и в случае Whisper-а, всегда паддятся до 30 секунд (видимо, из-за FlashAttention). На задачу определения таймстемпов не обучают т.к. говорят что можно определять их с помощью forced alignment-а.

## Результаты

Обучают на 64 A100 (40GB), примерное время обучения - 300 часов.

### Language Identification

![Screenshot 2024-02-29 at 19.11.57.png](OWSM-CTC%20An%20Open%20Encoder-Only%20Speech%20Foundation%20Model/Screenshot_2024-02-29_at_19.11.57.png)

### English Speech Recognition

![Screenshot 2024-02-29 at 19.12.39.png](OWSM-CTC%20An%20Open%20Encoder-Only%20Speech%20Foundation%20Model/Screenshot_2024-02-29_at_19.12.39.png)

### Multilingual Speech Recognition

![Screenshot 2024-02-29 at 19.13.23.png](OWSM-CTC%20An%20Open%20Encoder-Only%20Speech%20Foundation%20Model/Screenshot_2024-02-29_at_19.13.23.png)

### Speech Translation

![Screenshot 2024-02-29 at 19.14.53.png](OWSM-CTC%20An%20Open%20Encoder-Only%20Speech%20Foundation%20Model/Screenshot_2024-02-29_at_19.14.53.png)

![Screenshot 2024-02-29 at 19.15.05.png](OWSM-CTC%20An%20Open%20Encoder-Only%20Speech%20Foundation%20Model/Screenshot_2024-02-29_at_19.15.05.png)

### Long-form ASR

![Screenshot 2024-02-29 at 19.16.45.png](OWSM-CTC%20An%20Open%20Encoder-Only%20Speech%20Foundation%20Model/Screenshot_2024-02-29_at_19.16.45.png)

![Screenshot 2024-02-29 at 19.18.16.png](OWSM-CTC%20An%20Open%20Encoder-Only%20Speech%20Foundation%20Model/Screenshot_2024-02-29_at_19.18.16.png)

 В конце стоит еще сказать, что OWSM-CTC использует в 2 раза более сильный Downsampling в сравнении с OWSM v3.1. Снизу табличка из ablation.

![Screenshot 2024-02-29 at 19.22.00.png](OWSM-CTC%20An%20Open%20Encoder-Only%20Speech%20Foundation%20Model/Screenshot_2024-02-29_at_19.22.00.png)