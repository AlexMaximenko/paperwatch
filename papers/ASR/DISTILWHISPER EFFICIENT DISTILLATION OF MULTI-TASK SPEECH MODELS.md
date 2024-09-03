# DISTILWHISPER: EFFICIENT DISTILLATION OF MULTI-TASK SPEECH MODELS
VIA LANGUAGE-SPECIFIC EXPERTS

[https://arxiv.org/pdf/2311.01070.pdf](https://arxiv.org/pdf/2311.01070.pdf)

![Screenshot 2023-11-10 at 14.11.24.png](DISTILWHISPER%20EFFICIENT%20DISTILLATION%20OF%20MULTI-TASK%20SPEECH%20MODELS/Screenshot_2023-11-10_at_14.11.24.png)

Не путать с [Distil-Whisper](DISTIL-WHISPER%20538b89a8511b46c788ed991cb23da114.md). Прикольно, что эти две статьи вышли с разницей в 1 день.

В отличии от модели Distil-Whisper, где авторы пытаются задистиллировать Whisper на задачу распознавания английской речи, в этой статье предлагается подход к улучшению распознавания речи на разных языках маленькой моделькой **whisper-small.** 

![Untitled](DISTILWHISPER%20EFFICIENT%20DISTILLATION%20OF%20MULTI-TASK%20SPEECH%20MODELS/Untitled.png)

Основной целью является получение эффективной техники файнтюна, которая не приведет к catastrophical forgetting, но и не будет слишком компромиссной по качеству. Делать все это они предлагают с маленькой моделью **whisper-small.** Почему? Скорее всего чтоб люди с небольшими ресурсами могли повторить обучение (что спорно т.к. дистиллируют **whisper-large-v2**), либо чтоб эти же самые люди могли хотя бы инферить модель (хотя авторы не выкладывают веса). Еще есть вариант что с большим виспером не завелось.

## DISTILWHISPER

Идея статьи заключается в двух моментах. Первый - для каждого отдельного языка сделать свой адаптер с механизмом гейтинга (взяли подход из статьи [Share or Not? Learning to Schedule Language-Specific Capacity for Multilingual Translation](https://openreview.net/pdf?id=Wj4ODo0uyCF)).Второй - вместо банального CE-обучения добавить дистилляцию.

### CLSR module

Можно заметить на картинке с описанием модели что-то с названием CLSR Layer на каждом слое после Attention-ов.  Это и есть обучаемые  language-specific адаптеры, под каждым из которых также нарисован гейт. При обучении разморожены только CLSR-слои, вся остальная модель остается замороженной.

![Screenshot 2023-11-17 at 15.26.45.png](DISTILWHISPER%20EFFICIENT%20DISTILLATION%20OF%20MULTI-TASK%20SPEECH%20MODELS/Screenshot_2023-11-17_at_15.26.45.png)

Сам CLSR-Layer представляет из себя feed-forward, веса которого инициализируются из замороженных feed-forward слоев самой модели. К каждому CLSR-Layer в добавок идет обучаемый гейт. Он представляет из себя сигмоиду от двуслойного перцептрона со скалярным выходом, к которому еще накинули нум, увеличивающийся по мере тренировки модели.

![Screenshot 2023-11-17 at 15.33.16.png](DISTILWHISPER%20EFFICIENT%20DISTILLATION%20OF%20MULTI-TASK%20SPEECH%20MODELS/Screenshot_2023-11-17_at_15.33.16.png)

Шум нужен, по описанию авторов подхода, чтоб “Вначале модели было легко учиться и градиенты были нормальными, но с увеличением $t$ (число шагов обучения) мы будем увеличивать шум чтоб модель более явно форсила важность признаков”. Итоговая формула модуля вместе с гейтом следующая:

![Screenshot 2023-11-17 at 15.38.52.png](DISTILWHISPER%20EFFICIENT%20DISTILLATION%20OF%20MULTI-TASK%20SPEECH%20MODELS/Screenshot_2023-11-17_at_15.38.52.png)

$h^{shared}$ здесь - выход из оригинального замороженного feed-forward слоя, $h^{lang}$ - выход из CLSR-модуля.

Также предлагается в обучение докинуть регуляризационный лосс, который будет контролировать “объем трафика через CLSR”, т.е. долю токенов, для которых мы использовали CLSR-гейт. Формула следующая:

![Screenshot 2023-11-17 at 15.39.59.png](DISTILWHISPER%20EFFICIENT%20DISTILLATION%20OF%20MULTI-TASK%20SPEECH%20MODELS/Screenshot_2023-11-17_at_15.39.59.png)

Здесь штука сверху расписывается следующим образом:

$$
\mathcal{G}_{(X,Y)} = \sum_{x\in X} \sum_{m \in \mathcal {M}_{enc}}g_m(x) + \sum_{y\in y} \sum_{m \in \mathcal {M}_{dec}}g_m(y)
$$

Здесь $\mathcal{M}_{enc}$ и  $\mathcal{M}_{dec}$ - множество слоев энкодера и декодера соответственно, $X$ и $Y$ - пара аудио/текст из трейна.

Это есть суммарная доля токенов, прошедшая через CLSR-слои как в энкодере (первое слагаемое), так и в декодере (второе).

В выражении для лосса же мы делим “суммарный поток через CLSR-модули” на “обдий суммарный поток” и форсим это значение быть близким к $b$ (в обучении авторы выбрали $b=0.5$).

### Distillation

Для дистилляции используют расстояние Йенсена-Шеннона вместо дефолтного расстояния Кульбака-Лейблера. Она также представляет из себя неотрицательное расстояние между распределениями, но является симметриченым и в статье [“f-divergence minimization for sequence-level knowledge distillation”](https://arxiv.org/abs/2307.15190) показывается что дистилляция с ней приводит к лучшим метрикам студента. Определяется дивергенция Йенсена-Шеннона как полусумма дивергенций входящих распределений с их смесью, т.е. $M = \frac{1}{2}(P + Q)$.

![Screenshot 2023-11-17 at 15.49.09.png](DISTILWHISPER%20EFFICIENT%20DISTILLATION%20OF%20MULTI-TASK%20SPEECH%20MODELS/Screenshot_2023-11-17_at_15.49.09.png)

## Эксперименты

Обучали на нескольких языках из Common Voice. Выбрали языки для которых: 

- было минимум 10К сэмплов
- WER между whisper-large-v2 и whisper-small больше 11
- язык присутствует в FLEURS

Итоговый список языков: Catalan (ca), Czech (cs), Galician (gl), Hungarian (hu), Polish (pl), Thai (th), Tamil (ta) и Ukranian (uk).

Замеряли метрики на CV как in-domain test и FLEURS как out-of-domain test. Обучали 10 эпох с  $lr=1e-4$ и $batch\_size=16$.

![Screenshot 2023-11-17 at 15.55.07.png](DISTILWHISPER%20EFFICIENT%20DISTILLATION%20OF%20MULTI-TASK%20SPEECH%20MODELS/Screenshot_2023-11-17_at_15.55.07.png)

![Screenshot 2023-11-17 at 15.55.14.png](DISTILWHISPER%20EFFICIENT%20DISTILLATION%20OF%20MULTI-TASK%20SPEECH%20MODELS/Screenshot_2023-11-17_at_15.55.14.png)

![Screenshot 2023-11-17 at 15.55.20.png](DISTILWHISPER%20EFFICIENT%20DISTILLATION%20OF%20MULTI-TASK%20SPEECH%20MODELS/Screenshot_2023-11-17_at_15.55.20.png)