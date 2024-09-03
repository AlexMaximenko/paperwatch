# EXPLORING RWKV FOR MEMORY EFFICIENT AND LOW LATENCY%20STREAMING%20ASR

[https://arxiv.org/pdf/2309.14758.pdf](https://arxiv.org/pdf/2309.14758.pdf)

Speech Lab of DAMO Academy, Alibaba Group, China

В статье авторы берут вышедшую недавно убийцу трансформеров RWKW и сравнивают ее с конформером в качестве audio-encoder-а для RNN-T модели для задачи стримингового распознавания речи.

## RWKW

Пара услов об [убийце](https://arxiv.org/pdf/2305.13048.pdf). Является представителем Transformers-like RNN-like моделей, т.е. по представляет собой RNN, в которую добавили аналог аттеншна. Аттеншн в таких моделях, разумеется, казуальный. Можно почитать описание его последователя, выбившего лучшие perplexity-метрики на NLP-задачх [Retentive Network: A Successor to Transformer for Large Language Models](../GENERAL%20DL%20db60ceb048c5454ea175b5b39ece8f55/Retentive%20Network%20A%20Successor%20to%20Transformer%20for%20L%20ba3e96d72c6e47a494f2c65da19aef92.md), идея у них схожая.

![Screenshot 2023-10-20 at 15.22.45.png](EXPLORING%20RWKV%20FOR%20MEMORY%20EFFICIENT%20AND%20LOW%20LATENCY%20STREAMING%20ASR/Screenshot_2023-10-20_at_15.22.45.png)

Схема модели на рисунке выше. Все понятно и похоже на трансформеры, разобраться надо только с Channel Mixing и Time Mixing.

### Time mixing

Пусть на вход пришли $(x_1, x_2, \dots, x_T)$. Тогда выход будет считаться следующим способом:

$$
o_t = W_o \cdot (\sigma(r_t) \odot wkv_t)
$$

**Receptance**, или $r_t$, считается просто сглаживанием $x_t$ со своим предшественником $x_{t-1}$

$$
r_t = W_r \cdot (\mu_r x_t + (1 - \mu_r)x_{t-1})
$$

Похожая на **qkv** штука, $wkv_t$ является аналогом (условным) аттеншна и считается следующим образом:

$$
wkv_t = \frac{\sum_{i=1}^{t-1}e^{-(t-1-i)w + k_i}v_i + e^{u+ k_t}v_t}{e^{-(t-1-i)w + k_i} + e^{u+ k_t}}
$$

Как заметить аналогию с аттеншном: $v_i$ - value, $k_i$ - key, в самой формуле записан своего рода софтмакс. Query здесь отсутствует, за него - вектор $w$. Один на всех. Плюс к этому он не перемножается с key, а складывается. И вектор $u$, который заменяет вектор $w$ для value текущего токена. По формуле сразу видно ограничение: обуславливаемся только на прошлое + потенциально забываем информацию.

Ключи и значения $k_t$ и $v_t$ считаются аналогично receptance:

$$
k_t = W_k \cdot (\mu_k x_t + (1 - \mu_k)x_{t-1})
$$

$$
v_t = W_v \cdot (\mu_v x_t + (1 - \mu_v)x_{t-1})
$$

Важное свойство: эта формула может быть записана в RNN-like виде:

![Screenshot 2023-10-20 at 15.37.07.png](EXPLORING%20RWKV%20FOR%20MEMORY%20EFFICIENT%20AND%20LOW%20LATENCY%20STREAMING%20ASR/Screenshot_2023-10-20_at_15.37.07.png)

### Channel mixing

Тут опять receptance-like формулы + в конце квадрат ReLU. Красивую логику и мотивацию я пока не придумал, поэтому просто смотрим формулку и понимаем что все просто.

![Screenshot 2023-10-20 at 15.40.38.png](EXPLORING%20RWKV%20FOR%20MEMORY%20EFFICIENT%20AND%20LOW%20LATENCY%20STREAMING%20ASR/Screenshot_2023-10-20_at_15.40.38.png)

## Эксперименты

В экспериментах использовали Transducer модели: в качестве энкодеров брали chunk conformer с разными размерами чанков / длинной контекста в прошлое и RWKW разного размера (параметры в табличке ниже). Размер конформера: 12 layers + 15 conv_kernel_size + 8 head_num + 512 dim + 2048 ffn_dim = 90M параметров.

![Screenshot 2023-10-20 at 15.45.00.png](EXPLORING%20RWKV%20FOR%20MEMORY%20EFFICIENT%20AND%20LOW%20LATENCY%20STREAMING%20ASR/Screenshot_2023-10-20_at_15.45.00.png)

В качестве остальной части модели брали RNN-T (Transducer в табличках) и [BAT](https://arxiv.org/pdf/2305.11571.pdf) (BAT: Boundary aware transducer for memory-efficient and low-latency ASR). Про BAT не успел прочитать нормально. Суть - ускорение подсчета RNN-T lossa за счет уменьшения размерности латтисы. В сравнении с RNN-T немного проседает качество, но уменьшается время на обучение. В контексте этой статьи нам неважно как работает эта штука т.к. мы сравниваем энкодеры. Зачем вообще ее взяли в эту статью? - BAT написали авторы разбираемой статьи.

## Результаты

![Screenshot 2023-10-20 at 15.48.30.png](EXPLORING%20RWKV%20FOR%20MEMORY%20EFFICIENT%20AND%20LOW%20LATENCY%20STREAMING%20ASR/Screenshot_2023-10-20_at_15.48.30.png)

Модельки над горизонтальной лининей - взятые из других работ, под ней - эксперименты, проделанные авторами.

Экспы авторов: RWKW работает значимо лучше конформера, когда у него снижают размер чанка до 8 фреймов (320 мс), плюс к этому сам RWKW работает без задержки (здесь никак не учитывается время на инференс модели, но, кажется, RWKW должно работать достаточно быстро т.к. все операции там transformers-like и комьюнити уже написало свои эффективные cuda-kernel для нее). При увеличении размера чанка в 2 раза конформер становится лучше RWKW, но имеет уже довольно серьезное lattency в 640мс.

Выводы: прикольно, кажется на asr-задачах тоже могут работать убийцы. Причем даже без внедрения в них сверток. Интересно посмотреть как они скэйлятся - у авторов всего один эксп WenetSpeech и там результаты неоднозначные + не такая уж большая разница в размерах моделей. Еще интересно посмотреть как соотносятся время инференса конформера и RWKW + посмотреть как бы себя показала здесь Retentive Network, все-таки метрики на NLP у нее были значимо лучше.

![Untitled](EXPLORING%20RWKV%20FOR%20MEMORY%20EFFICIENT%20AND%20LOW%20LATENCY%20STREAMING%20ASR/Untitled.png)