# CONNECTING SPEECH ENCODER AND LARGE LANGUAGE MODEL FOR ASR

[https://arxiv.org/pdf/2309.13963.pdf](https://arxiv.org/pdf/2309.13963.pdf), 

Department of Electronic Engineering, Tsinghua University, 2 ByteDance

Авторы соединяют предобученный аудио-энкодер и предобученную LLM (Licuna) с помощью разных коннекторов и выбирают из них лучший.

Общая архитектура модели:

![Untitled](CONNECTING%20SPEECH%20ENCODER%20AND%20LARGE%20LANGUAGE%20MODEL%20FOR%20ASR/Untitled.png)

Аудио подается в замороженный  **Speech Encoder,** полученные эмбеддинги подаются в коннектор, там они переводятся в семантическое пространство **LLM** и идут к ней на вход. Дальше LLM делает свою работу.

Обозначим выходы из Speech Encoder-а как $X \in \mathcal{R}^{n_x \times d_x }$, входы в LLM, полученные с помощью коннектора как $T_{speech} \in \mathcal{R}^{n_t \times d_t }$ и скрытые состояния коннектора как $H \in \mathcal{R}^{n_h \times d_h }$.

## Коннекторы

### 1. Fully connected layers

Преобразование с помощью линейных слоев. Авторы сначала сжимают последовательность $X$ путем объединения соседних фреймов и получают $H$, а затем применяют два линейных слоя с ReLU:

$$
T_{speech} = \operatorname{Linear}(\operatorname{ReLU}(\operatorname{Linear}(H)))
$$

### 2. Multi-head cross-attention

Сжимают $X$ с помощью свертки и затем используют кросс-аттеншн с эмбеддингами из LLM (откуда их берут - непонятно, возможно берут все имеющиеся).

$$
H = \operatorname{Linear}(\operatorname{Conv1d}(X))
$$

$$
T_{speech} = \operatorname{MultiHead}(H, E, E)
$$

Здесь $H$ - query, а $E$ - эмбеддинги из LLM, использующиеся в качестве key и value.

### 3. Q-Former