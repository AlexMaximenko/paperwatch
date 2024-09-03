# Retentive Network: A Successor to Transformer for Large Language Models

# [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/pdf/2307.08621.pdf)

Microsoft Research, Tsinghua University

Авторы предлагают архитектуру модели Retentive Network, которая сочетает в себе мощь и параллелизуемость трансформеров с быстрым и дешевым инференсом рекуррентных моделей. На первой картинке изображен график, показывающий значительное превосходство модели над трансформером в эффективности инференса, а также график перплексии, показывающий лучшую масштабируемость Retentive Network в сравнении с трансформером.

![Untitled](../NLP/Retentive%20Network%20A%20Successor%20to%20Transformer/Untitled.png)

На второй картинке можно увидеть скромный график, показывающий всю крутизну модели: она объединяет в себе Training Parallelism, Low-Cost Inference и Strong Performance.

![Untitled](../NLP/Retentive%20Network%20A%20Successor%20to%20Transformer/Untitled%201.png)

## Retention mechanism

Главная часть статьи - механизм **Retention**, который заменит **self-attention**.
Суть подхода:

- Давайте сделаем RNN-ку, которую можно будет вычислять не только последовательно, но еще и параллельно (т.е. не считать каждый таймстемп новое скрытое состояние, а перемножить несколько матричек и получить тот же результат)
- А еще давайте сделаем это похожим на attention + найдем в нашей формуле отсылку к позиционным эмбеддингам (конкретно RoPE)

Математика подхода (можно пропустить):
Берем входные данные $X \in \mathbb{R}^{|x|*d_{model}}$. Начинаем думать о них как о последовательности $X_n$, домножаем эту последовательность на матрицу $W_V$ (аналог *Value* в Attention) и получаем последовательность векторов $v_n$. Хотим из этой последовательности получить другую, с контекстно-обусловленными векторами. Назовем ее $o_n$. Введем следующее рекурентное соотношение, позволяющее считать $o_n$, использующее при этом скрытую последовательность $s_n$:

$\hspace{1cm} s_n = As_{n-1} + K^T_nv_n,  \hspace{4cm}  A \in \mathbb{R}^{d\times d}, K_n \in \mathbb{R}^{1\times d}$$\hspace{1cm} o_n = Q_ns_n = \sum\limits_{m=1}^n Q_nA^{n-m}K^T_mv_m, \hspace{2.2cm} Q_n \in \mathbb{R}^{1\times d}$
Здесь $Q$ и $K$ - аналоги *Query* и *Key* в attention и считаются так:

$\hspace{1cm} Q = XW_Q, \hspace{1cm} K = XW_K$
Дальше авторы диагонализируют матрицу $A$, представляя ее в виде $A = \Lambda(\gamma e^{i\theta}) \Lambda^{-1}$, где $\gamma, \theta \in \mathbb{R}^d$. Подставляя это в уравнение на $o_n$, получим:

$\hspace{3.5cm} o_n = \sum\limits_{m=1}^n (Q_n(\gamma e^{i\theta})^n)(K_m (\gamma e^{i\theta})^{-m})^{\dag} v_m$.
Здесь замечаем, что $Q_n(\gamma e^{i\theta})^n$ и $K_m (\gamma e^{i\theta})^{-m}$ очень похожи на элементы из attention, к которым добавили RoPE (Rotary positional embeddings). Радуемся совпадению. Для конечной формулы осталось только упростить $\gamma$, приняв его скаляром, и получить:

$$
\hspace{0.5cm} o_n = \sum\limits_{m=1}^n \gamma^{n-m} (Q_ne^{ni\theta})(K_m e^{-im\theta})^{\dag} v_m
$$

## Способы вычисления Retention

### 1. Параллельное вычисление

Замечаем, что финальную формулу можно записать в виде произведения матричек. Матрички
составляем следующим образом:
$\hspace{1cm} Q = (XW_Q)\odot \Theta, \hspace{1cm} K = (XW_K)\odot \overline{\Theta}, \hspace{1cm} V = XW_V$

$\hspace{1cm} \Theta_n = e^{in\Theta}, \hspace{2cm} D_{nm} = \lambda^{n-m} \text{ при } n \ge m, 0 \text{ при } n < m$

$$
\operatorname{Retention}(X) = (QK^T \odot D)V
$$

К этой формуле прилагается картинка 3.

![Untitled](../NLP/Retentive%20Network%20A%20Successor%20to%20Transformer/Untitled%202.png)

И псевдокод:

![Untitled](../NLP/Retentive%20Network%20A%20Successor%20to%20Transformer/Untitled%203.png)

### 2. Последовательное вычисление

Этот способ вычисления был заложен в формулировке Retention. С учетом обозначений из пункта выше, можем переписать формулу:

$$
S_n = \gamma S_{n-1} + K^T_nV_n
$$

$$
 \operatorname{Retention}(X_n) = Q_nS_n 
$$

К этой формуле прилагается картинка 4.

![Untitled](../NLP/Retentive%20Network%20A%20Successor%20to%20Transformer/Untitled%204.png)

И псевдокод:

![Untitled](../NLP/Retentive%20Network%20A%20Successor%20to%20Transformer/Untitled%205.png)

### 3. Комбинированное вычисление

Предложенный механизм позволяет побить длинную входную последовательность на чанки, в кажом чанке посчитать локальный Retention с помощью параллельного способа, а затем, используя последовательный способ, передать информацию с предыдущих чанков следующим. Таким образом, не придется перемножать огромные матрицы и не придется рекуррентно считать состояния для такой длинной последовательности.
Формулы, описывающие этот способ:

$$
Q_{[i]} = Q_{Bi:B(i+1)}, \hspace{1cm} K_{[i]} = K_{Bi:B(i+1)}, \hspace{1cm} V_{[i]} = V_{Bi:B(i+1)}
$$

$$
R_i = K_{[i]}^TV_{[i]} + \gamma^B R_{i-1}
$$

$$
\operatorname{Retention}(X_{[i]}) = (Q_{[i]}K_{[i]}^T \odot D)V_{[i]} + (Q_{[i]}R_{i-1}) \odot \xi, \hspace{1cm} \xi_{ij} = \gamma^{i+1}
$$

В последнем равенстве первое слагаемое есть Retention внутри чанка, второе слагаемое - Cross-chunk чать.

Псевдокод:

![Untitled](../NLP/Retentive%20Network%20A%20Successor%20to%20Transformer/Untitled%206.png)

### Многоголовый Retention

Как и в *self-attention*, модуль *retention* можно и нужно делать многоголовым. Устанавливаем размерность головы $d$ и получаем $h = d_{model} / d$ голов. В каждой голове будут свои обучаемые матрицы $W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$. Также для каждой головы устанавливается свой параметр $\gamma$, это должно позволить разным головам учить разные свойства. Также авторы решили добавить еще $Swish$, $GroupNorm$ и пару линейных преобразований. В $GroupNorm$ выход из каждой головы нормируется отдельно. Итого, многоголовый *Retention* считается следующим образом:

$$
head_i = \operatorname{Retention}(X, \gamma_i)
$$

$$
Y = \operatorname {GroupNorm}_h(\operatorname{Concat}(head_1, \dots, head_h))
$$

$$
\operatorname{MSR}(X) = (\operatorname{swish}(XW_G) \odot Y)W_O
$$

Retentive Network

Сама сеть представляет из себя трансформер, в котором *self-attention* заменили на *retention*. То есть слой *Retention Network* есть последовательные *FeedForwardNetwork* и *RetentionNetwork*. И таких слоев можем настакать сколько угодно душе, как и в трансформерах.
Формула для наглядности:

$$
Y^l = \operatorname{MSR}(\operatorname{LN}(X^l)) + X^l
$$

$$
X^{l+1} = \operatorname{FFN}(\operatorname{LN}(Y^l)) + Y^l
$$

## Эксперименты, результаты и сравнение с другими моделями

Обучали на задачу Language Modeling, замеряли перплексию на валидационном сете и One/Few-Shot learning на нескольких других сетах.
Для экспериментов использовали 3 модели разного размера: 1.3B, 2.7B, 6.7B. Сравнивали с трансформерами такого же размера и с  "эффективными вариациями" трансформеров: **RWKW**, **Hyena**, **H3**, **Linear Transformer**.
Сравнение перплексии есть на самой первой картинке. 

![Untitled](../NLP/Retentive%20Network%20A%20Successor%20to%20Transformer/Untitled%207.png)

На картинке выше таблички сравнения **RetNet** и **Transformer** в one/few-shot learning и график сравнения производительности (потребление памяти, throughput) моделей при обучении. Все, естественно, в пользу модели **RetNet**.

![Untitled](../NLP/Retentive%20Network%20A%20Successor%20to%20Transformer/Untitled%208.png)

На картинке выше сравнение **RetNet** с эффективными вариациями трансформеров. Здесь **RetNet** опять лучше всех с заметным отрывом.

![Untitled](../NLP/Retentive%20Network%20A%20Successor%20to%20Transformer/Untitled%209.png)

На этой картинке табличка с ablation. Из интересного, использование $\gamma$ decay улучшает метрики, хотя по идее это должно плохо сказываться на способности модели учитывать длинный контекст.

## Выводы

Судя по метрикам и производительности, в качестве языковой модели эта архитектура действительно топ. Пока не очень понятно, насколько хорошо она будет учитывать длинный контекст, все-таки это RNN-ка и сигнал от векторов с далеких позиций будет затухать.

Использвать модель в качестве энкодера теоретически можно, убрав в матрице $D$ казуальность. Также можно добавить свертки и получить **ConvRetNet**. Но непонятно, будет ли это лучше обычных конформера/трансформера. Теоретически можно при обработке длинных последовательностей  использовать *Chunkwise Recuttent*-форму подсчета, но она дает заметное ускорение на последовательностях с длиной в несколько тысяч токенов, а у нас в ASR-е такого нет.
Кажется что использовать по полной крутые свойства архитектуры можно, используя ее как аудио-энкодер в streaming asr, т.е. в RNN-T модели. С нетерпением будем ждать соответствующие статьи.