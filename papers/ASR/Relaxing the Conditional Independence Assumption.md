# Relaxing the Conditional Independence Assumption of CTC-based ASR by Conditioning on Intermediate Predictions

[https://arxiv.org/pdf/2104.02724.pdf](https://arxiv.org/pdf/2104.02724.pdf)

![Screenshot 2024-02-23 at 15.54.17.png](Relaxing%20the%20Conditional%20Independence%20Assumption/Screenshot_2024-02-23_at_15.54.17.png)

Авторы предлагают крутой и интересный  вариант по ослаблению conditional-independence условия моделек, обучаемых с помощью CTC-лосса.

Статья простая, короткая и с классными результатами.

## Проблема и решение

Одна из главных проблем CTC-Loss-а — **conditional independence** предположение на аутпуты моделей, т.е. предсказания токенов модели обусловлены только на акустическую информацию и никак не учитывают предсказанные до этого символы. 

Эту проблему решают LAS и RNN-T модели, но делают это ценой более дорогого инференса. Товарищи Jumon и Tatsuya в своей работе показывают подход, с помощью которого можно внести в модель своего рода информацию о токенах, которые будут предсказываться, тем самым обусловив модель на свои будущие аутпуты. Звучит неочевидно, но рисуночек и дальшейшие комментарии все прояснят.

 

![Screenshot 2024-02-24 at 14.45.07.png](Relaxing%20the%20Conditional%20Independence%20Assumption/Screenshot_2024-02-24_at_14.45.07.png)

Внимание на картинку. Как видно из картинки слева — архитектурой модели является трансформер.

Сверху модели видем привычный нам CTC-Loss. Непривичный же нам CTC-Loss мы видим на картинке по центру. Кажется, что авторы хотят на некоторые слои энкодера навесить CTC-голову и считать CTC-Loss. Но это только половина их задумки. Они хотят обусловить модель на свои промежуточные предсказания и тем самым ослабить **conditional independence** свойство аутпутов.

Как они это делают:

- На выбранных слоях после Encoder Layer считают предсказания модели следующим образом:
    
    $$
    Z_l = \operatorname{SoftMax}(\operatorname{Linear}_{D \rightarrow |\mathcal{V}|}(\operatorname{LayerNorm}(X_l^{out}) )
    $$
    
    Здесь $X_l^{out}$ — выход из $l$-го слоя энкодера, а $\operatorname{LayerNorm}$ и $\operatorname{Linear}_{D \rightarrow |\mathcal{V}|}$ они берут из CTC-головы над самой моделью.
    
    Промежуточный CTC-Loss считается над этой штукой:
    
    $$
    \mathcal{L}_{l}^{inter} = − \log P_{CTC}(\bf{y} | Z_l)
    $$
    
- Лосс посчитали, круто. Теперь нужно обусловить модель на это предсказание. Чтобы это сделать авторы вводят новый линейный слой, который преобразует $Z_l$ из пространства размерности $|\mathcal{V}|$ обратно в размерность модели $D$. Затем это чудо прибавляется к нормированному выходу из предыдущего слоя энкодера:
    
    $$
    X_{l+1}^{in} = \operatorname{LayerNorm}(X_l^{out}) + \operatorname{Linear}_{ |\mathcal{V}| \rightarrow D} (Z_l)
    $$
    
    Таким образом мы обуславливаем дальнейшие состояние нашей модели на ее предсказание с предыдущих слоев (которое, в какой-то мере, коррелирует с настоящим).
    

## Эксперименты и результаты

В экспериментах сравнивали 4 важные модели:

- Encoder-Decoder Transformer: 12 encoder layers + 6 decoder layers
- Encoder-Only CTC-Model
- Encoder-Only CTC-Model + Intermediate CTC-Losses (InterCTC)
- Encoder-Only CTC-Model + Intermediate CTC-Losses + conditioning on intermediate predictions

Все модели учили на ASR на 3 сетах: WSJ (81h English), TEDLIUM2 (207h English), AISHELL-1 (170h Mandarin).

![Screenshot 2024-02-25 at 15.46.10.png](Relaxing%20the%20Conditional%20Independence%20Assumption/Screenshot_2024-02-25_at_15.46.10.png)

![Screenshot 2024-02-25 at 15.46.23.png](Relaxing%20the%20Conditional%20Independence%20Assumption/Screenshot_2024-02-25_at_15.46.23.png)

![Screenshot 2024-02-25 at 15.46.37.png](Relaxing%20the%20Conditional%20Independence%20Assumption/Screenshot_2024-02-25_at_15.46.37.png)

Результаты крутые, на всех сетах получаем большой отрыв от базовой CTC-модельки.