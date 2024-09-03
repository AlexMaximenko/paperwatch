# The Evolution of Multimodal Model Architectures (1)

![Screenshot 2024-06-20 at 17.23.09.png](The%20Evolution%20of%20Multimodal%20Model%20Architectures/Screenshot_2024-06-20_at_17.23.09.png)

Авторы предлагают немного упорядочить и структурировать весь зоопарк мультимодальных LLM-ок и разбить их на 4 категории по типу fusion-а модальностей, после чего выделить общие плюсы и минусы каждого из подходов. 

Категории, на которые авторы разбили модели:

- Type-A: стандартный cross-attention декодера LLM-ки на мультимодальные фичи
- Type-B: кастомные слои для fusion-а модальностей во внутренние слои декодера
- Type-C: использование и дообучение энкодера другой модальности для выравнивания его эмбеддингов с эмбеддингами LLM-ки
- Type-D: расширение словаря LLM-ки токенами из другой модальности

Плюсы и минусы рассматриваются в контексте необходимого для fusion-а компьюта, количества данных, сложности архитектур, масштабируемости, простоте добавления других модальностей и возможности any-to-any генерации.

![Screenshot 2024-06-20 at 18.26.18.png](The%20Evolution%20of%20Multimodal%20Model%20Architectures/Screenshot_2024-06-20_at_18.26.18.png)

# Обзоры классов и архитектур

## Type-A: Standard Cross-Attention based Deep Fusion

![Screenshot 2024-06-20 at 18.59.31.png](The%20Evolution%20of%20Multimodal%20Model%20Architectures/Screenshot_2024-06-20_at_18.59.31.png)

Первый тип - стандартный fusion через cross-attention на эмбеддинги из другой модальности. 

Как говорят авторы - вся мультимодальная тема началась с Flamingo ([статья](https://arxiv.org/abs/2204.14198), [paperwatch](https://www.youtube.com/watch?v=UvUjI9yC1Fo)). 

Общая схема работы:

- Берем предобученную LLM
- Берем предобученный modality Encoder
- Берем Resampler (perception / адаптер)
- Эмбеддинги из энкодера кормим в Resampler и получившиеся эмбеддинги подаем как Key и Values в Cross-attention LLM-ки

Модели, применяющие такой подход:

- **Flamingo**
    
    Разбор от Жоры
    
    [https://www.notion.so/Flamingo-0fbc1974dd6c4d2994ce03af3b70a32f?pvs=4](https://www.notion.so/0fbc1974dd6c4d2994ce03af3b70a32f?pvs=21)
    
    ![Screenshot 2024-06-20 at 19.35.28.png](The%20Evolution%20of%20Multimodal%20Model%20Architectures/Screenshot_2024-06-20_at_19.35.28.png)
    
    Брали предобученную LM и энкодер и дообучали на большом количестве данных:
    
    - 43 миллиона веб-страниц с image/text interleaving
    - 1.8B пар текст-картинка среднего качества (датасет ALIGN) + собрали еще 312 миллионов пар с картинками получше и caption-ами подлиннее
    - 27 миллионов видео (порядка 22 секунд) с описанием
- **Audio Flamingo**
    
    5.9M audio-text pairs; 18.1k hours
    
    ICL datasets: top-k closest training samples through kNN (LAION-CLAP embedding space); FAISS-gpu
    
    two training stages:
    
    - pre-training
    - supervised fine-tuning (SFT)
    
    маска в cross-attention позволяет обуславливаться на все предыдущие аудиозаписи
    
    - pre-training: train the audio representation transformation layers and the gated xattn-dense layers
    - SFT: unfreeze LM, учим все параметры, кроме audio feature extractor’а
    
    ![Untitled](The%20Evolution%20of%20Multimodal%20Model%20Architectures/Untitled.png)
    
- **OpenFlamingo**
    
    Опенсорсная реализация фламинго, обучена на открытых данных
    
- **Otter**
    
    Авторы собрали инструктивный image/text датасет и доучили OpenFlamingo на нем.
    
    Дообучали Perceiver и Cross-Attention. Недолго, всего на 4x3090.  Метрик нет, только demonstrations.
    
    Схема инструктивности внизу на схеме, <context> несет в себе текст и картинки, модели предлагается представить что она - GPT и ответить на инструкцию.
    
    ![Screenshot 2024-06-20 at 21.59.44.png](The%20Evolution%20of%20Multimodal%20Model%20Architectures/Screenshot_2024-06-20_at_21.59.44.png)
    
    ![Screenshot 2024-06-20 at 22.28.15.png](The%20Evolution%20of%20Multimodal%20Model%20Architectures/Screenshot_2024-06-20_at_22.28.15.png)
    
- **MultiModal-GPT**
    
    Еще один файнтюн OpenFlamingo на инструктивный сет, но с помощью LoRA в self-attention и cross-attention.
    
    Обучали как на Text-Image инструкции, так и на text-only.
    
    Для text-only использовали следующий шаблон и данные из датасетов Dolly 15K (Databricks, людской) и Alpaca GPT4 52K (синтетический). 
    
    ![Screenshot 2024-06-20 at 23.04.36.png](The%20Evolution%20of%20Multimodal%20Model%20Architectures/Screenshot_2024-06-20_at_23.04.36.png)
    
    Для Visual-Text инструкций брали следующий шаблон:
    
    ![Screenshot 2024-06-20 at 23.09.31.png](The%20Evolution%20of%20Multimodal%20Model%20Architectures/Screenshot_2024-06-20_at_23.09.31.png)
    
    ![Screenshot 2024-06-20 at 23.15.34.png](The%20Evolution%20of%20Multimodal%20Model%20Architectures/Screenshot_2024-06-20_at_23.15.34.png)
    
    Здесь, в отличии от Otter, использовали больше данных и больше компьюта (8 A100, 1 эпоха). В обучении использовали одинаковое количество visual-language сэмплов и text-only сэмплов.
    
    Во время обучения заметили, что часть Visual-Text датасетов не очень хорошего качества (ответы содержат 1-2 слова), что сильно ухудшало качество модели. По итогу убрали их.
    
- **PaLI-X (Google Research)**
- **IDEFICS**
    
    Еще одна open-source реализация flamingo.
    
- **Dolphins**
- **VL-Bart**
- **VL-T5**

## Type-B: Custom Layer based Deep Fusion (CLDF)

![Screenshot 2024-06-21 at 00.31.04.png](The%20Evolution%20of%20Multimodal%20Model%20Architectures/Screenshot_2024-06-21_at_00.31.04.png)

В целом подход похож на Type-A, здесь так же используется Deep Fusion, но с использованием кастомных слоев вместо стандартного cross-attention.

Примеры моделей:

- **LLaMA-Adapter**
    
    Не используют cross attention, вместо этого добавляют в слои декодера обучаемые “эмбеддинги-промпты”, которые могут содержать в себе информацию с другой модальности.
    
    ![Screenshot 2024-06-21 at 07.45.19.png](The%20Evolution%20of%20Multimodal%20Model%20Architectures/Screenshot_2024-06-21_at_07.45.19.png)
    
    ![Screenshot 2024-06-21 at 08.03.00.png](The%20Evolution%20of%20Multimodal%20Model%20Architectures/Screenshot_2024-06-21_at_08.03.00.png)
    
    ![Screenshot 2024-06-21 at 08.03.22.png](The%20Evolution%20of%20Multimodal%20Model%20Architectures/Screenshot_2024-06-21_at_08.03.22.png)
    
    Добавляют или через gating, чтоб модель не разлетелась в начале обучения.
    
    Интересно, что такой подход работает лучше чем LoRA и полный файнтюн:
    
    ![Screenshot 2024-06-21 at 08.04.58.png](The%20Evolution%20of%20Multimodal%20Model%20Architectures/Screenshot_2024-06-21_at_08.04.58.png)
    
    ![Screenshot 2024-06-21 at 08.07.48.png](The%20Evolution%20of%20Multimodal%20Model%20Architectures/Screenshot_2024-06-21_at_08.07.48.png)
    
    Интересно было в хоть каком-то виде получить сравнение с Flamingo, сделать это можно по COCO Caption. Здесь у авторов скор после файнтюна на этом датасете **122.2**, у фламинго - **138.1** после файнтюна и **113.8** в 32-shot режиме. Без неожиданностей.
    
- **LLaMA-Adapter-V2**
    
    Через месяц после выхода LLaMA-Adapter выходит LLaMA-Adapter-V2 от тех же авторов. В отличие от первой статьи, во второй авторы хотят сделать модель, которая будет уметь в мультимодальные инструкции (в первой части файнтюнили либо на инструкции, либо на image captioning). 
    
    Для этого предлагают:
    
    - Эмбеддинги от картинок подавать как адаптеры только на первый слой LLaMA
    - Обучаемые адаптеры вставить в остальные 30? слоев
    - Вместе с адаптерами обучать еще и bias / norm / scale в декодере, что все еще дает 0.04% обучаемых параметров
        
        ![Screenshot 2024-06-21 at 10.15.29.png](The%20Evolution%20of%20Multimodal%20Model%20Architectures/Screenshot_2024-06-21_at_10.15.29.png)
        
    
    Разделение параметров по обучаемым задачам влечет за собой “интерференцию” задачи и их последующее дополнение: модель оказывается способной в visual-instructions.
    
    ![Screenshot 2024-06-21 at 10.16.48.png](The%20Evolution%20of%20Multimodal%20Model%20Architectures/Screenshot_2024-06-21_at_10.16.48.png)
    
    Метрики приводят только для Image-Captioning-а, но почему-то они совпадают с LLaMA-Adapter.
    
    Также приводят что-то вроде SBS с ChatGPT и побеждают его😯
    
    ![Screenshot 2024-06-21 at 10.18.54.png](The%20Evolution%20of%20Multimodal%20Model%20Architectures/Screenshot_2024-06-21_at_10.18.54.png)
    
- **CogVLM**
    
    Еще один прикольный подход по deep fusion. Авторы берут картиночный энкодер (ViT), извлекают из него эмбеддинги, подают на вход LLM-ке вместе с текстом. Казалось бы тривиально, но есть 2 небольших дополнения:
    
    - В качестве позиции для RoPE всем этим эмбеддингам задается одно и то же значение (логично, это ведь одна и та же картинка, а именно “картиночные” позиционные эмбеддинги уже содержатся в эмбеддингах после ViT-а)
    - В MHA картиночные эмбеддинги используют свои матрицы QKV, а в FFN - свой FFN соответственно
    
    В таком подходе у нас никак не страдают способности языковой модели + на момент написания статьи вроде +- СОТА на text-image бенчмарках
    
    ![Screenshot 2024-06-21 at 10.34.49.png](The%20Evolution%20of%20Multimodal%20Model%20Architectures/Screenshot_2024-06-21_at_10.34.49.png)
    
    Претрейн: 1.5B image-text пар из опенсорса + собрали 40М visual-grounding dataset. 
    
    На первой стадии учатся чисто на Image Captioning (next-token-prediction.  
    
    На второй стадии добавляют задачу Referring Expression Comprehension (REC) - модели дают картину + слово-объект на нем и просят предсказать bounding box в виде [[x0, y0, x1, y1]].
    
    Инструктивный файнтюн: доучили 2 модели: одну на чаттинг, другую на bounding-боксы.
    
    В ablation показывают, что:
    
    - Их подход работает лучше, чем полный файнтюн
    - Картинкам выгоднее делать attention целиком, а не causal
    - Добавлять ssl-лосс на эмбеддинги картинок (чтоб каждый эмбединг предсказывал CLIP-вектор следующей позиции) нет смысла, метрики падают
- **mPLUG-Owl2**
    
    Моделька от Алибабы, вышедшая раньше CogVLM и уступающая ей по метрикам на всех сетах, где они обе были измерены.
    
- **CogAgent**
- **InternVL**
- **MM-Interleaved Tian**
- **CogCoM**
- **MoE-LLaVA**
- **LION**

## Type-C: Non-Tokenized Early Fusion (NTEF)

![Screenshot 2024-06-21 at 12.13.40.png](The%20Evolution%20of%20Multimodal%20Model%20Architectures/Screenshot_2024-06-21_at_12.13.40.png)

Наиболее часто встречающийся в статьях тип архитектуры:

- Берем modality-энкодер
- Берем projection-слой
- Берем LLM
- Выходы из modality-энкодера подаем в projection слой
- Выходы из projection-слоя подаем на уровень эмбеддингов LLM-ки

В отличии от типов A и B, здесь происходит Early fusion - мы учим modality-энкодер маппить эмбеддинги модальности прямиком во входное пространство модели. Если сильно махать руками, то тожно сказать что мы подстраиваем энкодер под LLM-ку, а не LLM-ку под новую модальность. Да, можно в LLM-ку навесить LoRA, но рукам кажется, что все равно это будет не “настоящее понимание”.

Модельки из этой категории:

- **LLaVA**
    
    ![Screenshot 2024-06-21 at 12.34.27.png](The%20Evolution%20of%20Multimodal%20Model%20Architectures/Screenshot_2024-06-21_at_12.34.27.png)
    
    Архитектура базовая - ViT в качестве энкодера, Vicuna - в качестве декодера.
    
    4 часа “претрейна” на 8 A100 и “10” часов файнтюна на 8 A100.
    
- **LLaVA-1.5**
    
    Долили к LLaVA новые данные, взяли CLIP покруче (выше разрешение) и LLM побольше (13B). Получили модель получше.🧑‍🔬
    
- **SLM**
    
    Разбирал Жора🔥
    
    ![Untitled](The%20Evolution%20of%20Multimodal%20Model%20Architectures/Untitled%201.png)
    
- **Pengi**
    
    Разбирал Георгий🔥
    
    ![Untitled](The%20Evolution%20of%20Multimodal%20Model%20Architectures/Untitled%202.png)
    
    ![Untitled](The%20Evolution%20of%20Multimodal%20Model%20Architectures/Untitled%203.png)
    
- **Qwen-Audio**
    
    Разбирал…  Георгий Господинов🔥
    
    ![Untitled](The%20Evolution%20of%20Multimodal%20Model%20Architectures/Untitled%204.png)
    
- **WavLLM**
    
    Разбирал Георгий Александрович Господинов🔥
    
    ![Untitled](The%20Evolution%20of%20Multimodal%20Model%20Architectures/Untitled%205.png)
    

Что по компьюту: модели из этого класса в среднем требуют меньше компьюта для обучения т.к. в большинстве из них обучается только адаптер.

## Type-D: Tokenized Early Fusion (TEF)

![Screenshot 2024-06-21 at 13.54.01.png](The%20Evolution%20of%20Multimodal%20Model%20Architectures/Screenshot_2024-06-21_at_13.54.01.png)

Последний из рассматриваемых подходов токенизация эмбеддингов из другой модальности и расширение ими словаря эмбеддингов.

Модели:

- **AudioPalm**
    
    Разбирал наш замечательный руководитель Георгий Александрович Господинов🔥
    
    Decoding audio tokens to raw waveform:
    
    1) audio tokens → SoundStream tokens:
    
    - autoregressive AudioLM-like
    - non-autoregressive SoundStorm
    
    2) SoundStream tokens → waveform:
    
    - convolutional decoder
        
        ![Screenshot 2024-06-21 at 14.20.43.png](The%20Evolution%20of%20Multimodal%20Model%20Architectures/Screenshot_2024-06-21_at_14.20.43.png)
        

Еще были некоторые Visual-Language модели, но я пока забил и не стал разбираться с токенизацией картинок. Модели: **LaVIT, TEAL, CM3Lean, SEED, Unicode, VL-GPT**