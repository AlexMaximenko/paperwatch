# pyannote

[**pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe**](https://www.isca-speech.org/archive/pdfs/interspeech_2023/bredin23_interspeech.pdf)

https://github.com/pyannote/pyannote-audio

three main stages:

- speaker segmentation (short sliding window)
- neural speaker embedding (local)
- agglomerative clustering (global)

- end-to-end speaker segmentation model
    
    [End-to-end speaker segmentation for overlap-aware resegmentation](https://arxiv.org/pdf/2104.04045.pdf) (InterSpeech, 2021)
    
    - Model:
        - SincNet trainable features
            
            [Speaker recognition from raw waveform with sincnet](https://arxiv.org/pdf/1808.00158.pdf)
            
            ![Untitled](pyannote/Untitled.png)
            
        - 4 x LSTM
        - 2 x feed-forward
        - $K_{max}$ output speakers
        - 16ms time resolution
    - Training:
        - multi-label classification
            
            ![Untitled](pyannote/Untitled%201.png)
            
        - permutation-invariant loss:
            
            $$
            \mathcal{L}(y, \hat{y}) = \min_{\text{perm}(y)} \mathcal{L}(\text{perm}(y),\hat{y})
            $$
            
    - Inference:
        - Speaker Segmentation:
            
            $$
            y_t^k > \theta
            $$
            
            - more advanced:
                
                [Optimization of RNN-Based Speech Activity Detection](https://www-tlp.limsi.fr/public/talsp2018-gelly08100927.pdf)
                
                ![Untitled](pyannote/Untitled%202.png)
                
        - Voice Activity Detection:
            
            $$
            \hat{y}_t^{\text{VAD}} = \max_k \hat{y}_t^k
            $$
            
        - Overlapped Speech Detection:
            
            $$
            \hat{y}_t^{\text{OSD}} = {\max_k}_{\text{2nd}} \hat{y}_t^k
            $$
            
- neural speaker embedding

1. local neural speaker segmentation (5s window, 500ms step)
    
    sliding window ⇒ test-time augmentation
    
    ![Untitled](pyannote/Untitled%203.png)
    
2. binarization: $\hat{y}_t^k > \theta$
    
    ![Untitled](pyannote/Untitled%204.png)
    
3. local speaker embedding
    
    ![Untitled](pyannote/Untitled%205.png)
    
4. global clustering: agglomerative, $\delta$ threshold
    
    ![Untitled](pyannote/Untitled%206.png)
    
5. final aggregation
    1. $K_f$ : estimate number of speakers: sum binarized segmentation result (averaged over windows)
        
        ![Untitled](pyannote/Untitled%207.png)
        
    2. estimate score of each cluster: sum clustered speaker segmentation score over windows
        
        ![Untitled](pyannote/Untitled%208.png)
        
    3. select $K_f$  clusters with highest score
        
        ![Untitled](pyannote/Untitled%209.png)
        
    4. filling within-speaker gaps shorter than $\Delta$
        
        ![Untitled](pyannote/Untitled%2010.png)
        

### Results

Hyperparameters:

- $\theta$ (used for binarizing speaker segmentation) is the most important hyper-parameter to tune
- followed by $\Delta$ (for filling short intra-speaker gaps)
- and then only $\delta$ (that serves as stopping criterion for the clustering stage)

![Untitled](pyannote/Untitled%2011.png)

## Evaluation: Diarization Error Rate (DER)

![image (9).png](pyannote/image_(9).png)

![image (7).png](pyannote/image_(7).png)

![image (8).png](pyannote/image_(8).png)

![image (7).png](pyannote/image_(7)%201.png)

![image (8).png](pyannote/image_(8)%201.png)

![image (9).png](pyannote/image_(9)%201.png)

![image (8).png](pyannote/image_(8)%202.png)

![image (7).png](pyannote/image_(7)%202.png)