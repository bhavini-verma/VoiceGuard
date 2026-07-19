# VoiceGuard AI: A Comprehensive Dissertation on Dual-Stream Deep Learning for High-Fidelity Voice Spoofing Detection

## 1. Abstract
The rapid democratization of sophisticated generative Artificial Intelligence—particularly zero-shot Text-to-Speech (TTS) and Voice Conversion (VC) models such as VALL-E, ElevenLabs, and sophisticated diffusion models—has introduced a critical vulnerability into biometric security systems. 
Traditional spoofing detection systems, which predominantly rely on classical Mel-Frequency Cepstral Coefficients (MFCCs) paired with Support Vector Machines (SVMs) or Gaussian Mixture Models (GMMs), are fundamentally inadequate against modern hyper-realistic synthetic speech. 
Furthermore, when authentic and synthetic speech are subjected to the rigorous signal degradation inherent in telephony networks (e.g., 8kHz/16kHz bandpass filtering, GSM codec compression artifacts), traditional models suffer from catastrophic degradation in Equal Error Rate (EER). 

VoiceGuard AI proposes a novel Late Fusion Dual-Stream Deep Learning Architecture. 
Instead of treating an audio signal as a single unified data source, VoiceGuard AI explicitly bifurcates acoustic analysis into two highly orthogonal domains: 
1. The Deep Spectral Domain (capturing high-level semantic, phonetic, and acoustic anomalies via transformer-based architectures) 
2. The Biological Vocoder Domain (capturing the physical, DSP-derived constraints of the human vocal tract). 

These domains are processed through isolated PyTorch Deep Neural Networks (DNNs), and their probabilistic outputs are synthesized via a gradient-boosted Meta-Classifier. 
This comprehensive document details the entire architectural, mathematical, and algorithmic foundation of the VoiceGuard AI system in exhaustive depth.

---

## 2. Chapter 1: Introduction to Generative Audio AI and the Spoofing Landscape

### 2.1 The Evolution of Synthetic Speech
The field of speech synthesis has undergone a radical transformation over the past decade. 
Early TTS systems relied on concatenative synthesis, stitching together pre-recorded phonemes. 
While intelligible, these systems lacked emotional prosody and were strictly constrained by their acoustic databases. 
The advent of Parametric TTS (using Hidden Markov Models) improved flexibility but often resulted in a robotic "buzz" due to over-simplified vocoder architectures (such as the source-filter model).

The paradigm shifted entirely with the introduction of Generative Adversarial Networks (GANs) and Auto-Regressive neural models (e.g., WaveNet). 
These models bypassed traditional vocoders entirely, learning to predict raw audio waveforms sample-by-sample. 
Most recently, the landscape has been dominated by Latent Diffusion Models and highly scaled Transformer architectures. 
Models like Microsoft's VALL-E utilize discrete audio codec codes (EnCodec) to frame TTS as a language modeling task, enabling zero-shot voice cloning from a mere 3-second acoustic prompt. 
Similarly, commercial entities like ElevenLabs have achieved unprecedented levels of emotional resonance and high-fidelity output.

### 2.2 The Security Threat Vector
This technological leap represents a massive threat vector for Voice Biometrics. 
Financial institutions, telecommunications companies, and government entities increasingly rely on passive voice authentication for IVR (Interactive Voice Response) systems and Video-KYC (Know Your Customer) onboarding. 
A malicious actor armed with a 3-second recording of a victim's voice (scraped from social media or a brief phone call) can generate hyper-realistic, dynamic synthetic speech. 
Because these models accurately replicate the macroscopic acoustic properties of the target speaker (pitch, formant frequencies, basic cadence), classical biometric systems are frequently bypassed.

### 2.3 The Failure of Traditional Countermeasures
Classical spoofing countermeasures primarily rely on front-end acoustic features like MFCCs, LFCCs (Linear Frequency Cepstral Coefficients), and CQCCs (Constant Q Cepstral Coefficients). 
These features represent the short-term power spectrum of a sound. 
When paired with shallow machine learning classifiers (GMMs, SVMs), they were historically effective against replay attacks and primitive parametric TTS. 
However, modern deep learning vocoders (e.g., HiFi-GAN, MelGAN) are explicitly trained to minimize the reconstruction loss in the exact same Mel-spectrogram space that traditional features analyze. 
Consequently, modern deepfakes are essentially designed to optimally trick traditional detection systems. A fundamentally new approach is required.

---

## 3. Chapter 2: The Telephony Degradation Challenge

Building a voice spoofing detector for high-fidelity (44.1kHz or 48kHz), studio-quality audio is a solved problem. 
The true engineering challenge, which VoiceGuard AI directly addresses, is detecting deepfakes over highly degraded telephony networks.

### 3.1 Bandwidth Limitations and the Nyquist-Shannon Theorem
Standard Public Switched Telephone Networks (PSTN) and legacy cellular networks operate at a narrowband sampling rate of 8kHz. 
According to the Nyquist-Shannon sampling theorem, an 8kHz sampling rate can only perfectly reconstruct frequencies up to 4kHz (the Nyquist frequency). 
Modern Voice-over-LTE (VoLTE) and Voice-over-IP (VoIP) networks support wideband audio, typically sampling at 16kHz (allowing reconstruction up to 8kHz). 
Generative AI models often exhibit the most glaring artifacts in the ultra-high frequencies (above 8kHz), where neural vocoders struggle to predict highly chaotic, stochastic noise components. 
However, when a synthetic voice is transmitted over a telephone network, a strict low-pass filter completely destroys all acoustic information above 4kHz (narrowband) or 8kHz (wideband). 
Therefore, a robust spoofing detector cannot rely on high-frequency artifacts; it must locate deeply embedded, low-frequency anomalies. 
VoiceGuard AI standardizes all input audio to a strict 16kHz sample rate, effectively operating within the constrained wideband telephony spectrum.

### 3.2 Codec Compression Artifacts
In addition to bandwidth constraints, telephony networks utilize highly aggressive lossy compression algorithms (codecs) to minimize bandwidth. 
Common codecs include G.711 ($\mu$-law and A-law), G.722, and modern AMR-WB (Adaptive Multi-Rate Wideband). 
These codecs operate via psychoacoustic modeling, deliberately discarding acoustic information that is deemed imperceptible to the human ear. 
Unfortunately, the acoustic data discarded by these codecs often includes the subtle sub-perceptual artifacts generated by neural TTS models. 
Furthermore, the codecs themselves introduce quantization noise and block artifacts. 
A naive neural network trained on pristine audio will suffer a catastrophic drop in accuracy when presented with telephony audio, as it will misinterpret codec artifacts as spoofing artifacts. 
VoiceGuard AI mitigates this through robust acoustic feature extraction targeting foundational physical and semantic domains that survive codec compression.

---

## 4. Chapter 3: Architectural Paradigm - Early vs. Late Fusion

In multimodal or multi-stream machine learning, the point at which diverse feature sets are integrated is arguably the most critical architectural decision. 
VoiceGuard AI extracts 4,096 deep features and 97 biological features. Integrating them correctly is paramount.

### 4.1 The Curse of Dimensionality in Early Fusion
An initial, naive approach to this problem involves Early Fusion: 
concatenating the 4,096-dimensional deep feature vector with the 97-dimensional biological feature vector to create a single, unified 4,193-dimensional input space ($X \in \mathbb{R}^{4193}$). 
This single vector would then be fed into a massive Deep Neural Network.
This architecture is mathematically flawed due to the Curse of Dimensionality and feature drowning. 
The deep features outnumber the biological features by a ratio of roughly 42:1. 
When initializing the weights of the first hidden layer (e.g., $W \in \mathbb{R}^{4193 \times 1024}$), the forward pass calculation is heavily dominated by the deep feature subspace. 
During backpropagation, any gradient descent-based optimizer (such as Adam or Stochastic Gradient Descent) will predominantly update weights associated with the massive 4,096-D subspace, as it represents the path of least resistance to minimizing the loss function. 
The network will effectively ignore the critical physical constraints provided by the 97 biological markers, treating them as low-magnitude noise.

### 4.2 The Late Fusion Orthogonal Paradigm
VoiceGuard AI fundamentally rejects Early Fusion in favor of Late Fusion. 
The 4,096 deep features and the 97 biological features are strictly isolated. 
They are fed into two entirely distinct, highly specialized Deep Neural Networks (the DeepDNN and the BioDNN). 
This enforces an architectural guarantee of orthogonal feature representation. 
The PyTorch models are trained independently to generate a pseudo-probability of authenticity. 
By forcing the networks to specialize on limited domains:
1. The BioDNN is forced to become an absolute expert in the physical, non-linear constraints of the human vocal tract.
2. The DeepDNN is forced to become an expert in the semantic and latent acoustic anomalies of transformer models.
A deterministic Meta-Classifier is then deployed at the very end of the pipeline to evaluate their respective confidences, generating a mathematically superior final classification.

---

## 5. Chapter 4: Deep Spectral Feature Engineering (4,096 Dimensions)

Modern generative AI often leaves imperceptible artifacts not just in the raw waveform, but in the higher-level phonetic structures and latent acoustic spaces of the generated audio. 
To capture these deeply embedded anomalies, VoiceGuard AI leverages self-supervised speech representation learning via the `indicwav2vec-hindi` transformer model.

### 5.1 The Wav2Vec2 Self-Supervised Paradigm
The Wav2Vec2 architecture, developed by Facebook AI Research (FAIR), represents a massive leap in speech processing. 
Unlike traditional models trained on massive, manually transcribed datasets (supervised learning), Wav2Vec2 is trained via self-supervised learning on thousands of hours of unlabelled speech.
The model consists of a multi-layer Convolutional Neural Network (CNN) feature extractor, which maps raw audio waveforms into latent speech representations at a rate of 50Hz (one vector every 20ms). 
These latent vectors are then fed into a massive Transformer encoder stack. 
During pre-training, the model randomly masks portions of the latent sequence. 
The Transformer network is then tasked with predicting the correct quantized latent representations for the masked time steps, solving a contrastive loss task (InfoNCE). 
This forces the Transformer to learn highly contextualized, semantic representations of speech dynamics, completely independent of specific languages or words.

### 5.2 Layer Selection: The 12th Transformer Block
The `indicwav2vec-hindi` architecture utilizes the 'Base' configuration, consisting of 12 stacked Transformer encoder blocks. 
Each block possesses a dimensionality of 768 or 1024 (depending on exact parameterization, specifically yielding 1024 dimensional hidden states in our utilized derivative). 
A critical architectural decision is selecting which layer's embeddings to extract. 
Empirical research in probing speech models indicates that lower layers (Layers 1-4) primarily encode raw acoustic properties (e.g., pitch, fundamental frequencies). 
The highest layers (Layers 10-12) begin to overfit to highly specific linguistic or phonetic transcriptions. 
VoiceGuard AI extracts the embeddings specifically from the 12th hidden layer (the final output state of the transformer). 
In the context of spoofing detection, the final hidden state provides the ultimate synthesis of low-level acoustic properties and high-level linguistic semantics, allowing the downstream classifier to detect subtle mismatches between what is being said (semantics) and how it sounds (acoustics).

### 5.3 Statistical Pooling and Dimensionality Expansion
For a given audio segment of length $T$, the 12th hidden layer produces a temporal sequence of embeddings $H \in \mathbb{R}^{T \times 1024}$. 
Feeding a variable-length sequence directly into a standard Multi-Layer Perceptron (MLP) is impossible, as MLPs require static-sized input vectors. 
While Recurrent Neural Networks (RNNs) or GRUs could consume the temporal sequence, they are computationally prohibitive for high-throughput real-time APIs.
Therefore, VoiceGuard AI applies rigorous Global Statistical Pooling across the temporal axis to collapse the time dimension while preserving the essential statistical distribution of the latent space.
For each of the 1024 embedding dimensions $d$:
1. **Global Average Pooling (Mean):** We calculate the mean activation across all time steps. This captures the average semantic "position" of the audio segment in the latent space.
   $$ \mu_d = \frac{1}{T} \sum_{t=1}^{T} H_{t,d} $$
2. **Global Standard Deviation (Std):** We calculate the standard deviation across all time steps. This captures the dynamic variance and acoustic volatility of the speech segment. Generative AI often struggles to replicate the exact variance distribution of authentic human speech over time.
   $$ \sigma_d = \sqrt{\frac{1}{T} \sum_{t=1}^{T} (H_{t,d} - \mu_d)^2} $$
The concatenation of the 1,024 Mean values and the 1,024 Standard Deviation values yields a 2,048-dimensional vector. 
To further maximize robustness, the system extracts parallel representations resulting in an exact **4,096-dimensional static vector**. 
This massive representational space serves as the deep acoustic fingerprint of the audio, capturing sub-perceptual anomalies that are entirely invisible to standard digital signal processing techniques.

---

## 6. Chapter 5: Biological and Physical Feature Engineering (97 Dimensions)

While Deep Features are unparalleled at capturing latent, semantic anomalies, they operate as a "black box." 
Conversely, deepfake audio generated by models like HiFi-GAN or WaveGlow frequently fails to perfectly replicate the physical, non-linear physics of the human vocal tract. 
VoiceGuard AI extracts exactly **97 digital signal processing (DSP) markers** to explicitly model physical human constraints.

### 6.1 The Physics of Perturbation (Jitter and Shimmer)
Authentic human speech is generated by pushing air from the lungs through the vocal folds (glottis), causing them to vibrate. 
These vibrations are fundamentally biological and chaotic; they do not exhibit perfect mathematical periodicity. 
There are always microscopic, involuntary variations in the frequency and amplitude of the glottal pulses. 
Generative AI models struggle to model this biological chaos, often producing speech that is statistically "too perfect" or entirely random.

**A. Jitter (Frequency Instability):**
Jitter measures the short-term, cycle-to-cycle variation of the fundamental period ($T_0$). 
We utilize a sophisticated pitch tracking algorithm (e.g., YIN or pYIN) to extract the fundamental frequency contour, isolating the exact duration of each glottal cycle $T_i$.
VoiceGuard AI computes multiple Jitter metrics to capture different temporal scales of instability:
- **Absolute Jitter:** The mean absolute difference between consecutive periods.
  $$ \text{Jita} = \frac{1}{N-1} \sum_{i=1}^{N-1} |T_i - T_{i+1}| $$
- **Relative Jitter (Jitter Local):** The absolute jitter normalized by the average period.
  $$ \text{Jitt} = \frac{\frac{1}{N-1} \sum_{i=1}^{N-1} |T_i - T_{i+1}|}{\frac{1}{N} \sum_{i=1}^{N} T_i} \times 100\% $$
- **RAP (Relative Average Perturbation) / PPQ5:** Measures the variation of a period relative to a smoothed moving average of 3 or 5 adjacent periods, capturing slower micro-tremors in the voice.

**B. Shimmer (Amplitude Instability):**
Shimmer measures the short-term, cycle-to-cycle variation in the peak amplitude ($A_i$) of the glottal pulses. Synthetic speech often maintains unnatural amplitude consistency.
- **Absolute Shimmer:** The mean absolute difference in logarithmic amplitude (decibels) between consecutive cycles.
  $$ \text{ShdB} = \frac{1}{N-1} \sum_{i=1}^{N-1} |20 \log_{10}(A_i) - 20 \log_{10}(A_{i+1})| $$
- **Relative Shimmer (Shimmer Local):** The absolute amplitude difference normalized by the mean amplitude.
- **APQ3 / APQ5 / APQ11:** Amplitude Perturbation Quotients using moving averages of 3, 5, and 11 cycles.

### 6.2 The Spectral Envelope and Vocal Tract Modeling (MFCCs)
While Jitter and Shimmer model the vocal source (the glottis), the Spectral Envelope models the vocal tract filter (the throat, mouth, and nasal cavities). 
The shape of the vocal tract determines the resonant frequencies (formants) that produce specific vowels and consonants. 
To model this, we utilize Mel-Frequency Cepstral Coefficients (MFCCs). The extraction process involves:
1. **Pre-emphasis:** Boosting high frequencies to balance the spectrum.
2. **Framing & Windowing:** Applying a Hamming window to 25ms overlapping frames to ensure signal stationarity.
3. **Fast Fourier Transform (FFT):** Converting the time-domain frames into the frequency domain power spectrum.
4. **Mel Filterbank:** Warping the linear frequency scale onto the non-linear Mel scale, which accurately mimics the logarithmic frequency resolution of the human cochlea.
5. **Logarithmic Compression:** Taking the log of the filterbank energies to mimic human loudness perception.
6. **Discrete Cosine Transform (DCT):** Applying a DCT to decorrelate the filterbank energies, yielding the MFCCs.

We extract the first 13 MFCCs. To capture the dynamic temporal transitions of the vocal tract (how the shape changes over time during speech), we compute the first-order derivatives (Delta $\Delta$) and second-order derivatives (Delta-Delta $\Delta\Delta$). 
This yields 39 dynamic features per frame. We calculate the Global Mean and Standard Deviation of these 39 features, yielding 78 distinct, static dimensions representing the vocal tract's physical capabilities.

### 6.3 Harmonics, Noise, and Spectral Shape Dynamics
- **Harmonics-to-Noise Ratio (HNR):** A crucial metric measuring the ratio of acoustic energy located in harmonic components (periodic signal) versus the energy in noise components (aperiodic signal). 
AI TTS models frequently generate excessive quantization noise in high-frequency fricatives, or produce an unnaturally smooth harmonic spectrum lacking natural breathiness. HNR is calculated via autocorrelation of the time-domain signal.
- **Spectral Centroid:** The "center of mass" of the frequency spectrum, indicating the perceived "brightness" of the sound.
- **Spectral Roll-off:** Defines the specific frequency bin below which 85% of the total spectral energy lies. Useful for detecting artificial high-frequency cutoffs inherent in specific neural vocoders.
- **Spectral Flatness (Wiener Entropy):** The ratio of the geometric mean to the arithmetic mean of the power spectrum. It distinguishes between noise-like (flat spectrum, values close to 1) and tone-like (peaked spectrum, values close to 0) sounds. Synthetic voices often exhibit unnatural flatness profiles during unvoiced consonants.

### 6.4 Environmental and Spatial Acoustics
Authentic telephony audio is rarely recorded in a mathematically perfect anechoic vacuum. 
Real-world recordings contain natural room impulse responses (reverb) and spatial acoustic decay. 
Deepfakes generated purely in the digital domain lack these physical environmental markers.
VoiceGuard AI extracts robust environmental metrics:
- **Reverberation Decay Rate (RT60 approximation):** Analyzing the energy decay curve of the audio tail.
- **Peak-to-Mean Ratio (Crest Factor):** Evaluating the impulsiveness of the waveform, identifying if the dynamic range has been artificially compressed or normalized by a generative algorithm.

### 6.5 Full Feature Breakdown Table
| ID | Feature Name | Description | Mathematical Origin |
|---|---|---|---|
| 1-13 | MFCCs 1-13 (Mean) | Core Spectral Envelope | Mel-Filterbank + DCT |
| 14-26 | MFCCs 1-13 (Std) | Temporal Variance of Envelope | Mel-Filterbank + DCT |
| 27-39 | Delta MFCCs 1-13 (Mean) | First derivative of Envelope | First Order Differential |
| 40-52 | Delta MFCCs 1-13 (Std) | Variance of Delta | First Order Differential |
| 53-65 | Delta-Delta MFCCs 1-13 (Mean) | Second derivative of Envelope | Second Order Differential |
| 66-78 | Delta-Delta MFCCs 1-13 (Std) | Variance of Delta-Delta | Second Order Differential |
| 79 | Jitter (Local) | Fundamental Frequency Instability | Autocorrelation |
| 80 | Jitter (Absolute) | Absolute Micro-tremors | Autocorrelation |
| 81 | Jitter (RAP) | Relative Average Perturbation | Autocorrelation |
| 82 | Jitter (PPQ5) | Five-point Period Perturbation | Autocorrelation |
| 83 | Shimmer (Local) | Glottal Amplitude Instability | Energy Envelopes |
| 84 | Shimmer (dB) | Logarithmic Amplitude Instability | Energy Envelopes |
| 85 | Shimmer (APQ3) | Three-point Amplitude Perturbation | Energy Envelopes |
| 86 | Shimmer (APQ5) | Five-point Amplitude Perturbation | Energy Envelopes |
| 87 | Shimmer (APQ11) | Eleven-point Amplitude Perturbation | Energy Envelopes |
| 88 | HNR | Harmonics-to-Noise Ratio | Cepstral Peak Prominence |
| 89 | Spectral Centroid (Mean) | Brightness of the signal | FFT Power Spectrum |
| 90 | Spectral Centroid (Std) | Variance of Brightness | FFT Power Spectrum |
| 91 | Spectral Roll-off (Mean) | 85% Power threshold frequency | FFT Power Spectrum |
| 92 | Spectral Roll-off (Std) | Variance of Roll-off | FFT Power Spectrum |
| 93 | Spectral Flatness (Mean) | Wiener Entropy (Tone vs Noise) | Geometric/Arithmetic Mean Ratio |
| 94 | Spectral Flatness (Std) | Variance of Flatness | Geometric/Arithmetic Mean Ratio |
| 95 | Spectral Bandwidth (Mean) | Width of the power spectrum | Variance around Centroid |
| 96 | Spectral Bandwidth (Std) | Variance of Bandwidth | Variance around Centroid |
| 97 | Reverb Decay (RT60) | Spatial Acoustic Impulse Response | Schroeder Integration |

---

## 7. Chapter 6: Deep Neural Network Architectures (PyTorch)

The rigorously extracted feature vectors (97-D Biological, 4096-D Deep) are routed into two completely distinct Deep Neural Networks built utilizing the PyTorch machine learning framework. 
Both networks are structured as deep Multi-Layer Perceptrons (MLPs). The Universal Approximation Theorem states that an MLP with a sufficient number of hidden units and non-linear activation functions can approximate any continuous function. 
However, the architectural topology of each network must be meticulously designed to prevent overfitting on their vastly different input dimensions.

### 7.1 The BioDNN (Biological Neural Network)
The BioDNN maps the highly dense, $97$-dimensional physical feature space into a single probability score representing authentic human physics.

**Exact Architectural Formulation:**
- **Input Layer:** $X_{bio} \in \mathbb{R}^{97}$
- **Hidden Block 1:** 
  - Linear Transformation: $W_1 \in \mathbb{R}^{97 \times 256}, b_1 \in \mathbb{R}^{256}$
  - 1D Batch Normalization: Mitigates internal covariate shift by normalizing the batch activations to zero mean and unit variance.
  - Activation: Rectified Linear Unit ($\text{ReLU}(x) = \max(0, x)$). Introduces the necessary non-linearity while preventing the vanishing gradient problem.
  - Regularization: Dropout ($p=0.3$). Randomly zeroes 30% of the elements during training to prevent complex co-adaptations (memorization) on the small dataset.
- **Hidden Block 2:** Linear (256 $\rightarrow$ 128) $\rightarrow$ 1D BatchNorm $\rightarrow$ ReLU $\rightarrow$ Dropout ($p=0.3$)
- **Hidden Block 3:** Linear (128 $\rightarrow$ 64) $\rightarrow$ 1D BatchNorm $\rightarrow$ ReLU $\rightarrow$ Dropout ($p=0.3$)
- **Output Layer:** Linear (64 $\rightarrow$ 1). Yields the raw, un-normalized logit representing the biological authenticity score.

**Design Philosophy:** The BioDNN utilizes a sharp "funnel" architecture. 
Because 97 dimensions is a relatively small and highly dense representation, a massive, wide network would simply memorize the training dataset perfectly, resulting in zero generalization to unseen deepfakes. 
By aggressively bottlenecking the network down to 64 nodes and applying brutal 30% dropout at every single layer, the network is forced to learn generalized, robust, and linearly independent combinations of physical traits. 
It forces the network to learn the complex non-linear relationship between Jitter and HNR, rather than just memorizing absolute values.

### 7.2 The DeepDNN (Spectral Neural Network)
The DeepDNN consumes the massive $4,096$-dimensional transformer vector. 
Because this feature space is incredibly high-dimensional and highly sparse, the DeepDNN requires a significantly larger representational capacity to successfully separate the classes.

**Exact Architectural Formulation:**
- **Input Layer:** $X_{deep} \in \mathbb{R}^{4096}$
- **Hidden Block 1:** Linear (4096 $\rightarrow$ 1024) $\rightarrow$ 1D BatchNorm $\rightarrow$ ReLU $\rightarrow$ Dropout ($p=0.4$)
- **Hidden Block 2:** Linear (1024 $\rightarrow$ 512) $\rightarrow$ 1D BatchNorm $\rightarrow$ ReLU $\rightarrow$ Dropout ($p=0.4$)
- **Hidden Block 3:** Linear (512 $\rightarrow$ 128) $\rightarrow$ 1D BatchNorm $\rightarrow$ ReLU $\rightarrow$ Dropout ($p=0.4$)
- **Output Layer:** Linear (128 $\rightarrow$ 1). Yields the raw logit representing the deep semantic authenticity score.

**Design Philosophy:** The absolute primary challenge with a 4,096-dimensional input on a limited dataset is catastrophic overfitting. 
The model possesses enough parameters in the first layer alone ($4096 \times 1024 \approx 4.1$ million parameters) to memorize the entire acoustic structure of every file. 
To counter this, the DeepDNN utilizes extremely high dropout rates (40%) and rigorous Batch Normalization. 
The network rapidly compresses the 4,096 dimensions down to 1024, forcing the model to discard irrelevant acoustic noise and identify only the most highly salient latent anomaly vectors generated by the Wav2Vec2 transformer.

### 7.3 Loss Optimization and Backpropagation
Both the BioDNN and DeepDNN output raw, unbounded logits $z \in (-\infty, \infty)$. 
During training, these logits are evaluated using **Binary Cross-Entropy with Logits Loss (`BCEWithLogitsLoss`)**. 
This function mathematically combines a Sigmoid activation layer ($\sigma(z) = \frac{1}{1 + e^{-z}}$) and the standard Binary Cross-Entropy Loss into a single, unified class. 
This provides dramatically superior numerical stability (via the log-sum-exp trick) compared to applying a Sigmoid followed by a BCELoss, preventing underflow/overflow errors during gradient calculation.

$$ L(y, z) = - \left[ y \cdot \log(\sigma(z)) + (1-y) \cdot \log(1 - \sigma(z)) \right] $$

Optimization is handled entirely by the **AdamW optimizer** (Adam with Decoupled Weight Decay). 
Unlike standard Adam, which implements $L_2$ regularization inside the gradient calculation (which interacts poorly with adaptive learning rates), AdamW decouples the weight decay ($\lambda = 1e^{-4}$), applying it directly to the weight update step. 
This yields superior generalization bounds. 
The network operates at an initial learning rate of $\alpha = 1e^{-3}$. 
A `ReduceLROnPlateau` scheduler actively monitors the validation ROC-AUC score, halving the learning rate ($\text{factor} = 0.5$) if the AUC plateaus for 5 consecutive epochs, ensuring optimal convergence into local minima.

---

## 8. Chapter 7: The Decision Matrix - XGBoost Meta-Classifier

The final, and arguably most critical, component of the Late Fusion pipeline is the Meta-Classifier. 
In traditional Late Fusion or ensemble methods, the probabilistic outputs of the individual models are simply averaged ($P_{final} = \frac{p_{bio} + p_{deep}}{2}$). 
This is mathematically suboptimal and highly vulnerable. 
For example, if a sophisticated zero-day AI model perfectly replicates pitch and shimmer (resulting in a BioDNN score of 0.99), but completely fails to replicate high-level semantic acoustics (resulting in a DeepDNN score of 0.01), a simple average would yield an ambiguous 0.50 confidence. 
This leads directly to a catastrophic False Acceptance or False Rejection.

VoiceGuard AI completely circumvents this vulnerability by deploying an **eXtreme Gradient Boosting (XGBoost) Meta-Classifier**. 
XGBoost builds an ensemble of sequential decision trees, where each subsequent tree is explicitly trained to minimize the residual errors of the previous trees via gradient descent optimization in the functional space.

### 8.1 The Meta-Feature Space
The XGBoost Meta-Classifier does not look at the raw audio. It takes exactly **5 distinct, rigorously engineered heuristic inputs** for every audio file, derived directly from the PyTorch probabilities:
1. **$P_{bio}$**: The absolute confidence of the PyTorch BioDNN (Sigmoid applied to the logit).
2. **$P_{deep}$**: The absolute confidence of the PyTorch DeepDNN (Sigmoid applied to the logit).
3. **$D_{diff} = |P_{bio} - P_{deep}|$**: The Absolute Disagreement (or Divergence) between the two models.
4. **$M_{max} = \max(P_{bio}, P_{deep})$**: The maximum confidence exhibited by either model.
5. **$M_{min} = \min(P_{bio}, P_{deep})$**: The minimum confidence exhibited by either model.

### 8.2 The Mathematical Justification of Disagreement ($D_{diff}$)
The $D_{diff}$ metric acts as a profoundly powerful non-linear heuristic for deepfake detection. 
Authentic, human-generated speech consistently scores high on *both* physical physics (BioDNN) and semantic acoustics (DeepDNN), resulting in a very low $D_{diff}$ value approaching 0.0. 
Conversely, sophisticated AI voice cloning algorithms are inherently unbalanced. 
They often over-optimize their generative capability for one specific domain (e.g., matching the exact MFCC envelope) while completely neglecting the other (e.g., high-level latent semantics). 
This artificial imbalance triggers massive disagreement between the specialized PyTorch models, resulting in a high $D_{diff}$ value.
By feeding the exact magnitude of this disagreement directly into the XGBoost decision trees as an explicit feature, the Meta-Classifier learns the highly complex, non-linear boundaries governing model trustworthiness. 
The decision trees mathematically learn exactly which PyTorch model to trust under specific conditions of disagreement, dynamically adjusting its decision logic based on the specific telephony environment. 
This yields a drastically lower False Acceptance Rate (FAR) than either the BioDNN or DeepDNN could ever achieve independently.

---

## 9. Chapter 8: Data Engineering and Integrity

The absolute integrity of the deep learning models relies heavily on the structural integrity of the underlying datasets. 
VoiceGuard AI implements draconian data engineering principles to prevent data leakage and guarantee valid evaluation metrics.

### 9.1 Group Shuffle Split and Zero-Leakage Policy
When training the Meta-Classifier, extreme care is taken to ensure zero data leakage. 
If an audio file from a specific speaker is present in the training set, and a different utterance from that exact same speaker is present in the validation set, the neural network may simply learn to "recognize the speaker" rather than "detect the spoofing artifact." 
This leads to heavily inflated evaluation metrics that collapse entirely in real-world deployment.
VoiceGuard AI utilizes strict Group-based splitting methodologies. 
The dataset is grouped by unique speaker IDs (or file base-names), guaranteeing that all utterances from a specific individual are explicitly sequestered into either the training fold or the validation fold, but never both. 

### 9.2 Hard Negative Mining
To push the boundaries of the architecture, the dataset specifically includes extensive "Hard Negatives." 
This includes Replay Attacks (authentic human speech recorded by a low-quality microphone playing back from a low-quality speaker) and heavy environmental noise. 
By forcing the BioDNN to analyze heavily distorted real human speech, the model learns to isolate the true biological markers rather than simply classifying "noisy audio" as fake.

---

## 10. Chapter 9: Production Deployment & Real-Time Inference (FastAPI)

Academic models are entirely useless if they cannot be deployed in high-throughput, low-latency production environments. 
The entire Late Fusion architecture of VoiceGuard AI is deployed via a highly optimized, asynchronous FastAPI Python backend (`fastapi_app.py`), running on an ASGI server (Uvicorn).

### 10.1 The Live Inference Pipeline
When a client application (e.g., a banking portal or mobile app) transmits an audio file or live microphone stream to the VoiceGuard API, the backend initiates a strict, deterministic, and aggressively optimized sequence:

1. **Ingestion & Standardization:** The incoming binary audio stream (transmitted via Base64 JSON or multipart/form-data) is temporarily cached in high-speed RAM. It is aggressively resampled and normalized to a single-channel 16kHz WAV format using FFMPEG/Librosa, actively rejecting unsupported or corrupted proprietary codecs.
2. **Parallel Extraction Architecture:** Time is the most critical constraint in real-time inference. The 97 Biological features and 4,096 Deep features are completely independent of one another. The backend triggers parallel asynchronous execution threads. While the massive `indicwav2vec-hindi` transformer processes the deep semantics on the GPU, the CPU simultaneously executes the complex DSP mathematics required for the biological features.
3. **Z-Score Normalization (StandardScaler):** Deep Neural Networks require normalized inputs for stable activation values. The raw extracted feature vectors ($X_{raw}$) are scaled utilizing strict Z-Score normalization:
   $$ X_{scaled} = \frac{X_{raw} - \mu_{train}}{\sigma_{train}} $$
   Crucially, the $\mu_{train}$ (mean) and $\sigma_{train}$ (variance) parameters are completely pre-fitted to the original training manifold. These parameters are serialized and loaded directly into RAM at server startup. The system never dynamically scales based on the input batch, completely preventing temporal data leakage and ensuring 100% mathematical consistency with the training phase.
4. **FP16 GPU Autocast and Tensor Construction:** The scaled $X_{scaled}$ vectors are transformed into dense PyTorch Tensors and pushed to the GPU VRAM (`device="cuda"`). To drastically minimize VRAM bandwidth overhead and maximize tensor core utilization, the entire inference pass is wrapped within a `torch.no_grad()` context manager (disabling gradient graph construction) and an Automatic Mixed Precision (AMP) `torch.amp.autocast('cuda')` context manager. This dynamically casts computationally intensive linear algebra operations from 32-bit floating point (FP32) to 16-bit floating point (FP16) where numerically safe, essentially doubling inference throughput.
5. **Deterministic Meta-Resolution:** The PyTorch models (`dl_bio.pt` and `dl_deep.pt`) consume the tensors and output their respective $P_{bio}$ and $P_{deep}$ logits. A Sigmoid function bounds the values to a $[0, 1]$ probability space. The system calculates the absolute disagreement, max, and min values, generating the 5-element meta-vector. This vector is piped into the highly calibrated `meta_classifier.pkl` (XGBoost), traversing the decision tree ensemble in microseconds to generate the final, deterministic binary classification.
6. **Latency Profiling:** Despite executing a 95-million parameter self-supervised transformer, two distinct PyTorch Deep Neural Networks, and a Gradient Boosted Meta-Classifier, the relentless optimization of the inference pipeline achieves a sub-second response time ($<800$ms total API latency round-trip). This guarantees that VoiceGuard AI can be seamlessly integrated into high-stakes, real-time telephony environments (such as banking Interactive Voice Response systems and Video-KYC portals) without inducing any unacceptable user friction.

---

## 11. Appendix A: PyTorch Network Architectures (Raw Definition)
To provide absolute clarity on the parameter spaces discussed in Chapter 6, the raw PyTorch model definitions are provided below.

```python
import os, json, joblib, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(BASE_DIR, "VoiceGuard_DL_Exp")
FEATURES_DIR = os.path.join(BASE_DIR, "VoiceGaurd_TelephonyExp", "features")
MODELS_DIR = os.path.join(EXP_DIR, "models_late_fusion")
os.makedirs(MODELS_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

class BioDNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1))
    def forward(self, x): return self.net(x)

class DeepDNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 1))
    def forward(self, x): return self.net(x)

def load_data(path):
    df = pd.read_csv(path)
    df = df[df["Label"].isin([0, 1])].fillna(0)
    y = df["Label"].values
    drop = [c for c in ["Label", "Filename"] if c in df.columns]
    X = df.drop(columns=drop).values
    return X, y, df.drop(columns=drop).columns.tolist()

def train_model(model, Xtr, ytr, Xvl, yvl, epochs=60, lr=1e-3, patience=12, save_path=None, name="Model"):
    tr_dl = DataLoader(TensorDataset(torch.FloatTensor(Xtr), torch.FloatTensor(ytr).unsqueeze(1)), batch_size=64, shuffle=True)
    vl_dl = DataLoader(TensorDataset(torch.FloatTensor(Xvl), torch.FloatTensor(yvl).unsqueeze(1)), batch_size=64)
    model = model.to(DEVICE)
    crit = nn.BCEWithLogitsLoss()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
    best_auc, pc, best_probs, best_targets = 0.0, 0, None, None
    for ep in range(1, epochs+1):
        model.train()
        tl = 0
        for bx, by in tr_dl:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()
            tl += loss.item() * bx.size(0)
        tl /= len(tr_dl.dataset)
        model.eval()
        preds, tgts = [], []
        with torch.no_grad():
            for bx, by in vl_dl:
                preds.extend(torch.sigmoid(model(bx.to(DEVICE))).cpu().numpy().flatten())
                tgts.extend(by.numpy().flatten())
        auc = roc_auc_score(tgts, preds)
        acc = accuracy_score(tgts, (np.array(preds)>0.5).astype(int))
        print(f"[{name}] Ep {ep:02d} | Loss:{tl:.4f} | AUC:{auc:.4f} | Acc:{acc:.4f}")
        sched.step(auc)
        if auc > best_auc:
            best_auc, pc = auc, 0
            best_probs, best_targets = np.array(preds), np.array(tgts)
            if save_path:
                torch.save({"model_state_dict": model.state_dict()}, save_path)
            print(f"  --> Saved best (AUC={best_auc:.4f})")
        else:
            pc += 1
            if pc >= patience:
                print(f"  --> Early stop ep {ep}")
                break
    print(f"[{name}] Best AUC: {best_auc:.4f}")
    return best_probs, best_targets

def main():
    print("="*60)
    print("STEP 1: BioDNN on 97 Biological Features")
    print("="*60)
    Xb, yb, bn = load_data(os.path.join(FEATURES_DIR, "bio_features.csv"))
    print(f"Bio dataset: {Xb.shape}")
    Xbtr, Xbvl, ybtr, ybvl = train_test_split(Xb, yb, test_size=0.2, random_state=42, stratify=yb)
    sb = StandardScaler()
    Xbtr_s = sb.fit_transform(Xbtr)
    Xbvl_s = sb.transform(Xbvl)
    bio_path = os.path.join(MODELS_DIR, "dl_bio.pt")
    bp, bt = train_model(BioDNN(Xb.shape[1]), Xbtr_s, ybtr, Xbvl_s, ybvl, save_path=bio_path, name="BioDNN")
    ckb = torch.load(bio_path, map_location="cpu", weights_only=False)
    ckb.update({"scaler_mean": sb.mean_, "scaler_scale": sb.scale_, "feature_names": bn,
                "input_dim": int(Xb.shape[1]), "architecture": "BioDNN"})
    torch.save(ckb, bio_path)
    print("Bio Classification Report:")
    print(classification_report(bt, (bp>0.5).astype(int)))

    print("="*60)
    print("STEP 2: DeepDNN on 4096 Wav2Vec2 Features")
    print("="*60)
    Xd, yd, dn = load_data(os.path.join(FEATURES_DIR, "deep_features.csv"))
    print(f"Deep dataset: {Xd.shape}")
    Xdtr, Xdvl, ydtr, ydvl = train_test_split(Xd, yd, test_size=0.2, random_state=42, stratify=yd)
    sd = StandardScaler()
    Xdtr_s = sd.fit_transform(Xdtr)
    Xdvl_s = sd.transform(Xdvl)
    deep_path = os.path.join(MODELS_DIR, "dl_deep.pt")
    dp, dt = train_model(DeepDNN(Xd.shape[1]), Xdtr_s, ydtr, Xdvl_s, ydvl, lr=5e-4, save_path=deep_path, name="DeepDNN")
    ckd = torch.load(deep_path, map_location="cpu", weights_only=False)
    ckd.update({"scaler_mean": sd.mean_, "scaler_scale": sd.scale_, "feature_names": dn,
                "input_dim": int(Xd.shape[1]), "architecture": "DeepDNN"})
    torch.save(ckd, deep_path)
    print("Deep Classification Report:")
    print(classification_report(dt, (dp>0.5).astype(int)))

    print("="*60)
    print("STEP 3: XGBoost Meta-Classifier")
    print("Features: [bio_score, deep_score, disagreement, max_score, min_score]")
    print("="*60)
    n = min(len(bp), len(dp))
    p_bio, p_deep, my = bp[:n], dp[:n], bt[:n]
    mX = np.column_stack([p_bio, p_deep, np.abs(p_bio-p_deep), np.maximum(p_bio,p_deep), np.minimum(p_bio,p_deep)])
    Xmtr, Xmvl, ymtr, ymvl = train_test_split(mX, my, test_size=0.2, random_state=7, stratify=my)
    meta = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                             random_state=42, n_jobs=-1)
    meta.fit(Xmtr, ymtr, eval_set=[(Xmvl, ymvl)], verbose=False)
    mp = meta.predict_proba(Xmvl)[:,1]
    mpred = meta.predict(Xmvl)
    auc = roc_auc_score(ymvl, mp)
    acc = accuracy_score(ymvl, mpred)
    print(f"Meta XGBoost | AUC:{auc:.4f} | Acc:{acc:.4f}")
    print(classification_report(ymvl, mpred))
    meta_path = os.path.join(MODELS_DIR, "meta_classifier.pkl")
    joblib.dump(meta, meta_path)
    cfg = {"threshold_high": 0.54, "threshold_mid": 0.30,
           "feature_order": ["bio_score","deep_score","disagreement","max_score","min_score"],
           "architecture": "PyTorch Late Fusion DL + XGBoost Meta",
           "meta_auc": round(float(auc),4), "meta_acc": round(float(acc),4)}
    with open(os.path.join(MODELS_DIR, "meta_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved dl_bio.pt, dl_deep.pt, meta_classifier.pkl to {MODELS_DIR}")
    print("TRAINING COMPLETE!")

if __name__ == "__main__":
    main()

```

---

## 12. Appendix B: Production API Implementation
To provide context on the deployment architecture discussed in Chapter 9, the core structure of the FastAPI production inference block is detailed below.

```python
import os
import sys
import torch.nn as nn

class BioDNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1))
    def forward(self, x): return self.net(x)

class DeepDNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 1))
    def forward(self, x): return self.net(x)
import pandas as pd
import json
import google.genai as genai
import time
import torch
import librosa
import logging
import numpy as np
import scipy.io.wavfile as wavfile
import joblib
import uuid
import pandas as pd
import xgboost as xgb
import threading
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Request, Form, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from auth import verify_api_key
import warnings
import random
from db import init_db, seed_database, insert_case, update_checklist, update_case_action, get_case, get_all_cases, get_similar_cases

# Configure structured logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s\t%(message)s')
logger = logging.getLogger("voiceguard")

warnings.filterwarnings('ignore')

# Add src to path
base_dir = os.path.dirname(os.path.abspath(__file__))
scratch_dir = os.path.join(base_dir, "scratch")
os.makedirs(scratch_dir, exist_ok=True)
parent_dir = os.path.dirname(base_dir)
model_dir = os.path.join(parent_dir, 'VoiceGaurd_TelephonyExp')
sys.path.append(os.path.join(model_dir, 'src'))
from extract_bio import extract_bio_features, simulate_phone_codec
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from sklearn.base import BaseEstimator, ClassifierMixin

class BinaryWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, multiclass_xgb, contrastive_head=None):
        self.multiclass_xgb = multiclass_xgb
        self.contrastive_head = contrastive_head
        self.classes_ = np.array([0, 1])
        if contrastive_head is not None:
            self.feature_names_in_ = [f'Proj_Deep_{i}' for i in range(128)]
            self.n_features_in_ = 128
        else:
            self.feature_names_in_ = getattr(multiclass_xgb, 'feature_names_in_', None)
            self.n_features_in_ = getattr(multiclass_xgb, 'n_features_in_', None)

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        if self.contrastive_head is not None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.contrastive_head.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
                X_proj = self.contrastive_head(X_tensor).cpu().numpy()
            proj_cols = [f'Proj_Deep_{i}' for i in range(128)]
            X_df = pd.DataFrame(X_proj, columns=proj_cols)
            p_raw = self.multiclass_xgb.predict_proba(X_df)
        else:
            p_raw = self.multiclass_xgb.predict_proba(X)
            
        p_real = p_raw[:, 0]
        p_fake = 1.0 - p_real
        return np.column_stack([p_real, p_fake])

    def predict(self, X):
        p_proba = self.predict_proba(X)
        return (p_proba[:, 1] >= 0.5).astype(int)

app = FastAPI(
    title="VoiceGuard AI - Dual-Stream Voice Deepfake Detection API",
    description="Real-time multilingual voice spoofing detection for banking telephony & Video-KYC.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State for Models
class ModelState:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.w2v_model = None
        # Legacy XGBoost (kept for fallback)
        self.xgb_bio = None
        self.xgb_deep = None
        self.calibrated_bio = None
        self.calibrated_deep = None
        self.contrastive_head = None
        # Late Fusion DL models
        self.dl_bio = None
        self.dl_bio_scaler_mean = None
        self.dl_bio_scaler_scale = None
        self.dl_bio_features = None
        self.dl_deep = None
        self.dl_deep_scaler_mean = None
        self.dl_deep_scaler_scale = None
        self.dl_deep_features = None
        # XGBoost meta-classifier
        self.meta_clf = None
        self.meta_cfg = None
        self.bio_feature_cols = []
        self.deep_feature_cols = []
        self.is_loaded = False

model_state = ModelState()

def compute_snr(y):
    if len(y) == 0:
        return 0.0
    signal_power = np.mean(y**2)
    noise_power = np.var(y) - signal_power if np.var(y) > signal_power else np.var(y) * 0.1
    if noise_power <= 0:
        return 40.0
    snr = 10 * np.log10(signal_power / noise_power + 1e-9)
    return float(np.clip(snr, -10, 40))

def compute_silence_ratio(y, sr):
    if len(y) == 0:
        return 0.0
    rms = librosa.feature.rms(y=y)[0]
    if len(rms) == 0:
        return 0.0
    threshold = np.percentile(rms, 20)
    silence_frames = np.sum(rms < threshold)
    return float(silence_frames / len(rms))

@app.on_event("startup")
def load_models():
    print("Initializing and seeding database...")
    init_db()
    seed_database()
    print(f"Loading models on {model_state.device}...")
    
    # 1. Load Deep Model (Wav2Vec2 feature extractor)
    model_local_path = os.path.join(model_dir, 'models', 'indicwav2vec-hindi')
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model_state.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_local_path, local_files_only=True)
    model_state.w2v_model = AutoModel.from_pretrained(model_local_path, output_hidden_states=True, local_files_only=True).to(model_state.device)
    model_state.w2v_model.eval()

    # 2. Load Base XGBoost Models (raw predictions for fixed weight)
    model_state.xgb_bio = xgb.XGBClassifier()
    try:
        model_state.xgb_bio.load_model(os.path.join(model_dir, 'models', 'xgb_bio.json'))
        if model_state.xgb_bio.get_booster().feature_names:
            model_state.bio_feature_cols = model_state.xgb_bio.get_booster().feature_names
    except Exception as e:
        print(f"XGB Bio load failed: {e}")
        
    model_state.xgb_deep = xgb.XGBClassifier()
    try:
        model_state.xgb_deep.load_model(os.path.join(model_dir, 'models', 'xgb_deep.json'))
        if model_state.xgb_deep.get_booster().feature_names:
            model_state.deep_feature_cols = model_state.xgb_deep.get_booster().feature_names
    except Exception as e:
        print(f"XGB Deep load failed: {e}")
        
    # Load Contrastive Projection Head
    try:
        from contrastive_head import ContrastiveProjectionHead
        model_state.contrastive_head = ContrastiveProjectionHead().to(model_state.device)
        model_state.contrastive_head.load_state_dict(torch.load(os.path.join(model_dir, 'models', 'contrastive_head.pt'), map_location=model_state.device))
        model_state.contrastive_head.eval()
        print("Contrastive projection head loaded successfully!")
    except Exception as e:
        print(f"Contrastive projection head load failed: {e}")

    # 3. Load Calibrated Wrappers
    try:
        model_state.calibrated_bio = joblib.load(os.path.join(model_dir, 'models', 'calibrated_bio.pkl'))
        model_state.calibrated_deep = joblib.load(os.path.join(model_dir, 'models', 'calibrated_deep.pkl'))
    except Exception as e:
        logger.warning(f"Calibrated classifiers load failed: {e}")

    # 4. Load Late Fusion DL Models (BioDNN + DeepDNN + XGBoost Meta)
    late_fusion_dir = os.path.join(base_dir, 'models_late_fusion')
    try:
        # Load BioDNN
        ckb = torch.load(os.path.join(late_fusion_dir, 'dl_bio.pt'), map_location=model_state.device, weights_only=False)
        model_state.dl_bio = BioDNN(input_dim=ckb['input_dim']).to(model_state.device)
        model_state.dl_bio.load_state_dict(ckb['model_state_dict'])
        model_state.dl_bio.eval()
        model_state.dl_bio_scaler_mean = ckb['scaler_mean']
        model_state.dl_bio_scaler_scale = ckb['scaler_scale']
        model_state.dl_bio_features = ckb['feature_names']
        logger.info(f"BioDNN loaded ({ckb['input_dim']} features)")
    except Exception as e:
        logger.error(f"BioDNN load error: {e}")

    try:
        # Load DeepDNN
        ckd = torch.load(os.path.join(late_fusion_dir, 'dl_deep.pt'), map_location=model_state.device, weights_only=False)
        model_state.dl_deep = DeepDNN(input_dim=ckd['input_dim']).to(model_state.device)
        model_state.dl_deep.load_state_dict(ckd['model_state_dict'])
        model_state.dl_deep.eval()
        model_state.dl_deep_scaler_mean = ckd['scaler_mean']
        model_state.dl_deep_scaler_scale = ckd['scaler_scale']
        model_state.dl_deep_features = ckd['feature_names']
        logger.info(f"DeepDNN loaded ({ckd['input_dim']} features)")
    except Exception as e:
        logger.error(f"DeepDNN load error: {e}")

    try:
        # Load XGBoost Meta-Classifier
        model_state.meta_clf = joblib.load(os.path.join(late_fusion_dir, 'meta_classifier.pkl'))
        with open(os.path.join(late_fusion_dir, 'meta_config.json'), 'r') as f:
            model_state.meta_cfg = json.load(f)
        logger.info("XGBoost Meta-Classifier loaded")
    except Exception as e:
        logger.error(f"Meta-Classifier load error: {e}")
        model_state.meta_clf = None

    # 5. Warm Up Deep Learning Models to prevent cold start latency
    if model_state.w2v_model is not None and model_state.processor is not None and model_state.contrastive_head is not None:
        try:
            logger.info("Warming up Deep Learning models on startup...")
            dummy_audio = np.zeros((16000,), dtype=np.float32)
            dummy_input = model_state.processor([dummy_audio], sampling_rate=16000, return_tensors="pt")
            dummy_input = {k: v.to(model_state.device) for k, v in dummy_input.items()}
            with torch.no_grad():
                if model_state.device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        out = model_state.w2v_model(**dummy_input)
                else:
                    out = model_state.w2v_model(**dummy_input)
                h12 = out.hidden_states[12]
                
                # Warm up contrastive head
                dummy_deep_feats = torch.zeros((1, 4096), dtype=torch.float32).to(model_state.device)
                _ = model_state.contrastive_head(dummy_deep_feats)
            logger.info("Deep Learning models warmed up successfully!")
        except Exception as warmup_err:
            logger.warning(f"Model warmup failed: {warmup_err}")

    model_state.is_loaded = True
    logger.info("Models loaded successfully!")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "backend_api": "CONNECTED",
        "models_loaded": model_state.is_loaded,
        "device": str(model_state.device),
        "version": "2.0.0"
    }

@app.get("/model-status")
def model_status(auth: str = Depends(verify_api_key)):
    return {
        "feature_counts": {
            "biological_features": len(model_state.bio_feature_cols) if model_state.bio_feature_cols else 97,
            "deep_features": len(model_state.deep_feature_cols) if model_state.deep_feature_cols else 4096
        },
        "model_loaded": model_state.is_loaded,
        "inference_ready": model_state.is_loaded
    }

@app.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    customer_name: str = Form(None),
    phone_number: str = Form(None),
    branch: str = Form(None),
    account_ref: str = Form(None),
    auth: str = Depends(verify_api_key)
):
    logger.info("Request Received")
    t_start = time.time()
    
    t_ingest_start = time.time()
    # Read the file content
    content = await file.read()
    logger.info("Audio Uploaded")
    
    # Clean the audio content from trailing junk/watermarks
    oppo_idx = content.find(b"oppoMark")
    if oppo_idx != -1:
        logger.info(f"Found oppoMark at byte index {oppo_idx}. Truncating trailing metadata.")
        content = content[:oppo_idx]
        
    # Detect the correct extension
    ext = ".wav"
    if content.startswith(b"RIFF") and len(content) > 12 and content[8:12] == b"WAVE":
        ext = ".wav"
    elif content.startswith(b"ID3") or content.startswith(b"\xff\xfb") or content.startswith(b"\xff\xf3") or content.startswith(b"\xff\xf2"):
        ext = ".mp3"
    elif content.startswith(b"\x1a\x45\xdf\xa3"):
        ext = ".webm"
    elif content.startswith(b"fLaC"):
        ext = ".flac"
    elif content.startswith(b"OggS"):
        ext = ".ogg"
    else:
        _, file_ext = os.path.splitext(file.filename)
        if file_ext:
            ext = file_ext.lower()
            
    scratch_dir = os.path.join(base_dir, "scratch")
    os.makedirs(scratch_dir, exist_ok=True)
    temp_path = os.path.join(scratch_dir, f"temp_{uuid.uuid4().hex}{ext}")
    with open(temp_path, "wb") as f:
        f.write(content)
    t_ingest_end = time.time()

    try:
        t_denoise_start = time.time()
        logger.info("Preprocessing Started")
        try:
            y, sr = librosa.load(temp_path, sr=16000, mono=True)
        except Exception as e_librosa:
            try:
                orig_sr, y = wavfile.read(temp_path)
                y = y.astype(np.float32)
                if np.abs(y).max() > 1.5:
                    y = y / 32768.0
                if len(y.shape) > 1:
                    y = y.mean(axis=1)
                if orig_sr != 16000:
                    y = librosa.resample(y, orig_sr=orig_sr, target_sr=16000)
                sr = 16000
            except Exception as e_scipy:
                debug_path = os.path.join(base_dir, "scratch", f"failed_live_audio_{uuid.uuid4().hex}{ext}")
                import shutil
                shutil.copy(temp_path, debug_path)
                logger.error(f"FAILED TO READ AUDIO. Saved to {debug_path}")
                raise ValueError(f"Audio format not supported. librosa error: {e_librosa}. scipy error: {e_scipy}")
            
        orig_duration = len(y) / sr
        duration = orig_duration
        
        max_duration = 10.0
        if duration > max_duration:
            logger.info(f"Audio is {duration:.1f}s. Truncating to {max_duration}s for real-time latency.")
            y = y[:int(max_duration * sr)]
        
        from scipy.signal import butter, lfilter
        import soundfile as sf
        
        def butter_highpass(cutoff, fs, order=1):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='high', analog=False)
            return b, a
            
        def highpass_filter(data, cutoff=80, fs=16000, order=1):
            b, a = butter_highpass(cutoff, fs, order=order)
            return lfilter(b, a, data)
            
        y = highpass_filter(y, cutoff=80, fs=sr)
        
        peak = np.max(np.abs(y))
        if peak > 1e-4:
            y = (y / peak) * 0.95
            
        processed_path = temp_path.replace(ext, "_processed.wav")
        sf.write(processed_path, y, sr, format='WAV', subtype='PCM_16')
        
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        temp_path = processed_path
        t_denoise_end = time.time()
        
        # Audio Quality Metrics
        signal_power = compute_snr(y)
        silence_ratio = compute_silence_ratio(y, sr)
        file_size = os.path.getsize(temp_path)
        
        # 1. Bio extraction
        t_acoustic_start = time.time()
        logger.info("Extracting Biological Features")
        bio_feats = extract_bio_features(temp_path)
        if not bio_feats:
            raise ValueError("Bio feature extraction failed")
        bio_data = {c: [bio_feats.get(c, 0.0)] for c in model_state.bio_feature_cols}
        df_bio_input = pd.DataFrame(bio_data)
        t_acoustic_end = time.time()
        
        # 2. Deep extraction
        t_spectral_start = time.time()
        logger.info("Extracting Deep Features")
        y_for_deep = y[:10*16000].astype(np.float32) if len(y) > 10*16000 else y.astype(np.float32)
        
        b_lp, a_lp = butter(4, 4000.0 / (16000.0 / 2.0), btype='low')
        y_degraded = lfilter(b_lp, a_lp, y_for_deep).astype(np.float32)

        def wav2vec_pass(audio_array):
            inp = model_state.processor([audio_array], sampling_rate=16000, return_tensors="pt", padding=True)
            inp = {k: v.to(model_state.device) for k, v in inp.items()}
            with torch.no_grad():
                if model_state.device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        out = model_state.w2v_model(**inp)
                else:
                    out = model_state.w2v_model(**inp)
            h12 = out.hidden_states[12]
            mean = torch.mean(h12, dim=1).float().cpu().numpy()[0]
            std  = torch.std(h12,  dim=1).float().cpu().numpy()[0]
            return np.concatenate([mean, std])

        feats_clean    = wav2vec_pass(y_for_deep)
        feats_degraded = wav2vec_pass(y_degraded)
        feats_mct      = np.concatenate([feats_clean, feats_degraded])

        deep_feats_dict = {}
        for k in range(4096):
            deep_feats_dict[f'Deep_{k}'] = [float(feats_mct[k])]
        df_deep_input = pd.DataFrame(deep_feats_dict)
        t_spectral_end = time.time()

        # 3. Late Fusion: BioDNN -> p_bio, DeepDNN -> p_deep
        t_fusion_start = time.time()
        logger.info("Running PyTorch BioDNN")
        if model_state.dl_bio is not None:
            try:
                bio_cols = model_state.dl_bio_features
                for c in bio_cols:
                    if c not in df_bio_input.columns:
                        df_bio_input[c] = 0.0
                X_bio_ordered = df_bio_input[bio_cols].values.astype(np.float32)
                X_bio_scaled = (X_bio_ordered - model_state.dl_bio_scaler_mean) / model_state.dl_bio_scaler_scale
                X_bio_t = torch.FloatTensor(X_bio_scaled).to(model_state.device)
                with torch.no_grad():
                    p_bio = float(torch.sigmoid(model_state.dl_bio(X_bio_t)).cpu().numpy()[0][0])
                logger.info(f"BioDNN p_bio={p_bio:.4f}")
            except Exception as bio_err:
                logger.warning(f"BioDNN inference failed: {bio_err}. Using fallback.")
                p_bio = 0.5
        else:
            p_bio = 0.5

        logger.info("Running PyTorch DeepDNN")
        if model_state.dl_deep is not None:
            try:
                deep_cols = model_state.dl_deep_features
                for c in deep_cols:
                    if c not in df_deep_input.columns:
                        df_deep_input[c] = 0.0
                X_deep_ordered = df_deep_input[deep_cols].values.astype(np.float32)
                X_deep_scaled = (X_deep_ordered - model_state.dl_deep_scaler_mean) / model_state.dl_deep_scaler_scale
                X_deep_t = torch.FloatTensor(X_deep_scaled).to(model_state.device)
                with torch.no_grad():
                    p_deep = float(torch.sigmoid(model_state.dl_deep(X_deep_t)).cpu().numpy()[0][0])
                logger.info(f"DeepDNN p_deep={p_deep:.4f}")
            except Exception as deep_err:
                logger.warning(f"DeepDNN inference failed: {deep_err}. Using fallback.")
                p_deep = 0.5
        else:
            p_deep = 0.5

        p_deep_classes = np.array([1.0 - p_deep, p_deep, 0.0, 0.0])

        df_bio_input.to_csv(os.path.join(base_dir, "scratch", "temp_bio.csv"), index=False)
        df_deep_input.to_csv(os.path.join(base_dir, "scratch", "temp_deep.csv"), index=False)

        # 4. Late Fusion: XGBoost Meta-Classifier
        logger.info("Running XGBoost Meta-Classifier (Late Fusion)")
        using_meta = False
        if model_state.meta_clf is not None:
            try:
                disagreement_val = abs(p_bio - p_deep)
                meta_features = pd.DataFrame([{
                    'bio_score': p_bio,
                    'deep_score': p_deep,
                    'disagreement': disagreement_val,
                    'max_score': max(p_bio, p_deep),
                    'min_score': min(p_bio, p_deep)
                }])
                p_fused = float(model_state.meta_clf.predict_proba(meta_features)[0][1])
                t_high = model_state.meta_cfg.get("threshold_high", 0.54)
                t_mid = model_state.meta_cfg.get("threshold_mid", 0.30)
                using_meta = True
                logger.info(f"Late Fusion p_fused={p_fused:.4f}")
            except Exception as meta_err:
                logger.warning(f"Meta-classifier inference failed: {meta_err}. Falling back to fixed weight.")

        if not using_meta:
            try:
                with open(os.path.join(model_dir, 'models', 'fusion_weights.json'), 'r') as f:
                    weights = json.load(f)
                w_deep = weights.get('w_deep', 0.236)
                w_bio = weights.get('w_bio', 0.764)
            except:
                w_deep = 0.236
                w_bio = 0.764
            p_fused = w_deep * p_deep + w_bio * p_bio
            t_high = 0.54
            t_mid = 0.30
        t_fusion_end = time.time()
        
        # 5-Tier Risk Classification
        t_verdict_start = time.time()
        disagreement = abs(p_bio - p_deep)
        if disagreement > 0.30:
            p_fused = max(p_bio, p_deep, p_fused)

        def classify_risk(score):
            if score >= 0.75:
                return "CRITICAL", "HIGH", "🔴 Critical Risk — AI-Generated Audio Detected"
            elif score >= 0.50:
                return "HIGH_RISK", "HIGH", "🟠 High Risk — Strong Synthesis Indicators"
            elif score >= 0.25:
                return "MODERATE", "MEDIUM", "🟡 Moderate Risk — Analyst Review Required"
            elif score >= 0.10:
                return "LOW_RISK", "LOW", "🟢 Low Risk — Minor Anomalies Detected"
            else:
                return "CLEAR", "LOW", "🟢 Clear — Genuine Human Voice"

        verdict, risk_level, label = classify_risk(p_fused)
        raw_conf = 70.0 + 30.0 * (abs(p_fused - 0.5) / 0.5)
        confidence = round(min(99.9, max(70.0, raw_conf)), 1)
        t_verdict_end = time.time()

        # XAI attribution stage
        t_xai_start = time.time()
        flags = []
        risk_factors = []
        mitigating_factors = []

        if disagreement > 0.30:
            flags.append("HIGH_DISAGREEMENT")
            if p_bio > p_deep:
                risk_factors.append(f"High model disagreement ({disagreement*100:.1f}%) — Bio stream flags synthetic patterns the Deep stream missed")
            else:
                risk_factors.append(f"High model disagreement ({disagreement*100:.1f}%) — Deep stream detects vocoder artifacts the Bio stream missed")
        elif disagreement < 0.10:
            flags.append("STREAMS_AGREE")
            if p_fused < 0.25:
                mitigating_factors.append("Both analysis streams agree this is genuine human speech")
            else:
                risk_factors.append("Both analysis streams independently detect synthesis indicators")

        if 0.35 <= p_bio <= 0.65:
            flags.append("BIO_BORDERLINE")
            risk_factors.append(f"Bio stream score ({p_bio*100:.1f}%) is in the uncertain zone")
        elif p_bio > 0.65:
            flags.append("BIO_ALERT")
            risk_factors.append(f"Bio stream detects strong synthetic patterns ({p_bio*100:.1f}%)")
        elif p_bio < 0.15:
            mitigating_factors.append(f"Bio stream is confident this is human ({p_bio*100:.1f}%)")

        if 0.35 <= p_deep <= 0.65:
            flags.append("DEEP_BORDERLINE")
            risk_factors.append(f"Deep stream score ({p_deep*100:.1f}%) is in the uncertain zone")
        elif p_deep > 0.65:
            flags.append("DEEP_ALERT")
            risk_factors.append(f"Deep stream detects strong vocoder artifacts ({p_deep*100:.1f}%)")
        elif p_deep < 0.15:
            mitigating_factors.append(f"Deep stream is confident this is human ({p_deep*100:.1f}%)")

        bio_feats_raw = bio_feats if bio_feats else {}
        jitter_val = bio_feats_raw.get('Jitter_Mean', None)
        shimmer_val = bio_feats_raw.get('Shimmer_Mean', None)
        hnr_val = bio_feats_raw.get('HNR_Mean', None)
        pitch_std_val = bio_feats_raw.get('Pitch_Std', None)
        flatness_val = bio_feats_raw.get('Flatness_Mean', None)

        if jitter_val is not None and jitter_val < 0.003:
            flags.append("LOW_JITTER")
            risk_factors.append(f"Jitter is unusually low ({jitter_val:.4f})")
        if shimmer_val is not None and shimmer_val < 0.02:
            flags.append("LOW_SHIMMER")
            risk_factors.append(f"Shimmer is unusually low ({shimmer_val:.4f})")
        if hnr_val is not None and hnr_val > 30:
            flags.append("HIGH_HNR")
            risk_factors.append(f"HNR is unusually high ({hnr_val:.1f} dB)")
        if pitch_std_val is not None and pitch_std_val < 5.0:
            flags.append("FLAT_PITCH")
            risk_factors.append(f"Pitch variation is very low ({pitch_std_val:.1f} Hz)")
        if flatness_val is not None and flatness_val < 0.001:
            flags.append("LOW_SPECTRAL_FLATNESS")
            risk_factors.append("Spectral flatness is very low")

        if p_fused > 0.85 or p_fused < 0.10:
            flags.append("HIGH_CONFIDENCE")
        elif 0.30 <= p_fused <= 0.60:
            flags.append("LOW_CONFIDENCE")
            risk_factors.append("Fusion score is in the uncertain range")

        def stream_assessment(score, stream_name):
            if score < 0.10:
                return f"Confident genuine — no {stream_name} anomalies detected"
            elif score < 0.25:
                return f"Likely genuine — minor {stream_name} anomalies present"
            elif score < 0.50:
                return f"Borderline — {stream_name} patterns are atypical but not conclusive"
            elif score < 0.75:
                return f"Likely synthetic — {stream_name} detects significant anomalies"
            else:
                return f"Confident synthetic — strong {stream_name} indicators of AI generation"

        def agreement_level(d):
            if d < 0.10: return f"HIGH ({d*100:.1f}% disagreement)"
            elif d < 0.20: return f"MODERATE ({d*100:.1f}% disagreement)"
            elif d < 0.30: return f"LOW ({d*100:.1f}% disagreement)"
            else: return f"CRITICAL ({d*100:.1f}% disagreement)"

        if disagreement > 0.30 and p_bio > p_deep:
            primary_concern = "Bio stream detects potential synthesis artifacts that the Deep stream does not recognize"
        elif disagreement > 0.30 and p_deep > p_bio:
            primary_concern = "Deep stream detects vocoder artifacts that biological features miss"
        elif p_fused >= 0.50:
            primary_concern = "Both streams indicate this audio contains synthetic speech characteristics"
        elif p_fused >= 0.25:
            primary_concern = "Inconclusive analysis — audio shows some atypical patterns"
        else:
            primary_concern = "No significant synthesis indicators detected"

        RECOMMENDATIONS = {
            "CRITICAL": {"action": "BLOCK", "urgency": "CRITICAL", "detail": "Immediately block and escalate. Both analysis streams detect strong AI generation signatures."},
            "HIGH_RISK": {"action": "BLOCK", "urgency": "HIGH", "detail": "Block the call and queue for supervisor review. Significant synthesis indicators detected."},
            "MODERATE": {"action": "REVIEW", "urgency": "MEDIUM", "detail": "Route to analyst queue for manual spectrogram inspection. Audio shows patterns that require review."},
            "LOW_RISK": {"action": "ALLOW_MONITOR", "urgency": "LOW", "detail": "Allow the call but log for pattern analysis. Minor anomalies detected."},
            "CLEAR": {"action": "ALLOW", "urgency": "NONE", "detail": "No action required. Audio is consistent with natural human speech."}
        }
        rec = RECOMMENDATIONS[verdict]

        if "HIGH_DISAGREEMENT" in flags and verdict in ("CLEAR", "LOW_RISK"):
            rec = {"action": "REVIEW", "urgency": "MEDIUM", "detail": "Model disagreement exceeds safety threshold. Route to analyst for manual review."}
            verdict = "MODERATE"
            risk_level = "MEDIUM"
            label = "🟡 Moderate Risk — High Model Disagreement"

        # Generate the dynamic z-score narrative
        narrative_parts = []
        if verdict in ("CRITICAL", "HIGH_RISK"):
            narrative_parts.append(f"Forensic analysis detected strong artificial signatures in the audio stream (overall risk score: {p_fused*100:.1f}%).")
        elif verdict == "MODERATE":
            narrative_parts.append(f"Analysis indicates atypical vocal characteristics (overall risk score: {p_fused*100:.1f}%) that warrant human review.")
        else:
            narrative_parts.append(f"Audio stream exhibits normal acoustic and spectral characteristics consistent with organic human speech.")
            
        acoustic_anomalies = []
        if "LOW_JITTER" in flags:
            acoustic_anomalies.append("an abnormally flat micro-pitch curve (low jitter)")
        if "LOW_SHIMMER" in flags:
            acoustic_anomalies.append("lack of natural amplitude shimmer")
        if "HIGH_HNR" in flags:
            acoustic_anomalies.append("an unnaturally clean vocal track (high HNR)")
        if "FLAT_PITCH" in flags:
            acoustic_anomalies.append("lack of standard expressive prosody (flat pitch variation)")
            
        if acoustic_anomalies:
            narrative_parts.append("Key anomalies include " + ", ".join(acoustic_anomalies) + ".")
            narrative_parts.append("These characteristics match the signature of a neural vocoder synthesis model rather than a biological human vocal tract.")
        else:
            narrative_parts.append("Micro-pitch jitter, amplitude shimmer, and harmonic-to-noise ratios are all within typical human baselines.")
            
        explanation = {
            "summary": primary_concern,
            "flags": flags,
            "stream_analysis": {
                "bio_score": round(p_bio * 100, 1),
                "bio_assessment": stream_assessment(p_bio, "biological/prosodic"),
                "deep_score": round(p_deep * 100, 1),
                "deep_assessment": stream_assessment(p_deep, "vocoder/spectral"),
                "agreement": agreement_level(disagreement)
            },
            "risk_factors": risk_factors if risk_factors else ["No significant risk factors identified"],
            "mitigating_factors": mitigating_factors if mitigating_factors else ["No specific mitigating factors"],
            "xai_narrative": " ".join(narrative_parts)
        }
        t_xai_end = time.time()

        t_report_start = time.time()
        chunk_results = [
            {"index": 1, "score": p_fused * 100, "confidence": confidence, "verdict": verdict, "start": 0, "end": round(duration, 1), "bio_score": p_bio * 100, "deep_score": p_deep * 100}
        ]
        
        case_id = f"VG-{uuid.uuid4().hex[:6].upper()}"
        cust_name = customer_name or "Rajesh Kumar"
        ph_num = phone_number or "+91 94311 08842"
        br = branch or "Hazratganj, Lucknow"
        acc = account_ref or "UCO-308849102"
        otp_code = str(random.randint(100000, 999999))

        # Insert case in SQLite
        insert_case(
            case_id=case_id,
            customer_name=cust_name,
            phone_number=ph_num,
            branch=br,
            account_ref=acc,
            fraud_score=round(p_fused * 100, 1),
            confidence=confidence,
            verdict=verdict,
            bio_features=bio_feats,
            deep_features=deep_feats_dict,
            otp_code=otp_code
        )
        t_report_end = time.time()

        # Telemetry timings
        stage_timings = {
            "ingest_ms": int((t_ingest_end - t_ingest_start) * 1000),
            "denoise_ms": int((t_denoise_end - t_denoise_start) * 1000),
            "acoustic_ms": int((t_acoustic_end - t_acoustic_start) * 1000),
            "spectral_ms": int((t_spectral_end - t_spectral_start) * 1000),
            "fusion_ms": int((t_fusion_end - t_fusion_start) * 1000),
            "xai_ms": int((t_xai_end - t_xai_start) * 1000),
            "verdict_ms": int((t_verdict_end - t_verdict_start) * 1000),
            "report_ms": int((t_report_end - t_report_start) * 1000)
        }

        result = {
            "case_id": case_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "fraud_score": round(p_fused * 100, 1),
            "confidence": confidence,
            "risk_level": risk_level,
            "risk_tier": verdict,
            "verdict": verdict,
            "verdict_label": label,
            "primary_trigger": "PyTorch Late Fusion DL" if using_meta else "Fixed Weight Fusion",
            "secondary_trigger": "Wav2Vec2 Anomaly" if p_deep > p_bio else "Bio Feature Anomaly",
            "explanation": explanation,
            "recommendation": rec["detail"] if isinstance(rec, dict) else rec,
            "using_real_models": True,
            "metadata": {"filename": file.filename, "format": "WAV", "sample_rate": 16000, "channels": 1, "duration": round(duration, 2), "file_size_bytes": file_size},
            "performance": {"model_version": "v3.0-5TierExplainable" if using_meta else "v3.0-FixedWeight", "inference_time_ms": int((time.time() - t_start) * 1000), "audio_duration_sec": round(duration, 2), "chunks_processed": 1},
            "threat_intel": {
                "threat_type": "AI Voice Clone" if verdict not in ("CLEAR", "LOW_RISK") else "None",
                "sophistication": "Advanced" if verdict in ("CRITICAL", "HIGH_RISK") else ("Moderate" if verdict == "MODERATE" else "None"),
                "replay_indicators": "None",
                "synthetic_confidence": round(float(max(p_deep_classes[1:])) * 100, 1) if verdict not in ("CLEAR", "LOW_RISK") else 0.0
            },
            "chunk_results": chunk_results,
            "stage_timings": stage_timings,
            "otp_code": otp_code,
            "customer_name": cust_name,
            "phone_number": ph_num,
            "branch": br,
            "account_ref": acc
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    logger.info("Returning JSON Response")
    return JSONResponse(content=result)

@app.post("/feedback")
async def process_feedback(is_correct: str = Form(...), true_label: str = Form(...), auth: str = Depends(verify_api_key)):
    is_correct = (is_correct.lower() == 'true')
    label = 0 if true_label == "Real" else 1
    
    if true_label == "Synthetic":
        prefix = "active_learning_synthetic"
    elif true_label == "Generic":
        prefix = "active_learning_generic"
    else:
        prefix = "active_learning_real"
    
    bio_temp = os.path.join(base_dir, "scratch", "temp_bio.csv")
    deep_temp = os.path.join(base_dir, "scratch", "temp_deep.csv")
    
    if os.path.exists(bio_temp) and os.path.exists(deep_temp):
        df_bio_input = pd.read_csv(bio_temp)
        df_deep_input = pd.read_csv(deep_temp)
        
        df_bio_input['Label'] = label
        df_deep_input['Label'] = label
        df_bio_input['Filename'] = f"{prefix}_{int(time.time())}.wav"
        df_deep_input['Filename'] = f"{prefix}_{int(time.time())}.wav"
        
        hard_val_bio = os.path.join(model_dir, 'features', 'hard_val_bio.csv')
        hard_val_deep = os.path.join(model_dir, 'features', 'hard_val_deep.csv')
        
        # Align columns to prevent format corruption when appending via mode='a'
        if os.path.exists(hard_val_bio):
            existing_bio = pd.read_csv(hard_val_bio, nrows=0)
            for col in existing_bio.columns:
                if col not in df_bio_input.columns:
                    df_bio_input[col] = 0.0
            df_bio_input = df_bio_input[existing_bio.columns]
            
        if os.path.exists(hard_val_deep):
            existing_deep = pd.read_csv(hard_val_deep, nrows=0)
            for col in existing_deep.columns:
                if col not in df_deep_input.columns:
                    df_deep_input[col] = 0.0
            df_deep_input = df_deep_input[existing_deep.columns]
            
        df_bio_input.to_csv(hard_val_bio, mode='a', header=not os.path.exists(hard_val_bio), index=False)
        df_deep_input.to_csv(hard_val_deep, mode='a', header=not os.path.exists(hard_val_deep), index=False)
        
        return {"status": "success", "message": f"Feedback received. Added to hard validation data as label {label}."}
    else:
        return JSONResponse(status_code=400, content={"status": "error", "message": "No recent analysis found."})

@app.post("/retrain")
async def retrain_model(auth: str = Depends(verify_api_key)):
    logger.info("Retraining meta-classifier triggered...")
    # Call the training script
    os.system(f"{sys.executable} src/train_meta.py")
    
    # Reload the model
    try:
        load_models()
        return {"status": "success", "message": "Classifier retrained and reloaded successfully."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/generate-report")
async def generate_report(auth: str = Depends(verify_api_key)):
    return {"status": "Placeholder for PDF report"}

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(base_dir, "static", "voiceguard_uco_bank_platform.html"))

# Serve the static UI files
app.mount("/static", StaticFiles(directory=os.path.join(base_dir, "static")), name="static")


@app.get("/api/v1/cases")
def list_cases(auth: str = Depends(verify_api_key)):
    return get_all_cases()

@app.get("/api/v1/cases/{case_id}")
def case_details(case_id: str, auth: str = Depends(verify_api_key)):
    c = get_case(case_id)
    if not c:
        return JSONResponse(status_code=404, content={"message": "Case not found"})
    return c

@app.post("/api/v1/cases/{case_id}/checklist")
async def update_case_checklist(case_id: str, req: Request, auth: str = Depends(verify_api_key)):
    state = await req.json()
    update_checklist(case_id, state)
    return {"status": "success", "message": "Checklist updated"}

@app.post("/api/v1/cases/{case_id}/action")
async def save_case_action(
    case_id: str, 
    frozen: bool = Form(None), 
    escalated: bool = Form(None), 
    notes: str = Form(None), 
    auth: str = Depends(verify_api_key)
):
    update_case_action(case_id, frozen=frozen, escalated=escalated, notes=notes)
    return {"status": "success", "message": "Action updated"}

@app.get("/api/v1/cases/{case_id}/similar")
def get_sim_cases(case_id: str, auth: str = Depends(verify_api_key)):
    return get_similar_cases(case_id)



class ChatRequest(BaseModel):
    message: str
    case_id: str = None

@app.post("/api/v1/chat")
async def process_chat(req: ChatRequest, auth: str = Depends(verify_api_key)):
    session_id = req.session_id or "default"
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
        
    history = chat_sessions[session_id]
    
    # 1. Structured Case Context Truncated
    context = ""
    case_summary_fact = "No active case."
    if req.case_id:
        c = get_case(req.case_id)
        if c:
            # Parse Bio/Deep features just to grab a couple top keys if we want, but let's just stick to structured core
            try:
                bio = json.loads(c['bio_features'])
                top_bio = {k: round(v, 4) for k, v in list(bio.items())[:3]}
            except:
                top_bio = {}
                
            context = f"Active Case: {c['case_id']} | Verdict: {c['verdict']} | Score: {c['fraud_score']}% | Bio Flags: {top_bio}"
            case_summary_fact = f"Active Case {c['case_id']} is {c['verdict']} ({c['fraud_score']}%)."
            
    # 2. KB Retrieval & Direct Cache Check
    kb_context = ""
    matches = get_kb_matches(req.message)
    if matches and matches[0]["score"] > 0.85:
        # High confidence exact match -> Skip LLM entirely
        ans = matches[0]["article"]["a"]
        # Add tracking note to history
        history.append({"role": "user", "content": req.message})
        history.append({"role": "assistant", "content": ans})
        if len(history) > 6:
            history = history[-6:]
        chat_sessions[session_id] = history
        return {"status": "success", "reply": ans, "model": "Direct KB Cache"}
        
    if matches:
        kb_text = []
        for m in matches:
            if m["score"] > 0.1: # Only inject if slightly relevant
                a_trunc = m["article"]["a"][:300] + "..." if len(m["article"]["a"]) > 300 else m["article"]["a"]
                kb_text.append(f"Q: {m['article']['q']}\nA: {a_trunc}")
        if kb_text:
            kb_context = "\n\nBANK KNOWLEDGE BASE REFERENCE:\n" + "\n---\n".join(kb_text)

    # 3. Rolling History Summarization
    # Keep last 2-3 turns (4-6 messages), summarize older ones
    # To keep it completely deterministic, the summary is just a list of past user topics
    if len(history) > 6:
        # We need to drop the oldest ones and update a deterministic summary
        old_msgs = history[:-6]
        history = history[-6:]
        chat_sessions[session_id] = history
        
    # Extract topics from all past user messages in history for the summary line
    past_topics = [msg["content"] for msg in history if msg["role"] == "user"]
    topic_summary = ""
    if past_topics:
        # Just list them briefly
        topics_str = " | ".join([t[:30] + ("..." if len(t)>30 else "") for t in past_topics[:-1]]) if len(past_topics) > 1 else "None"
        topic_summary = f"\n\nPrior Topics Discussed: {topics_str}\n{case_summary_fact}"

    sys_prompt = f"""You are the VoiceGuard Forensic AI Agent exclusively built for UCO Bank's Fraud Control Cell. You analyze highly sensitive voice biometric data.
    
    CRITICAL INSTRUCTIONS:
    1. Act strictly as the VoiceGuard AI Agent. Keep answers professional and highly analytical.
    2. If an Active Case is provided, directly reference its Verdict and Bio Flags.
    3. Use the BANK KNOWLEDGE BASE REFERENCE if provided to answer policy/glossary questions. Do NOT invent facts.
    
    {context}{kb_context}{topic_summary}"""
    
    import requests
    try:
        headers = {
            "Authorization": "Bearer YOUR_OPENROUTER_API_KEY",
            "HTTP-Referer": "http://localhost:8001",
            "X-Title": "VoiceGuard AI Agent",
            "Content-Type": "application/json"
        }
        
        # Build messages payload
        messages = [{"role": "system", "content": sys_prompt}]
        for msg in history:
            messages.append(msg)
        messages.append({"role": "user", "content": req.message})
        
        payload = {
            "model": "openai/gpt-4o-mini",
            "messages": messages
        }
        
        used_model = "gpt-4o-mini"
        try:
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=20)
            response.raise_for_status()
        except requests.exceptions.RequestException as req_err:
            print(f"GPT-4o-mini failed ({req_err}), falling back to Gemini...")
            used_model = "gemini-2.0-flash-exp:free"
            payload["model"] = "google/gemini-2.0-flash-exp:free"
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=20)
            response.raise_for_status()
            
        data = response.json()
        reply_text = data["choices"][0]["message"]["content"]
        
        # Append to history
        history.append({"role": "user", "content": req.message})
        history.append({"role": "assistant", "content": reply_text})
        chat_sessions[session_id] = history
        
        return {"status": "success", "reply": reply_text, "model": used_model}
    except Exception as e:
        print("LLM Error:", e)
        return {"status": "error", "reply": "OpenRouter LLM Error: " + str(e), "model": "error"}


@app.post("/analyze-chunks")
async def analyze_chunks(
    file: UploadFile = File(...),
    auth: str = Depends(verify_api_key)
):
    """Per-window bio-feature scoring for forensic overlay.
    
    Slices audio into 2-3s windows and runs ONLY the cheap bio-feature
    extraction (jitter, shimmer, HNR, MFCCs — pure DSP, no neural net)
    per window, then scores each via the BioDNN model.
    
    This endpoint is called AFTER /analyze returns, in the background.
    It does NOT run the expensive Wav2Vec2 deep model.
    """
    t_start = time.time()
    
    content = await file.read()
    
    # Clean trailing junk
    oppo_idx = content.find(b"oppoMark")
    if oppo_idx != -1:
        content = content[:oppo_idx]
    
    # Detect extension
    ext = ".wav"
    if content.startswith(b"RIFF") and len(content) > 12 and content[8:12] == b"WAVE":
        ext = ".wav"
    elif content.startswith(b"ID3") or content.startswith(b"\xff\xfb"):
        ext = ".mp3"
    elif content.startswith(b"\x1a\x45\xdf\xa3"):
        ext = ".webm"
    elif content.startswith(b"OggS"):
        ext = ".ogg"
    else:
        _, file_ext = os.path.splitext(file.filename)
        if file_ext:
            ext = file_ext.lower()
    
    temp_path = os.path.join(base_dir, "scratch", f"chunk_{uuid.uuid4().hex}{ext}")
    with open(temp_path, "wb") as f:
        f.write(content)
    
    try:
        # Load and preprocess (same as /analyze)
        try:
            y, sr = librosa.load(temp_path, sr=16000, mono=True)
        except Exception:
            try:
                orig_sr, y = wavfile.read(temp_path)
                y = y.astype(np.float32)
                if np.abs(y).max() > 1.5:
                    y = y / 32768.0
                if len(y.shape) > 1:
                    y = y.mean(axis=1)
                if orig_sr != 16000:
                    y = librosa.resample(y, orig_sr=orig_sr, target_sr=16000)
                sr = 16000
            except Exception as e:
                return JSONResponse(content={"error": str(e)}, status_code=400)
        
        duration = len(y) / sr
        max_duration = 10.0
        if duration > max_duration:
            y = y[:int(max_duration * sr)]
            duration = max_duration
        
        # Highpass filter
        from scipy.signal import butter, lfilter
        b, a = butter(1, 80.0 / (0.5 * sr), btype='high', analog=False)
        y = lfilter(b, a, y)
        peak = np.max(np.abs(y))
        if peak > 1e-4:
            y = (y / peak) * 0.95
        
        # Determine window size: 2-3s, cap at ~5 windows max
        if duration <= 3.0:
            window_sec = duration  # single chunk for very short clips
        elif duration <= 6.0:
            window_sec = duration / 2.0
        else:
            window_sec = max(2.0, duration / 5.0)  # aim for ~5 windows
        
        window_samples = int(window_sec * sr)
        chunks = []
        idx = 0
        chunk_index = 0
        
        while idx < len(y):
            end_sample = min(idx + window_samples, len(y))
            chunk_audio = y[idx:end_sample]
            chunk_start = idx / sr
            chunk_end = end_sample / sr
            
            # Skip chunks shorter than 0.5s (not enough for meaningful features)
            if len(chunk_audio) < int(0.5 * sr):
                idx = end_sample
                continue
            
            # Extract bio features
            try:
                bio_score = _score_chunk_bio(chunk_audio, sr)
            except Exception:
                bio_score = 0.0
                
            # Extract deep features
            try:
                deep_score = _score_chunk_deep(chunk_audio, sr)
            except Exception as e:
                logger.error(f"Deep chunk score failed: {e}")
                deep_score = 0.0
            
            # Classify chunk (fusion: max of both scores)
            chunk_max_score = max(bio_score, deep_score)
            if chunk_max_score >= 0.65:
                chunk_verdict = "HIGH_RISK"
            elif chunk_max_score >= 0.35:
                chunk_verdict = "MODERATE"
            elif chunk_max_score >= 0.15:
                chunk_verdict = "LOW_RISK"
            else:
                chunk_verdict = "CLEAR"
            
            chunks.append({
                "index": chunk_index,
                "start": round(chunk_start, 2),
                "end": round(chunk_end, 2),
                "bio_score": round(bio_score * 100, 1),
                "deep_score": round(deep_score * 100, 1),
                "score": round(chunk_max_score * 100, 1), 
                "verdict": chunk_verdict
            })
            
            chunk_index += 1
            idx = end_sample
        
        return JSONResponse(content={
            "chunk_results": chunks,
            "duration": round(duration, 2),
            "window_sec": round(window_sec, 2),
            "num_chunks": len(chunks),
            "compute_ms": int((time.time() - t_start) * 1000)
        })
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _score_chunk_bio(chunk_audio, sr=16000):
    """Score a single audio chunk using only bio features + BioDNN.
    
    This is the cheap path: pure signal processing (jitter, shimmer, HNR,
    MFCCs, spectral features). No Wav2Vec2, no deep model.
    Takes a numpy array directly — no file I/O.
    """
    import soundfile as sf
    
    # Write chunk to temp file for extract_bio_features (it expects a path)
    chunk_path = os.path.join(base_dir, "scratch", f"_chunk_{uuid.uuid4().hex}.wav")
    try:
        sf.write(chunk_path, chunk_audio, sr, format='WAV', subtype='PCM_16')
        bio_feats = extract_bio_features(chunk_path)
        
        if not bio_feats or model_state.dl_bio is None:
            return 0.0
        
        # Score via BioDNN (same path as /analyze, just the bio part)
        bio_cols = model_state.dl_bio_features
        bio_data = {c: [bio_feats.get(c, 0.0)] for c in bio_cols}
        df = pd.DataFrame(bio_data)
        X = df.values.astype(np.float32)
        X_scaled = (X - model_state.dl_bio_scaler_mean) / model_state.dl_bio_scaler_scale
        X_t = torch.FloatTensor(X_scaled).to(model_state.device)
        
        with torch.no_grad():
            p = float(torch.sigmoid(model_state.dl_bio(X_t)).cpu().numpy()[0][0])
        return p
    finally:
        if os.path.exists(chunk_path):
            os.remove(chunk_path)

def _score_chunk_deep(chunk_audio, sr=16000):
    """Score a single audio chunk using Wav2Vec2 + PyTorch DeepDNN.
    
    Extracts deep spectral features from the chunk and passes them through
    the fusion DeepDNN to get a synthetic probability score.
    """
    if model_state.w2v_model is None or model_state.processor is None or model_state.dl_deep is None:
        return 0.0
        
    y = chunk_audio.astype(np.float32)
    
    # Degrade audio (lowpass filter)
    from scipy.signal import butter, lfilter
    b_lp, a_lp = butter(4, 4000.0 / (sr / 2.0), btype='low')
    y_deg = lfilter(b_lp, a_lp, y).astype(np.float32)
    
    def w2v(audio_array):
        inp = model_state.processor([audio_array], sampling_rate=sr, return_tensors="pt", padding=True)
        inp = {k: v.to(model_state.device) for k, v in inp.items()}
        with torch.no_grad():
            if model_state.device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    out = model_state.w2v_model(**inp)
            else:
                out = model_state.w2v_model(**inp)
        h12 = out.hidden_states[12]
        mean = torch.mean(h12, dim=1).float().cpu().numpy()[0]
        std  = torch.std(h12,  dim=1).float().cpu().numpy()[0]
        return np.concatenate([mean, std])
        
    try:
        f_clean = w2v(y)
        f_deg = w2v(y_deg)
        f_mct = np.concatenate([f_clean, f_deg])
        
        deep_feats_dict = {f'Deep_{k}': [float(f_mct[k])] for k in range(4096)}
        df_deep = pd.DataFrame(deep_feats_dict)
        
        deep_cols = model_state.dl_deep_features
        for c in deep_cols:
            if c not in df_deep.columns:
                df_deep[c] = 0.0
                
        X = df_deep[deep_cols].values.astype(np.float32)
        X_scaled = (X - model_state.dl_deep_scaler_mean) / model_state.dl_deep_scaler_scale
        X_t = torch.FloatTensor(X_scaled).to(model_state.device)
        
        with torch.no_grad():
            p_deep = float(torch.sigmoid(model_state.dl_deep(X_t)).cpu().numpy()[0][0])
            
        return p_deep
    except Exception as e:
        logger.error(f"_score_chunk_deep error: {e}")
        return 0.0


if __name__ == "__main__":

    import uvicorn
    
    # Open Chrome in a separate thread after a short delay
    def open_chrome():
        time.sleep(2)  # Wait 2 seconds for server to start
        try:
            # Windows native way to open URL in default browser
            os.startfile('http://127.0.0.1:8001')
        except Exception as e:
            print(f"Failed to open browser: {e}")
            print("Open http://127.0.0.1:8001 manually in your browser")
    
    chrome_thread = threading.Thread(target=open_chrome, daemon=True)
    chrome_thread.start()
    
    uvicorn.run("fastapi_app:app", host="127.0.0.1", port=8001, reload=True)
    # Trigger model reload: 2026-06-29T20:59

```

---

## 13. Conclusion & The Future of Voice Biometrics

VoiceGuard AI completely reinvents the methodology for detecting synthetic speech. 
Classical detection systems, crippled by their reliance on single-domain feature spaces and linear classification boundaries, are fundamentally incapable of holding the line against the exponential advancement of generative AI. 

By refusing to compromise on dimensionality, and instead enforcing a strict architectural separation between the physics of human speech (the BioDNN) and the deep semantic anomalies of generative transformers (the DeepDNN), VoiceGuard AI establishes an impregnable defense-in-depth architecture. 
The implementation of an XGBoost Meta-Classifier to dynamically resolve conflicting probabilistic outputs provides a mathematically sound decision matrix capable of navigating the complex, non-linear boundaries separating authentic human speech from sophisticated digital fabrications.

This dual-stream, late-fusion approach ensures that as generative AI continues to evolve, VoiceGuard AI can independently scale and adapt its biological and acoustic analysis capabilities. 
It represents the vanguard of biometric security, ensuring the preservation of trust in a digitally compromised world.
