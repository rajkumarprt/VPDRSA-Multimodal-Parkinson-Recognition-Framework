# VPDRSA: Vision-based Parkinson Disease Recognition and Severity Assessment

## Repository Overview

This repository contains the official implementation of the **Vision-based Parkinson Disease Recognition and Severity Assessment (VPDRSA)** framework proposed in our research. The framework presents a multimodal deep learning architecture for automated Parkinson disease (PD) recognition by integrating complementary visual and audio information extracted from publicly available datasets. The proposed approach combines advanced convolutional neural networks, temporal sequence modeling, multimodal feature fusion, hyperparameter optimization, and ensemble machine learning to achieve robust and reliable PD detection.

The implementation is developed using Python and PyTorch, with Optuna employed for hyperparameter optimization and XGBoost for final classification. The repository is intended to facilitate reproducible research and serves as a reference implementation for researchers working on computer vision, speech processing, multimodal learning, and intelligent healthcare systems.

---

---

## Dataset Description

The proposed VPDRSA framework leverages two complementary publicly available datasets to capture diverse motor and speech characteristics associated with Parkinson disease. The **YouTubePD** dataset provides multimodal information comprising both video and speech, while the **Turning-in-Place Parkinson Disease** dataset focuses on clinically relevant turning movements for detailed gait analysis.

### YouTubePD Dataset

The **YouTubePD** dataset is a publicly available **multimodal Parkinson disease** dataset consisting of unconstrained YouTube videos of individuals with Parkinson disease and healthy controls. The videos are collected under real-world conditions, exhibiting variations in camera viewpoints, illumination, backgrounds, recording quality, and subject movements, making the dataset representative of practical clinical screening scenarios.

Each recording contains synchronized **RGB video** and **speech audio**, enabling simultaneous analysis of motor and vocal impairments associated with Parkinson disease. The dataset captures diverse activities such as spontaneous speech, facial expressions, and upper-body movements, allowing comprehensive multimodal feature learning.

#### Features Utilized

##### Visual Features
- Facial expressions and facial masking
- Body posture and movement coordination
- Temporal motion representations extracted from video sequences

##### Audio Features
- Mel-Frequency Cepstral Coefficients (MFCCs)
- Temporal speech characteristics
- Vocal articulation and phonation patterns
- Speech rhythm and acoustic variations


### Turning-in-Place Parkinson Disease Dataset

The **Turning-in-Place Parkinson Disease** dataset is a publicly available clinical dataset specifically designed to analyze **turning movements**, which are among the earliest motor impairments observed in Parkinson disease. The dataset contains videos of participants performing standardized turning-in-place tasks under controlled experimental conditions.

Unlike conventional gait datasets that primarily capture straight-line walking, this dataset emphasizes rotational movements that frequently reveal clinically significant symptoms such as **Freezing of Gait (FoG)**, impaired balance, reduced turning speed, postural instability, and motor coordination deficits.

Since this dataset consists exclusively of video recordings, only visual information is utilized in the proposed framework.

#### Features Utilized

- Turning motion patterns
- Body orientation changes
- Rotational gait characteristics
- Postural stability
- Motor coordination during turning
- Temporal movement dynamics
- Indicators associated with Freezing of Gait (FoG)

---

## Framework Overview

The proposed VPDRSA framework consists of the following major components:

### • Multimodal Data Acquisition

* Video data from the YouTubePD dataset.
* Turning-in-Place gait videos.
* Speech recordings for audio analysis.

### • Data Preprocessing

#### Video Preprocessing

* Video decoding and frame extraction
* Uniform temporal frame sampling (16 consecutive frames)
* Frame resizing to **224 × 224** pixels
* Pixel normalization
* Data quality verification and removal of corrupted samples

#### Audio Preprocessing

* Audio extraction from video recordings
* Noise reduction and resampling
* Extraction of **40-dimensional Mel-Frequency Cepstral Coefficients (MFCCs)**
* Temporal sequence standardization (300 frames)

### • Dataset Prepraration
* Temporal frame standardization.
* Image normalization.
* MFCC feature extraction from speech signals.
* Dataset cleaning and train-validation-test splitting.

### • Visual Feature Extraction

* EfficientNetV2-S for extracting gait and facial movement features from YouTubePD videos.
* MobileNetV3-Small for learning turning and freezing-of-gait representations.

### • Audio Feature Extraction

* Bidirectional GRU encoder for modeling temporal speech characteristics using MFCC features.

### • Temporal Modeling

* Temporal Attention Bidirectional LSTM for capturing long-range temporal dependencies in visual sequences.

### • Multimodal Feature Fusion

* Feature-level concatenation of visual and audio embeddings.
* Fully connected projection layer with dropout regularization.

### • Hyperparameter Optimization

* Optuna Tree-structured Parzen Estimator (TPE).
* Automatic optimization of learning rate, dropout, hidden dimensions, batch size, weight decay, and XGBoost parameters.

### • Classification

* XGBoost classifier trained on optimized multimodal latent features.
* Binary Parkinson disease recognition.

### • Performance Evaluation

* Accuracy
* Precision
* Recall
* F1-score
* AUROC
* False Positive Rate
* Confusion Matrix
* ROC Curve
* Precision–Recall Curve
* Stratified Five-Fold Cross-Validation
* Paired t-test for statistical significance

---

## Repository Contents

The repository includes the complete implementation required to reproduce the experiments presented in the manuscript, including

```text
├── Data preprocessing scripts
├── Video frame extraction
├── Audio preprocessing (MFCC extraction)
├── EfficientNetV2-S feature extractor
├── MobileNetV3-Small feature extractor
├── Bidirectional GRU audio encoder
├── Temporal Attention Bi-LSTM
├── Multimodal feature fusion module
├── Optuna hyperparameter optimization
├── XGBoost classifier
├── Model training scripts
├── Model evaluation scripts
├── Cross-validation
├── Statistical analysis
└── Visualization utilities
```

---

## Experimental Configuration

The implementation follows the experimental settings reported in the manuscript, including

* Image size: 224 × 224
* Video length: 16 frames
* Audio representation: 40-dimensional MFCC
* Audio sequence length: 300 frames
* Batch size: 4
* Mixed precision training
* Gradient clipping
* Optuna-TPE optimization
* XGBoost classifier
* Stratified 5-fold cross-validation

---

## Reproducibility

The repository has been developed to ensure reproducibility of the experimental results reported in the associated publication. The provided scripts allow users to preprocess the datasets, train the multimodal deep learning model, perform Optuna-based hyperparameter optimization, train the XGBoost classifier, evaluate the trained models, and reproduce the reported quantitative performance metrics.

---

## Citation

If you use this repository in your research, please cite the associated publication:

> Rajavel R., Sankaranarayanan S., Dhanushree D., *"Vision-based Parkinson Disease Recognition and Severity Assessment using Multimodal Feature Fusion and Deep Learning."* (Add journal details after publication.)

---

## Acknowledgement

This implementation was developed for research purposes to advance intelligent healthcare systems for early Parkinson disease recognition and severity assessment. The framework utilizes publicly available datasets, including the YouTubePD dataset (https://uiuc-yuxiong-lab.github.io/YouTubePD/) and the Turning-in-Place Parkinson Disease dataset (https://figshare.com/articles/dataset/A_public_dataset_of_video_acceleration_and_angular_velocity_in_individuals_with_Parkinson_s_disease_during_the_turning-in-place_task/14984667), together with open-source machine learning libraries such as PyTorch, XGBoost, and Optuna. We gratefully acknowledge the creators of these datasets and software tools for enabling reproducible research in artificial intelligence for healthcare.
