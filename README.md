A multimodal deep learning framework for Parkinson’s disease detection from video and audio information, combining complementary spatial, temporal, and acoustic representations with adaptive feature fusion and an XGBoost-based classifier.

Overview

This project develops a multimodal Parkinson’s disease detection system that combines visual and temporal information extracted from video data with acoustic information from speech-oriented videos.

The proposed architecture integrates:

EfficientNetV2 for visual feature extraction

MobileNetV3 for complementary visual representation learning

Bidirectional GRU (Bi-GRU) for temporal sequence modeling

Attention-based Bidirectional LSTM (Attention Bi-LSTM) for learning important temporal patterns

Adaptive Fusion for combining multimodal deep representations

Softmax as the baseline classifier

XGBoost as the optimized nonlinear classifier

Optuna with Tree-structured Parzen Estimator (TPE) for XGBoost hyperparameter optimization

The framework is evaluated using Accuracy, Precision, Recall, F1-score, False Positive Rate (FPR), AUROC, Loss, and computation time.

Research Objective

The primary objective is to investigate whether combining complementary visual, temporal, and audio representations can improve automated Parkinson’s disease detection.

The system is designed around the following principle:

Deep neural networks learn meaningful multimodal representations, adaptive fusion combines these representations, and XGBoost learns nonlinear decision boundaries from the resulting fused feature vector.

EfficientNetV2 and MobileNetV3 provide complementary visual representations, while Bi-GRU and Attention Bi-LSTM model temporal dependencies. Adaptive Fusion integrates the learned representations before classification.

Dataset

The project uses two datasets:

1. YouTube Dataset

The YouTube dataset contains Parkinson’s disease-related videos divided into:

Youtube dataset/
├── Negative/
└── Positive/

The dataset contains both positive and negative samples and provides video and audio information.

2. Turning Dataset

The Turning dataset contains Parkinson’s-related turning movement videos.

Turning dataset/
└── PDFE*.mp4

The turning videos provide movement-related visual information and are used as part of the broader multimodal Parkinson’s disease detection research.

Dataset Location

The raw dataset was organized under:

/content/drive/MyDrive/ParkinsonDataset

with the structure:

ParkinsonDataset/
├── Turning dataset/
│   ├── PDFE*.mp4
│   └── ...
│
└── Youtube dataset/
    ├── Negative/
    │   ├── *.mp4
    │   └── ...
    │
    └── Positive/
        ├── *.mp4
        └── ...

Data Processing

The videos are processed into frame sequences for visual feature extraction.

The preprocessing pipeline includes:

Loading video files.

Extracting video frames.

Resizing frames to the required input resolution.

Normalizing visual inputs.

Extracting audio from videos where available.

Preparing acoustic representations for the audio branch.

Constructing temporal sequences for recurrent networks.

Passing the resulting representations through the multimodal architecture.

The model therefore operates on both spatial and temporal information rather than treating a video as a single static image.

Proposed Architecture

                    VIDEO INPUT
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
        EfficientNetV2         MobileNetV3
              │                     │
              └──────────┬──────────┘
                         │
                  Visual Features
                         │
                         ▼
                     Bi-GRU
                         │
                         ▼
                Attention Bi-LSTM
                         │
                         ▼
                  Temporal Features

                    AUDIO INPUT
                         │
                         ▼
                 Audio Features
                         │
                         └──────────────┐
                                        │
                                        ▼
                              Adaptive Fusion
                                        │
                                        ▼
                            Fused Feature Vector
                                        │
                         ┌──────────────┴──────────────┐
                         │                             │
                         ▼                             ▼
                     Softmax                      XGBoost
                   Baseline Model              Optimized Model
                         │                             │
                         └──────────────┬──────────────┘
                                        ▼
                           Parkinson’s Classification

EfficientNetV2

EfficientNetV2 is used as a major visual feature extractor.

Its role is to learn discriminative spatial representations from video frames while maintaining an efficient architecture.

The extracted visual representations capture appearance and movement-related characteristics that can contribute to Parkinson’s disease classification.

MobileNetV3

MobileNetV3 provides a complementary visual representation.

Using a second visual backbone allows the system to learn different characteristics from the same video input instead of depending on a single feature extractor.

The EfficientNetV2 and MobileNetV3 representations are subsequently incorporated into the temporal and fusion stages of the framework.

Bidirectional GRU

The Bidirectional GRU (Bi-GRU) is used to model temporal dependencies across the extracted video representations.

Because the recurrent network operates bidirectionally, information from both forward and backward temporal directions can contribute to the learned representation.

This is useful for video-based analysis where clinically relevant movement patterns can occur across multiple frames.

Attention-based Bidirectional LSTM

The Attention Bi-LSTM extends temporal modeling by allowing the architecture to assign greater importance to informative temporal representations.

Instead of treating every temporal representation equally, the attention mechanism helps emphasize relevant portions of the sequence.

The resulting representation is used as part of the multimodal feature representation before adaptive fusion.

Adaptive Fusion

The multimodal representations are combined using an Adaptive Fusion mechanism.

The purpose of adaptive fusion is to learn how different representations should contribute to the final fused representation.

Conceptually:

Visual Features
      +
Temporal Features
      +
Audio Features
      │
      ▼
Adaptive Fusion
      │
      ▼
Fused Deep Representation

The fused representation provides the final classifier with a compact representation containing information learned from multiple branches.

Classification

Two classification configurations are evaluated.

Softmax Baseline

The baseline system uses a Softmax classification layer after adaptive fusion.

Deep Features
     ↓
Adaptive Fusion
     ↓
Softmax
     ↓
Positive / Negative

This configuration provides the baseline against which the optimized classifier is evaluated.

XGBoost Classifier

The optimized configuration replaces the Softmax classifier with XGBoost.

Deep Features
     ↓
Adaptive Fusion
     ↓
Fused Feature Vector
     ↓
XGBoost
     ↓
Positive / Negative

XGBoost is used because the final fused representation is a structured feature vector, allowing a tree-based model to learn nonlinear decision boundaries.

Optuna Hyperparameter Optimization

Optuna is used to optimize the XGBoost classifier.

The optimization uses the Tree-structured Parzen Estimator (TPE) approach.

The optimization process searches for suitable XGBoost hyperparameters while incorporating overfitting control.

The optimized classifier is then evaluated on the validation and test sets.

Experimental Configurations

Baseline Configuration

EfficientNetV2
        +
MobileNetV3
        +
Bi-GRU
        +
Attention Bi-LSTM
        +
Adaptive Fusion
        +
Softmax

Optimized Configuration

EfficientNetV2
        +
MobileNetV3
        +
Bi-GRU
        +
Attention Bi-LSTM
        +
Adaptive Fusion
        +
XGBoost

Evaluation Metrics

The following metrics are reported for model evaluation.

Metric

Description

Accuracy

Overall proportion of correctly classified samples

Precision

Proportion of predicted positive samples that are actually positive

Recall

Proportion of actual positive samples correctly identified

F1-score

Harmonic mean of Precision and Recall

FPR

Proportion of negative samples incorrectly classified as positive

AUROC

Area under the Receiver Operating Characteristic curve

Loss

Model classification loss

Time

Computation time for the corresponding evaluation stage

Experimental Results

Baseline — Softmax

Split

Loss

Accuracy

Precision

Recall

F1-score

FPR

AUROC

Time (s)

Train

0.1237

0.9718

0.9849

0.9609

0.9723

0.0167

0.9923

77.01

Validation

0.3634

0.9217

0.9310

0.9153

0.9231

0.0714

0.9885

11.69

Test

0.2098

0.9569

0.9825

0.9330

0.9573

0.0179

0.9952

344.94

Optimized — XGBoost

Split

Loss

Accuracy

Precision

Recall

F1-score

FPR

AUROC

Time (s)

Train

0.1237

0.9718

0.9849

0.9609

0.9723

0.0167

0.9923

97.10

Validation

0.3634

0.9217

0.9310

0.9153

0.9231

0.0714

0.9660

9.67

Test

0.2098

0.9569

0.9825

0.9330

0.9573

0.0179

0.9876

15.16

Ablation Study

An ablation study was performed to investigate the contribution of the major components of the proposed architecture.

The full model is compared against configurations where individual components are removed.

Ablation Configurations

Configuration

Architecture

No EfficientNetV2

MobileNetV3 + Bi-GRU + Attention Bi-LSTM + Adaptive Fusion + XGBoost

No MobileNetV3

EfficientNetV2 + Bi-GRU + Attention Bi-LSTM + Adaptive Fusion + XGBoost

No Bi-GRU

EfficientNetV2 + MobileNetV3 + Attention Bi-LSTM + Adaptive Fusion + XGBoost

No Attention Bi-LSTM

EfficientNetV2 + MobileNetV3 + Bi-GRU + Adaptive Fusion + XGBoost

No XGBoost

EfficientNetV2 + MobileNetV3 + Bi-GRU + Attention Bi-LSTM + Adaptive Fusion + Softmax

Full Model

EfficientNetV2 + MobileNetV3 + Bi-GRU + Attention Bi-LSTM + Adaptive Fusion + XGBoost

Ablation Results

Configuration

Accuracy

Precision

Recall

F1-score

FPR

AUROC

Loss

Time (s)

No EfficientNetV2

0.9310

0.9483

0.9167

0.9322

0.0536

0.9866

0.1780

37.52

No MobileNetV3

0.9224

0.9048

0.9500

0.9268

0.1071

0.9656

0.2969

50.25

No Bi-GRU

0.9310

0.9330

0.9330

0.9330

0.0714

0.9896

0.1645

107.38

No Attention Bi-LSTM

0.8879

0.8852

0.9000

0.8926

0.1250

0.9679

0.2672

242.76

No XGBoost

0.9224

0.9180

0.9333

0.9256

0.0893

0.9845

0.2287

140.38

Full Model

0.9310

0.9483

0.9167

0.9322

0.0536

0.9911

0.2187

105.73

Statistical Analysis

Multiple experimental runs were considered for comparing optimized and unoptimized configurations.

The statistical analysis focuses on:

Accuracy

AUROC

Mean performance

Standard deviation

Statistical significance testing

A Student's t-test is used to compare the optimized XGBoost configuration against the unoptimized Softmax configuration.

The reported analysis concluded that the difference was not statistically significant, because the best performance was already achieved by the unoptimized configuration.

Therefore, the optimization did not demonstrate a statistically significant improvement over the baseline in the reported experiments.

Explainability — SHAP

SHAP (SHapley Additive exPlanations) was considered for interpreting the XGBoost classifier.

The XGBoost classifier operates on the final fused representation rather than directly on raw video frames or audio spectrograms.

The fused representation contains 256 features generated by the preceding neural-network components.

Therefore, TreeSHAP can provide feature importance for the fused feature vector:

Video Frames ──► Neural Feature Extractors ──┐
                                             │
Audio ─────────► Neural Feature Extractors ──┤
                                             ▼
                                      Adaptive Fusion
                                             │
                                             ▼
                                   256-D Fused Features
                                             │
                                             ▼
                                          XGBoost
                                             │
                                             ▼
                                          SHAP

The SHAP explanation therefore describes the contribution of the fused features to the XGBoost prediction.

Directly mapping these SHAP values back to individual pixels, frequency bands, or specific time points is not straightforward because the fused features have passed through multiple nonlinear transformations, including EfficientNetV2, MobileNetV3, Bi-LSTM, Bi-GRU, and Adaptive Fusion.

End-to-End Pipeline

                    Raw Parkinson's Videos
                             │
                             ▼
                      Video Preprocessing
                             │
                  ┌──────────┴──────────┐
                  │                     │
                  ▼                     ▼
             Video Frames             Audio
                  │                     │
                  ▼                     ▼
            EfficientNetV2        Audio Features
                  │
                  ├──────────────┐
                  │              │
                  ▼              ▼
            MobileNetV3       Temporal Modeling
                                  │
                           ┌──────┴──────┐
                           ▼             ▼
                        Bi-GRU      Attention Bi-LSTM
                           │             │
                           └──────┬──────┘
                                  │
                                  ▼
                           Adaptive Fusion
                                  │
                                  ▼
                         Fused Deep Features
                                  │
                         ┌────────┴────────┐
                         │                 │
                         ▼                 ▼
                      Softmax           XGBoost
                       Baseline         Optimized
                         │                 │
                         └────────┬────────┘
                                  ▼
                         Parkinson's Prediction
                                  │
                                  ▼
                   Evaluation + Explainability

Project Structure

A recommended repository structure is:

multimodal-efficientnet-parkinsons/
│
├── README.md
│
├── data/
│   ├── youtube/
│   │   ├── Positive/
│   │   └── Negative/
│   │
│   └── turning/
│
├── notebooks/
│   ├── data_analysis/
│   ├── preprocessing/
│   ├── baseline/
│   ├── optimization/
│   ├── ablation/
│   ├── statistical_analysis/
│   └── explainability/
│
├── src/
│   ├── preprocessing/
│   ├── models/
│   │   ├── efficientnet.py
│   │   ├── mobilenet.py
│   │   ├── bigru.py
│   │   ├── attention_bilstm.py
│   │   └── fusion.py
│   │
│   ├── classifiers/
│   │   ├── softmax.py
│   │   └── xgboost.py
│   │
│   └── evaluation/
│
├── checkpoints/
│
├── results/
│   ├── baseline/
│   ├── optimized/
│   ├── ablation/
│   ├── statistical/
│   └── shap/
│
└── requirements.txt

Reproducibility

For reproducible experimentation:

Keep the raw datasets unchanged.

Maintain fixed train, validation, and test splits.

Record random seeds for every experiment.

Save model checkpoints after training.

Save extracted fused features when possible.

Store optimization results and selected hyperparameters.

Store all evaluation metrics.

Preserve ablation configurations.

Save statistical-analysis outputs.

Save SHAP results and plots.

This allows the analysis to be recovered without unnecessarily repeating expensive feature extraction and model training.

Key Findings

The reported experiments show that the multimodal architecture achieves strong classification performance.

The baseline Softmax configuration achieved:

Test Accuracy: 95.69%

Test Precision: 98.25%

Test Recall: 93.30%

Test F1-score: 95.73%

Test FPR: 1.79%

Test AUROC: 99.52%

The optimized XGBoost configuration achieved:

Test Accuracy: 95.69%

Test Precision: 98.25%

Test Recall: 93.30%

Test F1-score: 95.73%

Test FPR: 1.79%

Test AUROC: 98.76%

The ablation study indicates that removing individual components changes the overall performance, with the removal of Attention Bi-LSTM producing the largest reduction in Accuracy and F1-score among the reported ablation configurations.

The statistical analysis reported that the difference between the optimized and unoptimized configurations was not statistically significant.

Strengths

Multimodal representation learning

Complementary visual feature extraction

Bidirectional temporal modeling

Attention-based temporal representation

Adaptive feature fusion

Nonlinear XGBoost classification

Hyperparameter optimization using Optuna

Comprehensive evaluation using multiple classification metrics

Ablation-based component analysis

SHAP-based interpretation of fused XGBoost features

Limitations

The SHAP analysis explains the final fused feature vector rather than directly explaining individual video pixels or audio-frequency components.

The fused representation is produced through multiple nonlinear neural-network transformations, making direct attribution to the original raw modalities difficult.

The reported statistical analysis did not demonstrate a significant performance improvement from optimization over the baseline.

Dataset characteristics and modality availability should be considered when interpreting the generalizability of the results.

Research Summary

This project presents a multimodal Parkinson’s disease detection framework that combines deep visual representation learning, temporal sequence modeling, adaptive multimodal fusion, and nonlinear classification.

The architecture combines:

EfficientNetV2
      +
MobileNetV3
      +
Bi-GRU
      +
Attention Bi-LSTM
      +
Adaptive Fusion
      +
Softmax / XGBoost

The baseline Softmax model provides a strong reference point, while XGBoost is investigated as an alternative classifier operating on the fused deep representation.

The framework is further evaluated through ablation experiments, statistical comparison, and SHAP-based interpretation of the fused features.

Technologies

Python

PyTorch

TorchVision

Scikit-learn

XGBoost

Optuna

SHAP

NumPy

Pandas

Matplotlib

Google Colab

Google Drive
---

## Citation

If you use this repository in your research, please cite the associated publication:

> Rajavel R., Sankaranarayanan S., Dhanushree D., *"Vision-based Parkinson Disease Recognition and Severity Assessment using Multimodal Feature Fusion and Deep Learning."* (Add journal details after publication.)

---

## Acknowledgement

This implementation was developed for research purposes to advance intelligent healthcare systems for early Parkinson disease recognition and severity assessment. The framework utilizes publicly available datasets, including the YouTubePD dataset (https://uiuc-yuxiong-lab.github.io/YouTubePD/) and the Turning-in-Place Parkinson Disease dataset (https://figshare.com/articles/dataset/A_public_dataset_of_video_acceleration_and_angular_velocity_in_individuals_with_Parkinson_s_disease_during_the_turning-in-place_task/14984667), together with open-source machine learning libraries such as PyTorch, XGBoost, and Optuna. We gratefully acknowledge the creators of these datasets and software tools for enabling reproducible research in artificial intelligence for healthcare.
