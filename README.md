# VPDRSA-Multimodal-Parkinson-Recognition-Framework
This repository contains the official implementation of the Vision-based Parkinson Disease Recognition and Severity Assessment (VPDRSA) framework proposed in our research.\
The framework presents a multimodal deep learning architecture for automated Parkinson disease (PD) recognition by integrating complementary visual and audio information extracted from publicly available datasets.\
The proposed approach combines advanced convolutional neural networks, temporal sequence modeling, multimodal feature fusion, hyperparameter optimization, and ensemble machine learning to achieve robust and reliable PD detection.\
The implementation is developed using Python and PyTorch, with Optuna employed for hyperparameter optimization and XGBoost for final classification. The repository is intended to facilitate reproducible research and serves as a reference implementation for researchers working on computer vision, speech processing, multimodal learning, and intelligent healthcare systems.

Framework Overview\
The proposed VPDRSA framework consists of the following major components:
•	Multimodal Data Acquisition 
o	Video data from the YouTubePD dataset. 
o	Turning-in-Place gait videos. 
o	Speech recordings for audio analysis. 
•	Data Preprocessing 
o	Video frame extraction and resizing. 
o	Temporal frame standardization. 
o	Image normalization. 
o	MFCC feature extraction from speech signals. 
o	Dataset cleaning and train-validation-test splitting. 
•	Visual Feature Extraction 
o	EfficientNetV2-S for extracting gait and facial movement features from YouTubePD videos. 
o	MobileNetV3-Small for learning turning and freezing-of-gait representations. 
•	Audio Feature Extraction 
o	Bidirectional GRU encoder for modeling temporal speech characteristics using MFCC features. 
•	Temporal Modeling 
o	Temporal Attention Bidirectional LSTM for capturing long-range temporal dependencies in visual sequences. 
•	Multimodal Feature Fusion 
o	Feature-level concatenation of visual and audio embeddings. 
o	Fully connected projection layer with dropout regularization. 
•	Hyperparameter Optimization 
o	Optuna Tree-structured Parzen Estimator (TPE). 
o	Automatic optimization of learning rate, dropout, hidden dimensions, batch size, weight decay, and XGBoost parameters. 
•	Classification 
o	XGBoost classifier trained on optimized multimodal latent features. 
o	Binary Parkinson disease recognition. 
•	Performance Evaluation 
o	Accuracy 
o	Precision 
o	Recall 
o	F1-score 
o	AUROC 
o	False Positive Rate 
o	Confusion Matrix 
o	ROC Curve 
o	Precision–Recall Curve 
o	Stratified Five-Fold Cross-Validation 
o	Paired t-test for statistical significance 
________________________________________
Repository Contents
The repository includes the complete implementation required to reproduce the experiments presented in the manuscript, including
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
├── Visualization utilities
└── Example inference notebook
________________________________________
Experimental Configuration
The implementation follows the experimental settings reported in the manuscript, including
•	Image size: 224 × 224 
•	Video length: 16 frames 
•	Audio representation: 40-dimensional MFCC 
•	Audio sequence length: 300 frames 
•	Batch size: 4 
•	Mixed precision training 
•	Gradient clipping 
•	Optuna-TPE optimization 
•	XGBoost classifier 
•	Stratified 5-fold cross-validation 
________________________________________
Reproducibility
The repository has been developed to ensure reproducibility of the experimental results reported in the associated publication. The provided scripts allow users to preprocess the datasets, train the multimodal deep learning model, perform Optuna-based hyperparameter optimization, train the XGBoost classifier, evaluate the trained models, and reproduce the reported quantitative performance metrics.
________________________________________
Citation
If you use this repository in your research, please cite the associated publication:
Rajavel R., Sankaranarayanan S., Dhanushree S., "Vision-based Parkinson Disease Recognition and Severity Assessment using Multimodal Feature Fusion and Deep Learning." (Add journal details after publication.)
________________________________________
Dataset Used:
This implementation was developed for research purposes to advance intelligent healthcare systems for early Parkinson disease recognition and severity assessment. The framework utilizes publicly available datasets, including the YouTubePD dataset (https://uiuc-yuxiong-lab.github.io/YouTubePD/) and the Turning-in-Place Parkinson Disease dataset (https://figshare.com/articles/dataset/A_public_dataset_of_video_acceleration_and_angular_velocity_in_individuals_with_Parkinson_s_disease_during_the_turning-in-place_task/14984667), together with open-source machine learning libraries such as PyTorch, XGBoost, and Optuna. We gratefully acknowledge the creators of these datasets and software tools for enabling reproducible research in artificial intelligence for healthcare.

