# stress-analysis
Human Stress Classification Based on HRV Using Machine Learning
This repository contains a machine learning pipeline for classifying human stress levels using Heart Rate Variability (HRV) features extracted from physiological signals. The project explores and compares different ML models for accurate stress prediction, aiming to contribute to health monitoring and mental well-being solutions.

📌 Project Overview
Stress is a major factor affecting physical and mental health. Heart Rate Variability (HRV) is a physiological marker that reflects stress-related changes in the autonomic nervous system. This project leverages HRV data and machine learning algorithms to classify human stress levels into various categories (e.g., stressed vs. non-stressed).

🧠 Objectives
Extract meaningful HRV features from ECG or PPG data

Train and evaluate different machine learning models

Identify the best-performing model for stress classification

Analyze feature importance and model interpretability

📁 Dataset
Source: [e.g., WESAD, Stress Recognition in Automobile Drivers, or custom-collected data]

Data Includes:

RR intervals

Time-domain features (e.g., RMSSD, SDNN)

Frequency-domain features (e.g., LF, HF, LF/HF ratio)

Label (e.g., stressed, relaxed)

Note: You may need to preprocess raw ECG/PPG signals using a tool like NeuroKit2 or HRVAnalysis before using this repo.

🧰 Tools & Technologies
Python 3.8+

Libraries:

NumPy, Pandas, Matplotlib, Seaborn

Scikit-learn, XGBoost, LightGBM

HRVAnalysis or NeuroKit2 (for HRV extraction)

🚀 Features
HRV feature extraction from RR intervals

Binary and multi-class stress classification

Support for multiple ML algorithms:

Logistic Regression

Random Forest

Support Vector Machines

K-Nearest Neighbors

Gradient Boosting (XGBoost, LightGBM)

Cross-validation and performance metrics

Confusion matrix and ROC curve visualization

📊 Evaluation Metrics
Accuracy

Precision

Recall

F1-Score

ROC-AUC


