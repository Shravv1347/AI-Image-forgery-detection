AI Image Forgery Detection System

A machine learning–based system designed to detect forged or manipulated images by analyzing visual inconsistencies and compression artifacts. The system extracts multiple image features and applies several machine learning models to classify images as authentic or forged.

Project Overview

This project implements an AI-powered image forgery detection pipeline capable of identifying tampered images using feature analysis and machine learning classification.

The system evaluates:

Error Level Analysis (ELA) patterns

Noise inconsistencies

Color distribution anomalies

Texture characteristics

Compression artifacts

Image metadata features

Multiple models are trained and evaluated to achieve reliable detection performance.

Core Features
Feature Engineering

The system extracts and processes multiple image characteristics, including:

Error Level Analysis statistics

Noise distribution features

Color channel statistics

Texture and gradient features

Image dimension and metadata properties

Feature preprocessing includes:

Feature normalization and scaling

Dimensionality reduction

Feature selection methods

Feature aggregation and transformation

Machine Learning Models Implemented

The system supports multiple classification approaches:

Logistic Regression

Support Vector Machine (SVM)

K-Nearest Neighbors

Decision Tree

Random Forest

Naive Bayes

Gradient Boosting

Ensemble voting classifier

Each model is evaluated and compared to select optimal performance.

Evaluation Metrics

Models are evaluated using standard classification metrics:

Accuracy

Precision

Recall

F1 Score

ROC-AUC Score

Confusion Matrix

Probability confidence estimates

Performance plots and comparison visuals are automatically generated.
.
├── requirements.txt
├── README.md
│
├── feature_extraction.py
├── classification_models.py
├── regression_linear_models.py
├── distance_based_models.py
├── tree_rule_models.py
├── probabilistic_models.py
│
├── dataset_preparation.py
├── main_training.py
├── predict_forgery.py
│
└── dataset/
    ├── authentic/
    └── forged/

Installation
Requirements

Python 3.8 or higher

pip package manager

Available models include:

Logistic Regression

SVM

KNN

Decision Tree

Random Forest

Naive Bayes

Gradient Boosting

Ensemble

Output Files Generated

Training produces:

Model comparison plots

Confusion matrices

ROC curves

Serialized trained system file

Cached feature files

Batch prediction results

Technical Details
Feature Categories

The system extracts over 40 features including:

ELA Features
Statistical analysis of compression differences.

Noise Features
Noise distribution, variance, entropy, and energy.

Color Features
Statistical characteristics of RGB and HSV channels.

Texture Features
Gradient magnitude and edge density measures.

Metadata Features
Image dimensions and aspect ratio characteristics.