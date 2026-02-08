# AI Image Forgery Detection System

A comprehensive Machine Learning project implementing concepts from the complete ML syllabus for detecting forged/manipulated images.

## 📚 Project Overview

This project implements an AI-powered image forgery detection system that uses multiple machine learning algorithms to identify whether an image has been tampered with or manipulated. The system analyzes various features such as:

- Error Level Analysis (ELA) patterns
- Noise inconsistencies
- Color distribution anomalies
- Texture patterns
- Compression artifacts

## 🎓 Syllabus Concepts Implemented

### Unit 1: Introduction to Machine Learning

#### 1.1 - Fundamentals
- ✅ Machine Learning vs Designing
- ✅ Training vs Testing data splits
- ✅ Characteristics of ML (Predictive task)
- ✅ Binary Classification problem

#### 1.2 - Models and Features
- ✅ **Feature Extraction**: ELA features, noise patterns, color statistics, texture features
- ✅ **Feature Construction**: Gradient features, statistical aggregations
- ✅ **Feature Transformation**: Normalization (StandardScaler, MinMaxScaler), PCA
- ✅ **Feature Selection**:
  - Filter Method (SelectKBest with ANOVA F-test)
  - Wrapper Method (Recursive Feature Elimination)
  - Embedded Method (Random Forest feature importance)
- ✅ **Label Encoding**: Binary labels (authentic/forged)

#### 1.3 - Classification
- ✅ Binary Classification for forgery detection
- ✅ Performance Assessment:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix
  - ROC-AUC Score
  - Specificity, FPR, FNR
- ✅ Class Probability Estimation
- ✅ Cross-validation

### Unit 2: Regression and Linear Models

#### 2.1 - Regression
- ✅ Error measures (MSE, RMSE, MAE, R²)
- ✅ Overfitting/Underfitting analysis
- ✅ Bias-Variance tradeoff
- ✅ Polynomial Regression

#### 2.2 - Linear Models
- ✅ **Logistic Regression**: Multivariate classification
- ✅ **Regularized Regression**: Ridge, Lasso, ElasticNet
- ✅ **Perceptron**: Basic neural classifier
- ✅ **Support Vector Machines**:
  - Linear SVM
  - Kernel SVM (RBF, Polynomial)
  - Soft Margin SVM

#### 2.3 - Distance Based Models
- ✅ **K-Nearest Neighbors (KNN)**:
  - Classification
  - Optimal k selection
- ✅ **K-Means Clustering**:
  - Elbow method
  - Silhouette analysis
- ✅ **Hierarchical Clustering**:
  - Dendrogram visualization
- ✅ **DBSCAN**:
  - Density-based clustering
  - Noise detection

### Unit 3: Tree-Based and Probabilistic Models

#### 3.1 - Rule Based Models
- ✅ Association Rule Mining (Apriori algorithm demonstration)

#### 3.2 - Tree Based Models
- ✅ **Decision Trees**:
  - ID3-inspired implementation
  - Tree visualization
  - Feature importance
- ✅ **Random Forest**:
  - Ensemble of decision trees
  - Feature importance ranking
- ✅ **Regression Trees**
- ✅ **Gradient Boosting**

#### 3.3 - Probabilistic Models
- ✅ **Naive Bayes Classifier**:
  - Gaussian Naive Bayes
  - Class probability estimation
- ✅ **Normal Distribution**:
  - Parameter estimation
  - Geometric interpretations
  - Normality testing
- ✅ **Discriminative Learning**:
  - Linear Discriminant Analysis (LDA)
  - Quadratic Discriminant Analysis (QDA)
  - Maximum Likelihood estimation

## 📁 Project Structure

```
.
├── requirements.txt                    # Python dependencies
├── README.md                          # This file
│
├── feature_extraction.py              # Feature extraction & selection (Unit 1.2)
├── classification_models.py           # Binary/Multiclass classification (Unit 1.3)
├── regression_linear_models.py        # Regression & Linear models (Unit 2.1, 2.2)
├── distance_based_models.py           # KNN, K-Means, DBSCAN (Unit 2.3)
├── tree_rule_models.py               # Decision Trees, Random Forest (Unit 3.2)
├── probabilistic_models.py            # Naive Bayes, Normal Dist. (Unit 3.3)
│
├── dataset_preparation.py             # Dataset handling and preprocessing
├── main_training.py                   # Main training pipeline
├── predict_forgery.py                 # Prediction script for new images
│
└── dataset/                           # Image dataset
    ├── authentic/                     # Authentic images
    └── forged/                        # Forged images
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download this project

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### 1. Train the System

Run the main training script to:
- Create/load the dataset
- Extract features
- Train multiple ML models
- Evaluate performance
- Save the trained system

```bash
python main_training.py
```

This will:
- Generate a dataset with 50 authentic and 50 forged sample images
- Extract 40+ features from each image
- Train 7 different models
- Create performance comparison plots
- Save the trained system

### 2. Predict Single Image

Use the trained model to predict if a new image is forged:

```bash
python predict_forgery.py path/to/image.jpg
```

Example:
```bash
python predict_forgery.py dataset/forged/forged_000.jpg
```

### 3. Batch Prediction

Process multiple images in a directory:

```bash
python predict_forgery.py --batch path/to/directory/
```

This will:
- Process all images in the directory
- Generate predictions for each
- Save results to `batch_predictions.csv`

### 4. Use Specific Model

By default, predictions use the Ensemble model. To use a specific model:

```bash
python predict_forgery.py image.jpg --model "Random Forest"
```

Available models:
- Logistic Regression
- SVM
- KNN (K-Nearest Neighbors)
- Decision Tree
- Random Forest
- Naive Bayes
- Gradient Boosting
- Ensemble

## 📊 Output Files

After training, the system generates:

1. **model_comparison.png** - Visual comparison of all models
2. **confusion_matrix_*.png** - Confusion matrices for each model
3. **roc_curve_*.png** - ROC curves showing model performance
4. **forgery_detection_system.pkl** - Saved trained system
5. **features_cache.pkl** - Cached extracted features
6. **batch_predictions.csv** - Results from batch predictions

## 🔬 Technical Details

### Feature Extraction (40+ features)

#### Error Level Analysis (ELA) Features (8)
- Mean, Std, Max, Min, Median, Variance, Skewness, Kurtosis

#### Noise Pattern Features (5)
- Noise mean, std, variance, energy, entropy

#### Color Features (15)
- RGB channel statistics (mean, std, skewness, kurtosis)
- HSV component statistics

#### Texture Features (6)
- Gradient magnitude statistics
- Edge density
- Texture contrast and homogeneity

#### Metadata Features (5)
- Dimensions, aspect ratio, pixel count

### Models Trained

1. **Logistic Regression** - Linear probabilistic classifier
2. **Support Vector Machine** - Kernel-based margin maximizer
3. **K-Nearest Neighbors** - Instance-based learner
4. **Decision Tree** - Rule-based hierarchical classifier
5. **Random Forest** - Ensemble of decision trees
6. **Naive Bayes** - Probabilistic classifier
7. **Gradient Boosting** - Sequential ensemble method
8. **Ensemble** - Voting combination of top models

### Performance Metrics

All models are evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: Correct positive predictions
- **Recall**: Coverage of actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **Confusion Matrix**: Detailed prediction breakdown

## 🎯 Example Output

```
================================================================================
PREDICTION RESULTS
================================================================================
✗ Image appears to be FORGED

Probability Distribution:
  Authentic: 15.23%
  Forged:    84.77%

Confidence: 84.77%

Interpretation:
  High confidence in prediction
================================================================================
```

## 🔧 Customization

### Adjust Dataset Size

In `main_training.py`, modify:
```python
system.step1_prepare_dataset(n_authentic=100, n_forged=100)
```

### Change Feature Selection Method

In `main_training.py`, change method:
```python
system.step2_feature_engineering(method='embedded', k_features=25)
```

Options: `'filter'`, `'wrapper'`, `'embedded'`

### Modify Model Parameters

Edit the model initialization in `main_training.py`, for example:
```python
rf_model = RandomForestModel(n_estimators=200, max_depth=20)
```

## 📖 Using with Real Datasets

To use your own forgery dataset:

1. Organize images:
```
dataset/
  authentic/
    img1.jpg
    img2.jpg
    ...
  forged/
    img1.jpg
    img2.jpg
    ...
```

2. Run training:
```bash
python main_training.py
```

The system will automatically detect and use your images.

## 🎓 Learning Resources

This project implements concepts from:
- Machine Learning fundamentals
- Feature engineering
- Classification algorithms
- Ensemble methods
- Performance evaluation

Each module (`feature_extraction.py`, `classification_models.py`, etc.) includes detailed docstrings explaining the implemented concepts.

## ⚠️ Important Notes

1. **Sample Dataset**: The default dataset uses synthetic forgeries for demonstration. For production use, obtain real forgery datasets.

2. **Feature Extraction**: ELA and noise analysis work best on JPEG images due to compression analysis.

3. **Model Selection**: The Ensemble model typically provides the best performance by combining multiple algorithms.

4. **Confidence Threshold**: For critical applications, consider manual review for predictions with <75% confidence.

## 🐛 Troubleshooting

**Error: Cannot read image**
- Ensure image format is supported (.jpg, .png, .jpeg)
- Check file permissions

**Error: Model file not found**
- Run `python main_training.py` first to train the system

**Low accuracy**
- Increase dataset size
- Adjust feature selection parameters
- Try different models

## 📝 License

This project is for educational purposes as part of a Machine Learning course.

## 👨‍💻 Author

Created as a comprehensive ML project implementing the complete syllabus concepts.

## 🙏 Acknowledgments

- Scikit-learn library for ML implementations
- OpenCV for image processing
- Course syllabus for comprehensive ML concept coverage
