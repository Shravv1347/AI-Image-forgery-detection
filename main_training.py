"""
Main Training Script for AI Image Forgery Detection System
Integrates all ML concepts from the syllabus
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import all modules
from dataset_preparation import ForgeryDatasetPreparer
from feature_extraction import FeatureSelector, FeatureTransformer
from classification_models import BinaryClassifier
from regression_linear_models import LinearModelsForClassification
from distance_based_models import KNNClassifier
from tree_rule_models import DecisionTreeModel, RandomForestModel
from probabilistic_models import NaiveBayesClassifier

# Sklearn models
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier

import matplotlib.pyplot as plt
import seaborn as sns
import pickle


class ImageForgeryDetectionSystem:
    """
    Complete AI Image Forgery Detection System
    Implements concepts from all units of the syllabus
    """
    
    def __init__(self):
        self.data_preparer = ForgeryDatasetPreparer()
        self.feature_selector = None
        self.feature_transformer = None
        self.models = {}
        self.results = {}
        
    def step1_prepare_dataset(self, n_authentic=50, n_forged=50, use_existing=False):
        """
        Step 1: Dataset Preparation
        (Unit 1.1: Learning versus Designing, Training versus Testing)
        """
        print("\n" + "="*80)
        print("STEP 1: DATASET PREPARATION")
        print("="*80)
        
        if not use_existing:
            self.data_preparer.create_sample_dataset(n_authentic, n_forged)
        
        # Extract features
        features_df, labels, image_paths = self.data_preparer.load_and_extract_features()
        
        # Prepare train/val/test split
        X_train, X_val, X_test, y_train, y_val, y_test = \
            self.data_preparer.prepare_train_test_split(features_df, labels)
        
        # Store data
        self.X_train_raw = X_train
        self.X_val_raw = X_val
        self.X_test_raw = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.feature_names = list(features_df.columns)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def step2_feature_engineering(self, method='filter', k_features=25):
        """
        Step 2: Feature Selection and Transformation
        (Unit 1.2: Feature Selection Techniques - Filter, Wrapper, Embedded)
        """
        print("\n" + "="*80)
        print("STEP 2: FEATURE ENGINEERING")
        print("="*80)
        
        # Feature Selection
        self.feature_selector = FeatureSelector()
        
        print(f"\nApplying {method.upper()} method for feature selection...")
        if method == 'filter':
            X_train_selected = self.feature_selector.filter_method(
                self.X_train_raw, self.y_train, k=k_features
            )
        elif method == 'wrapper':
            X_train_selected = self.feature_selector.wrapper_method(
                self.X_train_raw, self.y_train, n_features=k_features
            )
        elif method == 'embedded':
            X_train_selected, importances = self.feature_selector.embedded_method(
                self.X_train_raw, self.y_train
            )
        else:
            raise ValueError("Method must be 'filter', 'wrapper', or 'embedded'")
        
        # Transform validation and test sets
        X_val_selected = self.feature_selector.transform(self.X_val_raw)
        X_test_selected = self.feature_selector.transform(self.X_test_raw)
        
        # Feature Transformation (Normalization)
        self.feature_transformer = FeatureTransformer()
        
        print("\nApplying feature normalization...")
        self.X_train = self.feature_transformer.normalize_features(
            X_train_selected, method='standard'
        )
        self.X_val = self.feature_transformer.transform(X_val_selected)
        self.X_test = self.feature_transformer.transform(X_test_selected)
        
        print(f"\nFeature engineering completed!")
        print(f"Original features: {self.X_train_raw.shape[1]}")
        print(f"Selected features: {self.X_train.shape[1]}")
        
        return self.X_train, self.X_val, self.X_test
    
    def step3_train_models(self):
        """
        Step 3: Train Multiple Classification Models
        Implements models from Units 1, 2, and 3
        """
        print("\n" + "="*80)
        print("STEP 3: TRAINING CLASSIFICATION MODELS")
        print("="*80)
        
        # 1. Logistic Regression (Unit 2.2)
        print("\n[1/7] Training Logistic Regression...")
        lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self.models['Logistic Regression'] = BinaryClassifier(lr_model, "Logistic Regression")
        self.models['Logistic Regression'].train(self.X_train, self.y_train)
        
        # 2. Support Vector Machine (Unit 2.2)
        print("\n[2/7] Training Support Vector Machine...")
        svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
        self.models['SVM'] = BinaryClassifier(svm_model, "SVM (RBF Kernel)")
        self.models['SVM'].train(self.X_train, self.y_train)
        
        # 3. K-Nearest Neighbors (Unit 2.3)
        print("\n[3/7] Training K-Nearest Neighbors...")
        knn_model = KNNClassifier(n_neighbors=5)
        knn_model.train(self.X_train, self.y_train)
        self.models['KNN'] = BinaryClassifier(knn_model.model, "K-Nearest Neighbors")
        self.models['KNN'].is_trained = True
        
        # 4. Decision Tree (Unit 3.2)
        print("\n[4/7] Training Decision Tree...")
        dt_model = DecisionTreeModel(task='classification', max_depth=10)
        dt_model.train(self.X_train, self.y_train)
        self.models['Decision Tree'] = BinaryClassifier(dt_model.model, "Decision Tree")
        self.models['Decision Tree'].is_trained = True
        
        # 5. Random Forest (Unit 3.2)
        print("\n[5/7] Training Random Forest...")
        rf_model = RandomForestModel(n_estimators=100, max_depth=15)
        rf_model.train(self.X_train, self.y_train)
        self.models['Random Forest'] = BinaryClassifier(rf_model.model, "Random Forest")
        self.models['Random Forest'].is_trained = True
        
        # 6. Naive Bayes (Unit 3.3)
        print("\n[6/7] Training Naive Bayes...")
        nb_model = NaiveBayesClassifier(variant='gaussian')
        nb_model.train(self.X_train, self.y_train)
        self.models['Naive Bayes'] = BinaryClassifier(nb_model.model, "Naive Bayes")
        self.models['Naive Bayes'].is_trained = True
        
        # 7. Gradient Boosting (Advanced ensemble)
        print("\n[7/7] Training Gradient Boosting...")
        gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
                                             max_depth=5, random_state=42)
        self.models['Gradient Boosting'] = BinaryClassifier(gb_model, "Gradient Boosting")
        self.models['Gradient Boosting'].train(self.X_train, self.y_train)
        
        print("\n" + "="*80)
        print("All models trained successfully!")
        print("="*80)
    
    def step4_evaluate_models(self):
        """
        Step 4: Evaluate and Compare Models
        (Unit 1.3: Assessing Classification performance)
        """
        print("\n" + "="*80)
        print("STEP 4: MODEL EVALUATION")
        print("="*80)
        
        for model_name, classifier in self.models.items():
            print(f"\nEvaluating {model_name}...")
            results = classifier.assess_performance(self.X_test, self.y_test, show_plots=False)
            self.results[model_name] = results
        
        # Create comparison table
        self._create_comparison_table()
        
        # Create comparison plots
        self._create_comparison_plots()
    
    def _create_comparison_table(self):
        """Create a comparison table of all models"""
        print("\n" + "="*80)
        print("MODEL COMPARISON TABLE")
        print("="*80)
        
        print(f"\n{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
        print("-" * 93)
        
        for model_name, results in self.results.items():
            roc_auc = results['roc_auc'] if results['roc_auc'] is not None else 0.0
            print(f"{model_name:<25} {results['accuracy']:<12.4f} {results['precision']:<12.4f} "
                  f"{results['recall']:<12.4f} {results['f1_score']:<12.4f} {roc_auc:<12.4f}")
    
    def _create_comparison_plots(self):
        """Create visual comparison of model performances"""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        model_names = list(self.results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics):
            values = [self.results[model][metric] for model in model_names]
            
            axes[idx].barh(model_names, values, color='skyblue', edgecolor='navy')
            axes[idx].set_xlabel(metric.replace('_', ' ').title(), fontsize=12)
            axes[idx].set_xlim([0, 1.0])
            axes[idx].grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(values):
                axes[idx].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10)
        
        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('./model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\nModel comparison plot saved as model_comparison.png")
    
    def step5_create_ensemble(self):
        """
        Step 5: Create Ensemble Model
        Combining multiple models for better performance
        """
        print("\n" + "="*80)
        print("STEP 5: CREATING ENSEMBLE MODEL")
        print("="*80)
        
        # Create voting classifier with top performing models
        voting_clf = VotingClassifier(
            estimators=[
                ('rf', self.models['Random Forest'].model),
                ('svm', self.models['SVM'].model),
                ('gb', self.models['Gradient Boosting'].model)
            ],
            voting='soft'
        )
        
        print("\nTraining Ensemble Model (Voting Classifier)...")
        self.models['Ensemble'] = BinaryClassifier(voting_clf, "Ensemble (Voting)")
        self.models['Ensemble'].train(self.X_train, self.y_train)
        
        print("\nEvaluating Ensemble Model...")
        ensemble_results = self.models['Ensemble'].assess_performance(
            self.X_test, self.y_test, show_plots=True
        )
        self.results['Ensemble'] = ensemble_results
    
    def predict_single_image(self, image_path, model_name='Ensemble'):
        """
        Predict whether a single image is authentic or forged
        (Demonstrates prediction pipeline)
        """
        print("\n" + "="*80)
        print("SINGLE IMAGE PREDICTION")
        print("="*80)
        print(f"Image: {image_path}")
        print(f"Model: {model_name}")
        
        # Extract features
        features = self.data_preparer.feature_extractor.extract_all_features(image_path)
        
        if features is None:
            print("Error: Could not extract features from image")
            return None
        
        # Convert to array
        import pandas as pd
        features_df = pd.DataFrame([features])
        X = features_df.values
        
        # Apply feature selection and transformation
        X_selected = self.feature_selector.transform(X)
        X_normalized = self.feature_transformer.transform(X_selected)
        
        # Make prediction
        if model_name not in self.models:
            print(f"Error: Model '{model_name}' not found")
            return None
        
        prediction = self.models[model_name].predict(X_normalized)[0]
        probabilities = self.models[model_name].predict_proba(X_normalized)[0]
        
        # Display results
        class_label = self.data_preparer.label_encoder.inverse_transform([prediction])[0]
        
        print("\n" + "-"*80)
        print(f"PREDICTION: {class_label.upper()}")
        print("-"*80)
        print(f"Probability of Authentic: {probabilities[0]:.4f}")
        print(f"Probability of Forged:    {probabilities[1]:.4f}")
        print(f"Confidence: {max(probabilities):.2%}")
        
        return {
            'prediction': class_label,
            'probabilities': probabilities,
            'confidence': max(probabilities)
        }
    
    def save_system(self, filename='forgery_detection_system.pkl'):
        """Save the complete trained system"""
        system_data = {
            'feature_selector': self.feature_selector,
            'feature_transformer': self.feature_transformer,
            'models': self.models,
            'label_encoder': self.data_preparer.label_encoder,
            'feature_names': self.feature_names
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(system_data, f)
        
        print(f"\nSystem saved to {filename}")
    
    def load_system(self, filename='forgery_detection_system.pkl'):
        """Load a previously trained system"""
        with open(filename, 'rb') as f:
            system_data = pickle.load(f)
        
        self.feature_selector = system_data['feature_selector']
        self.feature_transformer = system_data['feature_transformer']
        self.models = system_data['models']
        self.data_preparer.label_encoder = system_data['label_encoder']
        self.feature_names = system_data['feature_names']
        
        print(f"\nSystem loaded from {filename}")


def main():
    """
    Main execution function
    Complete workflow for AI Image Forgery Detection
    """
    print("\n" + "="*80)
    print("AI IMAGE FORGERY DETECTION SYSTEM")
    print("Machine Learning Project - Implementing Syllabus Concepts")
    print("="*80)
    
    # Initialize system
    system = ImageForgeryDetectionSystem()
    
    # Step 1: Prepare Dataset
    system.step1_prepare_dataset(n_authentic=50, n_forged=50, use_existing=False)
    
    # Step 2: Feature Engineering
    system.step2_feature_engineering(method='embedded', k_features=25)
    
    # Step 3: Train Models
    system.step3_train_models()
    
    # Step 4: Evaluate Models
    system.step4_evaluate_models()
    
    # Step 5: Create Ensemble
    system.step5_create_ensemble()
    
    # Save the system
    system.save_system()
    
    # Demo: Predict on a test image
    print("\n" + "="*80)
    print("DEMONSTRATION: Single Image Prediction")
    print("="*80)
    
    # Test on an image from the test set
    import os
    test_image = os.path.join('dataset', 'forged', 'forged_000.jpg')
    if os.path.exists(test_image):
        result = system.predict_single_image(test_image, model_name='Ensemble')
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print("\nGenerated Files:")
    print("- model_comparison.png: Visual comparison of all models")
    print("- confusion_matrix_*.png: Confusion matrices for each model")
    print("- roc_curve_*.png: ROC curves for models")
    print("- forgery_detection_system.pkl: Saved trained system")
    print("\nTo use the system for predictions, run:")
    print("  python predict_forgery.py <image_path>")


if __name__ == "__main__":
    main()
