"""
Classification Models Module
Implements: Binary Classification, Performance Assessment, Class Probability Estimation
(From Unit 1.3 of syllabus)
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns


class BinaryClassifier:
    """Base class for binary classification with performance assessment"""
    
    def __init__(self, model, model_name="Classifier"):
        self.model = model
        self.model_name = model_name
        self.is_trained = False
        self.training_history = {}
    
    def train(self, X_train, y_train):
        """Train the classifier"""
        print(f"\nTraining {self.model_name}...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"{self.model_name} training completed!")
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get class probability estimates"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        elif hasattr(self.model, 'decision_function'):
            # For SVM and similar models
            decision = self.model.decision_function(X)
            # Convert to probability-like scores
            proba = 1 / (1 + np.exp(-decision))
            return np.column_stack([1 - proba, proba])
        else:
            raise ValueError("Model does not support probability estimation")
    
    def assess_performance(self, X_test, y_test, show_plots=True):
        """
        Comprehensive performance assessment
        (From Unit 1.3: Assessing Classification performance)
        """
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # False Positive Rate and False Negative Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"Performance Assessment: {self.model_name}")
        print(f"{'='*60}")
        print(f"Accuracy:     {accuracy:.4f}")
        print(f"Precision:    {precision:.4f}")
        print(f"Recall:       {recall:.4f}")
        print(f"F1-Score:     {f1:.4f}")
        print(f"Specificity:  {specificity:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"True Negatives:  {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"True Positives:  {tp}")
        print(f"\nFalse Positive Rate: {fpr:.4f}")
        print(f"False Negative Rate: {fnr:.4f}")
        
        # ROC-AUC if probability estimation is available
        try:
            y_proba = self.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
            print(f"ROC-AUC Score: {roc_auc:.4f}")
            
            if show_plots:
                self._plot_roc_curve(y_test, y_proba)
        except:
            roc_auc = None
            print("ROC-AUC: Not available for this model")
        
        if show_plots:
            self._plot_confusion_matrix(cm)
        
        # Store results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'confusion_matrix': cm,
            'fpr': fpr,
            'fnr': fnr,
            'roc_auc': roc_auc
        }
        
        return results
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation"""
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        print(f"\n{self.model_name} - Cross-Validation Results ({cv}-fold):")
        print(f"Scores: {scores}")
        print(f"Mean Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        return scores
    
    def _plot_confusion_matrix(self, cm):
        """Plot confusion matrix heatmap"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Authentic', 'Forged'],
                    yticklabels=['Authentic', 'Forged'])
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'./confusion_matrix_{self.model_name.replace(" ", "_")}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved as confusion_matrix_{self.model_name.replace(' ', '_')}.png")
    
    def _plot_roc_curve(self, y_test, y_proba):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                 label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.model_name}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'./roc_curve_{self.model_name.replace(" ", "_")}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC curve saved as roc_curve_{self.model_name.replace(' ', '_')}.png")


class MulticlassClassifier:
    """
    Multiclass Classification Extension
    (From Unit 1.3 of syllabus)
    """
    
    def __init__(self, model, model_name="Multiclass Classifier"):
        self.model = model
        self.model_name = model_name
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """Train the multiclass classifier"""
        print(f"\nTraining {self.model_name}...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"{self.model_name} training completed!")
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get class probability estimates for all classes"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_proba(X)
    
    def assess_performance(self, X_test, y_test, class_names=None):
        """Assess multiclass classification performance"""
        y_pred = self.predict(X_test)
        
        # Overall metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\n{'='*60}")
        print(f"Multiclass Performance: {self.model_name}")
        print(f"{'='*60}")
        print(f"Accuracy:          {accuracy:.4f}")
        print(f"Weighted Precision: {precision:.4f}")
        print(f"Weighted Recall:    {recall:.4f}")
        print(f"Weighted F1-Score:  {f1:.4f}")
        
        # Detailed classification report
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self._plot_multiclass_confusion_matrix(cm, class_names)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
        
        return results
    
    def _plot_multiclass_confusion_matrix(self, cm, class_names):
        """Plot multiclass confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names if class_names else range(len(cm)),
                    yticklabels=class_names if class_names else range(len(cm)))
        plt.title(f'Multiclass Confusion Matrix - {self.model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'./multiclass_cm_{self.model_name.replace(" ", "_")}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
