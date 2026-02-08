"""
Regression and Linear Models Module
Implements: Linear Models (Least Squares, Multivariate Regression, Regularized Regression)
Perceptron, Support Vector Machines
(From Unit 2.1 and 2.2 of syllabus)
"""

import numpy as np
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, 
    LogisticRegression, Perceptron
)
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


class RegressionModels:
    """
    Regression Models with Error Assessment
    (From Unit 2.1 of syllabus)
    """
    
    def __init__(self):
        self.models = {}
        self.predictions = {}
        self.metrics = {}
    
    def train_linear_regression(self, X_train, y_train, model_name="Linear Regression"):
        """Ordinary Least Squares Linear Regression"""
        model = LinearRegression()
        model.fit(X_train, y_train)
        self.models[model_name] = model
        print(f"{model_name} trained successfully")
        return model
    
    def train_regularized_regression(self, X_train, y_train, alpha=1.0, method='ridge'):
        """
        Regularized Regression: Ridge, Lasso, ElasticNet
        (From Unit 2.2 of syllabus)
        """
        if method.lower() == 'ridge':
            model = Ridge(alpha=alpha)
            model_name = f"Ridge Regression (alpha={alpha})"
        elif method.lower() == 'lasso':
            model = Lasso(alpha=alpha)
            model_name = f"Lasso Regression (alpha={alpha})"
        elif method.lower() == 'elasticnet':
            model = ElasticNet(alpha=alpha, l1_ratio=0.5)
            model_name = f"ElasticNet Regression (alpha={alpha})"
        else:
            raise ValueError("Method must be 'ridge', 'lasso', or 'elasticnet'")
        
        model.fit(X_train, y_train)
        self.models[model_name] = model
        print(f"{model_name} trained successfully")
        return model
    
    def assess_regression_performance(self, model_name, X_test, y_test):
        """
        Assess Regression Performance using Error Measures
        (From Unit 2.1: Error measures, Overfitting, Underfitting, Bias, Variance)
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Train it first.")
        
        model = self.models[model_name]
        y_pred = model.predict(X_test)
        
        # Error measures
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Bias-Variance analysis
        residuals = y_test - y_pred
        bias = np.mean(residuals)
        variance = np.var(residuals)
        
        print(f"\n{'='*60}")
        print(f"Regression Performance: {model_name}")
        print(f"{'='*60}")
        print(f"Mean Squared Error (MSE):  {mse:.6f}")
        print(f"Root MSE (RMSE):           {rmse:.6f}")
        print(f"Mean Absolute Error (MAE): {mae:.6f}")
        print(f"R² Score:                  {r2:.6f}")
        print(f"Bias:                      {bias:.6f}")
        print(f"Variance:                  {variance:.6f}")
        
        # Store results
        self.predictions[model_name] = y_pred
        self.metrics[model_name] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'bias': bias,
            'variance': variance
        }
        
        return self.metrics[model_name]
    
    def plot_residuals(self, model_name, X_test, y_test):
        """Plot residuals to check for overfitting/underfitting"""
        if model_name not in self.predictions:
            raise ValueError("Run assess_regression_performance first")
        
        y_pred = self.predictions[model_name]
        residuals = y_test - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title(f'Residual Plot - {model_name}')
        axes[0].grid(alpha=0.3)
        
        # Actual vs Predicted
        axes[1].scatter(y_test, y_pred, alpha=0.5)
        axes[1].plot([y_test.min(), y_test.max()], 
                     [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1].set_xlabel('Actual Values')
        axes[1].set_ylabel('Predicted Values')
        axes[1].set_title(f'Actual vs Predicted - {model_name}')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'./residuals_{model_name.replace(" ", "_")}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Residual plots saved")


class LinearModelsForClassification:
    """
    Linear Models for Classification
    (From Unit 2.2: Perceptron, SVM)
    """
    
    def __init__(self):
        self.models = {}
    
    def train_perceptron(self, X_train, y_train, max_iter=1000):
        """
        Perceptron Algorithm
        (From Unit 2.2 of syllabus)
        """
        model = Perceptron(max_iter=max_iter, random_state=42, eta0=0.1)
        model.fit(X_train, y_train)
        self.models['Perceptron'] = model
        print("Perceptron trained successfully")
        return model
    
    def train_logistic_regression(self, X_train, y_train, C=1.0):
        """
        Logistic Regression (Multivariate Linear Model)
        (From Unit 2.2 of syllabus)
        """
        model = LogisticRegression(C=C, max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        self.models['Logistic Regression'] = model
        print("Logistic Regression trained successfully")
        return model
    
    def train_svm(self, X_train, y_train, kernel='rbf', C=1.0, gamma='scale'):
        """
        Support Vector Machine with Kernel Methods
        (From Unit 2.2: SVM, Soft Margin SVM, Kernel methods for non-Linearity)
        """
        model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=42)
        model.fit(X_train, y_train)
        model_name = f'SVM ({kernel} kernel)'
        self.models[model_name] = model
        print(f"{model_name} trained successfully")
        return model
    
    def train_linear_svm(self, X_train, y_train, C=1.0):
        """
        Linear SVM (Soft Margin)
        (From Unit 2.2 of syllabus)
        """
        model = LinearSVC(C=C, max_iter=2000, random_state=42)
        model.fit(X_train, y_train)
        self.models['Linear SVM'] = model
        print("Linear SVM trained successfully")
        return model
    
    def get_model(self, model_name):
        """Retrieve a trained model"""
        if model_name in self.models:
            return self.models[model_name]
        else:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")


class PolynomialRegressionModel:
    """
    Polynomial Regression
    (From Unit 2.1 of syllabus)
    """
    
    def __init__(self, degree=2):
        self.degree = degree
        self.model = LinearRegression()
        self.poly_features = None
    
    def _create_polynomial_features(self, X):
        """Create polynomial features"""
        from sklearn.preprocessing import PolynomialFeatures
        if self.poly_features is None:
            self.poly_features = PolynomialFeatures(degree=self.degree)
            return self.poly_features.fit_transform(X)
        else:
            return self.poly_features.transform(X)
    
    def train(self, X_train, y_train):
        """Train polynomial regression model"""
        X_poly = self._create_polynomial_features(X_train)
        self.model.fit(X_poly, y_train)
        print(f"Polynomial Regression (degree={self.degree}) trained successfully")
    
    def predict(self, X):
        """Make predictions"""
        X_poly = self._create_polynomial_features(X)
        return self.model.predict(X_poly)
    
    def assess_performance(self, X_test, y_test):
        """Assess polynomial regression performance"""
        y_pred = self.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nPolynomial Regression (degree={self.degree}) Performance:")
        print(f"MSE:  {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"R²:   {r2:.6f}")
        
        return {'mse': mse, 'rmse': rmse, 'r2': r2}
