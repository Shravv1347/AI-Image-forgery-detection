"""
Feature Extraction Module
Implements: Feature types, Feature Construction and Transformation, Feature Selection
(From Unit 1.2 of syllabus)
"""

import cv2
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier


class ImageFeatureExtractor:
    """Extract various features from images for forgery detection"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def extract_ela_features(self, image_path, quality=90):
        """Error Level Analysis - detects compression artifacts"""
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Save with specific quality
        temp_path = 'temp_ela.jpg'
        cv2.imwrite(temp_path, original, [cv2.IMWRITE_JPEG_QUALITY, quality])
        compressed = cv2.imread(temp_path)
        
        # Calculate difference
        ela = cv2.absdiff(original, compressed)
        ela_gray = cv2.cvtColor(ela, cv2.COLOR_BGR2GRAY)
        
        # Statistical features from ELA
        features = {
            'ela_mean': np.mean(ela_gray),
            'ela_std': np.std(ela_gray),
            'ela_max': np.max(ela_gray),
            'ela_min': np.min(ela_gray),
            'ela_median': np.median(ela_gray),
            'ela_variance': np.var(ela_gray),
            'ela_skewness': stats.skew(ela_gray.flatten()),
            'ela_kurtosis': stats.kurtosis(ela_gray.flatten())
        }
        
        return features
    
    def extract_noise_features(self, image_path):
        """Extract noise pattern features"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply median filter to estimate noise
        median_filtered = cv2.medianBlur(img, 5)
        noise = img.astype(float) - median_filtered.astype(float)
        
        features = {
            'noise_mean': np.mean(np.abs(noise)),
            'noise_std': np.std(noise),
            'noise_variance': np.var(noise),
            'noise_energy': np.sum(noise**2),
            'noise_entropy': stats.entropy(np.histogram(noise, bins=50)[0] + 1e-10)
        }
        
        return features
    
    def extract_color_features(self, image_path):
        """Extract color distribution features"""
        img = cv2.imread(image_path)
        
        # Color channel statistics
        features = {}
        for i, color in enumerate(['blue', 'green', 'red']):
            channel = img[:, :, i]
            features[f'{color}_mean'] = np.mean(channel)
            features[f'{color}_std'] = np.std(channel)
            features[f'{color}_skewness'] = stats.skew(channel.flatten())
            features[f'{color}_kurtosis'] = stats.kurtosis(channel.flatten())
        
        # HSV features
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for i, component in enumerate(['hue', 'saturation', 'value']):
            channel = hsv[:, :, i]
            features[f'{component}_mean'] = np.mean(channel)
            features[f'{component}_std'] = np.std(channel)
        
        return features
    
    def extract_texture_features(self, image_path):
        """Extract texture features using GLCM-like statistics"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Calculate gradients
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # LBP-inspired features
        lbp_mean = np.mean(gradient_magnitude)
        lbp_std = np.std(gradient_magnitude)
        
        features = {
            'gradient_mean': np.mean(gradient_magnitude),
            'gradient_std': np.std(gradient_magnitude),
            'gradient_max': np.max(gradient_magnitude),
            'edge_density': np.sum(gradient_magnitude > np.mean(gradient_magnitude)) / gradient_magnitude.size,
            'texture_contrast': np.std(gradient_magnitude),
            'texture_homogeneity': 1 / (1 + np.var(gradient_magnitude))
        }
        
        return features
    
    def extract_metadata_features(self, image_path):
        """Extract image metadata features"""
        img = cv2.imread(image_path)
        
        features = {
            'width': img.shape[1],
            'height': img.shape[0],
            'aspect_ratio': img.shape[1] / img.shape[0],
            'total_pixels': img.shape[0] * img.shape[1],
            'channels': img.shape[2] if len(img.shape) > 2 else 1
        }
        
        return features
    
    def extract_all_features(self, image_path):
        """Extract all features from an image"""
        features = {}
        
        try:
            features.update(self.extract_ela_features(image_path))
            features.update(self.extract_noise_features(image_path))
            features.update(self.extract_color_features(image_path))
            features.update(self.extract_texture_features(image_path))
            features.update(self.extract_metadata_features(image_path))
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None
        
        return features


class FeatureSelector:
    """
    Feature Selection using Filter, Wrapper, and Embedded methods
    (From Unit 1.2 of syllabus)
    """
    
    def __init__(self):
        self.selected_features = None
        self.selector = None
    
    def filter_method(self, X, y, k=20):
        """Filter method: SelectKBest with ANOVA F-statistic"""
        self.selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = self.selector.fit_transform(X, y)
        self.selected_features = self.selector.get_support(indices=True)
        print(f"Filter method selected {k} features")
        return X_selected
    
    def wrapper_method(self, X, y, n_features=15):
        """Wrapper method: Recursive Feature Elimination"""
        estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        self.selector = RFE(estimator, n_features_to_select=n_features, step=1)
        X_selected = self.selector.fit_transform(X, y)
        self.selected_features = self.selector.get_support(indices=True)
        print(f"Wrapper method selected {n_features} features")
        return X_selected
    
    def embedded_method(self, X, y, threshold=0.01):
        """Embedded method: Feature importance from Random Forest"""
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        importances = rf.feature_importances_
        self.selected_features = np.where(importances > threshold)[0]
        X_selected = X[:, self.selected_features]
        
        print(f"Embedded method selected {len(self.selected_features)} features")
        return X_selected, importances
    
    def transform(self, X):
        """Transform new data using selected features"""
        if self.selector is not None:
            return self.selector.transform(X)
        elif self.selected_features is not None:
            return X[:, self.selected_features]
        else:
            return X


class FeatureTransformer:
    """
    Feature Construction and Transformation
    (From Unit 1.2 of syllabus)
    """
    
    def __init__(self):
        self.scaler = None
        self.pca = None
    
    def normalize_features(self, X, method='standard'):
        """Normalize features using different scaling methods"""
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Method must be 'standard' or 'minmax'")
        
        X_normalized = self.scaler.fit_transform(X)
        return X_normalized
    
    def apply_pca(self, X, n_components=0.95):
        """Apply PCA for dimensionality reduction"""
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X)
        print(f"PCA reduced features from {X.shape[1]} to {X_pca.shape[1]}")
        print(f"Explained variance ratio: {sum(self.pca.explained_variance_ratio_):.4f}")
        return X_pca
    
    def transform(self, X):
        """Transform new data"""
        if self.scaler is not None:
            X = self.scaler.transform(X)
        if self.pca is not None:
            X = self.pca.transform(X)
        return X
