"""
Dataset Preparation Module
Handles dataset creation, loading, and preprocessing for image forgery detection
"""

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from feature_extraction import ImageFeatureExtractor
import pickle


class ForgeryDatasetPreparer:
    """Prepare dataset for image forgery detection"""
    
    def __init__(self, dataset_path='dataset'):
        self.dataset_path = dataset_path
        self.feature_extractor = ImageFeatureExtractor()
        self.label_encoder = LabelEncoder()
        
    def create_sample_dataset(self, n_authentic=50, n_forged=50):
        """
        Create a sample dataset with synthetic forgery indicators
        This is for demonstration - in production, use real forgery datasets
        """
        print("\n" + "="*60)
        print("Creating Sample Dataset for Image Forgery Detection")
        print("="*60)
        
        # Create directories
        os.makedirs(f'{self.dataset_path}/authentic', exist_ok=True)
        os.makedirs(f'{self.dataset_path}/forged', exist_ok=True)
        
        print(f"\nGenerating {n_authentic} authentic images...")
        for i in range(n_authentic):
            # Create authentic image (natural patterns)
            img = self._generate_authentic_image(i)
            cv2.imwrite(f'{self.dataset_path}/authentic/authentic_{i:03d}.jpg', img)
        
        print(f"Generating {n_forged} forged images...")
        for i in range(n_forged):
            # Create forged image (copy-paste artifacts, compression inconsistencies)
            img = self._generate_forged_image(i)
            cv2.imwrite(f'{self.dataset_path}/forged/forged_{i:03d}.jpg', img)
        
        print(f"\nDataset created successfully!")
        print(f"Location: {self.dataset_path}/")
        print(f"Total images: {n_authentic + n_forged}")
    
    def _generate_authentic_image(self, seed):
        """Generate synthetic authentic image"""
        np.random.seed(seed)
        
        # Create base image with natural gradients
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Add natural color variations
        for c in range(3):
            gradient = np.linspace(50, 200, 256)
            for i in range(256):
                img[:, i, c] = gradient[i] + np.random.randint(-20, 20)
        
        # Add some natural noise
        noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        
        # Add some geometric shapes (natural patterns)
        cv2.circle(img, (128, 128), 50, (200, 150, 100), -1)
        cv2.rectangle(img, (50, 50), (100, 100), (100, 150, 200), 2)
        
        # Apply slight blur (natural camera effect)
        img = cv2.GaussianBlur(img, (3, 3), 0.5)
        
        return img
    
    def _generate_forged_image(self, seed):
        """Generate synthetic forged image with forgery artifacts"""
        np.random.seed(seed + 1000)
        
        # Start with an authentic-like image
        img = self._generate_authentic_image(seed)
        
        # Introduce forgery artifacts:
        
        # 1. Copy-paste region (creating inconsistent noise patterns)
        region = img[50:100, 50:100].copy()
        img[150:200, 150:200] = region  # Paste in different location
        
        # 2. Compress part of the image differently (JPEG compression artifacts)
        temp_path = 'temp_forgery.jpg'
        
        # Compress a region with different quality
        region_to_compress = img[100:200, 100:200].copy()
        cv2.imwrite(temp_path, region_to_compress, [cv2.IMWRITE_JPEG_QUALITY, 50])
        compressed_region = cv2.imread(temp_path)
        img[100:200, 100:200] = cv2.resize(compressed_region, (100, 100))
        
        # 3. Add inconsistent lighting (splicing artifact)
        brightness_adjustment = np.random.randint(30, 60)
        img[0:128, :] = np.clip(img[0:128, :] + brightness_adjustment, 0, 255).astype(np.uint8)
        
        # 4. Add sharp edges from splicing
        cv2.line(img, (0, 128), (255, 128), (255, 255, 255), 2)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return img
    
    def load_and_extract_features(self, use_cache=True, cache_file='features_cache.pkl'):
        """
        Load images and extract features
        Implements: Feature extraction from Unit 1.2
        """
        print("\n" + "="*60)
        print("Loading Dataset and Extracting Features")
        print("="*60)
        
        # Check for cached features
        if use_cache and os.path.exists(cache_file):
            print(f"\nLoading cached features from {cache_file}...")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            print("Cached features loaded successfully!")
            return data['features'], data['labels'], data['image_paths']
        
        features_list = []
        labels = []
        image_paths = []
        
        # Load authentic images
        authentic_dir = f'{self.dataset_path}/authentic'
        if os.path.exists(authentic_dir):
            print(f"\nProcessing authentic images from {authentic_dir}...")
            authentic_files = [f for f in os.listdir(authentic_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            
            for i, filename in enumerate(authentic_files):
                filepath = os.path.join(authentic_dir, filename)
                features = self.feature_extractor.extract_all_features(filepath)
                
                if features is not None:
                    features_list.append(features)
                    labels.append('authentic')
                    image_paths.append(filepath)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(authentic_files)} authentic images")
        
        # Load forged images
        forged_dir = f'{self.dataset_path}/forged'
        if os.path.exists(forged_dir):
            print(f"\nProcessing forged images from {forged_dir}...")
            forged_files = [f for f in os.listdir(forged_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            
            for i, filename in enumerate(forged_files):
                filepath = os.path.join(forged_dir, filename)
                features = self.feature_extractor.extract_all_features(filepath)
                
                if features is not None:
                    features_list.append(features)
                    labels.append('forged')
                    image_paths.append(filepath)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(forged_files)} forged images")
        
        print(f"\nTotal images processed: {len(features_list)}")
        print(f"Authentic: {labels.count('authentic')}, Forged: {labels.count('forged')}")
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Cache the features
        if use_cache:
            print(f"\nCaching features to {cache_file}...")
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'features': features_df,
                    'labels': labels,
                    'image_paths': image_paths
                }, f)
            print("Features cached successfully!")
        
        return features_df, labels, image_paths
    
    def prepare_train_test_split(self, features_df, labels, test_size=0.2, val_size=0.1):
        """
        Prepare train/validation/test splits
        Implements: Training versus Testing from Unit 1.1
        """
        print("\n" + "="*60)
        print("Preparing Train/Validation/Test Split")
        print("="*60)
        
        # Encode labels (authentic=0, forged=1)
        y = self.label_encoder.fit_transform(labels)
        X = features_df.values
        
        print(f"\nLabel Encoding:")
        for i, class_name in enumerate(self.label_encoder.classes_):
            print(f"{class_name}: {i}")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: separate validation from training
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        print(f"\nDataset Split:")
        print(f"Training set:   {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"Test set:       {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        print(f"\nClass Distribution:")
        print(f"Training   - Authentic: {np.sum(y_train==0)}, Forged: {np.sum(y_train==1)}")
        print(f"Validation - Authentic: {np.sum(y_val==0)}, Forged: {np.sum(y_val==1)}")
        print(f"Test       - Authentic: {np.sum(y_test==0)}, Forged: {np.sum(y_test==1)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_processed_data(self, X_train, X_val, X_test, y_train, y_val, y_test, 
                           filename='processed_data.pkl'):
        """Save processed data for later use"""
        data = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'label_encoder': self.label_encoder
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"\nProcessed data saved to {filename}")
    
    def load_processed_data(self, filename='processed_data.pkl'):
        """Load previously processed data"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        self.label_encoder = data['label_encoder']
        
        print(f"\nProcessed data loaded from {filename}")
        
        return (data['X_train'], data['X_val'], data['X_test'],
                data['y_train'], data['y_val'], data['y_test'])
