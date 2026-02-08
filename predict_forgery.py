"""
Standalone Prediction Script
Use trained model to predict if an image is authentic or forged
"""

import sys
import os
import pickle
import cv2
import numpy as np
import pandas as pd
from feature_extraction import ImageFeatureExtractor


def load_system(model_path='forgery_detection_system.pkl'):
    """Load the trained forgery detection system"""
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please train the model first by running: python main_training.py")
        return None
    
    with open(model_path, 'rb') as f:
        system = pickle.load(f)
    
    print("Forgery Detection System loaded successfully!")
    return system


def predict_image(image_path, system, model_name='Ensemble'):
    """Predict whether an image is authentic or forged"""
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return None
    
    # Check if image is readable
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image file '{image_path}'")
        return None
    
    print("\n" + "="*80)
    print("IMAGE FORGERY DETECTION")
    print("="*80)
    print(f"Image: {image_path}")
    print(f"Image size: {img.shape[1]}x{img.shape[0]} pixels")
    
    # Extract features
    print("\nExtracting features...")
    feature_extractor = ImageFeatureExtractor()
    features = feature_extractor.extract_all_features(image_path)
    
    if features is None:
        print("Error: Could not extract features from image")
        return None
    
    # Convert to DataFrame
    features_df = pd.DataFrame([features])
    X = features_df.values
    
    # Apply feature selection
    print("Applying feature selection...")
    feature_selector = system['feature_selector']
    X_selected = feature_selector.transform(X)
    
    # Apply normalization
    print("Normalizing features...")
    feature_transformer = system['feature_transformer']
    X_normalized = feature_transformer.transform(X_selected)
    
    # Make prediction
    print(f"Making prediction using {model_name}...")
    
    if model_name not in system['models']:
        print(f"Warning: Model '{model_name}' not found. Using Ensemble model.")
        model_name = 'Ensemble'
    
    model = system['models'][model_name]
    prediction = model.predict(X_normalized)[0]
    probabilities = model.predict_proba(X_normalized)[0]
    
    # Get class label
    label_encoder = system['label_encoder']
    class_label = label_encoder.inverse_transform([prediction])[0]
    
    # Display results
    print("\n" + "="*80)
    print("PREDICTION RESULTS")
    print("="*80)
    
    if class_label == 'authentic':
        print("✓ Image appears to be AUTHENTIC")
    else:
        print("✗ Image appears to be FORGED")
    
    print("\nProbability Distribution:")
    print(f"  Authentic: {probabilities[0]:.2%}")
    print(f"  Forged:    {probabilities[1]:.2%}")
    print(f"\nConfidence: {max(probabilities):.2%}")
    
    # Interpretation
    confidence = max(probabilities)
    print("\nInterpretation:")
    if confidence > 0.9:
        print("  Very high confidence in prediction")
    elif confidence > 0.75:
        print("  High confidence in prediction")
    elif confidence > 0.6:
        print("  Moderate confidence in prediction")
    else:
        print("  Low confidence - image may need manual inspection")
    
    print("="*80)
    
    return {
        'prediction': class_label,
        'probabilities': {
            'authentic': probabilities[0],
            'forged': probabilities[1]
        },
        'confidence': confidence
    }


def batch_predict(image_directory, system, model_name='Ensemble'):
    """Predict forgery for all images in a directory"""
    
    if not os.path.isdir(image_directory):
        print(f"Error: Directory '{image_directory}' not found!")
        return None
    
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(image_directory) 
                   if f.lower().endswith(image_extensions)]
    
    if not image_files:
        print(f"No image files found in '{image_directory}'")
        return None
    
    print(f"\nFound {len(image_files)} images to process")
    
    results = []
    
    for i, filename in enumerate(image_files, 1):
        image_path = os.path.join(image_directory, filename)
        print(f"\n[{i}/{len(image_files)}] Processing: {filename}")
        
        result = predict_image(image_path, system, model_name)
        
        if result:
            result['filename'] = filename
            results.append(result)
    
    # Summary
    print("\n" + "="*80)
    print("BATCH PREDICTION SUMMARY")
    print("="*80)
    
    authentic_count = sum(1 for r in results if r['prediction'] == 'authentic')
    forged_count = sum(1 for r in results if r['prediction'] == 'forged')
    
    print(f"Total images processed: {len(results)}")
    print(f"Predicted as Authentic: {authentic_count}")
    print(f"Predicted as Forged:    {forged_count}")
    
    # Save results to CSV
    output_file = 'batch_predictions.csv'
    import pandas as pd
    df = pd.DataFrame([
        {
            'filename': r['filename'],
            'prediction': r['prediction'],
            'prob_authentic': r['probabilities']['authentic'],
            'prob_forged': r['probabilities']['forged'],
            'confidence': r['confidence']
        }
        for r in results
    ])
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return results


def main():
    """Main function for command-line usage"""
    
    if len(sys.argv) < 2:
        print("AI Image Forgery Detection - Prediction Tool")
        print("\nUsage:")
        print("  Single image:  python predict_forgery.py <image_path>")
        print("  Batch mode:    python predict_forgery.py --batch <directory_path>")
        print("\nOptions:")
        print("  --model <name>  Specify model to use (default: Ensemble)")
        print("                  Available: Logistic Regression, SVM, KNN, Decision Tree,")
        print("                             Random Forest, Naive Bayes, Gradient Boosting, Ensemble")
        print("\nExamples:")
        print("  python predict_forgery.py dataset/forged/forged_000.jpg")
        print("  python predict_forgery.py --batch dataset/forged/")
        print("  python predict_forgery.py image.jpg --model 'Random Forest'")
        return
    
    # Load the trained system
    system = load_system()
    if system is None:
        return
    
    # Parse arguments
    model_name = 'Ensemble'
    if '--model' in sys.argv:
        model_idx = sys.argv.index('--model')
        if model_idx + 1 < len(sys.argv):
            model_name = sys.argv[model_idx + 1]
    
    # Batch or single prediction
    if '--batch' in sys.argv:
        batch_idx = sys.argv.index('--batch')
        if batch_idx + 1 < len(sys.argv):
            directory = sys.argv[batch_idx + 1]
            batch_predict(directory, system, model_name)
    else:
        image_path = sys.argv[1]
        predict_image(image_path, system, model_name)


if __name__ == "__main__":
    main()
