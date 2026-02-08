"""
Flask Web Application for AI Image Forgery Detection System
Provides a modern web interface for image forgery detection
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import pickle
import cv2
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns

# Import project modules
from feature_extraction import ImageFeatureExtractor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# Global variable to store loaded system
SYSTEM = None
FEATURE_EXTRACTOR = ImageFeatureExtractor()


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_detection_system():
    """Load the trained forgery detection system"""
    global SYSTEM
    
    if SYSTEM is None:
        model_path = 'forgery_detection_system.pkl'
        
        if not os.path.exists(model_path):
            return None, "Model file not found. Please train the model first by running: python main_training.py"
        
        try:
            with open(model_path, 'rb') as f:
                SYSTEM = pickle.load(f)
            return SYSTEM, None
        except Exception as e:
            return None, f"Error loading model: {str(e)}"
    
    return SYSTEM, None


def create_visualization(image_path, prediction, probabilities, features_dict):
    """Create visualization charts for the prediction"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Original Image
    ax1 = fig.add_subplot(gs[0, 0])
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax1.imshow(img_rgb)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 2. Probability Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    classes = ['Authentic', 'Forged']
    colors = ['#2ecc71', '#e74c3c']
    bars = ax2.barh(classes, probabilities, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_xlim([0, 1])
    ax2.set_xlabel('Probability', fontsize=12)
    ax2.set_title('Prediction Probabilities', fontsize=14, fontweight='bold')
    
    # Add percentage labels
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        ax2.text(prob + 0.02, i, f'{prob*100:.1f}%', va='center', fontsize=11, fontweight='bold')
    
    ax2.grid(axis='x', alpha=0.3)
    
    # 3. Prediction Result
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    
    result_color = '#e74c3c' if prediction == 'forged' else '#2ecc71'
    result_text = 'FORGED' if prediction == 'forged' else 'AUTHENTIC'
    confidence = max(probabilities) * 100
    
    ax3.text(0.5, 0.6, result_text, ha='center', va='center', 
             fontsize=32, fontweight='bold', color=result_color,
             bbox=dict(boxstyle='round,pad=0.5', facecolor=result_color, alpha=0.2, edgecolor=result_color, linewidth=3))
    
    ax3.text(0.5, 0.3, f'Confidence: {confidence:.1f}%', ha='center', va='center',
             fontsize=16, fontweight='bold')
    
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    
    # 4. ELA Features
    ax4 = fig.add_subplot(gs[1, 0])
    ela_features = {k: v for k, v in features_dict.items() if k.startswith('ela_')}
    if ela_features:
        feature_names = [k.replace('ela_', '').title() for k in ela_features.keys()]
        feature_values = list(ela_features.values())
        
        ax4.barh(feature_names, feature_values, color='#3498db', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Value', fontsize=11)
        ax4.set_title('Error Level Analysis Features', fontsize=13, fontweight='bold')
        ax4.grid(axis='x', alpha=0.3)
    
    # 5. Color Features
    ax5 = fig.add_subplot(gs[1, 1])
    color_features = {k: v for k, v in features_dict.items() if any(c in k for c in ['red', 'green', 'blue'])}
    if color_features:
        # Get mean values for RGB
        rgb_means = [
            features_dict.get('red_mean', 0),
            features_dict.get('green_mean', 0),
            features_dict.get('blue_mean', 0)
        ]
        colors_rgb = ['#e74c3c', '#2ecc71', '#3498db']
        ax5.bar(['Red', 'Green', 'Blue'], rgb_means, color=colors_rgb, alpha=0.7, edgecolor='black', linewidth=2)
        ax5.set_ylabel('Mean Value', fontsize=11)
        ax5.set_title('Color Channel Statistics', fontsize=13, fontweight='bold')
        ax5.set_ylim([0, 255])
        ax5.grid(axis='y', alpha=0.3)
    
    # 6. Noise Features
    ax6 = fig.add_subplot(gs[1, 2])
    noise_features = {k: v for k, v in features_dict.items() if k.startswith('noise_')}
    if noise_features:
        feature_names = [k.replace('noise_', '').title() for k in noise_features.keys()]
        feature_values = list(noise_features.values())
        
        ax6.barh(feature_names, feature_values, color='#9b59b6', alpha=0.7, edgecolor='black')
        ax6.set_xlabel('Value', fontsize=11)
        ax6.set_title('Noise Pattern Features', fontsize=13, fontweight='bold')
        ax6.grid(axis='x', alpha=0.3)
    
    # 7. Texture Features
    ax7 = fig.add_subplot(gs[2, 0])
    texture_features = {k: v for k, v in features_dict.items() if 'gradient' in k or 'texture' in k or 'edge' in k}
    if texture_features:
        feature_names = [k.replace('gradient_', '').replace('texture_', '').replace('edge_', '').title() for k in texture_features.keys()]
        feature_values = list(texture_features.values())
        
        ax7.barh(feature_names, feature_values, color='#e67e22', alpha=0.7, edgecolor='black')
        ax7.set_xlabel('Value', fontsize=11)
        ax7.set_title('Texture & Gradient Features', fontsize=13, fontweight='bold')
        ax7.grid(axis='x', alpha=0.3)
    
    # 8. Feature Importance (Top 10)
    ax8 = fig.add_subplot(gs[2, 1:])
    # Sort features by absolute value and get top 10
    sorted_features = sorted(features_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    top_names = [k for k, v in sorted_features]
    top_values = [v for k, v in sorted_features]
    
    colors_bar = ['#e74c3c' if v < 0 else '#2ecc71' for v in top_values]
    ax8.barh(top_names, top_values, color=colors_bar, alpha=0.7, edgecolor='black')
    ax8.set_xlabel('Feature Value', fontsize=11)
    ax8.set_title('Top 10 Feature Values', fontsize=13, fontweight='bold')
    ax8.grid(axis='x', alpha=0.3)
    
    # Add overall title
    fig.suptitle(f'Image Forgery Detection Analysis - {result_text}', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Save figure
    output_path = 'static/results/analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/about')
def about():
    """Render the about page"""
    return render_template('about.html')


@app.route('/batch')
def batch():
    """Render the batch processing page"""
    return render_template('batch.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for single image prediction"""
    
    # Check if model is loaded
    system, error = load_detection_system()
    if error:
        return jsonify({'success': False, 'error': error}), 500
    
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, BMP, TIFF'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get selected model
        model_name = request.form.get('model', 'Ensemble')
        
        # Extract features
        features = FEATURE_EXTRACTOR.extract_all_features(filepath)
        
        if features is None:
            return jsonify({'success': False, 'error': 'Failed to extract features from image'}), 500
        
        # Store original features for visualization
        features_dict = features.copy()
        
        # Convert to DataFrame
        features_df = pd.DataFrame([features])
        X = features_df.values
        
        # Apply feature selection
        feature_selector = system['feature_selector']
        X_selected = feature_selector.transform(X)
        
        # Apply normalization
        feature_transformer = system['feature_transformer']
        X_normalized = feature_transformer.transform(X_selected)
        
        # Make prediction
        if model_name not in system['models']:
            model_name = 'Ensemble'
        
        model = system['models'][model_name]
        prediction = model.predict(X_normalized)[0]
        probabilities = model.predict_proba(X_normalized)[0]
        
        # Get class label
        label_encoder = system['label_encoder']
        class_label = label_encoder.inverse_transform([prediction])[0]
        
        # Create visualization
        viz_path = create_visualization(filepath, class_label, probabilities, features_dict)
        
        # Get image info
        img = cv2.imread(filepath)
        height, width = img.shape[:2]
        file_size = os.path.getsize(filepath)
        
        # Prepare response
        response = {
            'success': True,
            'prediction': class_label,
            'probabilities': {
                'authentic': float(probabilities[0]),
                'forged': float(probabilities[1])
            },
            'confidence': float(max(probabilities)),
            'model_used': model_name,
            'image_info': {
                'filename': filename,
                'width': int(width),
                'height': int(height),
                'size': f'{file_size / 1024:.2f} KB',
                'total_pixels': int(width * height)
            },
            'visualization': viz_path
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """API endpoint for batch image prediction"""
    
    # Check if model is loaded
    system, error = load_detection_system()
    if error:
        return jsonify({'success': False, 'error': error}), 500
    
    # Check if files are present
    if 'files[]' not in request.files:
        return jsonify({'success': False, 'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('files[]')
    
    if len(files) == 0:
        return jsonify({'success': False, 'error': 'No files selected'}), 400
    
    model_name = request.form.get('model', 'Ensemble')
    results = []
    
    for file in files:
        if file and allowed_file(file.filename):
            try:
                # Save file
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Extract features
                features = FEATURE_EXTRACTOR.extract_all_features(filepath)
                
                if features is None:
                    results.append({
                        'filename': filename,
                        'success': False,
                        'error': 'Failed to extract features'
                    })
                    continue
                
                # Convert and process
                features_df = pd.DataFrame([features])
                X = features_df.values
                
                X_selected = system['feature_selector'].transform(X)
                X_normalized = system['feature_transformer'].transform(X_selected)
                
                # Predict
                if model_name not in system['models']:
                    model_name = 'Ensemble'
                
                model = system['models'][model_name]
                prediction = model.predict(X_normalized)[0]
                probabilities = model.predict_proba(X_normalized)[0]
                
                class_label = system['label_encoder'].inverse_transform([prediction])[0]
                
                results.append({
                    'filename': filename,
                    'success': True,
                    'prediction': class_label,
                    'confidence': float(max(probabilities)),
                    'probabilities': {
                        'authentic': float(probabilities[0]),
                        'forged': float(probabilities[1])
                    }
                })
                
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'success': False,
                    'error': str(e)
                })
    
    return jsonify({
        'success': True,
        'total': len(files),
        'processed': len([r for r in results if r.get('success')]),
        'results': results
    })


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of available models"""
    system, error = load_detection_system()
    
    if error:
        return jsonify({'success': False, 'error': error}), 500
    
    models = list(system['models'].keys())
    
    return jsonify({
        'success': True,
        'models': models,
        'default': 'Ensemble'
    })


@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files"""
    return send_from_directory('static', path)


if __name__ == '__main__':
    print("\n" + "="*80)
    print("AI IMAGE FORGERY DETECTION SYSTEM - WEB INTERFACE")
    print("="*80)
    print("\nStarting web server...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    print("\nPress Ctrl+C to stop the server")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
