"""
QUICK START GUIDE
AI Image Forgery Detection System
"""

# ============================================================================
# STEP-BY-STEP GUIDE TO RUN THE PROJECT
# ============================================================================

"""
STEP 1: INSTALLATION
--------------------
1. Make sure Python 3.8+ is installed
2. Open terminal/command prompt
3. Navigate to the project directory:
   cd path/to/project

4. Install requirements:
   pip install -r requirements.txt

   This will install:
   - numpy, pandas (data manipulation)
   - scikit-learn (machine learning)
   - opencv-python (image processing)
   - matplotlib, seaborn (visualization)
   - scipy (statistical functions)


STEP 2: TRAIN THE SYSTEM
-------------------------
Run the main training script:

   python main_training.py

What happens:
- Creates a synthetic dataset (50 authentic + 50 forged images)
- Extracts 40+ features from each image
- Trains 7 different ML models
- Evaluates and compares all models
- Creates an ensemble model
- Saves the trained system

Time: 2-5 minutes
Output files:
- forgery_detection_system.pkl
- model_comparison.png
- confusion_matrix_*.png
- roc_curve_*.png


STEP 3: TEST PREDICTIONS
------------------------
Test the trained system on a single image:

   python predict_forgery.py dataset/forged/forged_000.jpg

Or test on multiple images:

   python predict_forgery.py --batch dataset/forged/

Or use a specific model:

   python predict_forgery.py image.jpg --model "Random Forest"


STEP 4: USE YOUR OWN IMAGES
---------------------------
To detect forgery in your own images:

1. Create dataset structure:
   dataset/
     authentic/  (put real images here)
     forged/     (put tampered images here)

2. Train on your data:
   python main_training.py

3. Predict on new images:
   python predict_forgery.py your_image.jpg


============================================================================
UNDERSTANDING THE OUTPUT
============================================================================

When you run predictions, you'll see:

✓ Image appears to be AUTHENTIC
or
✗ Image appears to be FORGED

Probability Distribution:
  Authentic: XX.XX%
  Forged:    XX.XX%

Confidence: XX.XX%

Interpretation guide:
- >90% confidence: Very reliable
- 75-90% confidence: Reliable
- 60-75% confidence: Moderate - double check
- <60% confidence: Low - manual inspection recommended


============================================================================
PROJECT FILES EXPLAINED
============================================================================

Core Modules (implementing syllabus concepts):
- feature_extraction.py      → Feature engineering (Unit 1.2)
- classification_models.py    → Classification (Unit 1.3)
- regression_linear_models.py → Linear models (Unit 2.1, 2.2)
- distance_based_models.py    → KNN, Clustering (Unit 2.3)
- tree_rule_models.py         → Trees, Random Forest (Unit 3.2)
- probabilistic_models.py     → Naive Bayes (Unit 3.3)

Utilities:
- dataset_preparation.py      → Dataset handling
- main_training.py            → Main training pipeline
- predict_forgery.py          → Prediction interface
- requirements.txt            → Dependencies
- README.md                   → Full documentation


============================================================================
CUSTOMIZATION OPTIONS
============================================================================

1. Change dataset size:
   In main_training.py, line ~280:
   system.step1_prepare_dataset(n_authentic=100, n_forged=100)

2. Change feature selection:
   In main_training.py, line ~283:
   system.step2_feature_engineering(method='embedded', k_features=30)
   
   Options: 'filter', 'wrapper', 'embedded'

3. Modify model parameters:
   In main_training.py, around line 310-360, edit model initialization:
   Example:
   rf_model = RandomForestModel(n_estimators=200, max_depth=20)


============================================================================
TROUBLESHOOTING
============================================================================

Problem: "ModuleNotFoundError"
Solution: Install requirements: pip install -r requirements.txt

Problem: "Model file not found"
Solution: Train first: python main_training.py

Problem: "Cannot read image"
Solution: 
- Check image format (.jpg, .png supported)
- Verify file path is correct
- Check file permissions

Problem: Low accuracy
Solution:
- Use more training images (100+ each class)
- Use real forgery dataset instead of synthetic
- Try different feature selection methods
- Adjust model parameters


============================================================================
ADVANCED USAGE
============================================================================

1. Access individual models programmatically:

   from main_training import ImageForgeryDetectionSystem
   
   system = ImageForgeryDetectionSystem()
   system.load_system('forgery_detection_system.pkl')
   
   result = system.predict_single_image('image.jpg', model_name='SVM')
   print(result)

2. Extract features only:

   from feature_extraction import ImageFeatureExtractor
   
   extractor = ImageFeatureExtractor()
   features = extractor.extract_all_features('image.jpg')
   print(features)

3. Train individual models:

   from tree_rule_models import RandomForestModel
   
   rf = RandomForestModel(n_estimators=100)
   rf.train(X_train, y_train)
   predictions = rf.predict(X_test)


============================================================================
CONCEPTS DEMONSTRATED
============================================================================

✓ Feature Extraction & Engineering
✓ Feature Selection (Filter, Wrapper, Embedded methods)
✓ Binary Classification
✓ Multiple ML algorithms (Logistic Regression, SVM, KNN, Trees, etc.)
✓ Ensemble Methods
✓ Cross-validation
✓ Performance Metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
✓ Bias-Variance Tradeoff
✓ Overfitting/Underfitting analysis
✓ Model Comparison
✓ Real-world application (Image Forgery Detection)


============================================================================
NEXT STEPS
============================================================================

1. ✓ Complete the Quick Start guide above
2. ✓ Run the system and examine outputs
3. ✓ Try with your own images
4. ✓ Experiment with different models
5. ✓ Review the code to understand ML concepts
6. ✓ Modify parameters to see effects on performance


For detailed documentation, see README.md
For concept explanations, see comments in each module
"""

if __name__ == "__main__":
    print(__doc__)
