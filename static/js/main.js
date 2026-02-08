// Main JavaScript for AI Image Forgery Detection System

let selectedFile = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('fileInput');
    const uploadBox = document.getElementById('uploadBox');
    
    // File input change event
    fileInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });
    
    // Drag and drop events
    uploadBox.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadBox.classList.add('dragover');
    });
    
    uploadBox.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadBox.classList.remove('dragover');
    });
    
    uploadBox.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadBox.classList.remove('dragover');
        
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
});

function handleFile(file) {
    // Validate file type
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff'];
    
    if (!allowedTypes.includes(file.type)) {
        alert('Invalid file type. Please upload a JPG, PNG, BMP, or TIFF image.');
        return;
    }
    
    // Validate file size (16MB max)
    if (file.size > 16 * 1024 * 1024) {
        alert('File size exceeds 16MB. Please upload a smaller image.');
        return;
    }
    
    selectedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = function(e) {
        const previewImg = document.getElementById('previewImg');
        const imagePreview = document.getElementById('imagePreview');
        const uploadContent = document.querySelector('.upload-content');
        
        previewImg.src = e.target.result;
        imagePreview.style.display = 'block';
        uploadContent.style.display = 'none';
        
        // Enable analyze button
        document.getElementById('analyzeBtn').disabled = false;
    };
    
    reader.readAsDataURL(file);
}

function removeImage() {
    selectedFile = null;
    
    const imagePreview = document.getElementById('imagePreview');
    const uploadContent = document.querySelector('.upload-content');
    const fileInput = document.getElementById('fileInput');
    
    imagePreview.style.display = 'none';
    uploadContent.style.display = 'block';
    fileInput.value = '';
    
    // Disable analyze button
    document.getElementById('analyzeBtn').disabled = true;
    
    // Hide results
    document.getElementById('resultsSection').style.display = 'none';
}

function analyzeImage() {
    if (!selectedFile) {
        alert('Please select an image first.');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('model', document.getElementById('modelSelect').value);
    
    // Show loading
    document.getElementById('loading').style.display = 'block';
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('analyzeBtn').disabled = true;
    
    // Send request
    fetch('/api/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('analyzeBtn').disabled = false;
        
        if (data.success) {
            displayResults(data);
        } else {
            alert('Error: ' + data.error);
        }
    })
    .catch(error => {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('analyzeBtn').disabled = false;
        alert('Error analyzing image: ' + error.message);
    });
}

function displayResults(data) {
    // Show results section
    document.getElementById('resultsSection').style.display = 'block';
    
    // Scroll to results
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
    
    // Update prediction card
    const predictionCard = document.getElementById('predictionCard');
    const predictionIcon = document.getElementById('predictionIcon');
    const predictionText = document.getElementById('predictionText');
    const confidenceText = document.getElementById('confidenceText');
    const modelUsed = document.getElementById('modelUsed');
    
    const isAuthentic = data.prediction === 'authentic';
    
    if (isAuthentic) {
        predictionCard.className = 'prediction-card authentic';
        predictionIcon.innerHTML = '<i class="fas fa-check-circle"></i>';
        predictionText.textContent = 'AUTHENTIC';
    } else {
        predictionCard.className = 'prediction-card forged';
        predictionIcon.innerHTML = '<i class="fas fa-exclamation-triangle"></i>';
        predictionText.textContent = 'FORGED';
    }
    
    confidenceText.textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
    modelUsed.textContent = data.model_used;
    
    // Update probability bars
    const probAuthentic = data.probabilities.authentic;
    const probForged = data.probabilities.forged;
    
    document.getElementById('probAuthentic').textContent = `${(probAuthentic * 100).toFixed(1)}%`;
    document.getElementById('probForged').textContent = `${(probForged * 100).toFixed(1)}%`;
    
    // Animate bars
    setTimeout(() => {
        document.getElementById('barAuthentic').style.width = `${probAuthentic * 100}%`;
        document.getElementById('barForged').style.width = `${probForged * 100}%`;
    }, 100);
    
    // Update image info
    document.getElementById('infoFilename').textContent = data.image_info.filename;
    document.getElementById('infoDimensions').textContent = `${data.image_info.width} × ${data.image_info.height}`;
    document.getElementById('infoSize').textContent = data.image_info.size;
    document.getElementById('infoPixels').textContent = data.image_info.total_pixels.toLocaleString();
    
    // Update visualization
    const visualizationImg = document.getElementById('visualizationImg');
    visualizationImg.src = '/' + data.visualization + '?t=' + new Date().getTime();
}

function resetAnalysis() {
    document.getElementById('resultsSection').style.display = 'none';
    removeImage();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function downloadReport() {
    // Create a simple text report
    const predictionText = document.getElementById('predictionText').textContent;
    const confidenceText = document.getElementById('confidenceText').textContent;
    const modelUsed = document.getElementById('modelUsed').textContent;
    const filename = document.getElementById('infoFilename').textContent;
    const dimensions = document.getElementById('infoDimensions').textContent;
    const probAuthentic = document.getElementById('probAuthentic').textContent;
    const probForged = document.getElementById('probForged').textContent;
    
    const report = `
AI IMAGE FORGERY DETECTION REPORT
=====================================

PREDICTION: ${predictionText}
${confidenceText}
Model Used: ${modelUsed}

IMAGE INFORMATION
-----------------
Filename: ${filename}
Dimensions: ${dimensions}

PROBABILITY DISTRIBUTION
------------------------
Authentic: ${probAuthentic}
Forged: ${probForged}

Report generated: ${new Date().toLocaleString()}

AI Image Forgery Detection System
    `;
    
    // Download as text file
    const blob = new Blob([report], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `forgery_report_${filename}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}
