// Batch Processing JavaScript

let selectedFiles = [];
let batchResults = [];

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    const batchFileInput = document.getElementById('batchFileInput');
    
    batchFileInput.addEventListener('change', function(e) {
        handleBatchFiles(Array.from(e.target.files));
    });
});

function handleBatchFiles(files) {
    // Validate and filter files
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff'];
    const validFiles = files.filter(file => allowedTypes.includes(file.type) && file.size <= 16 * 1024 * 1024);
    
    if (validFiles.length === 0) {
        alert('No valid image files selected. Please select JPG, PNG, BMP, or TIFF images under 16MB.');
        return;
    }
    
    if (validFiles.length > 50) {
        alert('Maximum 50 images allowed. Only the first 50 will be processed.');
        validFiles.splice(50);
    }
    
    selectedFiles = validFiles;
    displaySelectedFiles();
}

function displaySelectedFiles() {
    const selectedFilesDiv = document.getElementById('selectedFiles');
    const fileList = document.getElementById('fileList');
    const fileCount = document.getElementById('fileCount');
    
    fileList.innerHTML = '';
    fileCount.textContent = selectedFiles.length;
    
    selectedFiles.forEach((file, index) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <i class="fas fa-file-image"></i>
            <div class="file-item-name">${file.name}</div>
        `;
        fileList.appendChild(fileItem);
    });
    
    selectedFilesDiv.style.display = 'block';
    document.getElementById('batchAnalyzeBtn').disabled = false;
}

function clearFiles() {
    selectedFiles = [];
    document.getElementById('selectedFiles').style.display = 'none';
    document.getElementById('batchFileInput').value = '';
    document.getElementById('batchAnalyzeBtn').disabled = true;
    document.getElementById('batchResultsSection').style.display = 'none';
}

function analyzeBatch() {
    if (selectedFiles.length === 0) {
        alert('Please select images first.');
        return;
    }
    
    const formData = new FormData();
    selectedFiles.forEach(file => {
        formData.append('files[]', file);
    });
    formData.append('model', document.getElementById('batchModelSelect').value);
    
    // Show progress
    document.getElementById('progressSection').style.display = 'block';
    document.getElementById('batchResultsSection').style.display = 'none';
    document.getElementById('batchAnalyzeBtn').disabled = true;
    
    updateProgress(0, selectedFiles.length);
    
    // Send request
    fetch('/api/batch_predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('progressSection').style.display = 'none';
        document.getElementById('batchAnalyzeBtn').disabled = false;
        
        if (data.success) {
            batchResults = data.results;
            displayBatchResults(data);
        } else {
            alert('Error: ' + data.error);
        }
    })
    .catch(error => {
        document.getElementById('progressSection').style.display = 'none';
        document.getElementById('batchAnalyzeBtn').disabled = false;
        alert('Error processing batch: ' + error.message);
    });
    
    // Simulate progress (in real implementation, use websockets or polling)
    simulateProgress();
}

function simulateProgress() {
    let progress = 0;
    const interval = setInterval(() => {
        progress += 10;
        if (progress >= 90) {
            clearInterval(interval);
        }
        updateProgress(Math.floor(selectedFiles.length * progress / 100), selectedFiles.length);
    }, 300);
}

function updateProgress(current, total) {
    const percentage = Math.floor((current / total) * 100);
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const progressStatus = document.getElementById('progressStatus');
    
    progressBar.style.width = percentage + '%';
    progressText.textContent = percentage + '%';
    progressStatus.textContent = `Processing ${current} of ${total} images...`;
}

function displayBatchResults(data) {
    // Show results section
    document.getElementById('batchResultsSection').style.display = 'block';
    document.getElementById('batchResultsSection').scrollIntoView({ behavior: 'smooth' });
    
    // Calculate summary
    const total = data.total;
    const processed = data.processed;
    const authenticCount = data.results.filter(r => r.success && r.prediction === 'authentic').length;
    const forgedCount = data.results.filter(r => r.success && r.prediction === 'forged').length;
    const successRate = (processed / total * 100).toFixed(1);
    
    // Update summary cards
    document.getElementById('summaryTotal').textContent = total;
    document.getElementById('summaryAuthentic').textContent = authenticCount;
    document.getElementById('summaryForged').textContent = forgedCount;
    document.getElementById('summarySuccess').textContent = successRate + '%';
    
    // Populate results table
    const tableBody = document.getElementById('resultsTableBody');
    tableBody.innerHTML = '';
    
    data.results.forEach((result, index) => {
        const row = document.createElement('tr');
        
        if (result.success) {
            const statusClass = result.prediction === 'authentic' ? 'status-authentic' : 'status-forged';
            const statusText = result.prediction === 'authentic' ? 'Authentic' : 'Forged';
            
            row.innerHTML = `
                <td>${index + 1}</td>
                <td>${result.filename}</td>
                <td><span class="status-badge ${statusClass}">${statusText}</span></td>
                <td>${(result.confidence * 100).toFixed(1)}%</td>
                <td>${(result.probabilities.authentic * 100).toFixed(1)}%</td>
                <td>${(result.probabilities.forged * 100).toFixed(1)}%</td>
                <td><i class="fas fa-check-circle" style="color: #2ecc71;"></i> Success</td>
            `;
        } else {
            row.innerHTML = `
                <td>${index + 1}</td>
                <td>${result.filename}</td>
                <td colspan="4" style="color: #e74c3c;">Error: ${result.error}</td>
                <td><i class="fas fa-times-circle" style="color: #e74c3c;"></i> Failed</td>
            `;
        }
        
        tableBody.appendChild(row);
    });
}

function downloadCSV() {
    if (batchResults.length === 0) {
        alert('No results to download.');
        return;
    }
    
    // Create CSV content
    let csv = 'Filename,Prediction,Confidence,Authentic Probability,Forged Probability,Status\n';
    
    batchResults.forEach(result => {
        if (result.success) {
            csv += `"${result.filename}",${result.prediction},${(result.confidence * 100).toFixed(2)}%,${(result.probabilities.authentic * 100).toFixed(2)}%,${(result.probabilities.forged * 100).toFixed(2)}%,Success\n`;
        } else {
            csv += `"${result.filename}",Error,N/A,N/A,N/A,"${result.error}"\n`;
        }
    });
    
    // Download CSV
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `batch_results_${new Date().getTime()}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

function resetBatch() {
    clearFiles();
    document.getElementById('batchResultsSection').style.display = 'none';
    window.scrollTo({ top: 0, behavior: 'smooth' });
}
