document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const browseButton = document.getElementById('browse-button');
    const uploadForm = document.getElementById('upload-form');
    const urlForm = document.getElementById('url-form');
    const previewContainer = document.getElementById('preview-container');
    const previewImage = document.getElementById('preview-image');
    const removeImageButton = document.getElementById('remove-image');
    const resultsSection = document.getElementById('results-section');
    const resultImage = document.getElementById('result-image');
    const topName = document.getElementById('top-name');
    const topIdentity = document.getElementById('top-identity');
    const topConfidence = document.getElementById('top-confidence');
    const predictionsList = document.getElementById('predictions-list');
    const loadingSpinner = document.getElementById('loading-spinner');
    const errorAlert = document.getElementById('error-alert');
    const errorMessage = document.getElementById('error-message');

    // File Upload Handling
    browseButton.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });

    // Drag and Drop Handling
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            dropArea.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            dropArea.classList.remove('dragover');
        });
    });

    dropArea.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    });

    // File Preview
    function handleFiles(files) {
        if (files.length) {
            const file = files[0];
            if (file.type.match('image.*')) {
                const reader = new FileReader();
                
                reader.onload = (e) => {
                    previewImage.src = e.target.result;
                    previewContainer.classList.remove('d-none');
                    dropArea.classList.add('d-none');
                };
                
                reader.readAsDataURL(file);
            } else {
                showError('Please upload an image file (JPEG, PNG, etc.)');
            }
        }
    }

    // Remove Image
    removeImageButton.addEventListener('click', () => {
        fileInput.value = '';
        previewContainer.classList.add('d-none');
        dropArea.classList.remove('d-none');
        hideError();
    });

    // Form Submission
    uploadForm.addEventListener('submit', (e) => {
        e.preventDefault();
        if (fileInput.files.length) {
            submitForm('upload', new FormData(uploadForm));
        } else {
            showError('Please select an image to upload');
        }
    });

    urlForm.addEventListener('submit', (e) => {
        e.preventDefault();
        submitForm('url', new FormData(urlForm));
    });

    // API Calls
    function submitForm(type, formData) {
        // Hide previous results and errors
        resultsSection.classList.add('d-none');
        hideError();
        
        // Show loading spinner
        loadingSpinner.classList.remove('d-none');
        
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading spinner
            loadingSpinner.classList.add('d-none');
            
            if (data.error) {
                showError(data.error);
                return;
            }
            
            // Display results
            displayResults(data, type);
        })
        .catch(error => {
            loadingSpinner.classList.add('d-none');
            showError('Error connecting to server: ' + error.message);
        });
    }

    // Display Results
    function displayResults(data, type) {
        // Set top prediction
        topName.textContent = data.celebrity_name || 'Unknown';
        topIdentity.textContent = data.celebrity_id;
        topConfidence.textContent = (data.confidence * 100).toFixed(2) + '%';
        
        // Set result image
        if (type === 'upload') {
            if (data.image_path) {
                resultImage.src = '/' + data.image_path;
            } else {
                resultImage.src = previewImage.src;
            }
        } else {
            resultImage.src = document.getElementById('image-url').value;
        }
        
        // Generate predictions list
        predictionsList.innerHTML = '';
        data.top_predictions.slice(1).forEach((pred, index) => {
            const listItem = document.createElement('li');
            listItem.className = 'list-group-item d-flex justify-content-between align-items-center';
            
            // Calculate confidence color
            const confidence = pred.confidence * 100;
            let confidenceClass = 'bg-danger';
            if (confidence > 30) confidenceClass = 'bg-warning';
            if (confidence > 60) confidenceClass = 'bg-success';
            
            listItem.innerHTML = `
                <div>
                    <span class="fw-bold">${index + 2}. ${pred.celebrity_name || 'Unknown'}</span>
                    <small class="d-block text-muted">ID: ${pred.celebrity_id}</small>
                </div>
                <span class="badge ${confidenceClass} confidence-badge">${confidence.toFixed(2)}%</span>
            `;
            
            predictionsList.appendChild(listItem);
        });
        
        // Show results section with animation
        resultsSection.classList.remove('d-none');
        resultsSection.classList.add('fade-in');
        
        // Scroll to results
        resultsSection.scrollIntoView({behavior: 'smooth'});
    }

    // Error Handling
    function showError(message) {
        errorMessage.textContent = message;
        errorAlert.classList.remove('d-none');
    }
    
    function hideError() {
        errorAlert.classList.add('d-none');
    }
}); 