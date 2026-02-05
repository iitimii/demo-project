const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('image-input');
const previewContainer = document.getElementById('preview-container');
const previewImage = document.getElementById('preview');
const analyzeBtn = document.getElementById('analyze-btn');
const resultDiv = document.getElementById('result');

// Handle click to upload
uploadArea.addEventListener('click', () => fileInput.click());

// Handle file selection
fileInput.addEventListener('change', handleFileSelect);

// Drag and drop handlers
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
  uploadArea.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
  e.preventDefault();
  e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
  uploadArea.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
  uploadArea.addEventListener(eventName, unhighlight, false);
});

function highlight() {
  uploadArea.classList.add('dragover');
}

function unhighlight() {
  uploadArea.classList.remove('dragover');
}

uploadArea.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
  const dt = e.dataTransfer;
  const files = dt.files;

  // Update input files
  fileInput.files = files;

  // Trigger the same preview logic
  handleFileSelect({ target: { files } });
}

function handleFileSelect(event) {
  const file = event.target.files[0];
  if (!file) return;

  // Show preview
  const reader = new FileReader();
  reader.onload = function (e) {
    previewImage.src = e.target.result;
    previewContainer.style.display = 'block';
    resultDiv.style.display = 'none'; // Hide previous results

    // Update text
    const uploadText = document.querySelector('.upload-text');
    if (uploadText) uploadText.textContent = `Selected: ${file.name}`;
  };
  reader.readAsDataURL(file);
}

// Handle Analysis
analyzeBtn.addEventListener('click', async () => {
  const file = fileInput.files[0];
  if (!file) {
    alert('Please select an image first');
    return;
  }

  // UI Loading State
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = 'Analyzing...';
  resultDiv.style.display = 'none';

  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const data = await response.json();

    // Expect backend to return: { class: "Banana", confidence: 0.92 }
    // Fallbacks included so it doesn't explode if you named fields differently.
    const fruitLabel = data.class ?? data.label ?? 'Unknown';
    const confidence = Number(data.confidence ?? data.probability ?? 0);
    const confidencePercent = Math.max(0, Math.min(100, confidence * 100)).toFixed(1);

    // Update Result UI
    resultDiv.innerHTML = `
      <div class="result-header">Fruit Classification</div>
      <div class="result-value classification-result">${fruitLabel}</div>
      <div style="font-size: 0.9rem; color: #6b7280; margin-top:0.5rem;">
        Confidence: ${confidencePercent}%
      </div>
      <div class="confidence-bar-bg">
        <div class="confidence-bar-fill" style="width: ${confidencePercent}%"></div>
      </div>
    `;

    resultDiv.style.display = 'block';
  } catch (error) {
    console.error('Error:', error);
    alert('An error occurred during analysis. Make sure the backend is running.');
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = 'Analyze Fruit';
  }
});
