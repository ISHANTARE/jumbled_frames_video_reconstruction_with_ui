let currentVideoFile = null;
let outputFilePath = null;

// Initialize event listeners when page loads
document.addEventListener('DOMContentLoaded', function() {
    // File upload handling
    document.getElementById('videoInput').addEventListener('change', handleFileSelect);

    const uploadArea = document.getElementById('uploadArea');

    // Drag and drop handlers
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
});

function handleFileSelect(e) {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
}

function handleFile(file) {
    if (!file.type.startsWith('video/')) {
        showError('Please upload a video file (MP4, AVI, MOV, etc.)');
        return;
    }

    // Check file size (max 100MB)
    if (file.size > 100 * 1024 * 1024) {
        showError('File too large. Please upload a video smaller than 100MB.');
        return;
    }

    currentVideoFile = file;

    // Show preview
    const videoPreview = document.getElementById('videoPreview');
    videoPreview.src = URL.createObjectURL(file);
    document.getElementById('uploadPreview').style.display = 'block';

    // Scroll to preview
    videoPreview.scrollIntoView({ behavior: 'smooth' });

    hideError();
    hideResult();
}

async function reconstructVideo() {
    if (!currentVideoFile) {
        showError('Please select a video file first');
        return;
    }

    const reconstructBtn = document.getElementById('reconstructBtn');
    const progressBar = document.getElementById('progressBar');
    const progressFill = document.getElementById('progressFill');

    // Reset UI
    reconstructBtn.disabled = true;
    reconstructBtn.textContent = 'Processing...';
    progressBar.style.display = 'block';
    progressFill.style.width = '0%';
    hideError();
    hideResult();

    // Simulate progress (we can't get real progress from the backend easily)
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += Math.random() * 5;
        if (progress > 90) progress = 90;
        progressFill.style.width = progress + '%';
    }, 500);

    try {
        const formData = new FormData();
        formData.append('video', currentVideoFile);

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        clearInterval(progressInterval);
        progressFill.style.width = '100%';

        if (result.success) {
            showResult(result);
        } else {
            showError(result.error || 'Reconstruction failed');
        }
    } catch (error) {
        clearInterval(progressInterval);
        showError('Network error: ' + error.message);
    } finally {
        reconstructBtn.disabled = false;
        reconstructBtn.textContent = 'Reconstruct Video';
        setTimeout(() => {
            progressBar.style.display = 'none';
        }, 1000);
    }
}

function showResult(result) {
    outputFilePath = result.output_file;

    const resultInfo = document.getElementById('resultInfo');
    resultInfo.innerHTML = `
        <div class="success">
            Reconstruction completed in ${result.execution_time.toFixed(2)} seconds!<br>
            Processed ${result.frame_count} frames.
        </div>
    `;

    const outputPreview = document.getElementById('outputPreview');
    outputPreview.innerHTML = `
        <h4>Reconstructed Video Preview:</h4>
        <video class="video-preview" controls>
            <source src="/download/${outputFilePath}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    `;

    document.getElementById('result').style.display = 'block';
    document.getElementById('result').scrollIntoView({ behavior: 'smooth' });
}

function downloadVideo() {
    if (outputFilePath) {
        window.location.href = `/download/${outputFilePath}`;

        // Clean up the file after some time
        setTimeout(() => {
            fetch(`/cleanup/${outputFilePath}`)
                .catch(err => console.log('Cleanup failed:', err));
        }, 30000); // Clean up after 30 seconds
    }
}

function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    errorDiv.scrollIntoView({ behavior: 'smooth' });
}

function hideError() {
    document.getElementById('errorMessage').style.display = 'none';
}

function hideResult() {
    document.getElementById('result').style.display = 'none';
}

// Clean up object URLs when page unloads
window.addEventListener('beforeunload', function() {
    const videoPreview = document.getElementById('videoPreview');
    if (videoPreview && videoPreview.src) {
        URL.revokeObjectURL(videoPreview.src);
    }
});