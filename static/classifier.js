function onLoad() {
    let uploadForm = document.getElementById('upload-form');
    let fileInput = document.getElementById('file-input');
    
    // Listen for changes to the file input to show a preview.
    fileInput.addEventListener('change', function(e) {
        let file = e.target.files[0];
        if (file) {
            let reader = new FileReader();
            reader.onload = function(event) {
                let previewImg = document.getElementById('preview');
                if (!previewImg) {
                    previewImg = document.createElement('img');
                    previewImg.id = 'preview';
                    previewImg.style.maxWidth = '300px';
                    previewImg.style.display = 'block';
                    // Insert the preview image after the file input.
                    fileInput.parentNode.insertBefore(previewImg, uploadForm.querySelector('button'));
                }
                previewImg.src = event.target.result;
            };
            reader.readAsDataURL(file);
        }
    });
    
    // Override the default form submission with a fetch call.
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault(); // Prevent default form submission
        let formData = new FormData(uploadForm);
        
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Display response message from the server
            let statusDiv = document.getElementById('upload-status');
            if (!statusDiv) {
                statusDiv = document.createElement('div');
                statusDiv.id = 'upload-status';
                uploadForm.parentNode.appendChild(statusDiv);
            }
            statusDiv.innerText = data.message;
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });
}

document.addEventListener('DOMContentLoaded', onLoad);