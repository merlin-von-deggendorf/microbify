function onLoad() {
    let uploadForm = document.getElementById('upload-form');
    
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