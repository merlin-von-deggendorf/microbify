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

            // Display the classification result
            let resultDiv = document.getElementById('classification-result');
            if (!resultDiv) {
                resultDiv = document.createElement('div');
                resultDiv.id = 'classification-result';
                uploadForm.parentNode.appendChild(resultDiv);
            }
            resultDiv.innerText = `Predicted Class: ${data.class}`;
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });
}

document.addEventListener('DOMContentLoaded', onLoad);