function onLoad() {
    let uploadForm = document.getElementById('upload-form');
    let fileInput = document.getElementById('file-input');
    let dropZone = document.getElementById('drop-zone');
    let uploadBtn = document.getElementById('upload-btn');
    let previewBg = document.getElementById('preview-bg');
    let outputDiv = document.getElementById('output-div');

    // When drop zone is clicked, trigger file input.
    dropZone.addEventListener('click', function(e) {
        fileInput.click();
    });

    // When a file is selected, show preview and automatically upload.
    fileInput.addEventListener('change', function(e) {
        let file = e.target.files[0];
        if (file) {
            let reader = new FileReader();
            reader.onload = function(event) {
                // Show preview
                previewBg.src = event.target.result;
                previewBg.style.display = 'block';
                
                // Automatically upload the file to the server.
                let formData = new FormData();
                formData.append('file', file);

                fetch('/upload', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    outputDiv.innerHTML = data.message;
                })
                .catch(error => {
                    outputDiv.innerHTML = 'Error uploading file.';
                });
            };
            reader.readAsDataURL(file);
        }
    });
}

document.addEventListener('DOMContentLoaded', onLoad);