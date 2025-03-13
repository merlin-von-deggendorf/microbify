import os
from flask import Flask, render_template, jsonify, request
import resnet18
import io
from PIL import Image

app = Flask(__name__)

# Set the upload folder and create it if it doesn't exist.
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

grapes = resnet18.ClassificationModel(num_classes=4)
grapes.load_model('grapes')

@app.route('/')
def index():
    # Serve the classifier.html template
    return render_template('classifier.html')

@app.route('/get_data')
def get_data():
    # Return dynamic JSON data.
    data = {"message": "Hello from Flask!", "value": 42}
    return jsonify(data)

@app.route('/upload', methods=['POST'])
def upload():
    # Check if the POST request contains the file part.
    if 'file' not in request.files:
        return jsonify({'message': 'No file part in request.'})
    
    file = request.files['file']
    
    # Check if a file is selected.
    if file.filename == '':
        return jsonify({'message': 'No file selected.'})
    
    # Save the file in the uploads directory.
    try:
        # Read the file directly into memory and open as an image
        image_stream = io.BytesIO(file.read())
        image = Image.open(image_stream)
        
        # Optionally, if the model expects a different format, perform any necessary conversion here
        
        # Pass the image directly to the classifier.
        # You'll need to update classify_image in resnet18.ClassificationModel to work with a PIL Image.
        result = grapes.classify_ram_image(image)
        
        return jsonify({'message': f'Classified as {result}.'})
    except Exception as e:
        return jsonify({'message': f'Error during classification: {str(e)}'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)