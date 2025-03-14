import os
from flask import Flask, request, jsonify, render_template, flash
from werkzeug.utils import secure_filename
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from modelmodularizedA import ClassificationModel  # Corrected import

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Needed for flashing messages

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize the model
grape_model = ClassificationModel(num_classes=4)
grape_model.load_model('grapes')

@app.route('/')
def index():
    return render_template('classifier.html')

@app.route('/gallery')
def gallery():
    # Get list of filenames in the uploads folder.
    image_folder = os.path.join(app.config['UPLOAD_FOLDER'])
    images = os.listdir(image_folder) if os.path.exists(image_folder) else []
    return render_template('gallery.html', images=images)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part in request.'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No file selected.'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)

        predicted_class = grape_model.classify_image(path)
        print(f"Predicted class: {predicted_class}")

        return jsonify({'message': f'File successfully uploaded and classified as: {predicted_class}', 'class': predicted_class})
    else:
        return jsonify({'message': 'Allowed file types are png, jpg, jpeg, gif'}), 400

if __name__ == '__main__':
    app.run(debug=True)