import os
from flask import Flask, request, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models
from PIL import Image

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
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

train_dataset = datasets.ImageFolder('/Users/apple/Documents/grape_disease_test/dataset/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# Load the model
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features  # Get number of input features for final layer
model.fc = nn.Linear(num_ftrs, 4)  # Adjust the final layer to match the number of classes

model.load_state_dict(torch.load('/Users/apple/Documents/grape_disease_test/model/model.pth', map_location=device))
model = model.to(device)
model.eval()  # Set the model to evaluation mode

# Define class name mapping (same as in model.py)
class_names = {
    0: "Black Rot",
    1: "Esca (Black Measles)",
    2: "Leaf Blight (Isariopsis Leaf Spot)",
    3: "Healthy"
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)

            # Open and preprocess image
            image = Image.open(path).convert('RGB')
            image_transformed = transform(image).unsqueeze(0).to(device)

            # Inference
            with torch.no_grad():
                outputs = model(image_transformed)
                _, predicted = torch.max(outputs, 1)

            predicted_class = class_names[predicted.item()]  # Use class_names dictionary

            flash(f'File successfully uploaded and classified as: {predicted_class}')
            return redirect(url_for('index'))
        else:
            flash('Allowed file types are png, jpg, jpeg, gif')
            return redirect(request.url)
    
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
