from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import ssl
import urllib.request
from PIL import Image
import os
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Define transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load training dataset
train_dataset = datasets.ImageFolder('/Users/apple/Documents/grape_disease_test/dataset/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# Define the model (ResNet18 with modified final layer)
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)  # 4 classes

model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    total_iterations = len(train_loader)

    for iteration, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Iteration {iteration+1}/{total_iterations}, Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} completed. Avg Loss: {running_loss / len(train_loader):.4f}")

# Save the trained model
model_dir = '/Users/apple/Documents/grape_disease_test/model'
os.makedirs(model_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))
print("Model saved successfully.")

# Load test dataset and evaluate accuracy
model.eval()
test_dataset = datasets.ImageFolder('/Users/apple/Documents/grape_disease_test/dataset/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Model Accuracy on Test Set: {accuracy:.2f}%")

# Define class name mapping
class_names = {
    0: "Black Rot",
    1: "Esca (Black Measles)",
    2: "Leaf Blight (Isariopsis Leaf Spot)",
    3: "Healthy"
}

# Load and classify a single image
image_path = '/Users/apple/Documents/Git_grapeDisease_test/microbify/uploads/esca_IFV_1.jpg'
image = Image.open(image_path).convert('RGB')

# Apply transformations
image_transformed = transform(image).unsqueeze(0).to(device)

# Perform inference
model.eval()
with torch.no_grad():
    outputs = model(image_transformed)
    _, predicted = torch.max(outputs, 1)

# Get predicted disease name
predicted_class = class_names[predicted.item()]
print(f"Predicted Disease: {predicted_class}")
