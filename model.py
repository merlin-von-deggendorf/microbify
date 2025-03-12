from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import ssl
import urllib.request
from PIL import Image
import os

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# https://data.mendeley.com/datasets/j4xs3kh3fd/2
# decrease the resolution
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
train_dataset = datasets.ImageFolder('/Users/apple/Documents/grape_disease_test/dataset/train', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

import torch.nn as nn
import torchvision.models as models

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
# Adjust the final layer to the number of classes in your dataset.
model.fc = nn.Linear(num_ftrs, 4)  # num_classes is the number of disease categories

model = model.to(device)
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 25  # Increase the number of epochs

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    # number of iterations
    total_iterations = len(train_loader)
    iteration = 0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        print(f"Iteration {iteration}/{total_iterations}")
        iteration += 1
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# Save the model
model_dir = '/Users/apple/Documents/grape_disease_test/model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))

# test the model
model.eval()
test_dataset = datasets.ImageFolder('/Users/apple/Documents/grape_disease_test/dataset/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(f"Accuracy: {100 * correct / total}")

# Specify the path to the single image
image_path = '/Users/apple/Documents/Git_grapeDisease_test/microbify/uploads/esca_IFV_1.jpg'

# Open the image using PIL
image = Image.open(image_path).convert('RGB')

# Apply the same transforms used for training
image_transformed = transform(image)

# Add a batch dimension since our model expects a 4D tensor (batch_size, channels, height, width)
image_transformed = image_transformed.unsqueeze(0).to(device)

# Set the model to evaluation mode
model.eval()

# Perform inference without tracking gradients
with torch.no_grad():
    outputs = model(image_transformed)
    # Get the predicted class index
    _, predicted = torch.max(outputs, 1)

# If you want to see the corresponding class name, you can use the classes attribute from your dataset
predicted_class = train_dataset.classes[predicted.item()]
print(f"Predicted class: {predicted_class}")