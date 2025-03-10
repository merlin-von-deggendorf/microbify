from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
# https://data.mendeley.com/datasets/j4xs3kh3fd/2
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
train_dataset = datasets.ImageFolder('d:/microbify/weinreebe/kaggle/train', transform=transform)
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

num_epochs = 1

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    # number of iterations
    total_iterartions = len(train_loader)
    iteration=0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        print(f"Iteration {iteration}/{total_iterartions}")
        iteration+=1
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# test the model
model.eval()
test_dataset = datasets.ImageFolder('d:/microbify/weinreebe/kaggle/test', transform=transform)
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