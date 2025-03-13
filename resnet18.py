import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import tkinter as tk
from tkinter import filedialog

class ClassificationModel:
    def __init__(self, devicestr=None, num_classes=4, lr=0.001):
        # Set up device
        self.num_classes = num_classes
        if devicestr is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(devicestr)
        self.transform = transforms.Compose([
            transforms.Resize(256),            # Resize the smaller edge to 256 while keeping the aspect ratio.
            transforms.CenterCrop(224),        # Crop the center 224x224 region.
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        
        # Initialize the model with a pretrained ResNet18 and adjust the final layer weights=ResNet18_Weights.IMAGENET1K_V1
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        self.model = self.model.to(self.device)
        
        # Set loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)    
    
    def train(self, train_dir, batch_size, num_epochs=1):
        # ...existing code to train...
        self.train_dataset = datasets.ImageFolder(train_dir, transform=self.transform)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            total_iterations = len(self.train_loader)
            iteration = 0
            for inputs, labels in self.train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                print(f"Iteration {iteration}/{total_iterations}")
                iteration += 1
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / total_iterations}")
    
    def evaluate(self, test_dir, batch_size):
        # ...existing code to evaluate...
        self.test_dataset = datasets.ImageFolder(test_dir, transform=self.transform)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                print(f"Accuracy: {100 * correct / total}")
    
    def save_model(self, name):
        save_path = 'models/' + name + '.pth'
        torch.save(self.model.state_dict(), save_path)
    
    def load_model(self, name):
        """
        Loads model weights using the given name and sets the model to evaluation mode.
        
        Parameters:
            name (str): Name of the saved model (without path or extension).
        """
        load_path = 'models/' + name + '.pth'
        if not os.path.exists(load_path):
            return False
        self.model.load_state_dict(torch.load(load_path))
        return True
    
    def classify_image(self, image_path):
        """
        Classifies a single image and returns the predicted class as an integer.
        
        Parameters:
            image_path (str): Path to the image file.
        
        Returns:
            int: Predicted class label.
        """
        image = Image.open(image_path).convert("RGB")
        return self.classify_ram_image(image)
    def classify_ram_image(self, ram_image):
        """
        Classifies a single image and returns the predicted class as an integer.
        
        Parameters:
            image_path (str): Path to the image file.
        
        Returns:
            int: Predicted class label.
        """
        image = ram_image.convert("RGB")
        image = self.transform(image).unsqueeze(0)  # Add batch dimension
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image.to(self.device))
            _, predicted = torch.max(outputs, 1)
        return predicted.item()
    

def load_and_retrain_model(model_name:str,num_epochs=1,batch_size=256):
    train_dir = 'D:/microbify/weinreebe/kaggle/train'
    model_instance = ClassificationModel(num_classes=4)
    model_instance.load_model(model_name)
    model_instance.train(train_dir, batch_size=256, num_epochs=num_epochs)
    model_instance.save_model(model_name)
def load_and_evaluate_model(model_name:str):
    test_dir = 'D:/microbify/weinreebe/kaggle/test'
    model_instance = ClassificationModel(num_classes=4)
    model_instance.load_model(model_name)
    model_instance.evaluate(test_dir, batch_size=10)
def evaluate_image(model_name: str):
    # Create a hidden Tkinter window
    root = tk.Tk()
    root.withdraw()
    
    # Open file picker dialog for image selection
    image_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
    )
    
    if not image_path:
        print("No image selected.")
        return None
    
    model_instance = ClassificationModel(num_classes=4)
    model_instance.load_model(model_name)
    result = model_instance.classify_image(image_path)
    print(f"Predicted class: {result}")
    return result
# Example usage:
if __name__ == '__main__':
    # pass
    # load_and_retrain_model('grapes',num_epochs=3)
    # load_and_evaluate_model('grapes')
    evaluate_image('grapes')