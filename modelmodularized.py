import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

class ClassificationModel:
    def __init__(self, devicestr=None, num_classes=4, lr=0.001):
        # Set up device
        if devicestr is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(devicestr)
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize the model with a pretrained ResNet18 and adjust the final layer
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        self.model = self.model.to(self.device)
        
        # Set loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)    
    
    def train(self, train_dir, batch_size, num_epochs=1):
        # Prepare datasets and dataloaders
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
    
    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)
    
    def predict_single_image(self, image_path):
        """
        Classifies a single image and returns the predicted class.
        
        Parameters:
            image_path (str): Path to the input image.
            
        Returns:
            str: Predicted class label.
        """
        # Open the image and apply transformations
        image = Image.open(image_path).convert('RGB')
        image_transformed = self.transform(image)
        # Add a batch dimension
        image_transformed = image_transformed.unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image_transformed)
            _, predicted = torch.max(outputs, 1)
        
        # Assume the model was trained with ImageFolder where classes attribute exists.
        predicted_class = self.train_dataset.classes[predicted.item()]
        print(f"Predicted class: {predicted_class}")
        return predicted_class

def train_and_save_model():
    train_dir = 'd:/microbify/weinreebe/kaggle/train'
    test_dir = 'd:/microbify/weinreebe/kaggle/test'
    model_instance = ClassificationModel()
    model_instance.train(train_dir, batch_size=256, num_epochs=1)
    model_instance.save_model('d:/microbify/weinreebe/kaggle/model.pth')

def load_and_evaluate_model():
    test_dir = 'd:/microbify/weinreebe/kaggle/test'
    model_instance = ClassificationModel()
    model_instance.model.load_state_dict(torch.load('d:/microbify/weinreebe/kaggle/model.pth'))
    model_instance.evaluate(test_dir, batch_size=256)

# Example usage:
if __name__ == '__main__':
    # For evaluating the model
    load_and_evaluate_model()  
    # For predicting a single image, make sure to call predict_single_image with the image path
    # Example:
    # classifier = ClassificationModel()
    # classifier.train_dataset = datasets.ImageFolder('d:/microbify/weinreebe/kaggle/train', transform=classifier.transform)
    # classifier.predict_single_image('d:/microbify/weinreebe/kaggle/single_image.jpg')