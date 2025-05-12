import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import json

class ClassificationModel:
    def __init__(self, devicestr=None, num_classes=4, lr=0.001):
        # Set up device
        self.num_classes = num_classes
        if devicestr is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(devicestr)
        self.train_transform  =transforms.Compose([
                transforms.Resize(256),                      # make the smaller edge 256px
                transforms.CenterCrop(224),                  # or RandomResizedCrop(224) for more variation
                transforms.RandomHorizontalFlip(p=0.5),      # randomly flip left↔right
                transforms.RandomRotation(degrees=15),       # rotate ±15°
                transforms.ColorJitter(                     # random brightness/contrast/saturation
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1
                ),
                transforms.ToTensor(),                       # now turn PIL → [0–1] tensor
                transforms.Normalize(                        # finally normalize channels
                    mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]
                ),
            ])
        self.predict_transform = transforms.Compose([
            transforms.Resize(256),            # Resize the smaller edge to 256 while keeping the aspect ratio.
            transforms.CenterCrop(224),        # Crop the center 224x224 region.
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        self.model = self.model.to(self.device)
        self.classes = None
        # Set loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)    
    
    def train(self, train_dir, batch_size, num_epochs=1):
        # ...existing code to train...
        self.train_dataset = datasets.ImageFolder(train_dir, transform=self.train_transform)
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
    
    def evaluate(self, test_dir, batch_size,rate=1):
        # ...existing code to evaluate...
        self.test_dataset = datasets.ImageFolder(test_dir, transform=self.predict_transform)
        # if rate <1: split the dataset
        if rate < 1:
            test_size = int(len(self.test_dataset) * rate)
            indices = torch.randperm(len(self.test_dataset))[:test_size]
            self.test_dataset = torch.utils.data.Subset(self.test_dataset, indices)
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
        model_dir = os.path.join('models', name)
        os.makedirs(model_dir, exist_ok=True)
        save_path = os.path.join(model_dir, 'model.pth')
        torch.save(self.model.state_dict(), save_path)
        classes_file = os.path.join(model_dir,'classes.json')
        with open(classes_file, 'w') as f:
            json.dump(self.train_dataset.classes, f)
    
    def load_model(self, name):
        load_path = 'models/' + name + '/model.pth'
        if not os.path.exists(load_path):
            return False
        self.model.load_state_dict(torch.load(load_path,map_location=self.device))
        self.model.eval()
        self.classes, self.translation = self.load_classes(name)
    
    @staticmethod
    def load_classes(name):
        classes=None
        translation=None
        classes_file = os.path.join('models', name,'classes.json')
        translation_file = os.path.join('models', name,'translation.json')
        if os.path.exists(classes_file):
            with open(classes_file, 'r') as f:
                classes = json.load(f)
        if os.path.exists(translation_file):
            with open(translation_file, 'r') as f:
                translation = json.load(f)
        else:
            translation = classes
        # make sure the length of classes and translation is the same
        if len(classes) != len(translation):
            raise ValueError(f"Length of classes ({len(classes)}) and translation ({len(translation)}) do not match.")
        return classes, translation
    
    def classify_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return self.classify_ram_image(image)
    def classify_ram_image(self, ram_image):
        image = ram_image.convert("RGB")
        image = self.predict_transform(image).unsqueeze(0)  # Add batch dimension
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image.to(self.device))
            _, predicted = torch.max(outputs, 1)
        return predicted.item(), self.classes[predicted.item()], self.translation[predicted.item()]
    

def load_and_retrain_model(model_name:str,train_dir:str,num_epochs=1,batch_size=400,class_count=4):
    model_instance = ClassificationModel(num_classes=class_count)
    model_instance.load_model(model_name)
    model_instance.train(train_dir, batch_size=batch_size, num_epochs=num_epochs)
    model_instance.save_model(model_name)
def load_and_evaluate_model(model_name:str,test_dir:str,num_classes,batch_size=400,rate=1):
    model_instance = ClassificationModel(num_classes=num_classes)
    model_instance.load_model(model_name)
    model_instance.evaluate(test_dir, batch_size=batch_size,rate=rate)
# Example usage:
if __name__ == '__main__':
    # load_and_retrain_model('grapemehltau', 'D:/microbify/weinreebe/mixed', num_epochs=10, batch_size=256)
    # load_and_evaluate_model('grapemehltau', 'D:/microbify/weinreebe/mixed')
    # load_and_retrain_model('mehltau', 'D:/microbify/weinreebe/release', num_epochs=2, batch_size=256)
    # load_and_evaluate_model('mehltau', 'D:/microbify/weinreebe/release')
    # load_and_retrain_model('fullmix', 'D:/microbify/PlantVillage/', num_epochs=2, batch_size=256, class_count=15)
    # load_and_retrain_model('fullmix', 'D:/microbify/aggregate/', num_epochs=4, batch_size=400, class_count=27)
    load_and_evaluate_model('fullmix', 'D:/microbify/aggregate/',27, batch_size=400, rate=0.1)
  
    