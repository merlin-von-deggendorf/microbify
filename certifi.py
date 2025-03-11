import torch
from torchvision import models

model = models.resnet18(weights=None)  # Load without downloading
model.load_state_dict(torch.load("/path/to/resnet18-f37072fd.pth"))
