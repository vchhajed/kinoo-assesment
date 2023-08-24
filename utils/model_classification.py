import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader

# Load MobileNetV2 pretrained on ImageNet
model = models.mobilenet_v2(pretrained=True)
num_classes = 10  # Number of classes in your dataset

# Modify the classifier to match the number of classes in your dataset
model.classifier[1] = nn.Linear(model.last_channel, num_classes)

# Define transformations and load the custom dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root='path_to_train_data', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()  # Set the model in training mode
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Clear gradients

        # Forward pass
        outputs = model(inputs)  # Get model predictions
        loss = criterion(outputs, labels)  # Calculate loss
        
        # Backpropagation and optimization
        loss.backward()  # Compute gradients
        optimizer.step()  # Update model parameters

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')