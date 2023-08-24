import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

# Load MobileNetV2 model
model = models.mobilenet_v2(pretrained=True)
model.eval()

# Preprocess the image
image_path = 'path_to_your_image.jpg'
img = Image.open(image_path).convert('RGB')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
img_tensor = preprocess(img)
img_tensor = torch.unsqueeze(img_tensor, 0)

# Perform inference
with torch.no_grad():
    outputs = model(img_tensor)

# Load ImageNet class labels
with open('imagenet_classes.json') as f:
    class_labels = json.load(f)

# Get top predicted classes
_, indices = torch.topk(outputs, 5)
probabilities = torch.nn.functional.softmax(outputs, dim=1)[0] * 100

# Print top predicted classes and probabilities
for i in range(5):
    print(f"{class_labels[indices[0][i]]}: {probabilities[indices[0][i]].item():.2f}%")

# Save the model
model_save_path = 'mobilenetv2_model.pth'
torch.save(model.state_dict(), model_save_path)

# Report model performance
# Load and preprocess the same image for monitoring
img_monitor = preprocess(img)
img_monitor = torch.unsqueeze(img_monitor, 0)

# Load the saved model
model_loaded = models.mobilenet_v2(pretrained=False)
model_loaded.load_state_dict(torch.load(model_save_path))
model_loaded.eval()

# Perform inference for monitoring
with torch.no_grad():
    outputs_monitor = model_loaded(img_monitor)

# Predicted class and confidence
predicted_class = indices[0][0].item()
confidence = probabilities[indices[0][0]].item()

# Report to AI Platform Model Monitoring
project_id = 'your-project-id'
model_name = 'mobilenetv2'
model_version = 'v1'

credentials = GoogleCredentials.get_application_default()
service = discovery.build('ml', 'v1', credentials=credentials)

# Create the endpoint for model monitoring
endpoint = f'projects/{project_id}/locations/global/endpoints/{model_name}'

# Create the monitoring report
request_data = {
    "instances": [
        {"image": img_monitor[0].numpy().tolist()}
    ],
    "predictions": [
        {"score": confidence, "predicted_class": predicted_class}
    ]
}

# Report the monitoring data
service.projects().locations().endpoints().predict(
    name=endpoint,
    body=request_data
).execute()
