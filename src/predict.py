import torch
from src.models.model import CustomModel
from src.data.preprocessing import preprocess_image

# Load the trained model
model = CustomModel(num_classes=10)
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()

# Load and preprocess the image for inference
image_path = 'path_to_image.jpg'
img = preprocess_image(image_path)

# Perform inference
with torch.no_grad():
    output = model(img)

# Get the predicted class
_, predicted_class = torch.max(output, 1)
print(f"Predicted class: {predicted_class.item()}")
