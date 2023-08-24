import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from src.data.preprocessing import CustomDataset
from src.models.model import CustomModel
import sys

def validate_model(model, validation_loader):
    model.eval()
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for inputs, labels in validation_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            true_labels.extend(labels.numpy())
            predicted_labels.extend(predicted.numpy())

    accuracy = accuracy_score(true_labels, predicted_labels)
    return accuracy

if __name__ == "__main__":
    validation_dataset = CustomDataset(...)
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

    model = CustomModel(num_classes=10)
    model.load_state_dict(torch.load('trained_model.pth'))

    accuracy = validate_model(model, validation_loader)
    print(f"Validation Accuracy: {accuracy:.2f}")

    # Add test cases based on validation scores
    if accuracy >= 0.90:
        print("Validation Test: Passed")
    else:
        print("Validation Test: Failed")
        sys.exit(1)  # Exit with non-zero status to indicate test failure
