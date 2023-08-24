import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from src.data.preprocessing import CustomDataset
from src.models.model import CustomModel
import sys

def evaluate_model(model, test_loader):
    model.eval()
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            true_labels.extend(labels.numpy())
            predicted_labels.extend(predicted.numpy())

    report = classification_report(true_labels, predicted_labels)
    return report

if __name__ == "__main__":
    test_dataset = CustomDataset(...)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = CustomModel(num_classes=10)
    model.load_state_dict(torch.load('trained_model.pth'))

    evaluation_report = evaluate_model(model, test_loader)
    print("Evaluation Report:")
    print(evaluation_report)

    # Add test cases based on evaluation metrics
    expected_metrics = {
        "precision": 0.85,
        "recall": 0.80,
        "f1-score": 0.82
    }

    for metric, expected_value in expected_metrics.items():
        if metric in evaluation_report:
            metric_line = [line for line in evaluation_report.split("\n") if metric in line][0]
            actual_value = float(metric_line.split()[1])
            if actual_value >= expected_value:
                print(f"{metric.capitalize()} Test: Passed")
            else:
                print(f"{metric.capitalize()} Test: Failed")
                sys.exit(1)  # Exit with non-zero status to indicate test failure
