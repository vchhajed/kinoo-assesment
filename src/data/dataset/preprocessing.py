import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = [os.path.join(data_dir, img_name) for img_name in os.listdir(data_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = int(img_path.split("_")[0])  # Assuming labels are in the image filenames

        return image, label

if __name__ == "__main__":
    data_directory = "path_to_your_data_directory"
    
    # Define image transformations
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create the dataset
    dataset = CustomDataset(data_dir=data_directory, transform=data_transform)
    
    # Example usage
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    for inputs, labels in dataloader:
        print(inputs.shape, labels)
