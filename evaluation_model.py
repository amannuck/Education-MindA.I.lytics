import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import os
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.models as models
import os


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # The FER-2013 images are grayscale, so in_channels=1.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # FER-2013 images are 48x48 pixels, so after two pooling layers, the size will be 12x12.
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 4)  # 4 classes to classify
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)  # Flatten the layer
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def load_model(model_path):
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def evaluate_on_dataset(model, dataset_path, transform):
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    data_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on the dataset: {accuracy:.2f}%')


def predict_image(model, image_path, transform):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_label = class_labels[predicted.item()]
        print(f'Predicted class: {predicted_label}')


# Test class labels
class_labels = {
    0: "engaged",
    1: "happy",
    2: "neutral",
    3: "surprise"
}

if __name__ == '__main__':
    model_path = "C:\\Users\\User\\best_model.pth"
    dataset_path = "C:\\Users\\User\\Desktop\\472 project\\project 1\\COMP-472-Project-main\\COMP-472-Project-main\\datasets\\final_clean\\test"
    image_path = "C:\\Users\\User\\Desktop\\472 project\\project 1\\COMP-472-Project-main\\COMP-472-Project-main\\datasets\\final_clean\\test\\surprise\\PrivateTest_1338609.jpg"

    # transformations
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    model = load_model(model_path)
    evaluate_on_dataset(model, dataset_path, transform)  # Evaluate on the entire "test" dataset
    predict_image(model, image_path, transform)  # Predict a single image
