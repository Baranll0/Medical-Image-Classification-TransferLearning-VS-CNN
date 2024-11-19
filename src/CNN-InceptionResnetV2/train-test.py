import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pretrainedmodels
from torchvision import transforms, models
from PIL import Image

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
IMG_PATH = '/media/baran/Disk1/gonder/Brain-Tumor-Detection-VT-vs-CNN/dataset/yeni-dataset-kaggle/archive/brain_tumor_dataset'
TRAIN_DIR = 'TRAIN/'
VAL_DIR = 'VAL/'
TEST_DIR = 'TEST/'

# Custom Dataset
class BrainTumorDataset(Dataset):
    def __init__(self, dir_path, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform
        for idx, class_name in enumerate(sorted(os.listdir(dir_path))):
            if not class_name.startswith('.'):
                class_dir = os.path.join(dir_path, class_name)
                for file_name in os.listdir(class_dir):
                    if not file_name.startswith('.'):
                        self.images.append(os.path.join(class_dir, file_name))
                        self.labels.append(idx)  # 0 for 'no', 1 for 'yes'

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        # Read the image using cv2 (BGR format)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = Image.fromarray(image)  # Convert NumPy array to PIL image
        if self.transform:
            image = self.transform(image)
        return image, label

# Data transformations
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # InceptionResNetV2 input size
    transforms.ToTensor(),          # Convert PIL image to Tensor
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize for InceptionResNetV2
])

# Load datasets
train_dataset = BrainTumorDataset(os.path.join(IMG_PATH, TRAIN_DIR), transform)
val_dataset = BrainTumorDataset(os.path.join(IMG_PATH, VAL_DIR), transform)
test_dataset = BrainTumorDataset(os.path.join(IMG_PATH, TEST_DIR), transform)

print(f"Number of samples in train_dataset: {len(train_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Model
class BrainTumorClassifier(nn.Module):
    def __init__(self):
        super(BrainTumorClassifier, self).__init__()
        # Load InceptionResNetV2 model without pre-trained weights
        self.base_model = pretrainedmodels.inceptionresnetv2(pretrained=None)  # No pretrained weights
        self.base_model.last_linear = nn.Linear(self.base_model.last_linear.in_features, 1)  # 1 output unit for binary classification
        self.sigmoid = nn.Sigmoid()  # Sigmoid for binary classification

    def forward(self, x):
        x = self.base_model(x)
        return self.sigmoid(x)  # Sigmoid output for binary classification

model = BrainTumorClassifier().to(device)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            images, labels = images.to(device), labels.float().to(device)
            labels = labels.unsqueeze(1)  # Reshape labels to match output
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer)

# Evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = (outputs > 0.5).cpu().numpy()  # Apply threshold for binary classification
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds)
    cm = confusion_matrix(y_true, y_pred)
    print(classification_report(y_true, y_pred))
    return cm

cm = evaluate_model(model, test_loader)

# Plot confusion matrix
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', normalize=False):
    plt.figure(figsize=(6, 6))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    # Labeling the matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], '.2f' if normalize else 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Plot the normal confusion matrix
plot_confusion_matrix(cm, classes=["No Tumor", "Tumor"], normalize=False)

# Plot the normalized confusion matrix
plot_confusion_matrix(cm, classes=["No Tumor", "Tumor"], normalize=True)

# Save the model
torch.save(model.state_dict(), "brain_tumor_model.pth")
