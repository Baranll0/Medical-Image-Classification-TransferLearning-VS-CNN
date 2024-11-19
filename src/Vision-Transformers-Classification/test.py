import os
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from PIL import Image

# Test dizini
test_dir = '/dataset/yeni-dataset-kaggle/brain_tumor_dataset/test'

# Test verisini yükle
def load_image_data(data_dir):
    image_paths = []
    labels = []
    for label in ['no', 'yes']:
        folder_path = os.path.join(data_dir, label)
        for filename in os.listdir(folder_path):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                image_paths.append(os.path.join(folder_path, filename))
                labels.append(1 if label == 'yes' else 0)
    return image_paths, labels

# Test verisini yükle
image_paths, labels = load_image_data(test_dir)
test_data = Dataset.from_dict({'image': image_paths, 'label': labels})

# Görüntü işleme
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]

normalize = Normalize(mean=image_mean, std=image_std)
_test_transforms = Compose([
    Resize((size, size)),
    ToTensor(),
    normalize,
])


def transforms(examples):
    images = [Image.open(path).convert("RGB") for path in examples["image"]]
    examples["pixel_values"] = [_test_transforms(image) for image in images]
    return examples


# Test verisi üzerinde dönüşüm uygula
test_data = test_data.map(transforms, batched=True, batch_size=32)
test_data.set_format("torch", columns=["pixel_values", "label"])  # Important: set format to "torch"


# Kaydedilen modeli yükle
saved_model_path = "/media/baran/Disk1/gonder/Brain-Tumor-Detection-VT-vs-CNN/src/models/new-saved_model"
model = ViTForImageClassification.from_pretrained(saved_model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
model.to(device)

# Test için dataloader oluştur
test_loader = DataLoader(test_data, batch_size=32)


# Modeli değerlendirmeye al
model.eval()
predictions = []
true_labels = []

# Tahminler yap
with torch.no_grad():
    for batch in test_loader:
        inputs = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)

        outputs = model(inputs).logits
        preds = torch.argmax(outputs, dim=-1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Sonuçları yazdır
print("Classification Report:\n", classification_report(true_labels, predictions, target_names=["no", "yes"]))
print("Confusion Matrix:\n", confusion_matrix(true_labels, predictions))