import os
import random
from datasets import Dataset
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
from torchvision.transforms import Compose, Normalize, RandomRotation, RandomAdjustSharpness, Resize, ToTensor
import torch

# Veri dizinlerini ayarla
data_dir = '/dataset/yeni-dataset-kaggle/brain_tumor_dataset'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

# Eğitim ve test verilerini böl
def split_data(data_dir, train_ratio=0.8):
    for label in ['no', 'yes']:
        source_dir = os.path.join(data_dir, label)
        files = os.listdir(source_dir)
        random.shuffle(files)
        split_index = int(len(files) * train_ratio)
        train_files = files[:split_index]
        test_files = files[split_index:]
        os.makedirs(os.path.join(train_dir, label), exist_ok=True)
        os.makedirs(os.path.join(test_dir, label), exist_ok=True)
        for file in train_files:
            os.rename(os.path.join(source_dir, file), os.path.join(train_dir, label, file))
        for file in test_files:
            os.rename(os.path.join(source_dir, file), os.path.join(test_dir, label, file))

split_data(data_dir)

# Görüntü verilerini yükle
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

# Eğitim verilerini yükle
image_paths, labels = load_image_data(train_dir)
train_data = Dataset.from_dict({'image': image_paths, 'label': labels})

# Görüntü işleme
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]

normalize = Normalize(mean=image_mean, std=image_std)
_train_transforms = Compose([
    Resize((size, size)),
    RandomRotation(15),
    RandomAdjustSharpness(2),
    ToTensor(),
    normalize,
])

def transforms(examples, transform_func):
    images = [Image.open(path).convert("RGB") for path in examples["image"]]
    examples["pixel_values"] = [transform_func(image) for image in images]
    return examples

train_data = train_data.map(lambda examples: transforms(examples, _train_transforms), batched=True, batch_size=32)

# Modeli yapılandırma
id2label = {0: 'no', 1: 'yes'}
label2id = {'no': 0, 'yes': 1}
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)

# Eğitim argümanları
args = TrainingArguments(
    output_dir="Brain-Tumor-Detection",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir='logs',
    remove_unused_columns=False,
    report_to="none"
)

# Eğitim işlemi
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_data,
)

# Modeli eğit
trainer.train()

# Eğitilen modeli kaydet
model.save_pretrained("/media/baran/Disk1/gonder/Brain-Tumor-Detection-VT-vs-CNN/src/models/new-saved_model")
