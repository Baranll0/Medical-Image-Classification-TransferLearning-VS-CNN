
# Comparison of Transfer Learning and CNN Architectures for Medical Image Classification

This project explores the application of convolutional neural networks (CNNs) and transfer learning methods for classifying medical images. Specifically, it compares the performance of the InceptionResNetV2 architecture and Vision Transformer (ViT) on a brain tumor classification dataset.

---

## Dataset

The dataset used in this project consists of brain MRI images labeled as "Tumor" and "No Tumor." The dataset is split into training and testing subsets with an 80:20 ratio.

---

## Preprocessing

### Steps Performed on Images
Before training, images were preprocessed with the following techniques:
1. **Resizing:**
   All images were resized to a uniform size using cubic interpolation for compatibility with the CNN architecture.

2. **Gray-Scaling:**
   Images were converted to grayscale to reduce computational complexity and focus on essential features.

3. **Gaussian Blur:**
   Applied Gaussian blur to remove noise while preserving image edges.

4. **Thresholding:**
   Thresholding was used to segment the image into binary regions, followed by erosion and dilation to remove noise and small artifacts.

5. **Contour Detection and Cropping:**
   The largest contour was identified, and extreme points (leftmost, rightmost, topmost, bottommost) were used to crop the region of interest.

### Augmentation
Additional augmented images were generated to improve generalization. Augmentation methods included:
- Random rotations.
- Sharpness adjustments.

---

## Model Architectures

### CNN Model: InceptionResNetV2


![InceptionResnetV2 Architecture](https://raw.githubusercontent.com/Masterx-AI/Inception-ResNet-V2_Implementation/main/IR_v2.png)

The CNN model was implemented using the InceptionResNetV2 architecture. Pre-trained weights from ImageNet were used for transfer learning to speed up training and improve accuracy.

- **Train Accuracy:** 0.91
- **Validation Accuracy:** 0.86
- **Train Loss:** 0.20
- **Validation Loss:** 0.38

**Test Performance:**
- **Test Accuracy:** 0.86
- **Test Loss:** 0.38

### Transfer Learning: Vision Transformer (ViT)
ViT was utilized for image classification. The pre-trained `google/vit-base-patch16-224-in21k` model was fine-tuned for the task.

- **Train Accuracy:** 0.97
- **Validation Loss:** 0.34



## Results

### CNN (InceptionResNetV2):
- Moderate accuracy with good generalization.
- Effective in identifying core features of brain MRI images but slightly less robust compared to ViT.

### Transfer Learning (ViT):
- Superior performance with an accuracy of 0.97.
- Advanced feature extraction and generalization ability due to transformer-based architecture.

---

## Confusion Matrix

For Transfer Learning:
![Confusion Matrix](https://i.imgur.com/1cZUEWn.png)


---

## Conclusion

- **InceptionResNetV2** demonstrated good accuracy and efficiency for this task, making it a strong choice for real-time deployment scenarios.
- **ViT** surpassed InceptionResNetV2 in terms of accuracy and generalization, proving its strength in medical image classification tasks.

Both approaches showed promise, and future work can involve combining these methods or exploring other advanced architectures like EfficientNet or hybrid CNN-transformer models.

---

## How to Run

1. Clone the repository.
2. Install the required dependencies using:
   ```bash
   pip install -r requirements.txt
   ```
---

## Acknowledgements
- **Dataset:** [Brain MRI Images for Brain Tumor Detection (Kaggle)](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection).
- **Models:** Pre-trained architectures from Keras and Hugging Face Transformers.# Medical-Image-Classification-TransferLearning-VS-CNN
# Medical-Image-Classification-TransferLearning-VS-CNN2
