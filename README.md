<h1 align="center">MusaNet: Banana Leaf Disease Detection with Transfer Learning</h1>

<p align="center"> <img src="https://via.placeholder.com/600x300?text=Demo+of+MusaNet+Classifying+Banana+Leaves" alt="MusaNet Demo" /> </p>


MusaNet is a convolutional neural network (CNN) designed to detect diseases in banana leaves (Musa × paradisiaca) using transfer learning with MobileNetV2. Trained on a custom dataset of 478 images, the model achieves 91.66% accuracy in classifying three categories: healthy leaves, leaf spot, and Sigatoka. Built with TensorFlow and Python, this tool aids farmers and researchers in early disease detection.

<p>Read the full paper <a href="https://github.com/Hunnter7/Transfer-learning-to-create-MusaNet-CNN/blob/main/Transfer%20learning%20with%20convolutional%20neural%20networks%20for%20advanced%20image%20classification%20and%20detection%20of%20Musa%20%C3%97%20paradisiaca%20leaf%20disease.pdf" title="MusaNet Paper">here</a>.</p>

<p>Read the full google collab notebook <a href="https://github.com/Hunnter7/Transfer-learning-to-create-MusaNet-CNN/blob/main/Transfer_learning_to_create_MusaNet_CNN_model_.ipynb" title="MusaNet collab notebook">here</a>.</p>

Key Features
* Transfer Learning: Fine-tuned MobileNetV2 for rapid training on a small dataset.
* Data Augmentation: Enhanced dataset diversity using rotations, zooms, and shearing.
* High Accuracy: 91.66% overall accuracy in disease classification.
* User-Friendly: Preprocessing scripts and trained model included for easy deployment.

<h2>Materials and Methods</h2>

**Dataset Preparation**
* 478 images split into 3 classes: healthy leaves, leaf spot, and Sigatoka.
* Images standardized to 224x224 pixels with white backgrounds.
* Augmented using ImageDataGenerator with rotations (30°), zooms (0.5–1.5x), and shearing (15°).

**Model Architecture**
* MobileNetV2 base (pre-trained on ImageNet) with frozen layers.
* Custom top layer: 3-node dense layer with Softmax for classification.

```
# Model setup code snippet
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze layers

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(3, activation='softmax')
])

model.compile(optimizer=SGD(learning_rate=0.01), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
```

**Training**
* Trained for 10 epochs with SGD optimizer.
* 80/20 train-test split; achieved 91.66% test accuracy.

<h2>Results</h2>

| Class              | Test Accuracy  |
|--------------------|:--------------:|
| Healthy Leaves     | 95%            |
| Leaf Spot          | 90%            |
| Sigatoka           | 90%            |


**Limitations:**

* Struggles with wide-field images (e.g., full plantation views).
* Requires high-quality, close-up leaf images for reliable predictions.

<h2>Getting started</h2>

* Python 3.8+
* TensorFlow 2.9+
* Jupyter Notebook (optional)


**Installation**

1. Clone the repository:

```
git clone https://github.com/Hunnter7/Transfer-learning-to-create-MusaNet-CNN.git
```

2. Install dependencies

```
pip install -r requirements.txt
```

**Usage**


<h2>Acknowledgments</h2>

* Developed at Universidad Tecnológica La Salle as part of an IEEE-supported research initiative.
* Dataset includes images from field captures and open-source repositories.
