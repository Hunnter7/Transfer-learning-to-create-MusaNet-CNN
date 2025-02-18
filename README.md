<h1 align="center">MusaNet: Banana Leaf Disease Detection with Transfer Learning</h1>

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

This section explains how to use the MusaNet model for banana leaf disease detection. The model is trained and saved in the SavedModel format, located in the MusaNet folder. You can either load the model for predictions or retrain it using the provided Google Colab notebook.

1. Loading the Pre-Trained Model

The pre-trained model is saved in the MusaNet folder with the following structure:

```
MusaNet/
├── variables/
│   ├── variables.data-00000-of-00001
│   └── variables.index
├── saved_model.pb
└── keras_metadata.pb
```

To load the model and make predictions:

```
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('MusaNet')

# Make a prediction on a new image
def predict_leaf_disease(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(img_array)
    class_idx = tf.argmax(predictions, axis=1).numpy()[0]

    # Map class index to label
    class_labels = {0: 'Healthy Leaf', 1: 'Leaf Spot', 2: 'Sigatoka'}
    return class_labels[class_idx]

# Example usage
print(predict_leaf_disease('path_to_your_image.jpg'))
```

2. Retraining the Model (Google Colab)

If you want to retrain the model or explore the training process, use the provided Google Colab notebook <a href="https://github.com/Hunnter7/Transfer-learning-to-create-MusaNet-CNN/blob/main/Transfer_learning_to_create_MusaNet_CNN_model_.ipynb" title="MusaNet collab notebook">here</a>.</p>

**Steps to Run the Notebook:**

1. **Mount Google Drive:**
The notebook connects to Google Drive to access the dataset. Ensure your dataset is organized as follows:

```
/content/drive/MyDrive/DataSet/
├── Train_set/
│   ├── Healthy/
│   ├── Leaf_Spot/
│   └── Sigatoka/
└── Test_set/
    ├── Healthy/
    ├── Leaf_Spot/
    └── Sigatoka/
```

2. **Data Augmentation:**
The notebook uses ImageDataGenerator to augment the dataset with rotations, zooms, and shearing.

3. **Model Training:**
The model is trained using transfer learning with MobileNetV2 as the base model. Training logs and accuracy plots are displayed.

4. **Save the Model:**
After training, the model is saved in SavedModel format and zipped for download.

<h2>Notes</h2>

* Dataset: The dataset is not included in the repository due to size constraints. You can organize your dataset as shown above or modify the paths in the notebook.
* GPU Support: The notebook requests a GPU for faster training. Ensure you enable GPU in Colab (Runtime > Change runtime type > GPU).

<h2>Acknowledgments</h2>

* Developed at Universidad Tecnológica La Salle as part of an IEEE-supported research initiative.
* Dataset includes images from field captures and open-source repositories.
