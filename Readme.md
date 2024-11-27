# The study Data from Deep Learning foundations class in Leibniz University Hannover

## TASK 1: Convolutional Neural Networks

### Introduction

In this task, we are working with the Oxford-IIIT Pet Dataset. The images are of varying sizes, and each class includes pet breeds with diverse features. This dataset serves as a robust foundation for training and evaluating convolutional neural networks (CNNs) in tasks like classification and segmentation.

### Dataset Preprocessing

Before training the model, the dataset underwent the following preprocessing steps:
	1.	Resizing Images: All images were resized to a uniform dimension (e.g., 128x128 pixels) to ensure compatibility with the neural network input.
	2.	Data Augmentation: Techniques like random rotation, flipping, and cropping were applied to artificially expand the dataset and improve model generalization.
	3.	Normalization: Pixel values were normalized to a [0, 1] or [-1, 1] range to standardize the input data.

The dataset was then split into training, validation, and test sets to ensure fair evaluation of the model.

### Model Architecture

For this task, we used a pre-trained deep learning model. Pre-trained models are pre-trained on large datasets like ImageNet and provide an excellent starting point for fine-tuning on specific tasks, reducing the computational cost and time of training.

### Transfer Learning

We applied transfer learning to adapt a pre-trained model to the task of pet classification. Transfer learning involves leveraging a model trained on a different but related task and fine-tuning it to improve its performance on a new dataset.

### What is Transfer Learning?

Transfer learning is a machine learning technique that reuses a pre-trained model for a new problem. By exploiting the knowledge gained from solving one task, the model can generalize better for a related task. This approach is particularly useful when the new dataset is small or lacks diversity.

### Steps in Transfer Learning:

	1.	Select a Pre-Trained Source Model: For this task, we used a model pre-trained on the ImageNet dataset, which contains millions of labeled images across thousands of categories.
	2.	Adapt the Source Model: The final layer of the pre-trained model was replaced with a new layer tailored to the number of categories in the Oxford-IIIT Pet Dataset.
	3.	Fine-Tuning the Model:
	•	Some layers of the pre-trained model were frozen to retain learned features, while others were fine-tuned to adapt to the specific characteristics of the pet dataset.
	•	Optimization was performed using a lower learning rate to prevent drastic changes to the pre-trained weights.

### Fine-Tuning Strategies:

There are several strategies for fine-tuning a pre-trained model:
	•	Freezing Layers: For smaller datasets, most of the model layers are frozen to avoid overfitting, and only the final layers are trained.
	•	Partial Fine-Tuning: In cases where the dataset is relatively large, additional layers can be trained to enhance task-specific performance.
	•	Full Fine-Tuning: If the new dataset is extensive, the entire model can be fine-tuned to the new task.

### Benefits of Transfer Learning:

	•	Less Training Data Needed: The pre-trained model has already learned generic features, reducing the need for large training datasets.
	•	Better Initial Model: The starting weights are more relevant, enabling faster convergence.
	•	Faster Training: With fewer layers to train, training time is significantly reduced.
	•	Higher Accuracy: The model is better equipped to perform well after training.

### Example:

For instance, a pre-trained model trained on ImageNet to recognize objects can be adapted to predict animal species in the Oxford-IIIT Pet Dataset by leveraging its learned ability to identify distinguishing features of animals.

# Task 2 Recurrent Neural Networks
## Human Activity Recognition using Recurrent Neural Networks

This project implements a Recurrent Neural Network (RNN) to classify gyroscopic sensor data into one of six human activities: **walking, sitting, standing, and others**. The primary goal is to design a model that achieves over 80% accuracy on the test dataset.

## Project Overview

### Objectives:
1. **Load and Explore Data:**
   - Load preprocessed tensor data.
   - Analyze and document the dimensions and characteristics of the dataset.
2. **Design and Train RNN:**
   - Create an RNN model with at least one hidden layer.
   - Train the model for at least 5 epochs using a suitable loss function.
   - Perform validation after each epoch.
3. **Evaluate Model:**
   - Select the best model checkpoint based on validation performance.
   - Compute the final test accuracy.
   - Ensure a test accuracy of at least 80%.
4. **Additional Analysis:**
   - Evaluate additional metrics.
   - Visualize results using a confusion matrix.

## Dataset

### Data Files:
- `train.pt`: Training data.
- `test.pt`: Testing data.
- `val.pt`: Validation data.

### Dataset Structure:
The dataset is stored as dictionaries with the following keys:
- **`samples`**: A tensor containing gyroscopic data.
- **`labels`**: Corresponding activity labels.

### Data Dimensions:
- **Train Samples Shape:** `[5881, 3, 206]`
  - **5881:** Number of training examples.
  - **3:** Number of sensor channels (e.g., x, y, z axes).
  - **206:** Number of timesteps per sequence.
- **Train Labels Shape:** `[5881]` (One label per sample.)

Similar structures exist for test and validation datasets.

### Exploratory Data Analysis (EDA):
- Verified dataset dimensions.
- Plotted sample sensor data for analysis.
- Visualized sensor signal patterns for each activity.

## Model Design

### RNN Architecture:
- **Input:** 3 sensor channels.
- **RNN Layer:** Long Short-Term Memory (LSTM) with 256 hidden units and 2 layers.
- **Fully Connected Layers:**
  - First dense layer reduces dimensions by half with ReLU activation.
  - Final output layer maps features to 6 activity classes.
- **Loss Function:** CrossEntropyLoss (suitable for multi-class classification).
- **Optimizer:** Adam optimizer with a learning rate of 0.001.

### Key Features:
- Handles sequence input of shape `[batch_size, channels, timesteps]`.
- Uses the output of the last LSTM timestep for classification.

### Training Details:
- **Batch Size:** 64.
- **Epochs:** 10.
- Validation performed after each epoch.
- Best model selected based on validation accuracy.

## Results

### Model Performance:
- **Validation Accuracy:** Monitored during training to identify the best checkpoint.
- **Test Accuracy:** Achieved 83.5%, surpassing the 80% threshold.

### Evaluation Metrics:
- **Classification Report:** Includes precision, recall, F1-score for each activity.
- **Confusion Matrix:** Visualizes the performance of the model across different activity classes.

## Code Highlights

### Key Steps:
1. **Device Configuration:**
   - Utilized Apple M3 Pro GPU (MPS backend) for accelerated training.
   - Ensured all tensors were in `float32` format, as required by the MPS framework.

2. **Training Loop:**
   - Gradient-based optimization using backpropagation.
   - Validation accuracy monitored for checkpointing.

3. **Testing and Visualization:**
   - Evaluated classification performance.
   - Displayed confusion matrix for in-depth error analysis.

### Code Snippet for Confusion Matrix:
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Generate confusion matrix
cm = confusion_matrix(all_labels, all_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(num_classes))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
```

## Conclusion
- The RNN model successfully classified human activities with high accuracy.
- Validation and testing procedures ensured reliable performance.
- The project demonstrates the capability of RNNs in handling sequential sensor data.

## Future Improvements
1. Experiment with different RNN architectures (e.g., GRU, bidirectional LSTMs).
2. Use data augmentation to improve model generalization.
3. Implement regularization techniques to avoid overfitting.
4. Explore transformer-based models for sequential data processing.