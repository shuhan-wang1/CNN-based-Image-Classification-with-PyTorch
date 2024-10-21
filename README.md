# CNN-based Image Classification with PyTorch - MathTeXNet
This repository contains a convulutional nural network(CNN) model implemented in PyTorch for multi-class image classification. The project supports custom datasets and includes data augmentation, automatic mixed precision(AMP) for efficient training, early stopping, and learning rate scheduling.

## 1. Key Features
### 1.1 CNN Architecture:
The CNN model includes multiple convolutional lyers followed by batch normalization, max-pooling, and fully connected layers. Dropout is applied to prevent overfitting.

### 1.2 Custom Dataset loader:
A modified version of 'ImageFolder' is used to load custom dataset and map class names to indices. It supports flexible class labeling, including handing single and double character folder names.

### 1.3 Data Augmentation:
The project applies data augmentation techniques such as random rotation, horizontal flip, and affine transformations to improve generalization.

### 1.4 Mixed Precision Training:
Automatic mixed precision(AMP) is utilized for faster training with reduced memory usage.

### 1.5 Early Stopping: 
An early stopping mechanism monitors the validation loss to prevent overfitting and stop training when the model's performance stops improving.

### 1.6 Learning Rate Scheduler:
A learning rate shceduler 'ReduceLROnPlateau' adjusts the learning rate when validation loss plateaus.

## 2. Architecture
The model is a multi-layer CNN defined as follows:

### 2.1 Convolutional Layers:
Three layers, with 64, 128 and 256 filters respectively.

### 2.2 Batch Normalization:
Applied after each convolutional layer to stablize training.

### 2.3 Max-Pooling:
Reduces the spatial dimensions of the features.

### 2.4 Fully Connected Layers:
Two fully connected layers map the extracted features to the output classes.

### 2.5 Dropout:
Applied between fully connected layers to prevent overfitting.

## 3. Data Preprocessing
The script use several preprocessing techniques to improve model performance:

### 3.1 Grayscale Conversion:
Converts images to a single channel (Grayscale).

### 3.2 Resize:
Resize all images to a uniform size (64 times 64 by default)/

### 3.3 Random Augmentations:
Includes random rotations, flips, amd affine transformations to simulate variations in the training data.

### 3.4 Normalization:
Images are normalized to the range [-1, 1]

## 4. Training Procedure
### 4.1 Load Data:
Custom datasets are loaded from the specified directory using the 'CustomImageFolder' class, which handles the mapping of class names to labels.

### 4.2 Train the Model:
The model is trained using a combination of cross-entropy loss and the Adam optimizer. AMP is used to speed up training, and a GradScaler helps with gradient scaling.

### 4.3 Validate the Model:
After each epoch, the model is evaluated on the validation set to tract performance. If the validation loss improves, the model is saved.

### 4.4 Early Stopping:
If the validation loss stops improving for several epochs, training is halted to prevent overfitting.

### 4.5 Test the Model:
After training, the model is evaluated on the test set to calculate its accuracy and visualize predictions.

## 5. Usage
### 5.1 Install Dependencies:
'pip install torch torchvision scikit-learn matplotlib seaborn'

### 5.2 Train the Model:
Customize the data set directory and run the script:
'python train.py --data-dir path_to_your_data'

### 5.3 Test the Model:
After training, test the model on a test dataset:
'python test.py --model best_model.pth --data-dir path_to_test_data'

## Example Custom Data Loading
The custom data loader ('CustomImageFolder') handles various folder naming conventions, including:
- Single-digit folders (0-9)
- Single-letter folders (a-z)
- Double-character folderts that represents capital letters (aa, bb, etc.)

## 6. Contributions
Please feel free to open issues or submit pull requests to improve the code!!!
