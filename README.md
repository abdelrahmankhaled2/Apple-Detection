# Apple Detection using Convolutional Neural Network (CNN)

This repository contains code for a Convolutional Neural Network (CNN) model trained to detect apples and damaged apples in images. The project utilizes the TensorFlow and Keras frameworks for deep learning and computer vision tasks.

## Dependencies

- Python 3.x
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- Gradio

## Dataset

The training and testing datasets are loaded from CSV files (`train_annotations.csv` and `test_annotations.csv`). The images are organized in folders (`train` and `test`) corresponding to their respective datasets.

## Data Augmentation

Image data augmentation is applied using Keras' `ImageDataGenerator` to enhance the model's ability to generalize and improve performance.

## Model Architecture

The CNN model is constructed using Keras' Sequential API. It consists of convolutional layers, max-pooling layers, dropout layers, and dense layers. The model is compiled with the Adam optimizer and binary crossentropy loss.

## Training

The dataset is split into training and validation sets, and the model is trained using the `fit` method. Training history is stored for later analysis.

## Evaluation

The trained model is evaluated on the test set to assess its performance in terms of loss and accuracy.

## Gradio Interface

A Gradio interface is created to interactively test the model. Users can upload images, and the model makes predictions on whether the image contains an apple or a damaged apple.

## Usage

To run the Gradio interface, ensure all dependencies are installed, and execute the script. Upload an image to the interface to get real-time predictions.

