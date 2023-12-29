import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout

# Load the CSV files
train_annotations_path = r"C:\Users\abdel\PycharmProjects\Apple Detection\Dataset\csv files\train_annotations.csv"
test_annotations_path = r"C:\Users\abdel\PycharmProjects\Apple Detection\Dataset\csv files\test_annotations.csv"

df_train = pd.read_csv(train_annotations_path)
df_test = pd.read_csv(test_annotations_path)

# Use a lambda function to create a new column 'category' based on the 'class' column
df_train['category'] = df_train['class'].apply(lambda x: 'Apple' if 'apple' in x.lower() else 'Damaged')

# Create a count plot
sns.set_style('darkgrid')
sns.countplot(x='category', data=df_train, hue='class')
plt.show()

# Define image folders
train_folder = r"C:\Users\abdel\PycharmProjects\Apple Detection\Dataset\train"
test_folder = r"C:\Users\abdel\PycharmProjects\Apple Detection\Dataset\test"

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

# Convert string labels to numeric values
label_encoder = LabelEncoder()
df_train['class'] = label_encoder.fit_transform(df_train['class'])
df_test['class'] = label_encoder.transform(df_test['class'])

# Split the dataset into training, validation, and test sets
train_df, val_df = train_test_split(df_train, test_size=0.2, random_state=42)

# Create data generators
train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_folder,
    x_col="filename",
    y_col="class",
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary"
)

val_generator = datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=train_folder,
    x_col="filename",
    y_col="class",
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary"
)

test_generator = datagen.flow_from_dataframe(
    dataframe=df_test,
    directory=test_folder,
    x_col="filename",
    y_col="class",
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    shuffle=False
)

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout((0.3)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout((0.2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display model summary
model.summary()

# Train the model
history = model.fit(train_generator, epochs=5, validation_data=val_generator)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(test_generator)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

import gradio as gr
import PIL.Image

def predict(image):
    # Check image type
    print(type(image))

    # Convert PIL Image to NumPy array if needed
    if isinstance(image, PIL.Image.Image):
        image = np.array(image)

    # Check image validity
    if image is None or image.size == 0:
        print("Error: Empty or invalid image.")
        return "Error: Empty or invalid image."

    # Preprocess the input image
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    # Make predictions
    prediction = model.predict(image)

    # Convert the prediction to a class label
    predicted_class = "Apple" if prediction < 0.1 else "Damaged"

    print("The prediction:", predicted_class, "with confidence:", prediction)
    return predicted_class

# Create the Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=[gr.Image(type="pil", label="Upload Image")],
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    title="Prediction Apple",
    live=True
)

# Launch the interface
interface.launch()