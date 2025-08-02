import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
import cv2
import os

# Simulated dataset (replace with real satellite imagery dataset)
def load_data(data_dir='data'):
    images = []
    labels = []
    for i in range(100):  # Simulated 100 images
        img = np.random.rand(224, 224, 3)  # Replace with cv2.imread for real images
        images.append(img)
        labels.append(np.random.randint(0, 2))  # 0: no fire, 1: fire
    return np.array(images), np.array(labels)

# Load and preprocess data
X, y = load_data()
X = X / 255.0  # Normalize pixel values

# Build lightweight model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile and train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=5, batch_size=16, validation_split=0.2)

# Convert to TFLite for edge deployment
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save model
with open('forest_fire_model.tflite', 'wb') as f:
    f.write(tflite_model)