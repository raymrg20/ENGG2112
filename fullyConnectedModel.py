import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

# Define the model architecture
def create_edge_detection_classification_model(input_shape, num_classes=10):  # Assuming 10 classes arbitrarily
    base_input = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(base_input)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    edge_output = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='edge_output')(x)
    y = tf.keras.layers.Flatten()(x)
    y = tf.keras.layers.Dense(128, activation='relu')(y)
    y = tf.keras.layers.Dropout(0.5)(y)
    class_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='class_output')(y)
    model = tf.keras.Model(inputs=base_input, outputs=[edge_output, class_output])
    model.compile(optimizer='adam',
                  loss={'edge_output': 'binary_crossentropy', 'class_output': 'sparse_categorical_crossentropy'},
                  metrics={'edge_output': 'accuracy', 'class_output': 'accuracy'})
    return model

input_shape = (256, 256, 1)
model = create_edge_detection_classification_model(input_shape, num_classes=10)

# Directory containing images
folder_dir = 'Processed_Apples'

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    img = cv2.resize(img, (256, 256))
    img = np.expand_dims(img, axis=-1)
    img = img.astype('float32') / 255.0

    # Generating synthetic edges for demonstration, replace this with your method
    edges = cv2.Canny(img.astype(np.uint8), 100, 200)
    edges = np.expand_dims(edges, axis=-1)
    edges = edges.astype('float32') / 255.0

    return img, edges

# Create datasets
images = []
edge_labels = []

for image_name in os.listdir(folder_dir):
    image_path = os.path.join(folder_dir, image_name)
    img, edges = preprocess_image(image_path)
    if img is not None and edges is not None:
        images.append(img)
        edge_labels.append(edges)

images = np.array(images)
edge_labels = np.array(edge_labels)

X_train, X_val, y_train, y_val = train_test_split(images, edge_labels, test_size=0.2, random_state=42)

# Now use X_train, y_train for training and X_val, y_val for validation
history = model.fit(X_train, {'edge_output': y_train}, validation_data=(X_val, {'edge_output': y_val}), epochs=1, batch_size=10)

