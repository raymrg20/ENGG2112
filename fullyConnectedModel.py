import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

# Define the model architecture
def create_edge_detection_classification_model(input_shape, num_classes):  # Assuming 10 classes arbitrarily
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
model = create_edge_detection_classification_model(input_shape, num_classes=8)

# Directory containing images
folder_dir = 'Processed Fruits'
class_names = sorted(os.listdir(folder_dir))  # Sort to ensure label consistency
class_indices = {name: index for index, name in enumerate(class_names)}

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
class_labels = []

for class_name in class_names:
    class_dir = os.path.join(folder_dir, class_name)
    class_index = class_indices[class_name]
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        img, edges = preprocess_image(image_path)
        if img is not None and edges is not None:
            images.append(img)
            edge_labels.append(edges)
            class_labels.append(class_index)

images = np.array(images)
edge_labels = np.array(edge_labels)
class_labels = np.array(class_labels)

X_train, X_val, y_train, y_val = train_test_split(images, list(zip(edge_labels, class_labels)), test_size=0.2, random_state=42)
edge_labels_train, class_labels_train = zip(*y_train)
edge_labels_val, class_labels_val = zip(*y_val)

# Train model
history = model.fit(X_train, {'edge_output': np.array(edge_labels_train), 'class_output': np.array(class_labels_train)},
                    validation_data=(X_val, {'edge_output': np.array(edge_labels_val), 'class_output': np.array(class_labels_val)}),
                    epochs=1, batch_size=10)
model.save(r"C:\Users\CYBORG 15\Documents\GitHub\ENGG2112\my_model.keras")

