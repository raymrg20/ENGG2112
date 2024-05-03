import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# Function to create the edge detection and classification model
def create_edge_detection_classification_model(input_shape, num_classes):
    base_input = tf.keras.layers.Input(shape=input_shape)
    
    # Convolutional Base for feature extraction
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(base_input)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    
    # Edge detection output
    edge_output = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='edge_output')(x)
    
    # Fully connected layers for classification
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
num_classes = 10  # Example: Assume 10 different classes for classification

model = create_edge_detection_classification_model(input_shape, num_classes)
model.summary()

# Function to prepare the image and generate labels using Canny edge detector
def prepare_image_and_label(image_path, target_size=(256, 256)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}. Ensure the path is correct.")
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    edges = cv2.Canny((img * 255).astype('uint8'), 100, 200)
    edges = edges.astype('float32') / 255.0
    return img.reshape(target_size + (1,)), edges.reshape(target_size + (1,))

# Load image and prepare data
image_path = 'pear.jpg'  # Ensure the image is available at this path
image, edges = prepare_image_and_label(image_path)

# Convert to numpy arrays and add batch dimension
image = np.array([image])
edges = np.array([edges])

# Dummy classification label (since we don't have actual classes)
class_labels = np.array([0])  # assuming '0' is a class index

# Train the model
model.fit(image, {'edge_output': edges, 'class_output': class_labels}, epochs=5, batch_size=1)

# Predict using the same image
predictions = model.predict(image)
predicted_edges = (predictions[0][0, :, :, 0] > 0.3).astype(np.uint8)
predicted_class = np.argmax(predictions[1], axis=1)

# Display results
plt.figure(figsize=(18, 6))  # Adjust the figure size as needed

# Plot the original image
plt.subplot(1, 3, 1)
plt.imshow(image[0, :, :, 0], cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Plot the edge probability map from the model prediction
plt.subplot(1, 3, 2)
plt.imshow(predictions[0][0, :, :, 0], cmap='gray')  # Make sure this is capturing the right slice
plt.title('Edge Probability Map')
plt.axis('off')

# Plot the Canny edges for comparison
plt.subplot(1, 3, 3)
canny_edges = cv2.Canny((image[0, :, :, 0] * 255).astype(np.uint8), 100, 200)
plt.imshow(canny_edges, cmap='gray')
plt.title('Canny Edges')
plt.axis('off')

plt.show()
