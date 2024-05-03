import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to create the model for edge detection without fully connected Layers
def create_edge_detection_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.UpSampling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.UpSampling2D((2, 2)),
        tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')  # Output layer for edge detection
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Prepare the image and label for training
def prepare_image_and_label(image_path, target_size=(256, 256)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}. Ensure the path is correct.")
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    edges = cv2.Canny((img * 255).astype('uint8'), 100, 200)
    edges = edges.astype('float32') / 255.0
    return img.reshape(target_size + (1,)), edges.reshape(target_size + (1,))

# Load image
image_path = 'pear.jpg'
images, labels = prepare_image_and_label(image_path)

# Convert to numpy arrays and add batch dimension
images = np.array([images])
labels = np.array([labels])

# Create and train the model
model = create_edge_detection_model(input_shape=(256, 256, 1))
model.fit(images, labels, epochs=5, batch_size=1)

# Predict using the same image
predictions = model.predict(images)
predicted_edges = (predictions[0, :, :, 0] > 0.5).astype(np.uint8)

# Display results
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(images[0, :, :, 0], cmap='gray')  # Original
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(predicted_edges, cmap='gray')  # Predicted by CNN
plt.title('Predicted Edges')
plt.axis('off')

# Canny edges for comparison
canny_edges = cv2.Canny((images[0, :, :, 0] * 255).astype(np.uint8), 100, 200)
plt.subplot(1, 3, 3)
plt.imshow(canny_edges, cmap='gray')  # Canny Edge Detection
plt.title('Canny Edges')
plt.axis('off')

plt.show()
