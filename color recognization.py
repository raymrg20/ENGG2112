import cv2
import numpy as np
def pixelate(image, pixel_size):
    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Resize the image to smaller dimensions
    small_image = cv2.resize(image, (width // pixel_size, height // pixel_size))

    # Enlarge the small image to original size
    pixelated_image = cv2.resize(small_image, (width, height), interpolation=cv2.INTER_NEAREST)

    return pixelated_image


# Using SVM Classifier
def extract_image(image, low_threshold, high_threshold):
    gray_Img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_Img, low_threshold, high_threshold)
    kernel = np.ones((3, 3), np.uint8) # Make more visible
    edges = cv2.dilate(edges, kernel, iterations=1)
    features = edges.flatten() #Flatten the edge data to create a feature vector for each pixel

    return features

image = cv2.imread('apple.jpg')
features = extract_image(image, low_threshold=50, high_threshold=100)

appleedge = features.reshape(image.shape[0], image.shape[1])

cv2.imshow('appleedge', appleedge)
cv2.waitKey(0)
cv2.destroyAllWindows()

image_path = "appleedge.jpg"
original_image1 = cv2.imread(image_path)

# Set the pixel size for pixelation (adjust as needed)
pixel_size1 = 25

# Pixelate the image
pixelated_image = pixelate(original_image1, pixel_size1)

# Display the original and pixelated images
cv2.imshow("Original Image", original_image)
cv2.imshow("Pixelated Image", pixelated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Load the image
image_path = "apple.jpg"
original_image = cv2.imread(image_path)

# Set the pixel size for pixelation (adjust as needed)
pixel_size = 25

# Pixelate the image
pixelated_image = pixelate(original_image, pixel_size)

# Display the original and pixelated images
cv2.imshow("Original Image", original_image)
cv2.imshow("Pixelated Image", pixelated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
