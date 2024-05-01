import cv2
import numpy as np

# Using SVM Classifier
def extract_image(image, low_threshold, high_threshold):
    gray_Img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_Img, low_threshold, high_threshold)
    kernel = np.ones((2, 2), np.uint8) # Make more visible
    edges = cv2.dilate(edges, kernel, iterations=1)
    features = edges.flatten() #Flatten the edge data to create a feature vector for each pixel

    return features

image = cv2.imread('pear.jpg')
features = extract_image(image, low_threshold=50, high_threshold=100)

processed_image = features.reshape(image.shape[0], image.shape[1])
filename = 'processed_image3.png'
cv2.imshow('processed_image', processed_image)
cv2.imwrite(filename, processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


