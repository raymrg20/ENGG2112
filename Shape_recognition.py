import cv2
import numpy as np

def process_image(image_path):
    img = cv2.imread(image_path) #Load the image
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #gray scale
    edges = cv2.Canny(gray_img, threshold1=100, threshold2=200) #canny edge
    resized_img = cv2.resize(edges, (256, 256))
    img_array = resized_img.astype(np.float32) / 255.0  # Convert data type and scale pixel values to 0-1 and normalizing
    
    return img_array, resized_img

image_path = 'pear.jpg'
processed_image, display = process_image(image_path)

cv2.imshow('processed_image', display)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Second Method
import cv2
import numpy as np

# Using SVM Classifier
def extract_image(image, low_threshold, high_threshold):
    gray_Img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_Img, low_threshold, high_threshold)
    kernel = np.ones((3, 3), np.uint8) # Make more visible
    edges = cv2.dilate(edges, kernel, iterations=1)
    features = edges.flatten() #Flatten the edge data to create a feature vector for each pixel

    return features

image = cv2.imread('pear.jpg')
features = extract_image(image, low_threshold=50, high_threshold=100)

processed_image = features.reshape(image.shape[0], image.shape[1])

cv2.imshow('processed_image', processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


