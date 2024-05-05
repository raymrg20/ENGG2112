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



filename = "processed_image.png"
cv2.imwrite(filename, display)

cv2.imshow('processed_image', display)

cv2.waitKey(0)
cv2.destroyAllWindows()