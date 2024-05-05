#This processes all images in a folder using edge detection method Canny, then saves them into a new folder
import os
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



def process_images(input_folder, output_folder, low_threshold, high_threshold):
    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Process only image files
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)


            # Read the image
            image = cv2.imread(input_path)
            #print(image)
            # Process the image
            processed_image = extract_image(image, low_threshold, high_threshold)
            #print("Processed image:", processed_image)
            # Save the processed image

            processed_image = processed_image.reshape(image.shape[0], image.shape[1])


            cv2.imwrite(output_path, processed_image)

            print(f"Processed: {filename}")

# Example usage
input_folder = r"C:\Uni\AMME2112\fruits-360_dataset\fruits-360\Test\Apple Red 1"

print("Input folder:", input_folder)

output_folder = "Processed_Apples"
low_threshold = 50
high_threshold = 100
process_images(input_folder, output_folder, low_threshold, high_threshold)