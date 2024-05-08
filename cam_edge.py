import cv2
import numpy as np
def extract_image(image, low_threshold, high_threshold):
    
    gray_Img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_Img, low_threshold, high_threshold)
    kernel = np.ones((2, 2), np.uint8) # Make more visible
    edges = cv2.dilate(edges, kernel, iterations=1)
    features = edges.flatten() #Flatten the edge data to create a feature vector for each pixel

    return features


def main():
    # Open the default camera (0) for capturing video
    cap = cv2.VideoCapture(1)
    low_threshold=50
    high_threshold=100
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Apply edge detection
        edges = extract_image(frame,low_threshold,high_threshold)
        
        # Display the resulting frame
        cv2.imshow('Edge Detection', edges.reshape(frame.shape[0], frame.shape[1]))
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
