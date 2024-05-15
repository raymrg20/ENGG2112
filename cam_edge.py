import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("my_model.keras")
class_names = ['Apple', 'Banana', 'Blueberries', 'Lemon', 'Mandarine', 'Orange', 'Pear', 'Strawberries']

def extract_image(image, low_threshold, high_threshold):
    gray_Img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_Img, low_threshold, high_threshold)
    kernel = np.ones((2, 2), np.uint8) # Make more visible
    edges = cv2.dilate(edges, kernel, iterations=1)
    features = edges.flatten() #Flatten the edge data to create a feature vector for each pixel

    return features

def preprocess_frame(frame):
    # Resize the frame to match the model's expected input
    frame = cv2.resize(frame, (256, 256))  # Resize to 256x256
    frame = np.expand_dims(frame, axis=-1)  # Add channel dimension
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return frame

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

        # Preprocess the frame
        processed_frame = preprocess_frame(extract_image(frame,low_threshold,high_threshold).reshape(frame.shape[0], frame.shape[1]))

        # Predict with the model
        predictions = model.predict(processed_frame)
        class_prediction = predictions[1]  # Assuming the second output is class predictions
        class_id = np.argmax(class_prediction)
        fruit_name = class_names[class_id]
        fruit_confidence = class_prediction[0][class_id]
        text = f"{fruit_name} Detected: {fruit_confidence * 100:.2f}% Confidence"

        # Display the resulting frame
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        #For binary Classifications
        #class output index 7
        # fruit_present = predictions[1][0][6] > 0.5
        # print(fruit_present)
        # text = "Fruit Detected: Yes" if fruit_present else "Fruit Detected: No"
        # cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        
        # Display the resulting frame
        cv2.imshow('Live', frame)
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

