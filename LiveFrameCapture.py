import cv2
import numpy as np
from machine_learning_model import predicted_function  # Import your machine learning model function

def main():
    # Access camera
    cap = cv2.VideoCapture(0)  # 0 for default camera, you can change it to other indices if multiple cameras are available

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the captured frame
        cv2.imshow('Frame', frame)

        # Apply your machine learning model to the frame
        prediction = predict_function(frame)  # Use your machine learning model function here

        # Display prediction result
        print("Prediction:", prediction)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
