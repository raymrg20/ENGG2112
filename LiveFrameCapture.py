'''
import cv2
import numpy as np
#from machine_learning_model import predicted_function  # Import your machine learning model function

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
        #prediction = predict_function(frame)  # Use your machine learning model function here

        # Display prediction result
        #print("Prediction:", prediction)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
'''



# 2nd method:

import cv2
import numpy as np
#from tensorflow.keras.models import load_model
#from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
#from tensorflow.keras.applications.mobilenet_v2 import decode_predictions

# Load pre-trained MobileNetV2 model
#model = load_model('path_to_your_model.h5')

# Set up video capture
cap = cv2.VideoCapture(0)  # Change the index to the appropriate camera if there are multiple cameras

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    # Preprocess frame for the model
    resized_frame = cv2.resize(frame, (224, 224))  # Ensure the size matches the input size of the model
    resized_frame = np.expand_dims(resized_frame, axis=0)
    #processed_frame = preprocess_input(resized_frame)

    # Make predictions
    #predictions = model.predict(processed_frame)
    #label = decode_predictions(predictions)

    # Draw label on the frame
    #cv2.putText(frame, label[0][0][1], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Live Camera', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
