import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, classification_report, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc, RocCurveDisplay


model = load_model("my_model.keras")

# Prepare your test dataset in the correct format
folder_dir = 'Processed Fruits'
class_names = sorted(os.listdir(folder_dir))
class_indices = {name: index for index, name in enumerate(class_names)}

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    img = cv2.resize(img, (256, 256))
    img = np.expand_dims(img, axis=-1)
    img = img.astype('float32') / 255.0
    return img

# Load images for testing
test_images = []
test_labels = []
for class_name in class_names:
    class_dir = os.path.join(folder_dir, class_name)
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        img = preprocess_image(image_path)
        if img is not None:
            test_images.append(img)
            test_labels.append(class_indices[class_name])

test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Make predictions
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions[1], axis=1)

# Evaluate accuracy and precision
accuracy = accuracy_score(test_labels, predicted_classes)
precision = precision_score(test_labels, predicted_classes, average='macro')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")

# Display confusion matrix
conf_matrix = confusion_matrix(test_labels, predicted_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
plt.figure(figsize=(10,8))
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

# Show classification report
print(classification_report(test_labels, predicted_classes, target_names=class_names))


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(class_names)):
    fpr[i], tpr[i], _ = roc_curve(test_labels == i, predictions[1][:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for each class
plt.figure()
for i in range(len(class_names)):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for each class')
plt.legend(loc="lower right")
plt.show()
