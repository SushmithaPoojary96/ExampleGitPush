import numpy as np               # For numerical computations and array handling
import cv2                       # OpenCV library for image processing
from utils.preprocess import preprocess_image  # Preprocessing function (resize + normalize)
from utils.database import collection  # MongoDB collection where person data is stored
from utils.cnn_model import build_model   # Function to create CNN model architecture

# -----------------------------
# Initialize training data lists
# -----------------------------
X = []  # Will hold all preprocessed images
y = []  # Will hold corresponding labels for each image

# -----------------------------
# Fetch all registered persons from MongoDB
# -----------------------------
persons = collection.find()  # Returns a cursor over all documents in the 'persons' collection

# -----------------------------
# Loop over each person and their stored images
# -----------------------------
for person in persons:
    label = person["label"]  # Get unique label assigned to this person

    # Loop over each binary image stored for this person
    for img_binary in person["images"]:
        # Preprocess image: decode, resize to IMAGE_SIZE, normalize pixels
        img = preprocess_image(img_binary)

        # Add preprocessed image to training list
        X.append(img)

        # Add the corresponding label to the labels list
        y.append(label)

# -----------------------------
# Convert lists to NumPy arrays (required by Keras)
# -----------------------------
X = np.array(X)  # Shape: (num_samples, IMAGE_SIZE, IMAGE_SIZE, 3)
y = np.array(y)  # Shape: (num_samples,)

# -----------------------------
# Determine number of classes (unique persons) and build the CNN
# -----------------------------
num_classes = len(np.unique(y))  # Number of distinct people in dataset
model = build_model(num_classes)  # Create a CNN model with output neurons = num_classes

# -----------------------------
# Train the CNN model
# -----------------------------
model.fit(
    X,            # Input images
    y,            # Corresponding labels
    epochs=10,    # Number of times to iterate over the whole dataset
    batch_size=8  # Number of samples per gradient update
)

# -----------------------------
# Save the trained model to disk
# -----------------------------
model.save("model/face_model.h5")  # Saves architecture + weights + optimizer state
print("Model Trained Successfully")  # Confirmation message