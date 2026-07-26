from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from config import IMAGE_SIZE  # Import the fixed image size used for CNN input


def build_model(num_classes):
    """
    Build a Convolutional Neural Network (CNN) for face recognition.
    num_classes: number of unique persons (output classes)
    """

    # Initialize a Sequential model (stack of layers)
    model = Sequential()

    # -----------------------------
    # First Convolutional Block
    # -----------------------------
    # Conv2D: 32 filters, 3x3 kernel, ReLU activation
    # input_shape: IMAGE_SIZE x IMAGE_SIZE x 3 (RGB image)
    model.add(Conv2D(32, (3, 3), activation='relu',
                     input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))

    # MaxPooling2D: 2x2 pooling reduces spatial size by half
    # Helps reduce computation and makes features more robust
    model.add(MaxPooling2D((2, 2)))

    # -----------------------------
    # Second Convolutional Block
    # -----------------------------
    # Conv2D: 64 filters, 3x3 kernel, ReLU activation
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # MaxPooling2D: 2x2 pooling
    model.add(MaxPooling2D((2, 2)))

    # -----------------------------
    # Third Convolutional Block
    # -----------------------------
    # Conv2D: 128 filters, 3x3 kernel, ReLU activation
    model.add(Conv2D(128, (3, 3), activation='relu'))
    # MaxPooling2D: 2x2 pooling
    model.add(MaxPooling2D((2, 2)))

    # -----------------------------
    # Flattening
    # -----------------------------
    # Convert 2D feature maps into 1D feature vector for dense layers
    model.add(Flatten())

    # -----------------------------
    # Fully Connected Layers
    # -----------------------------
    # Dense layer with 128 neurons and ReLU activation
    model.add(Dense(128, activation='relu'))

    # Dropout layer (50%) to prevent overfitting
    model.add(Dropout(0.5))

    # Output layer: num_classes neurons, softmax activation for multi-class probability
    model.add(Dense(num_classes, activation='softmax'))

    # -----------------------------
    # Compile the model
    # -----------------------------
    # Optimizer: Adam (adaptive learning rate)
    # Loss: sparse categorical crossentropy (integer labels)
    # Metrics: accuracy
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Return the built and compiled model
    return model