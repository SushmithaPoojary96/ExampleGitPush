import cv2
import numpy as np
from config import IMAGE_SIZE  # Import the fixed image size for the CNN

def preprocess_image(image_bytes):
    """
    Preprocess an image for CNN input:
    1. Convert binary to NumPy array
    2. Decode into an image
    3. Resize to fixed dimensions
    4. Normalize pixel values
    """

    # Convert raw bytes from file or database to a 1D NumPy array of type uint8
    img_array = np.frombuffer(image_bytes, np.uint8)

    # Decode the NumPy array into an image (height x width x 3 color channels)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Resize image to fixed dimensions (IMAGE_SIZE x IMAGE_SIZE)
    # CNNs require consistent input size
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    # Normalize pixel values from 0-255 to 0.0-1.0 for faster and stable training
    img = img / 255.0

    # Return the preprocessed image ready for model input
    return img