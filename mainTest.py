import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

# Load the model
model = load_model('BrainTumor10Epochs.h5')

# Read the image using OpenCV
image = cv2.imread('D:\\Machine_learning_project\\Brain-TumorProjectdeeplearning\\pred\\pred5.jpg')

# Check if the image was loaded correctly
if image is None:
    raise ValueError("Failed to load image from the specified path.")

# Convert the image to RGB (OpenCV uses BGR by default)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert the image to a PIL Image object
img = Image.fromarray(image_rgb)

# Resize the image to (64, 64)
img = img.resize((64, 64))

# Convert the image back to a NumPy array
img = np.array(img)

# Normalize the image if required (example: scaling pixel values to [0, 1])
img = img / 255.0  # Apply this only if your model was trained with normalized images

# Expand dimensions to match the input shape of the model
input_img = np.expand_dims(img, axis=0)

# Make a prediction
results = model.predict(input_img)

# Convert the prediction to class label (0 or 1) for binary classification
# Assuming the model uses a sigmoid activation function
predicted_class = (results > 0.5).astype(int)

# Print the prediction result
print(f"Prediction probabilities: {results}")
print(f"Predicted class: {predicted_class[0][0]}")
