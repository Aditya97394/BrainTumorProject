from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import cv2
from keras.models import load_model
import os

app = Flask(__name__)

# Load the model
model = load_model('BrainTumor10Epochs.h5')

# Define the path for uploaded files
UPLOAD_FOLDER = 'uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def predict_image(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image to RGB (OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the image to a PIL Image object
    img = Image.fromarray(image_rgb)

    # Resize the image to (64, 64)
    img = img.resize((64, 64))

    # Convert the image back to a NumPy array
    img = np.array(img)

    # Normalize the image
    img = img / 255.0

    # Expand dimensions to match the input shape of the model
    input_img = np.expand_dims(img, axis=0)

    # Make a prediction
    results = model.predict(input_img)

    # Convert the prediction to class label (0 or 1) for binary classification
    predicted_class = (results > 0.5).astype(int)

    return predicted_class[0][0]


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Predict the image
            result = predict_image(file_path)

            # Prepare the result message
            if result == 1:
                result_message = "Tumor Detected"
            else:
                result_message = "No Tumor Detected"

            return render_template('result.html', result_message=result_message, image_file=filename)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
