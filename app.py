from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import os
import numpy as np

app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'C:\\Users\\PKN\\OneDrive\\Documents\\V sem\\ML\\cottonpredit_model.h5'

# Load your trained model
model = load_model(MODEL_PATH)

# Ensure the 'uploads' folder exists
uploads_folder = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(uploads_folder, exist_ok=True)

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    return preds

def get_class_label(predictions):
    # Adjust this function according to your label mapping
    label_mapping = {0: 'Diseased leaf', 1: 'Diseased plant', 2: 'Healthy leaf', 3: 'Healthy plant'}
    predicted_class = np.argmax(predictions)
    class_label = label_mapping.get(predicted_class, 'Unknown Class')

    return class_label

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        if f:
            # Save the file to the 'uploads' folder
            file_path = os.path.join(uploads_folder, secure_filename(f.filename))
            f.save(file_path)

            # Make prediction
            preds = model_predict(file_path, model)

            # Get the class label
            result = get_class_label(preds)

            # Render the result template
            return render_template('result.html', prediction=result)

    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
