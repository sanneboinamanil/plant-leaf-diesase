from flask import Flask, request, jsonify, render_template
import os
import numpy as np
from keras.models import load_model
from werkzeug.utils import secure_filename

from PIL import Image
import tensorflow as tf

app = Flask(__name__,template_folder='templates')
model = load_model('model/potatoes.h5')
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
INPUT_SHAPE = (256,256)

@app.route('/', methods=['GET', 'POST'])
def index():
    uploaded_image_path = None 
    result=""
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No file part'})
        
        image = request.files['image']

        if image.filename == '':
            return jsonify({'error': 'No selected file'})

        if image:
            filename = secure_filename(image.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(image_path)
            img = Image.open(image_path)
            uploaded_image_path = image_path
            
            img= img.resize(INPUT_SHAPE)
            img = np.array(img)/255.0
            #img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            #img = img / 255.0  # Normalize t
            # Preprocess the image (resize, normalize, etc.)
            # Make a prediction using your model
            # Return the prediction result as JSON
            #image_resized = tf.reshape(img, (-1, 256, 256, 3, 1))
            #image_resized = tf.squeeze(image_resized)
            prediction = model.predict(img)
            # Assuming the model outputs probabilities for multiple classes, you can get the class with the highest probability:
            class_index = np.argmax(prediction)
            class_labels = ['Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy']  # Replace with your actual class labels
            result = f"Disease: {class_labels[class_index]}"

    return render_template('index.html', result=result,uploaded_image=uploaded_image_path)

    

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)