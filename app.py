import os
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = load_model('autism (1).h5') 

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
    
        file = request.files['image']
        filename = secure_filename(file.filename)
        if file:
            img_path = os.path.join('static/uploads', file.filename)
            file.save(img_path)
            
            
            img_array = preprocess_image(img_path)
            prediction = model.predict(img_array)
            
            if prediction > 0.015:
                result = 'Non Autistic'
            else:
                result = 'Autistic'
            return render_template('index.html', img_path=img_path,filename=filename, result=result)
    
    return render_template('index.html', img_path=None, result=None)


if __name__ == '__main__':
    app.run(debug=True)
