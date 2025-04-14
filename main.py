from flask import Flask, render_template, send_from_directory, url_for, request
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

app.config['SECRET_KEY'] = "qwerty"
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'

model = load_model('Age_Gender_Classification_model.h5')

os.makedirs(app.config['UPLOADED_PHOTOS_DEST'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

class UploadForm(FlaskForm):
    photo = FileField(
        validators=[
            FileAllowed(ALLOWED_EXTENSIONS, "Only images are allowed"),
            FileRequired("File field should not be empty")
        ]
    )
    submit = SubmitField('Upload')

@app.route('/uploads/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'], filename)

def make_prediction(image_path):
    img = image.load_img(image_path[1:], target_size=(200, 200))  # Resize to match model input
    img_array = image.img_to_array(img)  # Convert to array
    img_array = img_array / 255.0  # Normalize pixel values (same as training)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    age_pred, gender_pred = model.predict(img_array)
    
    predicted_age = age_pred[0][0]  # Age is a single value (linear output)
    gender_prob = gender_pred[0][0]  # Sigmoid output (probability)
    predicted_gender = "Female" if gender_prob > 0.5 else "Male"  # Gender threshold at 0.5
    
    return {'age':int(predicted_age), 'gender':predicted_gender}


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    form = UploadForm()
    if form.validate_on_submit():
        file = form.photo.data
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename))
        file_url = url_for('get_file', filename=filename)
        prediction = make_prediction(file_url)
    else:
        file_url = None
        prediction = None
    
    return render_template('index.html', form=form, file_url=file_url, prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)