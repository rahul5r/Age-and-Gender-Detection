from flask import Flask, render_template, send_from_directory, url_for, request
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


app = Flask(__name__)

app.config['SECRET_KEY'] = "qwerty"
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'
app.config['UPLOADED_FACES_DEST'] = 'faces'

model = load_model('Age_Gender_Classification_model.h5')

os.makedirs(app.config['UPLOADED_PHOTOS_DEST'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

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

@app.route('/faces/<filename>')
def get_face_file(filename):
    return send_from_directory(app.config['UPLOADED_FACES_DEST'], filename)

def detect_faces(image_path):
    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    detected_faces = []

    counter = 1
    for (x, y, w, h) in faces:
        face_crop_bgr = img[y:y+h, x:x+w]
        face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        save_path = "faces/"+image_path.split('/')[-1].split('.')[0]+str(counter)+'.'+image_path.split('/')[-1].split('.')[-1]
        counter += 1
        cv2.imwrite(save_path, cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2BGR))
        
        detected_faces.append(save_path)
    return detected_faces

def make_prediction(image_path):
    faces = detect_faces(image_path[1:])
    predictions = []
    for img in faces:
        face_path = url_for('get_face_file', filename=img.split('/')[-1])
        img = image.load_img(img, target_size=(200, 200))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0 
        img_array = np.expand_dims(img_array, axis=0)
        
        age_pred, gender_pred = model.predict(img_array)
        
        predicted_age = age_pred[0][0]
        gender_prob = gender_pred[0][0]
        predicted_gender = "Female" if gender_prob > 0.5 else "Male"
        
        predictions.append({'age':int(predicted_age), 'gender':predicted_gender, 'face':face_path})
    return predictions


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    form = UploadForm()
    if form.validate_on_submit():
        file = form.photo.data
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename))
        file_url = url_for('get_file', filename=filename)
        predictions = make_prediction(file_url)
    else:
        file_url = None
        predictions = None
    
    return render_template('index.html', form=form, file_url=file_url, predictions=predictions)


if __name__ == "__main__":
    app.run(debug=True)