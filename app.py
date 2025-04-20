import os
import cv2
import numpy as np
import streamlit as st
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pandas as pd

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

model = load_model('Age_Gender_Classification_model.h5')

def detect_faces(image_path):
    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    detected_faces = []

    counter = 1
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            face_crop_bgr = img[y:y+h, x:x+w]
            face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
            save_path = "faces/"+image_path.split("\\")[-1].split('.')[0]+str(counter)+'.'+image_path.split('/')[-1].split('.')[-1]
            counter += 1
            cv2.imwrite(save_path, cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2BGR))
            detected_faces.append(save_path)
    return detected_faces

def make_prediction(image_path):
    faces = detect_faces(image_path)
    predictions = []
    if len(faces) > 0:
        for img in faces:
            img = image.load_img(img, target_size=(200, 200))
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0 
            img_array = np.expand_dims(img_array, axis=0)
            
            age_pred, gender_pred = model.predict(img_array)
            
            predicted_age = age_pred[0][0]
            gender_prob = gender_pred[0][0]
            predicted_gender = "Female" if gender_prob > 0.5 else "Male"
            predictions.append({'age':int(predicted_age), 'gender':predicted_gender, 'face':img})
    return predictions

st.title('Age and Gender Detection')
st.text('It predicts the age and gender of each detected face(s) in an Image.')

uploaded_file = st.file_uploader("Choose an Image", accept_multiple_files=False)
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", width=200)

enable = st.toggle("Capture from camera")

filepath = None

if enable:
    picture = st.camera_input("Take a picture")
    if picture is not None:
        bytes_data = picture.getvalue()
        np_img = np.frombuffer(bytes_data, np.uint8)
        cv2_img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        filename = f"camera_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(UPLOAD_DIR, filename)
        cv2.imwrite(filepath, cv2_img)

elif uploaded_file is not None:
    filename = uploaded_file.name
    filepath = os.path.join(UPLOAD_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.read())

if filepath is not None:
    with st.container():
        if st.button("Submit"):
            predictions = make_prediction(filepath)
            num_cols = 5

            for i in range(0, len(predictions), num_cols):
                row = predictions[i:i + num_cols]
                cols = st.columns(len(row))

                for col, prediction in zip(cols, row):
                    with col:

                        st.image(prediction['face'], width=120)
                        st.markdown(f"**Gender:** {prediction['gender']}")
                        st.markdown(f"**Age:** {prediction['age']}")

                    