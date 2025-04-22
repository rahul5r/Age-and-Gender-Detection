import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('Age_Gender_Classification_model.h5')

def detect_faces(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    detected_faces = []

    for (x, y, w, h) in faces:
        face_crop_bgr = img_array[y:y+h, x:x+w]
        face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        detected_faces.append(face_crop_rgb)
    return detected_faces

def make_prediction(img_array):
    faces = detect_faces(img_array)
    predictions = []
    for face_rgb in faces:
        face_resized = cv2.resize(face_rgb, (200, 200))
        img_array = face_resized / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        age_pred, gender_pred = model.predict(img_array)
        
        predicted_age = age_pred[0][0]
        gender_prob = gender_pred[0][0]
        predicted_gender = "Female" if gender_prob > 0.5 else "Male"
        predictions.append({
            'age': int(predicted_age),
            'gender': predicted_gender,
            'face': face_rgb
        })
    return predictions

st.title('Age and Gender Detection')
st.text('It predicts the age and gender of each detected face(s) in an Image.')

uploaded_file = st.file_uploader("Choose an Image", accept_multiple_files=False)
img_array = None

if uploaded_file:
    bytes_data = uploaded_file.read()
    np_img = np.frombuffer(bytes_data, np.uint8)
    img_array = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    st.image(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB), caption="Uploaded Image", width=200)

enable = st.toggle("Capture from camera")

if enable:
    picture = st.camera_input("Take a picture")
    if picture is not None:
        bytes_data = picture.getvalue()
        np_img = np.frombuffer(bytes_data, np.uint8)
        img_array = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB), caption="Captured Image", width=200)

if img_array is not None:
    with st.container():
        if st.button("Submit"):
            predictions = make_prediction(img_array)
            num_cols = 5

            for i in range(0, len(predictions), num_cols):
                row = predictions[i:i + num_cols]
                cols = st.columns(len(row))

                for col, prediction in zip(cols, row):
                    with col:
                        st.image(prediction['face'], width=120)
                        st.markdown(f"**Gender:** {prediction['gender']}")
                        st.markdown(f"**Age:** {prediction['age']}")