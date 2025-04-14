from flask import Flask, render_template
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FLaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField

app = Flask(__name__)

app.config['SECRET_KEY'] = "qwerty"
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)

class UploadForm(FLaskForm):
    photo = FileField(
        validators = [
            FileAllowed(photos, "Only images are allowed"),
            FileRequired("File field should not be empty")
        ]
    )
    submit = SubmitField('Upload')

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)