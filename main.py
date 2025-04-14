from flask import Flask, render_template, send_from_directory, url_for, request
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

app.config['SECRET_KEY'] = "qwerty"
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'

# Ensure the upload directory exists
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


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    form = UploadForm()
    if form.validate_on_submit():
        file = form.photo.data
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename))
        file_url = url_for('get_file', filename=filename)
    else:
        file_url = None
    
    return render_template('index.html', form=form, file_url=file_url)


if __name__ == "__main__":
    app.run(debug=True)