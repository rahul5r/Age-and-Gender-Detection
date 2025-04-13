from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def hello_word():
    return render_template('index.html')

@app.route('/<name>')
def rediction(name):
    return f"<h1> Hello! {name}. </h1>"


if __name__ == "__main__":
    app.run(debug=True)