from flask import Flask
from path import auth_endpoint
from flask_cors import CORS
from api.detect_image import detect_endpoint
from api.detect_face import detect_face_endpoint
import os

app = Flask(__name__)
cors = CORS(app)
app.register_blueprint(auth_endpoint)
app.register_blueprint(detect_endpoint)
app.register_blueprint(detect_face_endpoint)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def hello_world():
    
    return "<p>Hello, World!</p>"

# print("Hello Bagus")

# def main():
if __name__ == "__main__":
    app.run(debug=True, port=7101, host="0.0.0.0")
