from flask import Flask
from path import auth_endpoint

app = Flask(__name__)
app.register_blueprint(auth_endpoint)
@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

# print("Hello Bagus")

# def main():
if __name__ == "__main__":
    app.run(debug=True, port=7101, host="0.0.0.0")
