from flask import Blueprint, jsonify, request, Response
from datetime import datetime

auth_endpoint = Blueprint('auth_endpoint', __name__)

@auth_endpoint.route("/hello2")
def hello_2():
    # request.cookies.add("Nama", "Bagus Triwibowo")
    # Response.set_cookie(self=Response, key="nama", value="bagus triwibowo", domain="/", max_age=100)
    return jsonify({
        "name": __name__,
        "today": datetime.now()
    })