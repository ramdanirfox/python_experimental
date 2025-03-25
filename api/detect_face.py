from flask import Blueprint, current_app, jsonify, request, Response
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
import os, math


detect_face_endpoint = Blueprint('detect_face_endpoint', __name__)

# Initialize MTCNN and InceptionResnetV1
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device='cpu') # Use 'cuda' if you have GPU
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def cosine_similarity_basic(vector1, vector2):
    """
    Calculates cosine similarity between two vectors without NumPy.
    """
    dot_product = sum(a * b for a, b in zip(vector1, vector2))
    norm_vector1 = math.sqrt(sum(a * a for a in vector1))
    norm_vector2 = math.sqrt(sum(b * b for b in vector2))
    if norm_vector1 == 0 or norm_vector2 == 0:
        return 0 #handle zero norm vectors.
    return dot_product / (norm_vector1 * norm_vector2)

def cosine_similarity_numpy(vector1, vector2):
    """
    Calculates cosine similarity between two vectors using NumPy.
    """
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    if norm_vector1 == 0 or norm_vector2 == 0:
        return 0  # Handle zero-norm vectors
    return dot_product / (norm_vector1 * norm_vector2)

def get_face_embedding(image_path):
    """
    Extracts face embedding from an image.
    """
    img = Image.open(image_path)
    img_cropped = mtcnn(img)

    if img_cropped is not None:
        img_embedding = resnet(img_cropped.unsqueeze(0))
        return img_embedding.detach().numpy()
    else:
        return None  # No face detected

def compare_faces(image1_path, image2_path, threshold=0.7):
    """
    Compares two faces and returns True if they are similar, False otherwise.
    """
    embedding1 = get_face_embedding(image1_path)
    embedding2 = get_face_embedding(image2_path)
    if embedding1 is None or embedding2 is None:
        return False  # One or both images have no faces
    similarity = cosine_similarity_numpy(embedding1[0], embedding2[0])
    return similarity > threshold

@detect_face_endpoint.route("/detect_face", methods=['POST'])
def compare_faces_api():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Missing images. image1 and image2 required with multipart/formdata request'}), 400

    image1 = request.files['image1']
    image2 = request.files['image2']

    if image1.filename == '' or image2.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    image1_path = os.path.join(current_app.config['UPLOAD_FOLDER'], image1.filename)
    image2_path = os.path.join(current_app.config['UPLOAD_FOLDER'], image2.filename)

    image1.save(image1_path)
    image2.save(image2_path)

    try:
        result = compare_faces(image1_path, image2_path)
        return jsonify({'result': bool(result)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up uploaded files (optional)
        os.remove(image1_path)
        os.remove(image2_path)