import os
import cv2
import numpy as np
from flask import Flask, jsonify, request
from video_classifier import VideoFrameClassifier

app = Flask(__name__)

ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'gif']


@app.route('/')
def hello_world():
    return 'Hello, World!'


def is_image_file(file_name: str):
    """
    Description : Checks if the image is valid or not
    :param file_name: Nme of the file
    :return: bool
    """
    file_name = file_name.lower()
    if file_name.split('.')[-1] in ALLOWED_EXTENSIONS:
        return True
    return False


@app.route('/api/v1/getframetype', methods=['POST'])
def get_prediction():
    """
    Description : Detects the type of video frame from image and return its class
    :return: Returns a frame type from given image or return null
    """
    model_path = 'models/epoch_18_224.pth'
    output_path = 'predictions'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    file = request.files.get('file', '')
    image = file.read()
    frame_classifier = VideoFrameClassifier(model_path)
    if image and is_image_file(file.filename):
        np_img = np.fromstring(image, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        temp_image = img.copy()
        cv2.imwrite('temp.png',temp_image)
        img_h, img_w = img.shape[0], img.shape[1]
        frame_type = frame_classifier.classify_frame(temp_image, ground_truth=None)
        return jsonify({'frame_type': frame_type})
    else:
        return jsonify({'message':'not an image file'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
