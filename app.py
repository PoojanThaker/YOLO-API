import numpy as np
import cv2
import os
from flask import Flask, request, jsonify
import io
from PIL import Image
from model import Yolo

# Initialize the Flask application
app = Flask(__name__)


# route http posts to this method
@app.route('/api/', methods=['POST'])
def main():
    img = request.files["image"].read()
    img = Image.open(io.BytesIO(img))
    npimg = np.array(img)
    image = npimg.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    res = Yolo().get_predection(image)
    return jsonify(res)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
