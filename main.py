from flask import Flask, request, send_file, logging, Response
from flask_cors import CORS
import io
import os
import cv2
import numpy as np
from PIL import Image as Img, Image
from torchvision import transforms as T
import torch
import base64

from U2Net.model import U2NET

app = Flask(__name__)
CORS(app)

def remove_background(image):
    app.logger.info(f'Processing image of shape {image.shape} and type {image.dtype}')
    
    model_dir = os.path.join(os.getcwd(), 'U2Net/saved_models/u2net/u2net.pth')
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
    net.eval()

    img = Img.fromarray(image).convert('RGB')
    ow, oh = img.size
    transform = T.Compose([
        T.Resize((320, 320)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(img)
    img = img.unsqueeze(0)

    d1, d2, d3, d4, d5, d6, d7 = net(img)

    pred = d1[:, 0, :, :]
    pred = torch.sigmoid(pred)
    pred = pred.data.cpu().numpy().squeeze()
    pred = (pred - pred.min()) / (pred.max() - pred.min())

    pred = cv2.resize(pred, (ow, oh))

    thresh = post_process_mask(pred)

    orig = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    foreground = cv2.bitwise_and(orig, orig, mask=thresh)

    alpha = thresh  # directly use thresh as alpha

    r, g, b = cv2.split(foreground)
    rgba = [r, g, b, alpha]
    result = cv2.merge(rgba)  # merge to 4-channel image without specifying 4

    app.logger.info(f'Created result image of shape {result.shape} and type {result.dtype}')

    return result

def post_process_mask(mask):
    # Convert the mask to uint8
    mask = (mask * 255).astype(np.uint8)

    # Threshold the mask
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.dilate(mask, kernel, iterations = 1)
    mask = cv2.erode(mask, kernel, iterations = 1)

    return mask

@app.route('/remove_background', methods=['POST'])
def remove_background_api():
    file = request.files['image']  # get the image
    img = Image.open(file.stream)  # open the image
    img_array = np.array(img)  # convert the image to a numpy array
    result = remove_background(img_array)  # call the remove_background function

    # Convert the result to a PIL Image
    result_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA))

    # Save the result to a BytesIO object
    img_bytes = io.BytesIO()
    result_img.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    # Create a data URL from the bytes
    data_url = "data:image/png;base64," + base64.b64encode(img_bytes.getvalue()).decode('utf-8')

    # Return an HTML string with the img tag
    return f'<img src="{data_url}" alt="Processed Image">'

if __name__ == '__main__':
    app.run(debug=True)
