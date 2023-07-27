import os
import cv2
import numpy as np
from PIL import Image as Img, Image
from torchvision import transforms as T
import torch
from flask import Flask, request, send_file
import io

from U_2_Net import U2NET

def remove_background(image):
    model_dir = os.path.join(os.getcwd(), 'U-2-Net/saved_models/u2net/u2net.pth')
    net = U2NET(3,1)
    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    img = Img.fromarray(image).convert('RGB')
    ow, oh = img.size
    transform = T.Compose([
        T.Resize((320,320)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(img)
    img = img.unsqueeze(0)

    if torch.cuda.is_available():
        img = img.cuda()

    d1, d2, d3, d4, d5, d6, d7 = net(img)

    pred = d1[:,0,:,:]
    pred = torch.sigmoid(pred)
    pred = pred.data.cpu().numpy().squeeze()
    pred = (pred - pred.min()) / (pred.max() - pred.min())

    pred = cv2.resize(pred, (ow, oh))

    thresh = pred.copy()
    thresh[thresh >= 0.5] = 255
    thresh[thresh < 0.5] = 0
    thresh = thresh.astype(np.uint8)

    orig = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    foreground = cv2.bitwise_and(orig, orig, mask=thresh)

    alpha = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

    b, g, r = cv2.split(foreground)
    rgba = [b, g, r, alpha]
    result = cv2.merge(rgba, 4)

    return result

app = Flask(__name__)

@app.route('/remove_background', methods=['POST'])
def remove_background_api():
    file = request.files['image']  # get the image
    img = Image.open(file.stream)  # open the image
    img_array = np.array(img)  # convert the image to a numpy array
    result = remove_background(img_array)  # call the remove_background function

    # Convert the result to a PIL Image
    result_img = Image.fromarray(result)

    # Save the result to a BytesIO object
    img_bytes = io.BytesIO()
    result_img.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()

    # Return the result as a file
    return send_file(
        io.BytesIO(img_bytes),
        mimetype='image/png',
        as_attachment=True,
        attachment_filename='output.png')

if __name__ == '__main__':
    app.run(debug=True)
