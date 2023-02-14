import base64
import cv2
import flask
import json
import numpy as np
import os
import timm
import torch

from albumentations import (Normalize, Compose, Resize)
from albumentations.pytorch import ToTensorV2
from flask import request

# The flask app for serving predictions
app = flask.Flask(__name__)
cwd = os.getcwd()
best_model_path = cwd + "/saved_models/best_roc_auc_model.pth"

class ScoringService(object):
    def __init__(self, best_model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_model_path = best_model_path
        self.model = self.get_model()  # Where we keep the model when it's loaded

    def get_model(self):
        self.model = timm.create_model("efficientnet_b2", pretrained=False)
        self.model.reset_classifier(1)
        checkpoint = torch.load(self.best_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        return self.model

    def predict(self, image):
        with torch.no_grad():
            prediction = self.model.forward(image)
            prediction = torch.sigmoid(prediction)[0].item()
        return prediction

    @staticmethod
    def transform(image):
        """
        Define series of transformations which will be applied to raw image.
        """
        transformer = Compose([Resize(height=224, width=224, p=1),
                               Normalize(),
                               ToTensorV2()])
        image = transformer(image=image)["image"]
        return image

    def decode_image_to_torch(self, image_b64_encoded):
        """
        Decodes a base64 encoded image and converts it to the expected PyTorch tensor.
        :param image_b64_encoded: byte encoded image.
        :return image: image as torch tensor.
        """
        image_decoded = base64.b64decode(image_b64_encoded)
        image_as_np = np.frombuffer(image_decoded, dtype=np.uint8)
        image_as_np = cv2.imdecode(image_as_np, flags=1)
        image = self.transform(image_as_np)
        image = torch.tensor(image, device=self.device, dtype=torch.float32).clone().detach()
        image = torch.unsqueeze(image, 0)  # we need to add the batch dimension
        return image


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = scorer.model is not None  # You can insert a health check here
    status = 200 if health else 404
    return flask.Response(response=f"Status is: {status}", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def transformation():
    """Do an inference on a single batch of data."""
    data = json.loads(request.data)
    image_b64_encoded = data["image"]
    image = scorer.decode_image_to_torch(image_b64_encoded)
    prediction = scorer.predict(image)
    return str(prediction)

scorer = ScoringService(best_model_path)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6565, debug=True)
