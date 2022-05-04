import base64
import cv2
import flask
import json
import numpy as np
import pickle
import torch

from albumentations import (Normalize, Compose, RandomResizedCrop)
from albumentations.pytorch import ToTensorV2
from config import config
from flask import request
from models.efficientnet.hyperparameters import hyperparameters as ef_hp
from models.efficientnet.model import EfficientNetwork

# The flask app for serving predictions
app = flask.Flask(__name__)

label_encoder_sex = pickle.load(open("encoders/label_encoder_sex", 'rb'))
label_encoder_anatomy = pickle.load(open("encoders/label_encoder_anatomy", 'rb'))


class ScoringService(object):

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None  # Where we keep the model when it's loaded
        self.label_encoder_sex = label_encoder_sex
        self.label_encoder_anatomy = label_encoder_anatomy

    def get_model(self):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if self.model is None:
            self.model = EfficientNetwork(output_size=config.OUTPUT_SIZE,
                                          no_columns=ef_hp.NO_COLUMNS,
                                          b4=False, b2=True).to(self.device)
            self.model.load_state_dict(torch.load("model.pth", map_location=self.device))
        return self.model

    def predict(self, image, json_data):
        """For the input, do the predictions and return them."""
        model = self.get_model()
        model.eval()
        with torch.no_grad():
            prediction = model.forward(image, json_data, verbose=False)
            prediction = torch.sigmoid(prediction)[0].item()
        return prediction

    @staticmethod
    def transform(image):
        """
        Define series of transformations which will be applied to raw image.
        """
        transformer = Compose([RandomResizedCrop(height=224,
                                                 width=224,
                                                 scale=(0.4, 1.0)),
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
        image = cv2.imdecode(image_as_np, flags=1)
        image = self.transform(image)
        image = torch.tensor(image, device=self.device, dtype=torch.float32).clone().detach()
        image = torch.unsqueeze(image, 0)  # we need to add a dimension because of the DataLoader
        return image

    def transform_json_data(self, json_data):
        json_data["sex"] = self.label_encoder_sex.transform([json_data["sex"]])[0]
        json_data["anatomy"] = self.label_encoder_anatomy.transform([json_data["anatomy"]])[0]
        return json_data

    def convert_json_to_tensor(self, json_data):
        """
        Converts incoming JSON data to a PyTorch tensor.
        :param json_data: JSON payload
        :return torch_data: JSON payload converted to PyTorch tensor.
        """
        data = self.transform_json_data(json_data)
        data = torch.tensor(list(data.values()), device=self.device, dtype=torch.float32)
        data = torch.unsqueeze(data, 0)  # we need to add a dimension because of the DataLoader
        return data


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response=f"Status is: {status}", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    # Load data
    data = json.loads(request.data)
    image_b64_encoded = data["image"]
    image = scorer.decode_image_to_torch(image_b64_encoded)
    json_data = data["json_data"]
    json_data = scorer.convert_json_to_tensor(json_data)

    # Make prediction
    prediction = scorer.predict(image, json_data)
    return prediction


scorer = ScoringService()
scorer.get_model()

if __name__ == "__main__":
    app.run(debug=True)