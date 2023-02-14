import base64
import cv2
import numpy as np
import timm
import torch

from albumentations import (Normalize, Compose, CenterCrop)
from albumentations.pytorch import ToTensorV2


def build_payload(image_path):
    image_b64_encoded = encode_image_b64(image_path)
    payload = {
        "image": image_b64_encoded.decode("utf8")
    }
    return payload


def decode_image_to_torch(image_b64_encoded, device, verbose=False):
    """
    Decodes a base64 encoded image and converts it to the expected PyTorch tensor.
    :param image_b64_encoded: byte encoded image.
    :param device: device to load model, either "cuda" or "cpu".
    :return image: image as torch tensor.
    """
    image_decoded = base64.b64decode(image_b64_encoded) # decode byte-encoded image
    image_as_np = np.frombuffer(image_decoded, dtype=np.uint8) # convert the decoded image into numpy array
    image = cv2.imdecode(image_as_np, flags=1)
    if verbose:
        cv2.imshow('image', image)
        cv2.waitKey(2500)
        cv2.destroyWindow('image')
    image = transform(image) # apply relevant transformations to the image
    image = torch.tensor(image, device=device, dtype=torch.float32)
    image = torch.unsqueeze(image, 0)  # we need to add a dimension because of the DataLoader
    return image


def encode_image_b64(path):
    """
    Encodes an image as base64.
    :param path: path to image.
    :return: base64 encoded image
    """

    with open(path, "rb") as f:
        image = f.read()
    image_b64_encoded = base64.b64encode(image)
    return image_b64_encoded


def get_model(best_model_path, device):
    model = timm.create_model("efficientnet_b2", pretrained=False)
    model.reset_classifier(1)
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint)
    return model


def predict(model, image):
    """
    For the input, do the predictions and return them.
    :param model: trained PyTorch model.
    :param image:
    :param json_data:
    :return: prediction
    """
    with torch.no_grad():
        prediction = model.forward(image)
        prediction = torch.sigmoid(prediction)[0].item()
    return prediction


def transform(image):
    """
    Applies series of transformations to raw image.
    :param image: openCV image.
    :return: transformed image as a PyTorch tensor.
    """
    transformer = Compose([CenterCrop(height=224,
                                  width=224,
                                  p=1),
                           Normalize(),
                           ToTensorV2()])
    image = transformer(image=image)["image"]
    return image
