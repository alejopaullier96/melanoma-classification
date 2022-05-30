import base64
import cv2
import numpy as np
import torch

from albumentations import (Normalize, VerticalFlip, HorizontalFlip, Compose,
                            RandomBrightnessContrast, HueSaturationValue,
                            RandomResizedCrop, ShiftScaleRotate)
from albumentations.pytorch import ToTensorV2
from config import config
from dataset.hyperparameters import hyperparameters as ds_hp
from models.efficientnet.hyperparameters import hyperparameters as ef_hp
from models.efficientnet.model import EfficientNetwork
from preprocessing import preprocess


def build_payload(image_path, sex, age, anatomy):
    image_b64_encoded = encode_image_b64(image_path)
    columns = ["sex", "age", "anatomy"]
    minmax_scalers = {}
    # Load MinMaxScaler to scale features
    for column in columns:
        minmax_scalers[column] = preprocess.load_minmax_scaler("./encoders/minmax_scaler_" + column + ".pkl")

    json_file = {
        "sex": minmax_scalers["sex"].transform(np.array(sex).reshape(-1, 1)).item(),
        "age": minmax_scalers["age"].transform(np.array(age).reshape(-1, 1)).item(),
        "anatomy": minmax_scalers["anatomy"].transform(np.array(anatomy).reshape(-1, 1)).item()
    }
    payload = {
        "image": image_b64_encoded.decode("utf8"),
        "json_data": json_file
    }
    return payload


def convert_json_to_tensor(json_data, device):
    """
    Converts incoming JSON data to a PyTorch tensor.
    :param json_data: JSON payload.
    :param device: device to load model, either "cuda" or "cpu".
    :return torch_data: JSON payload converted to PyTorch tensor.
    """
    data = torch.tensor(list(json_data.values()), device=device, dtype=torch.float32)
    data = torch.unsqueeze(data, 0)  # we need to add a dimension because of the DataLoader
    return data


def decode_image_to_torch(image_b64_encoded, device, show_image=False):
    """
    Decodes a base64 encoded image and converts it to the expected PyTorch tensor.
    :param image_b64_encoded: byte encoded image.
    :param device: device to load model, either "cuda" or "cpu".
    :return image: image as torch tensor.
    """
    image_decoded = base64.b64decode(image_b64_encoded)
    image_as_np = np.frombuffer(image_decoded, dtype=np.uint8)
    image = cv2.imdecode(image_as_np, flags=1)
    if show_image:
        cv2.imshow('image', image)
        cv2.waitKey(2500)
        cv2.destroyWindow('image')
    image = transform(image)
    image = torch.tensor(image, device=device, dtype=torch.float32).clone().detach()
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


def get_model(path, device):
    """
    Get the model object for this instance, loading it if it's not already loaded. In this case the model's weight
    extension is ".pth".
    :param path: a string containing the path to the model's weights
    :param device: device to load model, either "cuda" or "cpu".
    :return: model with pre-trained weights.
    """
    model = EfficientNetwork(output_size=config.OUTPUT_SIZE,
                            no_columns=ef_hp.NO_COLUMNS,
                            b4=False, b2=True).to(device)
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    return model


def predict(model, image, json_data):
    """
    For the input, do the predictions and return them.
    :param model: trained PyTorch model.
    :param image:
    :param json_data:
    :return: prediction
    """
    TTA = 3
    model.eval()

    with torch.no_grad():
        predictions = 0
        for i in range(TTA):
            prediction = model.forward(image, json_data, verbose=False)
            prediction = torch.sigmoid(prediction)[0].item()
            predictions += prediction
        predictions /= TTA
    return predictions


def transform(image):
    """
    Applies series of transformations to raw image.
    :param image: openCV image.
    :return: transformed image as a PyTorch tensor.
    """
    transformer = Compose([RandomResizedCrop(height=224,
                                             width=224,
                                             scale=(0.4, 1.0)),
                           ShiftScaleRotate(rotate_limit=90,
                                            scale_limit=[0.8, 1.2]),
                           HorizontalFlip(p=ds_hp.HORIZONTAL_FLIP),
                           VerticalFlip(p=ds_hp.VERTICAL_FLIP),
                           HueSaturationValue(sat_shift_limit=[0.7, 1.3],
                                              hue_shift_limit=[-0.1, 0.1]),
                           RandomBrightnessContrast(brightness_limit=[0.7, 1.3],
                                                    contrast_limit=[0.7, 1.3]),
                           Normalize(),
                           ToTensorV2()])
    image = transformer(image=image)["image"]
    return image
