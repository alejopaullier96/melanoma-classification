import cv2
import json
import pickle
import torch


from albumentations import (Normalize, Compose, RandomResizedCrop)
from albumentations.pytorch import ToTensorV2
from config import config
from models.efficientnet.hyperparameters import hyperparameters as ef_hp
from models.efficientnet.model import EfficientNetwork


def sep():
    print("-"*100)


def transform_json_data(json):
    """
    Encodes input data with label encoders.
    :param json: json data with a "sex" and "anatomy" keys.
    :return: encoded features
    """
    json["sex"] = label_encoder_sex.transform([json["sex"]])[0]
    json["anatomy"] = label_encoder_anatomy.transform([json["anatomy"]])[0]
    return json


def load_json_to_tensor(location):
    """
    Loads a JSON payload from the specified location, transforms features and converts it to a PyTorch tensor.
    :param location: location where the JSON payload is stored
    :return: encoded features
    """
    f = open(location)
    data = json.load(f)
    data = transform_json_data(data)
    data = torch.tensor(list(data.values()), device=device, dtype=torch.float32)
    data = torch.unsqueeze(data, 0) # we need to add a dimension because of the DataLoader
    return data


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


def load_image_to_tensor(location):
    """
    Loads an image from specified location, transforms it and converts it to a PyTorch tensor.
    """
    image = cv2.imread(location)
    image = transform(image)
    image = torch.tensor(image, device=device, dtype=torch.float32)
    image = torch.unsqueeze(image, 0) # we need to add a dimension because of the DataLoader
    return image


def predict(model, image, data):
    """
    Makes a prediction for an image with metadata
    :param model: trained model weights.
    :param image: melanoma image.
    :param data: metadata in JSON format.
    :return:
    """
    model.eval()
    with torch.no_grad():
        prediction = model.forward(image, data, verbose=False)
        prediction = torch.sigmoid(prediction)
    return prediction


# Load pickled Label Encoders
label_encoder_sex = pickle.load(open("encoders/label_encoder_sex", 'rb'))
label_encoder_anatomy = pickle.load(open("encoders/label_encoder_anatomy", 'rb'))
# Load best model weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EfficientNetwork(output_size=config.OUTPUT_SIZE,
                         no_columns=ef_hp.NO_COLUMNS,
                         b4=False, b2=True).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
# Load image and JSON metadata
image = load_image_to_tensor("./data/images/" + filename + ".jpg")
data = load_json_to_tensor("./data/payloads/" + filename + ".json")
# Get a prediction
predict(model, image, data)
