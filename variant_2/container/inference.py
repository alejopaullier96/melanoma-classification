import torch

from variant_1.container import inference_utils

model_path = "model.pth"
image_path = "data/train_jpg/ISIC_5179549.jpg"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = inference_utils.build_payload(image_path,
                                     0,
                                     60,
                                     0)

model = inference_utils.get_model(model_path, device)
model.eval()

def infer(model, data):
    """
    Do an inference on a single payload. Since this functions intends to reproduce future use we expect data to come
    as a JSON payload where image metadata comes as a dictionary and the image is base64 encoded.
    :param model: trained model.
    :param data: JSON with an "image" and a "json_data" keys.
    :return: prediction (between 0 and 1).
    """
    # Load data
    image_b64_encoded = data["image"]
    image = inference_utils.decode_image_to_torch(image_b64_encoded, device)
    json_data = data["json_data"]
    json_data = inference_utils.convert_json_to_tensor(json_data, device)
    # Make prediction
    prediction = inference_utils.predict(model, image, json_data)
    return str(prediction)


if __name__ == "__main__":
    prediction = infer(model, data)
    print(f"Prediction: {prediction}")
