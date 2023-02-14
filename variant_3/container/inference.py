"""
This file is for local inference. Just specify the path to the best model and a test image.
"""
import torch

from variant_3.container import inference_utils

model_path = "./saved_models/best_roc_auc_model.pth"
image_path = "/Users/alejopaullier/p34028.png"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
data = inference_utils.build_payload(image_path)
model = inference_utils.get_model(model_path, device).to(device)
model.eval()

def infer(model, data):
    """
    Do an inference on a single payload. Since this functions intends to reproduce future use we expect data to come
    as a JSON payload where image metadata comes as a dictionary and the image is base64 encoded.
    :param model: trained model.
    :param data: JSON with an "image" key.
    :return: prediction (between 0 and 1).
    """
    # Load data
    image_b64_encoded = data["image"]
    image = inference_utils.decode_image_to_torch(image_b64_encoded, device)
    prediction = inference_utils.predict(model, image)
    return str(prediction)


if __name__ == "__main__":
    prediction = infer(model, data)
    print(f"Prediction: {prediction}")
