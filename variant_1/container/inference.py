import torch

from variant_1.container import inference_utils

model_path = "model.pth"
image_path = "data/train_jpg/ISIC_0528044.jpg"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = inference_utils.build_payload(image_path,
                                     1,
                                     80.0,
                                     0)

model = inference_utils.get_model(model_path, device)

def infer(model, data):
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
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
