# model = EfficientNetwork(output_size=config.OUTPUT_SIZE,
#                          no_columns=ef_hp.NO_COLUMNS,
#                          b4=False, b2=True).to(device)
# model.load_state_dict(torch.load(best_model_path))
# # Set the model in evaluation mode
# model.eval()
#
# with torch.no_grad():
#     out = model(images, csv_data)
#     pred = torch.sigmoid(out)
#     valid_preds[k * images.shape[0]: k * images.shape[0] + images.shape[0]] = pred


import os
import joblib

def predict_fn(input_object, model):
    ###########################################
    # Do your custom preprocessing logic here #
    ###########################################

    print("calling model")
    predictions = model.predict(input_object)
    return predictions


def model_fn(model_dir):
    print("loading model.joblib from: {}".format(model_dir))
    loaded_model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return loaded_model


def input_fn():
    return None


def output_fn():
    return None
