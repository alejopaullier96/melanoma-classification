import pandas as pd
import torch
import utils
import warnings
warnings.filterwarnings("ignore")


from config import paths
from dataset.dataset import MelanomaDataset
from models.efficientnet.model import EfficientNetwork
from preprocessing import preprocess
from torch.utils.data import DataLoader


def best_single_model(df, model, device):
    """
    This functions makes perdictions for a whole dataset given a model.
    :param df: pandas dataframe with the expected columns.
    :param model: trained model (architecture + weights).
    :param device:
    :return:
    """
    ds = MelanomaDataset(df, vertical_flip=0.5, horizontal_flip=0.5, is_train=False, is_valid=False, is_test=True)
    ds_loader = DataLoader(ds, batch_size=64, shuffle=False)
    predictions = []
    with torch.no_grad():
        for k, (images, csv_data) in enumerate(ds_loader):
            images = torch.tensor(images, device=device, dtype=torch.float32)
            csv_data = torch.tensor(csv_data, device=device, dtype=torch.float32)
            out = model(images, csv_data)
            # Covert to probabilities
            out = torch.sigmoid(out).cpu().detach().numpy().tolist()
            predictions += out
    predictions = pd.DataFrame(predictions)
    return predictions


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device available now: ', device)
df = pd.read_csv(paths.TRAIN_CSV, sep=',')
df = utils.add_path_column(df, "image_id", "path_jpg", paths.TRAIN_JPG_FOLDER)

# Load Label Encoders
le_sex = preprocess.load_label_encoder("encoders/label_encoder_sex.pkl")
le_anatomy = preprocess.load_label_encoder("encoders/label_encoder_anatomy.pkl")
# Load MinMax Scalers
minmax_sex = preprocess.load_minmax_scaler("encoders/minmax_scaler_sex.pkl")
minmax_age = preprocess.load_minmax_scaler("encoders/minmax_scaler_age.pkl")
minmax_anatomy = preprocess.load_minmax_scaler("encoders/minmax_scaler_anatomy.pkl")
# Create dictionaries
label_encoders = {
    "sex": le_sex,
    "anatomy": le_anatomy
}
minmax_scalers = {
    "sex": minmax_sex,
    "age": minmax_age,
    "anatomy": minmax_anatomy
}
# Label encode data
df = preprocess.label_encode_transform(df, label_encoders)
# Scale data
train_df = preprocess.minmax_scale_transform(df, minmax_scalers)
best_model = EfficientNetwork(output_size=1,
                              no_columns=3,
                              b4=False,
                              b2=True).to(device)
model_path = "model.pth"
best_model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
best_model.eval()
predictions = best_single_model(df, best_model, device)
predictions.to_csv("predictions.csv")
