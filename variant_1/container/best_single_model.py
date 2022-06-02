import numpy as np

from dataset.dataset import MelanomaDataset
from torch.utils.data import DataLoader


# Basics
import pandas as pd
import torch
import warnings
warnings.filterwarnings("ignore")

# Local utility code
from config import paths
from dataset.dataset import MelanomaDataset
from models.efficientnet.model import EfficientNetwork
from preprocessing import preprocess
import utils

def best_single_model(df, model, device):
    """
    This functions makes perdictions for a whole dataset given a model.
    :param df: pandas dataframe with the expected columns.
    :param model: trained model (architecture + weights).
    :param device:
    :return:
    """
    ds = MelanomaDataset(df, vertical_flip=0.5, horizontal_flip=0.5,
                           is_train=False, is_valid=False, is_test=True)
    ds_loader = DataLoader(ds, batch_size=64, shuffle=False)
    predictions = []

    with torch.no_grad():
        for k, (images, csv_data) in enumerate(ds_loader):
            images = torch.tensor(images, device=device, dtype=torch.float32)
            csv_data = torch.tensor(csv_data, device=device, dtype=torch.float32)
            out = model(images, csv_data)
            # Covert to probabilities
            out = torch.sigmoid(out).detach().numpy().tolist()
            predictions += out

        predictions = pd.DataFrame(predictions)
        return predictions


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device available now: ', device)
df = pd.read_csv(paths.TRAIN_CSV, sep = ',')[0:100]
df = utils.add_path_column(df, "image_id", "path_jpg", paths.TRAIN_JPG_FOLDER)
# Label encode data
df, label_encoders = preprocess.label_encode(df)
# Scale data
train_df, minmax_scalers = preprocess.minmax_scale(df)
best_model = EfficientNetwork(output_size=1,
                              no_columns=3,
                              b4=False,
                              b2=True).to(device)
best_model.load_state_dict(torch.load("model.pth", map_location=torch.device(device)))
best_model.eval()
predictions = torch.zeros(size = (len(df), 1), dtype=torch.float32, device=device)
preds = best_single_model(df, best_model, device)
preds.to_csv("preds.csv")
