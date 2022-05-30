import pandas as pd
import torch

from dataset.dataset import MelanomaDataset
from torch.utils.data import DataLoader


# Basics
import pandas as pd
import torch
import warnings
warnings.filterwarnings("ignore")

# Local utility code
from config import config, paths
from dataset.dataset import MelanomaDataset
from models.efficientnet.hyperparameters import hyperparameters as ef_hp
from models.efficientnet.model import EfficientNetwork
from preprocessing import preprocess
from train_function import train_function
import utils

def best_single_model(df, model, predictions, device, TTA=3):
    """
    This functions makes perdictions for a whole dataset given a model.
    :param df: pandas dataframe with the expected columns.
    :param model: trained model (architecture + weights).
    :param predictions: torch tensor of zeros with lenght = len(df).
    :param device:
    :param TTA:
    :return:
    """
    ds = MelanomaDataset(df, vertical_flip=0.5, horizontal_flip=0.5,
                           is_train=False, is_valid=False, is_test=True)
    ds_loader = DataLoader(ds, batch_size=64, shuffle=False)

    model.eval()

    with torch.no_grad():
        for i in range(TTA):
            for k, (images, csv_data) in enumerate(ds_loader):
                images = torch.tensor(images, device=device, dtype=torch.float32)
                csv_data = torch.tensor(csv_data, device=device, dtype=torch.float32)

                out = model(images, csv_data)
                # Covert to probablities
                out = torch.sigmoid(out)

                # ADDS! the prediction to the matrix we already created
                predictions[k * images.shape[0]: k * images.shape[0] + images.shape[0]] += out

        # Divide Predictions by TTA (to average the results during TTA)
        predictions /= TTA
        predictions = pd.DataFrame(predictions)
        return predictions


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device available now: ', device)
df = pd.read_csv(paths.TRAIN_CSV, sep = ',')
df = utils.add_path_column(df, "image_id", "path_jpg", paths.TRAIN_JPG_FOLDER)
# Label encode data
df, label_encoders = preprocess.label_encode(df)
# Scale data
train_df, minmax_scalers = preprocess.minmax_scale(df)
best_model = EfficientNetwork(output_size = 1,
                            no_columns=3,
                            b4=False,
                            b2=True).to(device)
best_model.load_state_dict(torch.load("model.pth", map_location=torch.device(device)))
predictions = torch.zeros(size = (len(df), 1), dtype=torch.float32, device=device)
preds = best_single_model(df, best_model, predictions, device, TTA=3)
preds.to_csv("preds.csv")
