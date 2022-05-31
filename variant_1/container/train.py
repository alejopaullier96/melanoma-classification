#!/usr/bin/env python

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

utils.set_seed()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device available now: ', device)
train_df = pd.read_csv(paths.TRAIN_CSV, sep = ',')
test_df = pd.read_csv(paths.TEST_CSV, sep = ',')
train_df = utils.add_path_column(train_df, "image_id", "path_jpg", paths.TRAIN_JPG_FOLDER)
test_df = utils.add_path_column(test_df, "image_id", "path_jpg", paths.TEST_JPG_FOLDER)
# Label encode data
train_df, label_encoders = preprocess.label_encode(train_df)
test_df = preprocess.label_encode_transform(test_df, label_encoders)
# Scale data
train_df, minmax_scalers = preprocess.minmax_scale(train_df)
test_df = preprocess.minmax_scale_transform(test_df, minmax_scalers)
# Save
preprocess.save_label_encoders(label_encoders)
preprocess.save_minmax_scalers(minmax_scalers)

# Create folds
folds = utils.create_folds(train_df, config.FOLDS)
# Predictions
predictions = torch.zeros(size = (len(test_df), 1), dtype=torch.float32, device=device)
# Create model instance
model = EfficientNetwork(output_size=config.OUTPUT_SIZE,
                         no_columns=ef_hp.NO_COLUMNS,
                         b4=False, b2=True).to(device)
version = utils.short_id()
# Train
oof, predictions = train_function(predictions,
                                   train_df,
                                   test_df,
                                   model,
                                   MelanomaDataset,
                                   folds,
                                   device,
                                   version=version)
oof = pd.DataFrame(oof)
predictions = pd.DataFrame(predictions)
# Save Out of Fold and Test predictions
oof.to_csv("oof.csv", index=False)
predictions.to_csv("predictions.csv", index=False)

# Uncomment if you want to keep the best model only:
# utils.keep_best_model("saved_models")
