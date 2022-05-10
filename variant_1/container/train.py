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
train_df = utils.add_path_column(train_df, "image_id", "path_jpg", "./data/train_jpg/")
test_df = utils.add_path_column(test_df, "image_id", "path_jpg", "./data/test_jpg/")
train_df, label_encoders = preprocess.label_encode(train_df)
test_df = preprocess.label_encode_transform(test_df, label_encoders)
preprocess.save_label_encoders(label_encoders)

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
# Keep best model only
utils.keep_best_model("saved_models")
