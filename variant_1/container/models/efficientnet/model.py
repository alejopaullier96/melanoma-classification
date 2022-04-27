import torch
import torch.nn as nn
import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet
from .hyperparameters import hyperparameters

class EfficientNetwork(nn.Module):
    def __init__(self, output_size, no_columns, b4=False, b2=False):
        super().__init__()

        self.b4 = b4
        self.b2 = b2
        self.no_columns = no_columns

        # Define Feature part (IMAGE)
        if b4:
            self.features = EfficientNet.from_pretrained('efficientnet-b4')
        elif b2:
            self.features = EfficientNet.from_pretrained('efficientnet-b2')
        else:
            self.features = EfficientNet.from_pretrained('efficientnet-b7')

        # (CSV)
        self.csv = nn.Sequential(nn.Linear(self.no_columns,
                                           hyperparameters.FFNN_HIDDEN_LAYER_SIZE),
                                 nn.BatchNorm1d(hyperparameters.FFNN_HIDDEN_LAYER_SIZE),
                                 nn.ReLU(),
                                 nn.Dropout(p=hyperparameters.DROPOUT),

                                 nn.Linear(hyperparameters.FFNN_HIDDEN_LAYER_SIZE,
                                           hyperparameters.FFNN_HIDDEN_LAYER_SIZE),
                                 nn.BatchNorm1d(hyperparameters.FFNN_HIDDEN_LAYER_SIZE),
                                 nn.ReLU(),
                                 nn.Dropout(p=hyperparameters.DROPOUT))

        # Define Classification part
        if b4:
            self.classification = nn.Sequential(nn.Linear(hyperparameters.EFFNET_B4_HIDDEN_LAYER_SIZE +
                                                          hyperparameters.FFNN_HIDDEN_LAYER_SIZE, 
                                                          output_size))
        elif b2:
            self.classification = nn.Sequential(nn.Linear(hyperparameters.EFFNET_B2_HIDDEN_LAYER_SIZE +
                                                          hyperparameters.FFNN_HIDDEN_LAYER_SIZE,
                                                          output_size))
        else:
            self.classification = nn.Sequential(nn.Linear(hyperparameters.EFFNET_B7_HIDDEN_LAYER_SIZE +
                                                          hyperparameters.FFNN_HIDDEN_LAYER_SIZE,
                                                          output_size))

    def forward(self, image, csv_data, verbose=False):

        if verbose: print('Input Image shape:', image.shape, '\n' +
                         'Input csv_data shape:', csv_data.shape)

        # IMAGE CNN
        image = self.features.extract_features(image)
        if verbose: print('Features Image shape:', image.shape)

        if self.b4:
            image = F.avg_pool2d(image, image.size()[2:]).reshape(-1, hyperparameters.EFFNET_B4_HIDDEN_LAYER_SIZE)
        elif self.b2:
            image = F.avg_pool2d(image, image.size()[2:]).reshape(-1, hyperparameters.EFFNET_B2_HIDDEN_LAYER_SIZE)
        else:
            image = F.avg_pool2d(image, image.size()[2:]).reshape(-1, hyperparameters.EFFNET_B7_HIDDEN_LAYER_SIZE)
        if verbose: print('Image Reshaped shape:', image.shape)

        # CSV FNN
        csv_data = self.csv(csv_data)
        if verbose: print('CSV Data:', csv_data.shape)

        # Concatenate
        image_csv_data = torch.cat((image, csv_data), dim=1)

        # CLASSIF
        out = self.classification(image_csv_data)
        if verbose: print('Out shape:', out.shape)

        return out