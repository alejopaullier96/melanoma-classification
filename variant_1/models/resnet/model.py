from hyperparameters import hyperparameters
from torchvision.models import resnet34, resnet50


class ResNet50Network(nn.Module):
    def __init__(self, output_size, no_columns):
        super().__init__()

        self.no_columns = no_columns
        self.output_size = output_size
        # Define Feature part (IMAGE)
        self.model = resnet50(pretrained=True)  # 1000 neurons out
        # (CSV data)
        self.csv = nn.Sequential(nn.Linear(self.no_columns,
                                           hyperparameters.FFNN_HIDDEN_LAYER_SIZE),
                                 nn.BatchNorm1d(hyperparameters.FFNN_HIDDEN_LAYER_SIZE),
                                 nn.ReLU(),
                                 nn.Dropout(p=hyperparameters.DROPOUT))

        # Define Classification part
        self.classification = nn.Linear(hyperparameters.RESNET_HIDDEN_LAYER_SIZE +
                                        hyperparameters.FFNN_HIDDEN_LAYER_SIZE,
                                        output_size)

    def forward(self, image, csv_data, verbose=False):

        if verbose: print('Input Image shape:', image.shape, '\n' +
                         'Input csv_data shape:', csv_data.shape)

        # Image CNN
        image = self.model(image)
        if verbose: print('Features Image shape:', image.shape)

        # CSV FNN
        csv_data = self.csv(csv_data)
        if verbose: print('CSV Data:', csv_data.shape)

        # Concatenate layers from image with layers from csv_data
        image_csv_data = torch.cat((image, csv_data), dim=1)

        # CLASSIF
        out = self.classification(image_csv_data)
        if verbose: print('Out shape:', out.shape)

        return out