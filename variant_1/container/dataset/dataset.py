import cv2
import numpy as np

# Data Augmentation for Image Preprocessing
from albumentations import (ToFloat, Normalize, VerticalFlip, HorizontalFlip, Compose, Resize,
                            RandomBrightnessContrast, HueSaturationValue, Blur, GaussNoise,
                            Rotate, RandomResizedCrop, Cutout, ShiftScaleRotate)
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class MelanomaDataset(Dataset):

    def __init__(self, dataframe, vertical_flip, horizontal_flip, is_train=True, is_valid=False, is_test=False):

        self.dataframe = dataframe
        self.is_train = is_train
        self.is_valid = is_valid
        self.vertical_flip = vertical_flip
        self.horizontal_flip = horizontal_flip

        # Data Augmentation (custom for each dataset type)
        # Crop dimensions are because of EfficientNet input layer. Checkout each EfficientNet variant resolution here:
        # https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
        if is_train or is_test:
            self.transform = Compose([RandomResizedCrop(height=224,
                                                        width=224,
                                                        scale=(0.4, 1.0)),
                                      ShiftScaleRotate(rotate_limit=90,
                                                       scale_limit=[0.8, 1.2]),
                                      HorizontalFlip(p=self.horizontal_flip),
                                      VerticalFlip(p=self.vertical_flip),
                                      HueSaturationValue(sat_shift_limit=[0.7, 1.3],
                                                         hue_shift_limit=[-0.1, 0.1]),
                                      RandomBrightnessContrast(brightness_limit=[0.7, 1.3],
                                                               contrast_limit=[0.7, 1.3]),
                                      Normalize(),
                                      ToTensorV2()])
        else:
            self.transform = Compose([RandomResizedCrop(height=224,
                                                        width=224,
                                                        scale=(0.4, 1.0)),
                                      Normalize(),
                                      ToTensorV2()])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        # Select path and read image
        image_path = self.dataframe['path_jpg'][index]
        image = cv2.imread(image_path)
        # For this image also import .csv information (sex, age, anatomy)
        csv_data = np.array(self.dataframe.iloc[index][['sex', 'age', 'anatomy']].values,
                            dtype=np.float32)

        # Apply transforms
        image = self.transform(image=image)
        # Extract image from dictionary
        image = image['image']

        # If train/valid: image + class | If test: only image
        if self.is_train or self.is_valid:
            return (image, csv_data), self.dataframe['target'][index]
        else:
            return (image, csv_data)
