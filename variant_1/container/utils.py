import cv2
import numpy as np
import os
import random
import string
import torch


from sklearn.model_selection import GroupKFold


def set_seed(seed=1234):
    '''
    Sets the seed of the entire project so results are the same every time we run.
    This is for REPRODUCIBILITY.
    '''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def save_image_to(location, destination, verbose=False):
    """
    Reads an image from a location path and writes it to a destination path.
    :param location: path where the image is stored
    :param destination: path where we want to save the image
    :param verbose: whether to print if saving was succesful or not.
    :return: -
    """
    image_cv2 = cv2.imread(location)
    result = cv2.imwrite(destination, image_cv2)
    if verbose==True:
        print(result)


def add_path_column(df, image_id_column, new_column_name, directory):
    """
    Creates a new column which contains the path to the images.
    :param df: a dataframe which has a column with images ids.
    :param image_id_column: the name of the column with images ids.
    :param new_column_name: the column name for the new column with images paths.
    :param directory: directory where images are stored.
    :return: df with the new column
    """
    # Create the paths
    path_train = directory + df[image_id_column] + '.jpg'
    # Append to the original dataframes
    df[new_column_name] = path_train

    return df


def keep_best_model(model_directory):
    model_names = []
    for _, _, files in os.walk(model_directory):
        model_names = files
    model_names = [y[:5] for y  in [x[-9:] for x in model_names]]
    scores = [float(score) for score in model_names]
    index_min = np.argmax(scores)
    best_model = files[index_min]
    # print(best_model)
    files.remove(best_model)
    for file in files:
        os.remove(model_directory + "/" + file)


def create_folds(df, k):
    """
    Creates folds for training.
    :param df: a dataframe with a "target" column and an "patient_id" column.
    :param k: number of folds
    :return: folds
    """
    # Create Object
    group_fold = GroupKFold(n_splits = k)

    length = len(df)

    # Generate indices to split data into training and test set.
    folds = group_fold.split(X = np.zeros(length),
                             y = df['target'],
                             groups = df['patient_id'].tolist())
    return folds


def create_oof(df):
    length = len(df)
    # Out of Fold Predictions
    oof = np.zeros(shape = (length, 1))


def short_id():
    alphabet = string.ascii_lowercase + string.digits
    return ''.join(random.choices(alphabet, k=12))
