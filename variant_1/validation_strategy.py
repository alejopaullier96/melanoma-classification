import numpy as np

from sklearn.model_selection import GroupKFold


def create_folds(df, k):
    """
    :param df: a dataframe with a "target" column and an "ID" column.
    :param k: number of folds
    :return: folds
    """
    # Create Object
    group_fold = GroupKFold(n_splits = k)

    length = len(df)

    # Generate indices to split data into training and test set.
    folds = group_fold.split(X = np.zeros(length),
                             y = df['target'],
                             groups = df['ID'].tolist())
    return folds


def create_oof(df):
    length = len(df)
    # Out of Fold Predictions
    oof = np.zeros(shape = (length, 1))
