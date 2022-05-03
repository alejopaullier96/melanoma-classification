import pickle

from sklearn.preprocessing import LabelEncoder


def label_encode(df):
    to_encode = ['sex', 'anatomy', 'diagnosis']
    encoded_all = []
    label_encoders ={}

    for column in to_encode:
        label_encoder = LabelEncoder()
        encoded = label_encoder.fit_transform(df[column])
        encoded_all.append(encoded)
        label_encoders[column] = label_encoder

    df['sex'] = encoded_all[0]
    df['anatomy'] = encoded_all[1]
    df['diagnosis'] = encoded_all[2]

    if 'benign_malignant' in df.columns:
        df.drop(['benign_malignant'], axis=1, inplace=True)

    return df, label_encoders


def label_encode_transform(df, label_encoders):
    to_encode = ['sex', 'anatomy']
    encoded_all = []

    for column in to_encode:
        label_encoder = label_encoders[column]
        encoded = label_encoder.transform(df[column])
        encoded_all.append(encoded)

    df['sex'] = encoded_all[0]
    df['anatomy'] = encoded_all[1]

    return df


def save_label_encoders(label_encoders):
    """
    Save encoders to the encoders folder as pickles.
    :param label_encoders: list with label encoders
    :return: -
    """
    to_encode = ['sex', 'anatomy']

    for column in to_encode:
        label_encoder = label_encoders[column]
        with open('encoders/label_encoder'+ "_" + column, 'wb') as f:
            pickle.dump(label_encoder, f)
