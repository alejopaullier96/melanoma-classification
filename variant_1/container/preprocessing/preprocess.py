import pickle

from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def label_encode(df):
    to_encode = ['sex', 'anatomy']
    encoded_all = []
    label_encoders = {}

    for column in to_encode:
        label_encoder = LabelEncoder()
        encoded = label_encoder.fit_transform(df[column])
        encoded_all.append(encoded)
        label_encoders[column] = label_encoder

    df['sex'] = encoded_all[0]
    df['anatomy'] = encoded_all[1]

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


def minmax_scale(df):
    to_scale = ['sex', 'age', 'anatomy']
    scaled_all = []
    minmax_scalers = {}

    for column in to_scale:
        minmax_scaler = MinMaxScaler()
        scaled = minmax_scaler.fit_transform(df[column].values.reshape(-1, 1))
        scaled_all.append(scaled)
        minmax_scalers[column] = minmax_scaler

    df['sex'] = scaled_all[0]
    df['age'] = scaled_all[1]
    df['anatomy'] = scaled_all[2]

    if 'benign_malignant' in df.columns:
        df.drop(['benign_malignant'], axis=1, inplace=True)

    return df, minmax_scalers


def minmax_scale_transform(df, minmax_scalers):
    to_scale = ['sex', 'age', 'anatomy']
    scaled_all = []

    for column in to_scale:
        minmax_scaler = minmax_scalers[column]
        scaled = minmax_scaler.transform(df[column].to_numpy().reshape(-1, 1))
        scaled_all.append(scaled)
    df['sex'] = scaled_all[0]
    df['age'] = scaled_all[1]
    df['anatomy'] = scaled_all[2]

    return df


def load_label_encoder(label_encoder_path):
    """
    Load LabelEncoder pickle from the encoders folder.
    :param label_encoder_path: path to pickled LabelEncoder.
    :return: LabelEncoder
    """
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

    return label_encoder


def load_minmax_scaler(minmax_scaler_path):
    """
    Load MinMaxScaler pickle from the encoders folder.
    :param minmax_scaler_path: path to pickled MinMaxScaler.
    :return: MinMaxScaler
    """
    with open(minmax_scaler_path, 'rb') as f:
        minmax_scaler = pickle.load(f)

    return minmax_scaler


def save_label_encoders(label_encoders):
    """
    Save encoders to the encoders folder as pickles.
    :param label_encoders: list with label encoders
    :return: -
    """
    to_encode = ['sex', 'anatomy']

    for column in to_encode:
        label_encoder = label_encoders[column]
        with open('encoders/label_encoder' + "_" + column + ".pkl", 'wb') as f:
            pickle.dump(label_encoder, f)


def save_minmax_scalers(minmax_scalers):
    """
    Saves MinMaxScalers to the encoders folder as pickles.
    :param minmax_scalers: list with minmax scalers.
    :return: -
    """
    to_scale = ['sex', 'age', 'anatomy']

    for column in to_scale:
        minmax_scaler = minmax_scalers[column]
        with open('encoders/minmax_scaler' + "_" + column + ".pkl", 'wb') as f:
            pickle.dump(minmax_scaler, f)
