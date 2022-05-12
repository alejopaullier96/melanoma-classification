class config:
    BATCH_SIZE_TRAIN = 64
    BATCH_SIZE_VALIDATION = 32
    BATCH_SIZE_TEST = 16
    EPOCHS = 3
    FOLDS = 2
    LEARNING_RATE = 0.0005
    LR_FACTOR = 0.4  # BY HOW MUCH THE LR IS DECREASING
    LR_PATIENCE = 1  # 1 MODEL NOT IMPROVING UNTIL LR IS DECREASING
    NUM_WORKERS = 1
    OUTPUT_SIZE = 1
    PATIENCE = 3
    TTA = 3
    WEIGHT_DECAY = 0.0

class paths:
    TRAIN_CSV = "./data/train.csv"
    TEST_CSV = "./data/test.csv"
    TRAIN_JPG_FOLDER = "./data/train_jpg/"
    TEST_JPG_FOLDER = "./data/test_jpg/"
