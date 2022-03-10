# Melanoma Classification
```
.
├── config.py
├── dataset
│   ├── dataset.py
│   ├── functions.py
│   └── hyperparameters.py
├── models
│   ├── efficientnet
│   │   ├── artifacts
│   │   ├── hyperparameters.py
│   │   └── model.py
│   └── resnet
│       ├── artifacts
│       ├── hyperparameters.py
│       └── model.py
├── requirements.txt
├── seed.py
├── train.py
└── validation_strategy.py
```

### Configuration

Parameters:

ReduceLROnPlateau(): Reduce learning rate when a metric has stopped improving. Here patience is set to 1, meaning that if 1 model doesn't improve, then the lr will decrease by a factor of 0.2.

- `BATCH_SIZE1`: 32
- `BATCH_SIZE2`: 16
- `DEVICE`: which hardware device to use. Use `cpu` for CPU and `cuda` for GPU.
- `EPOCHS`: the number of training iterations.
- `FOLDS`: how many folds to use on our K-fold cross-validation strategy.
- `LEARNING_RATE`: 0.0005
- `LR_FACTOR`: 0.4  # BY HOW MUCH THE LR IS DECREASING
- `LR_PATIENCE`: 1  # 1 MODEL NOT IMPROVING UNTIL LR IS DECREASING
- `NUM_WORKERS`: 8
- `OUTPUT_SIZE`: 1
- `PATIENCE`: Early Stopping Patience (how many epochs to wait with no improvement until it stops)
- `TTA`: Test Time Augmentation Rounds (creating multiple augmented copies of each image in the test set, having the model make a prediction for each, then returning an ensemble of those predictions)
- `WEIGHT_DECAY`: 0.0
