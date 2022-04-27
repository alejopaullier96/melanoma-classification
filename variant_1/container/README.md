# Melanoma Classification

Last update 2022-04-26.

### Introduction

This is variant 1 for Melanoma classification. A variant is a solution, starting point or MVP for classifying 
different melanomas. This variant uses this [notebook](https://www.kaggle.com/andradaolteanu/melanoma-competiton-aug-resnet-effnet-lb-0-91) as a starting point. 

This project intends to train and serve a Deep Learning computer vision model behind an API endpoint using HTTP 
requests, possibly as a Docker container.

### Docker (locally)

Build:
1. `cd` into the Dockerfile location. 
2. Build the Docker image from the Dockerfile using:

    `docker build -f Dockerfile -t image_name . `
    
    If you are running the `docker build` command on a Macbook with M1 chip this will raise an error. Try running the same 
    command with the `--platform linux/x86_64` flag instead:
    
    `docker build -f Dockerfile --platform linux/x86_64 -t image_name . `

To train the model run:

`docker run -ti image_name train`

### Docker (AWS)

1. Ensure your SageMaker EC2 instance role has the correct permissions. Under *Permissions and Encryption* you can see 
the role. Check that the role has `AmazonEC2ContainerRegistryFullAccess` and `AmazonSageMakerFullAccess` permissions.
2. Open a terminal and log in ECR running the command: 
`aws ecr get-login-password --region region | docker login --username AWS --password-stdin aws_account_id.dkr.ecr.region.amazonaws.com`
3. `cd` into the variant root.
4. Execute `bash build_and_push.sh docker-image-name`
5. Check if image is pulled in your SageMaker instance with: `docker image ls`. If image is not pulled, pull it with: 
   `docker pull image_URI`.
6. Run the docker container with: `docker run -ti image_URI /bin/bash`

### Data

This variant's data is store under a data folder with the following structure:
```
data
├── README.md
├── test.csv
├── test_jpg
    ├── image_ID.jpg
    └── ...
├── train.csv
└── train_jpg
    ├── image_ID.jpg
    └── ...
```

Files:
- `train_jpg/`: folder with `.jpg` images for training stage.
- `test_jpg/`: folder with `.jpg` images for test stage.
- `train.csv`: CSV file with image and patient metadata.
- `test.csv`: CSV file with image and patient metadata.

`train.csv` and `test.csv` attributes:

- `image_id`: unique identifier, points to filename of related DICOM image [str].
- `patient_id`: unique patient identifier [str].
- `sex`: the sex of the patient (when unknown, will be blank) [str].
- `age_approx`: approximate patient age at time of imaging [int].
- `anatom_site_general_challenge`: location of imaged site [str].
- `diagnosis`: detailed diagnosis information (train only) [str].
- `benign_malignant`: indicator of malignancy of imaged lesion [str].
- `target`: binarized version of the target variable [int].

### Structure

**Tree**: 
```
.
├── Dockerfile
├── README.md
├── config.py
├── dataset
│   ├── dataset.py
│   ├── functions.py
│   └── hyperparameters.py
├── logs
│   └── 
├── models
│   ├── efficientnet
│   │   ├── __init__.py
│   │   ├── hyperparameters.py
│   │   └── model.py
│   └── resnet
│       ├── hyperparameters.py
│       └── model.py
├── preprocessing
│   └── preprocess.py
├── requirements.txt
├── saved_models
│   └── 
├── train.py
├── utils.py
└── validation_strategy.py

```

**Folders:**
- `dataset`: contains the PyTorch Dataset class for the problem along and its hyperparameters.
- `logs`: contains logs from training stage.
- `models`: contains folders each one with its unique model and hyperparameters.
- `preprocessing`: these folders contains python scripts for pre-processing the tabular metadata that comes with the 
  images.
- `saved_models`: contains the best model weights which come from the training stage.

### Main files

- `config.py`: contains general configurations and hyperparameters for the training stage.
- `Dockerfile`: Dockerfile to build the docker image.
- `requirements.txt`: contains all necessary Python libraries which will be installed in the docker container.
- `train.py`: main function. This function trains the model and stores the best model's artifacts/weights. It also outputs the logs from the training process. 
- `utils.py`: some utility functions used in the training process.

### Configuration

The following are the configuration parameters' definitions:

- `BATCH_SIZE_TRAIN`: batch size for the train DataLoader.
- `BATCH_SIZE_VALIDATION`: batch size for the validation DataLoader.
- `BATCH_SIZE_TEST`: batch size for the test DataLoader.
- `DEVICE`: which hardware device to use. Use `cpu` for CPU and `cuda` for GPU.
- `EPOCHS`: the number of training iterations.
- `FOLDS`: how many folds to use on our K-fold cross-validation strategy.
- `LEARNING_RATE`: learning rate for the optimizer.
- `LR_FACTOR`: decreasing rate of the learning rate.
- `LR_PATIENCE`: -
- `NUM_WORKERS`: tells the data loader instance how many sub-processes to use for data loading.  
- `OUTPUT_SIZE`: size of the output cell. Since it is a binary classifier is set to 1.
- `PATIENCE`: Early Stopping Patience (how many epochs to wait with no improvement until it stops)
- `TTA`: Test Time Augmentation Rounds (creating multiple augmented copies of each image in the test set, having the model make a prediction for each, then returning an ensemble of those predictions)
- `WEIGHT_DECAY`: -

### Important module classes

**PyTorch**

Optimizers:
- [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)

Schedulers:
- [ReduceLROnPlateau](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#reducelronplateau)

Loss functions:
- [BCEWithLogitsLoss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)

Dataset:
- [Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)

DataLoader:
- [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)

**Scikit Learn**

Validation:
- [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)


### Future changes

- The Docker Image produced by the Dockerfile is huge (~6.51 GB). This must be reduced. PyTorch libraries are very large.
- Add some Flask script for the API.
