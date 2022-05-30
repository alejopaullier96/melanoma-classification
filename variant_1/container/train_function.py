# System
import datetime
import gc
import numpy as np
import os, os.path
import time
from tqdm import tqdm

# Sklearn
from sklearn.metrics import accuracy_score, roc_auc_score

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Local utility code
from config import config, paths
from dataset.hyperparameters import hyperparameters as ds_hp
from models.efficientnet.hyperparameters import hyperparameters as ef_hp
from models.efficientnet.model import EfficientNetwork

def train_function(predictions, train_df, test_df, model, MelanomaDataset, folds, device, version='v1'):
    """
    This function iterates over folds. On each fold, the original train dataset is split into a new train subset and a
    validation dataset. Test dataset is static through fold iterations. For each fold it trains the model for the
    specified epochs. Hence, training as well as evaluation on validation dataset is performed at an epoch level.
    The amount of iterations is therefore FOLDS*EPOCHS. After training is complete, the model is evaluated on the
    validation dataset and the model artifacts are saved if the metric is improved. Finally, at the last of every fold
    iteration the evaluation metric is computed for the test dataset. Please note that model selection is performed
    with respect to the validation metric and that no retraining on the original train dataset is performed.

    :param predictions: predictions for the test set.
    :param model: model architecture. Check available models on the /models for more information.
    :param MelanomaDataset: a custom dataset for this variant. Check /dataset for more information.
    :param folds: folds for training and validation. Check create_folds() function for more information.
    :param version: model version. Each time we train a new model we must create a new version.
    :return oof: Out of Fold predictions. In each fold we predict the Validation set, in consequence, as validation
    sets are non overlapping we end up with predictions for the whole train set.
    :return predictions: predictions for the test set
    """
    # Creates a .txt file that will contain the logs
    f = open(f"logs/logs_{version}.txt", "w+")

    # Out of Fold Predictions
    oof = np.zeros(shape=(len(train_df), 1))

    # Iterate over folds
    for fold, (train_index, valid_index) in enumerate(folds):
        # Append to .txt
        with open(f"logs/logs_{version}.txt", 'a+') as f:
            print('-' * 10, 'Fold:', fold + 1, '-' * 10, file=f)
        print('-' * 10, 'Fold:', fold + 1, '-' * 10)

        # --- Create Instances ---
        # Best ROC score in this fold
        best_roc = None
        # Reset patience before every fold. Check ReadMe.md for more information.
        patience_f = config.PATIENCE

        # Initiate the model
        model = model

        # Create optimizer. Check ReadMe.md for more information.
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.LEARNING_RATE,
                                     weight_decay=config.WEIGHT_DECAY)

        # Create scheduler. Check ReadMe.md for more information.
        scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                      mode='max',
                                      patience=config.LR_PATIENCE,
                                      verbose=True,
                                      factor=config.LR_FACTOR)

        # Create Loss. Check ReadMe.md for more information.
        criterion = nn.BCEWithLogitsLoss()

        # --- Read in Data ---
        train_data = train_df.iloc[train_index].reset_index(drop=True)
        valid_data = train_df.iloc[valid_index].reset_index(drop=True)

        # Create Data instances. Check ReadMe.md for more information.
        train = MelanomaDataset(train_data,
                                vertical_flip=ds_hp.VERTICAL_FLIP,
                                horizontal_flip=ds_hp.HORIZONTAL_FLIP,
                                is_train=True, is_valid=False, is_test=False)
        valid = MelanomaDataset(valid_data,
                                vertical_flip=ds_hp.VERTICAL_FLIP,
                                horizontal_flip=ds_hp.HORIZONTAL_FLIP,
                                is_train=False, is_valid=True, is_test=False)
        # Read in test data | Remember! We're using data augmentation like we use for Train data.
        test = MelanomaDataset(test_df,
                               vertical_flip=ds_hp.VERTICAL_FLIP,
                               horizontal_flip=ds_hp.HORIZONTAL_FLIP,
                               is_train=False, is_valid=False, is_test=True)

        # Create Dataloaders
        train_loader = DataLoader(train,
                                  batch_size=config.BATCH_SIZE_TRAIN,
                                  shuffle=True)
        # shuffle=False! Otherwise function won't work!!!
        valid_loader = DataLoader(valid,
                                  batch_size=config.BATCH_SIZE_VALIDATION,
                                  shuffle=False)
        test_loader = DataLoader(test,
                                 batch_size=config.BATCH_SIZE_TEST,
                                 shuffle=False)

        # === EPOCHS ===
        epochs = config.EPOCHS
        for epoch in range(epochs):
            start_time = time.time()
            correct = 0
            train_losses = 0

            # === TRAIN ===
            # Sets the module in training mode.
            model.train()

            # === Iterate over batches ===
            with tqdm(train_loader, unit="train_batch") as tqdm_train_loader:
                for (images, csv_data), labels in tqdm_train_loader:
                    # Save them to device
                    images = torch.tensor(images, device=device, dtype=torch.float32)
                    csv_data = torch.tensor(csv_data, device=device, dtype=torch.float32)
                    labels = torch.tensor(labels, device=device, dtype=torch.float32)

                    # Clear gradients first; very important, usually done BEFORE prediction
                    optimizer.zero_grad()

                    # Log Probabilities & Backpropagation
                    out = model(images, csv_data)
                    loss = criterion(out, labels.unsqueeze(1))
                    loss.backward()
                    optimizer.step()

                    # --- Save information after this batch ---
                    # Save loss
                    train_losses += loss.item()
                    # From log probabilities to actual probabilities
                    train_preds = torch.round(torch.sigmoid(out))  # 0 and 1
                    # Number of correct predictions
                    correct += (train_preds.cpu() == labels.cpu().unsqueeze(1)).sum().item()
            # Compute Train Accuracy
            train_acc = correct / len(train_index)

            # === EVAL ===
            # Sets the model in evaluation mode
            model.eval()

            # Create matrix to store evaluation predictions (for accuracy)
            valid_preds = torch.zeros(size=(len(valid_index), 1), device=device, dtype=torch.float32)

            # Disables gradients (we need to be sure no optimization happens)
            with torch.no_grad():
                for k, ((images, csv_data), labels) in enumerate(tqdm(valid_loader, unit="valid_batch")):
                    images = torch.tensor(images, device=device, dtype=torch.float32)
                    csv_data = torch.tensor(csv_data, device=device, dtype=torch.float32)
                    labels = torch.tensor(labels, device=device, dtype=torch.float32)
                    out = model(images, csv_data)
                    pred = torch.sigmoid(out)
                    valid_preds[k * images.shape[0]: k * images.shape[0] + images.shape[0]] = pred

                # Compute accuracy
                valid_acc = accuracy_score(valid_data['target'].values,
                                           torch.round(valid_preds.cpu()))
                # Compute ROC
                valid_roc = roc_auc_score(valid_data['target'].values,
                                          valid_preds.cpu())

                # Compute time on Train + Eval
                duration = str(datetime.timedelta(seconds=time.time() - start_time))[:7]

                # PRINT INFO
                # Append to .txt file
                with open(f"logs/logs_{version}.txt", 'a+') as f:
                    print('{} | Epoch: {}/{} | Loss: {:.4} | Train Acc: {:.3} | Valid Acc: {:.3} | ROC: {:.3}'. \
                          format(duration, epoch + 1, epochs, train_losses, train_acc, valid_acc, valid_roc), file=f)
                # Print to console
                print('{} | Epoch: {}/{} | Loss: {:.4} | Train Acc: {:.3} | Valid Acc: {:.3} | ROC: {:.3}'. \
                      format(duration, epoch + 1, epochs, train_losses, train_acc, valid_acc, valid_roc))

                # === SAVE MODEL ===

                # Update scheduler (for learning_rate)
                scheduler.step(valid_roc)

                # Update best_roc
                if not best_roc:  # If best_roc = None
                    best_roc = valid_roc
                    torch.save(model.state_dict(),
                               f"saved_models/Fold{fold + 1}_Epoch{epoch + 1}_ValidAcc_{valid_acc:.3f}_ROC_{valid_roc:.3f}.pth")
                    continue

                if valid_roc > best_roc:
                    best_roc = valid_roc
                    # Reset patience (because we have improvement)
                    patience_f = config.PATIENCE
                    torch.save(model.state_dict(),
                               f"saved_models/Fold{fold + 1}_Epoch{epoch + 1}_ValidAcc_{valid_acc:.3f}_ROC_{valid_roc:.3f}.pth")
                else:
                    # Decrease patience (no improvement in ROC)
                    patience_f = patience_f - 1
                    if patience_f == 0:
                        with open(f"logs/logs_{version}.txt", 'a+') as f:
                            print('Early stopping (no improvement since 3 models) | Best ROC: {}'. \
                                  format(best_roc), file=f)
                        print('Early stopping (no improvement since 3 models) | Best ROC: {}'. \
                              format(best_roc))
                        break

        # === INFERENCE ===
        # Choose model with best_roc in this fold
        best_model_path = 'saved_models/' + [file for file in os.listdir('saved_models') if
                                             str(round(best_roc, 3)) in file and 'Fold' + str(fold + 1) in file][0]
        # Using best model from Epoch Train
        model = EfficientNetwork(output_size=config.OUTPUT_SIZE,
                                 no_columns=ef_hp.NO_COLUMNS,
                                 b4=False, b2=True).to(device)
        model.load_state_dict(torch.load(best_model_path))
        # Set the model in evaluation mode
        model.eval()

        with torch.no_grad():
            # --- EVAL ---
            # Predicting again on Validation data to get preds for OOF
            valid_preds = torch.zeros(size=(len(valid_index), 1), device=device, dtype=torch.float32)

            for k, ((images, csv_data), _) in enumerate(tqdm(valid_loader, unit="oof_batch")):
                images = torch.tensor(images, device=device, dtype=torch.float32)
                csv_data = torch.tensor(csv_data, device=device, dtype=torch.float32)

                out = model(images, csv_data)
                pred = torch.sigmoid(out)
                valid_preds[k * images.shape[0]: k * images.shape[0] + images.shape[0]] = pred

            # Save info to OOF
            oof[valid_index] = valid_preds.cpu().numpy()

            # --- TEST ---
            # Now (Finally) prediction for our TEST data
            for i in range(config.TTA):
                for k, (images, csv_data) in enumerate(tqdm(test_loader, unit=f"test_loader_TTA_{i}")):
                    images = torch.tensor(images, device=device, dtype=torch.float32)
                    csv_data = torch.tensor(csv_data, device=device, dtype=torch.float32)

                    out = model(images, csv_data)
                    # Convert to probablities
                    out = torch.sigmoid(out)

                    # ADDS! the prediction to the matrix we already created
                    predictions[k * images.shape[0]: k * images.shape[0] + images.shape[0]] += out

            # Divide Predictions by TTA (to average the results during TTA)
            predictions /= config.TTA

        # === CLEANING ===
        # Clear memory
        del train, valid, train_loader, valid_loader, images, labels
        # Garbage collector
        gc.collect()

    return oof, predictions
