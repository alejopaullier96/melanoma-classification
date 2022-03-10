# --- Read in Data ---
train_data = train_df.iloc[train_index].reset_index(drop=True)
valid_data = train_df.iloc[valid_index].reset_index(drop=True)

# Create Data instances
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

# Dataloaders
train_loader = DataLoader(train,
                          batch_size=config.BATCH_SIZE_TRAIN,
                          shuffle=True,
                          num_workers=config.NUM_WORKERS)
# shuffle=False! Otherwise function won't work!!!
valid_loader = DataLoader(valid,
                          batch_size=config.BATCH_SIZE_TEST,
                          shuffle=False,
                          num_workers=config.NUM_WORKERS)
test_loader = DataLoader(test,
                         batch_size=config.BATCH_SIZE_TEST,
                         shuffle=False,
                         num_workers=config.NUM_WORKERS)