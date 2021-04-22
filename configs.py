DATA_CONFIGS = {
    'fit_asset' : 'AAPL', # TODO 'all' --> train a model on all the assets
                          # otherwise specify the ticker of the asset TODO or a list of tickers
    'freq' : '60min',   # '1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly'
    'input_width' : 35, # observations per window
    'label_width' : 1,  
    'shift' : 1,
    'batch_size' : 1,
    'split': [2/3,1/6,1/6], # fit on ratio_split*len(data) of asset time series
}


MODEL_CONFIGS = {
    # w = window length (number of periods) is defined in `data_configs`
    'k' : 2, # dimension of time
    # n = number of features at each point in time (without time features). This is implied by the dataset
    # TODO allow specifying which columns are of interest as features?
    'q' : 5, # number of queries (columns in W)
    'h' : 10, # number of heads
    'z' : 25, # number of features to be extracted from the q*h results created by the attention heads
    'layers': [25,12,5,1],
    'activ': "relu"
}

TRAINING_CONFIGS = {
    # TODO put the optimizer and scheduler in here
    'max_epochs' : 1,
    'patience' : 2,
    'val_freq' : 1,
    'batches_per_update' : 1,
    'lr' : 1e-3
}

STORAGE_CONFIGS = {
    'figs_dir' : 'figures',
    'losses_dir' : 'metrics',
    'models_dir' : 'models',
    'config_history_dir' : 'config_history'
}