import log
import logging
logger = logging.getLogger('root')

import tensorflow as tf

from Model import Model

from Data import fetch_asset_data, WindowGenerator

from tokens import DATA_PATH
from configs import DATA_CONFIGS, MODEL_CONFIGS, TRAINING_CONFIGS, STORAGE_CONFIGS

import numpy as np

import os
from os import path
import pickle

from tqdm import tqdm

## Disable GPU
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def get_window_generator():
    df = fetch_asset_data(DATA_CONFIGS['fit_asset'], DATA_CONFIGS['freq'], DATA_PATH)
    train_df, val_df, test_df = df.ts.split(DATA_CONFIGS['split'])
    wg = WindowGenerator(DATA_CONFIGS['input_width'], DATA_CONFIGS['label_width'], DATA_CONFIGS['shift'],
                         train_df, val_df, test_df,
                         'date', df.columns[:-1], df.columns[-2:-1], 1)
    return wg

def get_model_settings(wg):
    w = DATA_CONFIGS['input_width']
    k = MODEL_CONFIGS['k']
    n = len(wg.train_df.columns[:-1])
    q = MODEL_CONFIGS['q']
    h = MODEL_CONFIGS['h']
    z = MODEL_CONFIGS['z']
    layers = MODEL_CONFIGS['layers']
    activ = MODEL_CONFIGS['activ']
    return w, k, n, q, h, z, layers, activ


def init_model(wg):
    w, k, n, q, h, z, layers, activ = get_model_settings(wg)
    return Model(w, k, n, q, h, z, layers, activ)

def train_model(model, optimizer, wg, MAX_EPOCHS):
    PATIENCE, VAL_FREQ = TRAINING_CONFIGS['patience'], TRAINING_CONFIGS['val_freq']
    BATCHES_PER_UPDATE = TRAINING_CONFIGS['batches_per_update']

    assert PATIENCE > 0, f"Patience must be at least 1. Currently patience='{PATIENCE}'"

    TRAIN_LOSSES = []
    VAL_LOSSES = []

    batch_num = 0
    for epoch in range(1, MAX_EPOCHS+1):
        logger.debug(f"epoch={epoch}")
        # TODO time series weighted losses (longer warmup --> greater loss)
        with tqdm(total=len(wg.train), leave=True) as pbar:
            for epoch_batch in range(len(wg.train)):
                # TODO adjust the training loop... Account for batches per grad update
                # TODO this loop takes days to get through with any reasonable training set. Figure out how to reduce it
                # warm up
                x = model.frnn.init_hidden
                for i, ((I,t), (target, _)) in enumerate(wg.train):
                    if i < epoch_batch:
                        t = tf.expand_dims(t,-1)
                        _, x = model(I,t,x)
                    else:
                        break
                # train 
                with tf.GradientTape() as tape:
                    start_batch = epoch_batch
                    last_batch  = start_batch + BATCHES_PER_UPDATE - 1

                    loss = tf.zeros((1,1))
                    for i, ((I,t), (target, _)) in enumerate(wg.train):
                        # skip to current batch
                        if i < start_batch:
                            pass
                        elif start_batch <= i <= last_batch:
                            t = tf.expand_dims(t,-1)

                            yhat, x = model(I,t,x)
                            loss += tf.losses.MSE(yhat, target) / BATCHES_PER_UPDATE

                            batch_num += 1
                            epoch_batch += 1
                        else:
                            break
                    pbar.update(1)
                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))

                    loss = np.mean(loss).item()
                    pbar.set_description(f"Train Loss (Batch): {loss:.9}")
                    TRAIN_LOSSES.append((batch_num, loss))
        # validation
        with tqdm(total=len(wg.val), leave=True) as pbar:
            logger.debug("Validation")
            if epoch % VAL_FREQ == 0:
                LENGTH_VALIDATION_SET = len(wg.val)
                loss = tf.zeros((1,1))
                for i, ((I,t), (target, _)) in enumerate(wg.val):
                    t = tf.expand_dims(t,-1)

                    yhat = model(I, t, x)
                    loss += tf.losses.MSE(yhat, target) / LENGTH_VALIDATION_SET

                    pbar.update(1)
                loss = np.mean(loss).item()
                pbar.set_description(f"Validation Loss (Set): {loss:.9f}")
                VAL_LOSSES.append((batch_num, loss))
        # early stopping
        if len(VAL_LOSSES) >= PATIENCE+1:
            logger.debug(f"Patience. len(VAL_LOSSES)={len(VAL_LOSSES)}")
            if VAL_LOSSES[-1][-1] > VAL_LOSSES[-2][-1]:
                logger.debug(f"Loss went up. curr:{STORED_VAL_LOSSES[-1][-1]:.3f}; prev:{VAL_LOSSES[-2][-1]}")
                higher = True
                for i in range(2,PATIENCE+1):
                    if VAL_LOSSES[-i][-1] < VAL_LOSSES[-i-1][-1]:
                        higher = False
                        break
                if higher:
                    logger.debug(f"Validation loss has not decreased for '{PATIENCE}' epochs. Early stopping enforced on epoch='{epoch}'.")
                    break
        # checkpoint
        logger.debug("checkpoint")
        save(model, TRAIN_LOSSES, VAL_LOSSES)
    return TRAIN_LOSSES, VAL_LOSSES

def save(model, train_losses, val_losses):
    def save_model():
        # init directory
        MODEL_DIR = STORAGE_CONFIGS['models_dir']
        if not path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        # generate name
        unique_names = []
        for f in os.listdir(MODEL_DIR):
            ID = f[:2] # TODO make not hard coded for only 2 digit ids
            if ID not in unique_names: 
                unique_names.append(ID)
        _model_num = len(unique_names)
        MODEL_NAME = f"{_model_num:0=2d}"
        while os.path.exists(path.join(MODEL_DIR, f"{MODEL_NAME}.index")): # don't overwrite a previous model
            _model_num += 1
            MODEL_NAME = f"{_model_num:0=2d}"
        # save
        model.save_weights(path.join(MODEL_DIR, MODEL_NAME))
        return MODEL_NAME
    
    def save_losses():
        LOSSES_DIR = STORAGE_CONFIGS['losses_dir']
        name_obj_tups = [("train", train_losses),("val", val_losses)]
        _pickle_helper(LOSSES_DIR, name_obj_tups)
    
    def save_configs():
        SAVE_CONFIG_DIR = STORAGE_CONFIGS['config_history_dir']
        name_obj_tups = [("data", DATA_CONFIGS), ("model", MODEL_CONFIGS), ("training", TRAINING_CONFIGS)]
        _pickle_helper(SAVE_CONFIG_DIR, name_obj_tups)

    def _pickle_helper(DIR, NAME_OBJ_TUPS):
        if not path.exists(DIR):
            os.makedirs(DIR)

        for obj_name, obj in NAME_OBJ_TUPS:
            p = path.join(DIR, f"{MODEL_NAME}-{obj_name}.pickle")
            with open(p, 'wb') as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    MODEL_NAME = save_model()
    save_losses()
    save_configs()

def main():
    wg = get_window_generator()
    model = init_model(wg)
    optimizer = tf.keras.optimizers.Adam(learning_rate=TRAINING_CONFIGS['lr'])

    train_losses, val_losses = train_model(model, optimizer, wg, TRAINING_CONFIGS['max_epochs'])

    save(model, train_losses, val_losses)


if __name__ == "__main__":
    main()