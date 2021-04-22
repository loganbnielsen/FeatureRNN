import log
import logging
logger = logging.getLogger('root')

import tensorflow as tf

import numpy as np
import pandas as pd
import pandas_api_ext

import os
from os import path

def fetch_asset_data(asset_ticker, asset_freq, data_dir):
    """
        return df
    """
    p = None
    for f in os.listdir(data_dir):
        desc = path.splitext(f)[0].split("-")
        tic, freq = desc[0], desc[-1]
        if tic == asset_ticker and freq == asset_freq:
            p = path.join(data_dir,f)
            break
    assert p, f"Unable to find file for ticker='{asset_ticker}' and freq='{asset_freq}'"
    df = pd.read_csv(p, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df['t'] = (df["date"] - df.iloc[0,0]).dt.total_seconds() + 1 # TODO adjust for frequency? (shouldn't matter in theory)
    price_columns = df.columns[df.columns.str.contains("open|high|low|close", case=False)].to_list()
    returns = df.ts.create_delta_perc_vars(price_columns)
    returns['date'] = df['t']
    return returns.dropna()

class WindowGenerator:
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df, time_column,
                 inputs_columns, label_columns, batch_size):
        self.batch_size = batch_size
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df


        # Work out the columns
        self.inputs_columns = inputs_columns
        self.label_columns = label_columns
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        self.time_column = time_column


    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        inputs = tf.stack(
            [inputs[:, :, self.column_indices[name]] for name in self.inputs_columns],
            axis=-1)
        labels = features[:, self.labels_slice, :]
        labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)

        inputs_time = features[:, self.input_slice, self.column_indices[self.time_column]]
        labels_time = features[:, self.labels_slice, self.column_indices[self.time_column]]
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        # inputs_time.set_shape([None, 1, None])
        # labels_time.set_shape([None, 1, None])

        return (inputs, inputs_time), (labels, labels_time)

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=self.batch_size)

        return ds.map(self.split_window)

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
    
