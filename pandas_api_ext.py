import pandas as pd
import numpy as np

@pd.api.extensions.register_dataframe_accessor("ts")
class Functions:
    def __init__(self, pandas_obj):
        # self._validate(pandas_obj)
        self._obj = pandas_obj.sort_values("date")

    @staticmethod
    def _validate(obj):
        _required_columns = ["date","ticker"]
        for _col in _required_columns:
            if _col not in obj.columns:
                raise AttributeError(f"Must have '{_col}'.")

    def _add_cols(self, _delta_perc_cols):
        cols = _delta_perc_cols.columns
        self._obj[cols] = _delta_perc_cols
        return self._obj

    def create_delta_perc_vars(self, columns, lag=1, join=False, merge_date=False):
        _vars = np.array(columns)
        _lagged_cols = self.create_lagged_vars(columns, lag)
        _delta_perc_cols = (self._obj[columns] -_lagged_cols.values) / _lagged_cols.values * 100
        _delta_perc_cols.columns = np.char.add(f"delta{lag}_perc_" ,_vars)
        res = self._add_cols(_delta_perc_cols) if join else _delta_perc_cols
        if merge_date:
            res['date'] = self._obj['date']
        return res

    def create_lagged_vars(self, columns, lag=1, join=False, merge_date=False):
        _vars = np.array(columns)
        _lagged_cols = self._obj.groupby("ticker")[_vars].shift(lag)
        _lagged_cols.columns = np.char.add("lag_", _vars)
        res = self._add_cols(_lagged_cols) if join else _lagged_cols
        if merge_date:
            res['date'] = self._obj['date']
        return res

    def split(self, ratio=[3/4, 1/8, 1/8], drop_columns=[]):
        assert np.isclose(sum(ratio), 1), f"Ratios must add to one. Got {sum(ratio)}"
        
        splits = np.array(ratio)
        obs = len(self._obj) * splits
        cuts = np.ceil(np.cumsum(obs)).astype(int)
        frames = []
        prev=None
        for end in cuts:
            _df = self._obj.iloc[prev:end]
            frames.append(_df.drop(drop_columns,axis=1))
            prev = end
        assert sum([len(df) for df in frames]) == len(self._obj), f"Lost observations. Received '{sum([len(df) for df in frames])}', but got expected '{len(self._obj)}'"

        return frames