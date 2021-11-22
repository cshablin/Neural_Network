from typing import Dict

import dask
import pandas as pd
from dask.config import set

'''
              timestamp    id     name         x         y
    2000-01-01 00:00:00   967    Jerry -0.031348 -0.040633
    2000-01-01 00:00:01  1066  Michael -0.262136  0.307107
    2000-01-01 00:00:02   988    Wendy -0.526331  0.128641
    2000-01-01 00:00:03  1016   Yvonne  0.620456  0.767270
    2000-01-01 00:00:04   998   Ursula  0.684902 -0.463278
'''



class SomeService:
    def __init__(self):
        df = dask.datasets.timeseries()
        pd_df = df.compute()
        means = self.__get_mean_for_group(pd_df, 'name' ,'x')
        pd_df['mean'] = pd_df.apply(lambda x: means[x['name']], axis=1)
        self.df = pd_df

    def __get_mean_for_group(self, df : pd.DataFrame, gb_column: str, mean_column) -> Dict[str, float]:
        groups_by_name = df.groupby(gb_column)
        means = groups_by_name.mean()
        return means.to_dict()[mean_column]

    def get_top_10_rows(self, name: str) -> pd.DataFrame:
        result_df = self.df[self.df.name == name]
        result_df.sort_values(by=['x'], inplace=True, ascending=False)
        return result_df[:10]

