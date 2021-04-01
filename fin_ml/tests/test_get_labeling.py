import pickle
import os.path
from pathlib import PureWindowsPath

import pandas as pd
import matplotlib.pyplot as plt

from fin_ml.utils.filters import cusum
from fin_ml.utils.volatility import get_daily_vol

from fin_ml.labeling import data_labeling as dl


def test_main(data:pd.DataFrame):
    # 일일 변동성을 구한다.
    daily_vol = get_daily_vol(close=data['close'], span=100)

    # CUSUM 필터를 구한다.
    _, _, t_events = cusum(data['close'], threshold=0.05)

    # 버티컬 배리어를 구한다.
    vertical_barriers = dl.add_vertical_barrier(t_events=t_events, close=data['close'], timedelta=pd.Timedelta(days=1))

    # 트리플 배리어를 구한다.
    triple_barrier_events = dl.get_events(
        close=data['close'], t_events=t_events, pt_sl=[1, 1], target=daily_vol, min_ret=0.005, 
        num_threads=3, vertical_barrier_times=vertical_barriers, side=None)

    # 레이블을 구한다.
    labeled_data = dl.get_bins(triple_barrier_events, data['close'])
    labeled_data = dl.drop_labels(labeled_data)
    return labeled_data

    
if __name__ == '__main__':
    test_main()


