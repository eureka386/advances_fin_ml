import pickle
import os.path
from pathlib import PureWindowsPath

import pandas as pd
import matplotlib.pyplot as plt

from fin_ml.data_structures import imbalance_bar
from fin_ml.utils.filters import cusum
from fin_ml.tests import test_get_imbalance

def data_load(file_path):
    print(f'(*) loading data...."{file_path}"')
    df = pd.read_csv(file_path)
    print('(*) data loaded.')
    return df


def test_main(_run_type):
    # # 샘플 데이터 파일경로 정의
    file_path = PureWindowsPath(os.path.dirname(__file__))
    file_abs_path = os.path.join(file_path, '../data_samples/sp500_tick_compact.csv').replace("\\","/")

    # csv to dataframe
    df = data_load(file_abs_path)

    # imbalance bar 생성
    file_name = f'sp500_imbalance_{_run_type}-bar'
    _imbalance_bars = imbalance_bar.run(df, run_type=_run_type)
    pickle.dump(_imbalance_bars, open(f'{file_name}.pickle', 'wb'))
    _imbalance_bars = pickle.load(open(f'{file_name}.pickle', 'rb'))
    _imbalance_bars = _imbalance_bars.set_index('date_time')
    return _imbalance_bars
    
if __name__ == '__main__':
    test_main()
