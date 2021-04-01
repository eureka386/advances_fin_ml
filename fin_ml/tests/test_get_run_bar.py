import pickle
import os.path
from pathlib import PureWindowsPath

import pandas as pd
import matplotlib.pyplot as plt

from fin_ml.data_structures import run_bar
from fin_ml.utils.filters import cusum
from fin_ml.utils.volatility import get_daily_vol


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

    # run bar 생성
    file_name = f'sp500_run_{_run_type}-bar'
    _run_bars = run_bar.run(df, run_type=_run_type)
    pickle.dump(_run_bars, open(f'{file_name}.pickle', 'wb'))
    _run_bars = pickle.load(open(f'{file_name}.pickle', 'rb'))
    _run_bars = _run_bars.set_index('date_time')
    return _run_bars
    
if __name__ == '__main__':
    test_main()


