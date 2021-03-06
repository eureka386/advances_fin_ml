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
    df_types = {}

    # run bar 생성
    file_name = f'sp500_run_{_run_type}-bar'
    # _run_bars = run_bar.run(df, run_type=_run_type)
    # pickle.dump(_run_bars, open(f'{file_name}.pickle', 'wb'))
    _run_bars = pickle.load(open(f'{file_name}.pickle', 'rb'))
    df_types[_run_type] = _run_bars
    # 일일 변동성을 구한다.
    _close = _run_bars[['date_time', 'close']].set_index('date_time')
    daily_vol = get_daily_vol(_close['close'])

    # cusum filter -> 수익률 기준으로 임계치를 초과할 때마다 샘플을 추출한다.
    neg_events, pos_events = cusum(_run_bars['close'], threshold=0.05)
    
    # 샘플링된 데이터 인덱스 위치에 값을 넣는다.
    _run_bars['cusum_neg'] = _run_bars.iloc[neg_events]['close']
    _run_bars['cusum_pos'] = _run_bars.iloc[pos_events]['close']
    filtered_bars = pd.DatetimeIndex(_run_bars.iloc[neg_events + pos_events]['date_time'])
    

    # 날짜 필드를 인덱스 필드로 지정
    _run_bars = _run_bars.set_index('date_time')


    # t1 = run_bar.getVerticalBarriers(_run_bars['close'], filtered_bars, numDays = 1)
    # events = run_bar.getEvents(_run_bars['close'], tEvents=filtered_bars, ptSl=[1,1],  numThreads=1, trgt=daily_vol, minRet=0.01, side = None)

    

    # # 시각화 파라미터 지정
    # _run_bars[['close', 'cusum_neg', 'cusum_pos']].plot(
    #     ms=1,
    #     style=[
    #         '.-',               # close
    #         '.',                # cusum_neg
    #         '*',                # cusum_pos
    #         # '.-', '.-'
    #         ],
    #     color=[
    #         'black',            # close
    #         'blue',             # cusum_neg
    #         'red',
    #         # 'gray', 'gray'
    #         ],             # cusum_pos
    #     linewidth=0.1,
    #     rot=20                  # 텍스트 출력 시 기울기 지정
    # )
    # plt.savefig(f'{file_name}.png', dpi=800)
    return _run_bars
    
if __name__ == '__main__':
    test_main()


