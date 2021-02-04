import pickle
import os.path
from pathlib import PureWindowsPath

import pandas as pd
import matplotlib.pyplot as plt

from fin_ml.data_structures import imbalance_bar
from fin_ml.utils.filters import cusum
from fin_ml.utils.volatility import get_daily_vol


def data_load(file_path):
    print(f'(*) loading data...."{file_path}"')
    df = pd.read_csv(file_path)
    print('(*) data loaded.')
    return df


def test_main():
    # # 샘플 데이터 파일경로 정의
    file_path = PureWindowsPath(os.path.dirname(__file__))
    file_abs_path = os.path.join(file_path, '../data_samples/sp500_tick_compact100000.csv').replace("\\","/")

    # csv to dataframe
    df = data_load(file_abs_path)

    # imbalance bar 생성
    _imbalance_bars = imbalance_bar.run(df)
    # pickle.dump(_imbalance_bars, open('sp500_imbalance_bar.pickle', 'wb'))
    # _imbalance_bars = pickle.load(open('sp500_imbalance_bar.pickle', 'rb'))

    # 일일 변동성을 구한다.
    _close = _imbalance_bars[['date_time', 'close']].set_index('date_time')
    daily_vol = get_daily_vol(_close['close'])

    # cusum filter -> 수익률 기준으로 임계치를 초과할 때마다 샘플을 추출한다.
    neg_events, pos_events = cusum(_imbalance_bars['close'], threshold=0.05)

    # 샘플링된 데이터 인덱스 위치에 값을 넣는다.
    _imbalance_bars['cusum_neg'] = _imbalance_bars.iloc[neg_events]['close']
    _imbalance_bars['cusum_pos'] = _imbalance_bars.iloc[pos_events]['close']

    # 날짜 필드를 인덱스 필드로 지정
    _imbalance_bars = _imbalance_bars.set_index('date_time')

    # 시각화 파라미터 지정
    _imbalance_bars[['close', 'cusum_neg', 'cusum_pos']].plot(
        style=[
            '.-',               # close
            '.',                # cusum_neg
            '*'],               # cusum_pos
        color=[
            'black',            # close
            'blue',             # cusum_neg
            'red'],             # cusum_pos
        rot=20                  # 텍스트 출력 시 기울기 지정
    )
    plt.savefig('sp500_imbalance_bar.png')
    plt.show()

if __name__ == '__main__':
    test_main()


