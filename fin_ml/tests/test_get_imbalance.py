import pickle
import os.path
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path, PureWindowsPath
from fin_ml.data_structures import imbalance_bar
from fin_ml.utils.filters import cusum


def data_load(file_path):
    print(f'(*) loading data...."{file_path}"')
    df = pd.read_csv(file_path)
    print('(*) data loaded.')
    return df


def test_main():
    # 샘플 데이터 파일경로 정의
    file_path = PureWindowsPath(os.path.dirname(__file__))
    file_abs_path = os.path.join(file_path, '../data_samples/sp500_tick_compact.csv').replace("\\","/")

    # csv to dataframe
    df = data_load(file_abs_path)

    # imbalance bar 생성
    imbalace_bar = imbalance_bar.run(df)
    # pickle.dump(imbalace_bar, open('sp500_imbalance_bar.pickle', 'wb'))
    # imbalace_bar = pickle.load(open('sp500_imbalance_bar.pickle', 'rb'))

    # cusum filter -> 수익률 기준으로 임계치를 초과할 때마다 샘플을 추출한다.
    neg_events, pos_events = cusum(imbalace_bar['close'], threshold=0.05)

    
    imbalace_bar['cusum_neg'] = imbalace_bar.iloc[neg_events]['close']
    imbalace_bar['cusum_pos'] = imbalace_bar.iloc[pos_events]['close']
    imbalace_bar = imbalace_bar.set_index('date_time')

    imbalace_bar[['close', 'cusum_neg', 'cusum_pos']].plot(style=['.-', '.', '*'], color=['black', 'blue', 'red'], rot=20)
    plt.savefig('sp500_imbalance_bar.png')
    plt.show()

if __name__ == '__main__':
    test_main()


