import pickle
import progressbar
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from fin_ml.utils.ewm import ewm_mean

MAX_OUTLIER_CHECK_SIZE = 20000

def _get_expected_imbalance(signs:list, expected_num_ticks:float, window:int=100):
    """기대 불균형 값을 구한다.

    window: 최대 윈도우 크기
    """

    if len(signs) < expected_num_ticks:
        return
    else:
        ewma_window = min(len(signs), window)
        return ewm_mean(signs[-ewma_window:], span=ewma_window)[-1]


def _get_exp_num_ticks(num_ticks_bar:list, num_prev_bars:int, min_max:list):
    """E[T]를 구한다.
    min_max: 바 샘플 수렴 제어 목적 (참고:https://github.com/hudson-and-thames/mlfinlab/blob/master/mlfinlab/data_structures/imbalance_data_structures.py)
    """
    assert len(min_max) == 2 and min_max[0] < min_max[1], 'value error'     # validation check
    _min = min_max[0]; _max=min_max[1]
    exp_num_ticks = ewm_mean(data=num_ticks_bar[-num_prev_bars:], span=num_prev_bars)[-1]
    return min(max(exp_num_ticks, _min), _max)

def is_outlier(past_value, check_value):
    if len(past_value) > MAX_OUTLIER_CHECK_SIZE:            # tick 사이즈가 커질수록 속도 저하되기 때문에 통계 최대치 지정
        q1, q3 = np.percentile(past_value[-MAX_OUTLIER_CHECK_SIZE:], [25, 75]) 
    else:
        q1, q3 = np.percentile(past_value, [25, 75]) 
    iqr = q3 - q1
    _size = iqr * 100
    lower_bound = q1 - _size
    upper_bound = q3 + _size
    if not lower_bound < check_value < upper_bound:
        # print(f"lower_bound:{lower_bound}, check_value:{check_value} , upper_bound:{upper_bound}")
        return True
    return False


def run(df:pd.DataFrame, expected_imbalance_window:int=20, num_prev_bars:int=3,
        expected_num_ticks:int=20, expected_num_ticks_min_max:list=[10, 30],
        run_type:str='tick')->pd.DataFrame:
    """불균형바를 구한다
    df:                                   틱 데이터의 pandas.DataFrame 객체 입력
    expected_imbalance_window:            기대 불균형의 최대 윈도우 크기
    num_prev_bars:                        E[T]의 지수가중평균을 구할 때의 window 및 span 크기
    expected_num_ticks:
    expected_num_ticks_min_max:           제한을 두지 않을 경우는 [0, np.inf]로 설정
    run_type:                             tick->틱 불균형바, volume->거래량 불균형바, dollar->달러(원) 불균형바
    """

    print(f'(*) {run_type} imbalance bar를 생성합니다.')
    
    # 바 추출 타입 체크
    _run_type = ('tick', 'volume', 'dollar').index(run_type)

    # 바 추출 이후 초기화 되지 않을 변수들
    signs = []
    list_bars = []
    num_ticks_bar = []
    tick_num = 0
    prv_sign = 0
    expected_imbalance = None
    
    # 바 추출 이후 초기화될 변수들
    price_open = price_close = prv_price = None
    price_high, price_low = -np.inf, np.inf
    cum_theta = cum_tick = cum_dollar = cum_volume = 0
    
    sample_size = len(df)
    data = df.values
    data_len = len(data)

    # 진행률 체크용 progress 바 생성
    bar = progressbar.ProgressBar(maxval=data_len, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    
    same_time_idx = 0
    prv_date_time = None
    for d in data:
        tick_num += 1
        bar.update(tick_num)
        date_time = _date_time = d[0]
        # 해상도가 초단위이기 때문에 추후 일일 변동성 구할 때 중복된 값이 추출될 수 있음
        # 단위 시간 내 중복이 있는 경우 tick 순서대로 microsecond 단위로 유니크한 값을 기록하도록 함
        if prv_date_time == _date_time:
            same_time_idx += 1
            date_time += '.{:06d}'.format(same_time_idx)
        else:
            same_time_idx = 0
            date_time += '.{:06d}'.format(same_time_idx)
        prv_date_time = _date_time
        price = d[1]
        volume = d[2]
        ## make ohlc ##
        if price_open is None: price_open = price       # open
        if price > price_high: price_high = price       # high
        if price < price_low: price_low = price         # low
        price_close = price                             # close
        
        ## 누적 tick / dollar / volume
        cum_tick += 1
        cum_dollar += price
        cum_volume += volume

        ############## tick_delta 초기값 설정 #################
        if prv_price is not None: tick_delta = price - prv_price    # 이전 값이 있는 경우 delta 구함
        else: tick_delta = 0                                        # None인 경우 0으로 세팅

        ############## imbalance 계산 ####################
        if tick_delta != 0:
            _sign = 0
            if tick_delta > 0: _sign = 1
            elif tick_delta < 0: _sign = -1
        else:
            _sign = prv_sign

        if _run_type == 0:       # tick type
            imbalance = _sign
        elif _run_type == 1:     # volume type
            imbalance = _sign * volume
        elif _run_type == 2:     # dollar type
            imbalance = _sign * volume * price

        signs.append(imbalance)         # 불균형 바 list
        cum_theta += imbalance          # tick_delta == 0인 경우는 기존 값 누적
        prv_price = price
        prv_sign = _sign
        ############# 초기 기대 불균형 값 세팅 #################
        if not list_bars and expected_imbalance is None:
            expected_imbalance = _get_expected_imbalance(signs, expected_num_ticks, expected_imbalance_window)
        
        ############# bar 추출 #############
        if expected_imbalance is not None:
            observed_value = expected_num_ticks * np.abs(expected_imbalance)
            if (np.abs(cum_theta) > observed_value):
                #### bar 생성 ####
                bar_info = dict(date_time=date_time, tick_num=tick_num, open=price_open, high= price_high, low=price_low, close=price_close, 
                cum_vol=cum_volume, cum_dallar=cum_dollar, threshold=observed_value, observed_value=np.abs(cum_theta))
                # print(f'{tick_num}/{sample_size}\t{bar_info}')

                # 관측 값이 과도하게 크거나 작을 경우 이상치로 판단하여 무시함
                if _run_type and is_outlier(signs[:-1], imbalance):
                    _ = signs.pop()         # 입력된 이상치 제거
                    continue
                list_bars.append(bar_info)
                num_ticks_bar.append(cum_tick)

                #### 기존 값 초기화 ####
                expected_num_ticks = _get_exp_num_ticks(num_ticks_bar, num_prev_bars, expected_num_ticks_min_max)           # E[T]의 기대 크기
                expected_imbalance = _get_expected_imbalance(signs, expected_num_ticks, expected_imbalance_window)          # 기대 불균형

                # 바 추출 이후 초기화될 변수들
                price_open = price_close = None
                price_high, price_low = -np.inf, np.inf
                cum_theta = cum_tick = cum_dollar = cum_volume = 0
    

    bar.finish()
    df = pd.DataFrame(list_bars)
    df['date_time'] = pd.to_datetime(df['date_time'])
    return df

    
