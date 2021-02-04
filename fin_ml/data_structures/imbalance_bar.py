import pickle

import pandas as pd
import numpy as np
from fin_ml.utils.ewm import ewm_mean


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


def run(df:pd.DataFrame, expected_imbalance_window:int=100, num_prev_bars:int=3, expected_num_ticks:int=100, expected_num_ticks_min_max:list=[20, 200])->pd.DataFrame:
    """불균형바를 구한다
    df:                                   틱 데이터의 pandas.DataFrame 객체 입력
    expected_imbalance_window:            기대 불균형의 최대 윈도우 크기
    num_prev_bars:                        E[T]의 지수가중평균을 구할 때의 window 및 span 크기
    expected_num_ticks:
    expected_num_ticks_min_max:           제한을 두지 않을 경우는 [0, np.inf]로 설정
    """

    print('(*) imbalance bar를 생성합니다.')
    
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
    for _idx in df.index:
        tick_num += 1
        date_time = df.iloc[_idx]['date_time']
        price = df.iloc[_idx]['price']
        volume = df.iloc[_idx]['volume']

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
            prv_sign = _sign
        else:
            _sign = prv_sign

        signs.append(_sign)         # 불균형 바 list
        cum_theta += _sign          # tick_delta == 0인 경우는 기존 값 누적
        prv_price = price
        
        ############# 초기 기대 불균형 값 세팅 #################
        if not list_bars and expected_imbalance is None:
            expected_imbalance = _get_expected_imbalance(signs, expected_num_ticks, expected_imbalance_window)
        
        ############# bar 추출 #############
        if (np.abs(cum_theta) > expected_num_ticks * np.abs(expected_imbalance) if expected_imbalance is not None else False):
            #### bar 생성 ####
            bar_info = dict(date_time=date_time, tick_num=tick_num, open=price_open, high= price_high, low=price_low, close=price_close, 
            cum_vol=cum_volume, cum_dallar=cum_dollar)
            print(f'{tick_num}/{sample_size}\t{bar_info}')
            list_bars.append(bar_info)
            num_ticks_bar.append(cum_tick)

            #### 기존 값 초기화 ####
            expected_num_ticks = _get_exp_num_ticks(num_ticks_bar, num_prev_bars, expected_num_ticks_min_max)           # E[T]의 기대 크기
            expected_imbalance = _get_expected_imbalance(signs, expected_num_ticks, expected_imbalance_window)          # 기대 불균형

            # 바 추출 이후 초기화될 변수들
            price_open = price_close = None
            price_high, price_low = -np.inf, np.inf
            cum_theta = cum_tick = cum_dollar = cum_volume = 0
    
    df = pd.DataFrame(list_bars)
    df['date_time'] = pd.to_datetime(df['date_time'])
    return df

    
