import pickle
import progressbar
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from fin_ml.utils.ewm import ewm_mean
from fin_ml.utils.mp_pandas import mpPandasObj

MAX_OUTLIER_CHECK_SIZE = 20000

def _get_expected_imbalance(signs:list, expected_num_ticks:float, window:int=100, warm_up:bool=False):
    """기대 불균형 값을 구한다.

    window: 최대 윈도우 크기
    """

    if len(signs) < expected_num_ticks and warm_up:
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


def getEvents(close, tEvents, ptSl, trgt, minRet, numThreads=1, t1=False, side=None):
    
    '''
    tEvents : 트리플- 배리어 시드가 될 타임스탬프 값을 가진 pandas time index 
    
    pts1 : 음이 아닌 실수로 두 배리어의 넓이를 설정한다. 0은  호리존탈배리어가 없다는걸 의미한다.
    
    t1 : 버티컬 배리어의 타임스탬프를 가진 pandas series , False 면 버티컬 배리어를 지정않했다는 의미이다.
    
    trgt : 수익률의 절대값으로 표현된 목표의 pandas seires ??? 
    
    minRet : 트리플 베리어를 검색하기 위한 최소 목표 수익률
    
    numThreads : 함수에서 현재 동시에 사용되고 있는 스레드 수 
    
    '''
    # 1) 목표 구하기
    
    trgt = trgt.reindex(index = tEvents) # 이벤트 발생한 날짜 일별 변동성 추출 <- 수익률? 
    trgt = trgt[trgt > minRet] # 일별 변동성이 minret 보다 큰 경우 추출 
    
    # 2) get t1 (max holding period)
    
    # t1 은 버티컬 라인 시간 
    if t1 is False: t1 = pd.Series(pd.NaT, index=tEvents)  # 버티컬라인 입력안하면 NA 로 
    
    # 3) form events object, apply stop loss on t1
    
    side_ = pd.Series(1.,index=trgt.index)
    if side is None: # side 는 호리즌 라인 
        side_, ptSl_ = pd.Series(1.0, index=trgt.index), [ptSl[0], ptSl[0]]
    else:
        side_, ptSl_ = side.loc[trgt.index], ptSl[:2] # 호리즌 라인이 존재하면   위에서 존재하는 일변 변동성 날짜를 호리즌 라인에 넣음 
    
    
    events = pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis=1).dropna(subset=['trgt'])
    
    df0 = mpPandasObj(func=applyPtSlOnT1, pdObj=('molecule', events.index), numThreads=numThreads, close=close, events=events, ptSl=ptSl_)
    events['t1'] = df0.dropna(how='all').min(axis=1) # pd.min ignores NaN
    if side is None:
        events = events.drop('side', axis=1)

    # store for later
    events['pt'] = ptSl[0]
    events['sl'] = ptSl[1]

    return events


def applyPtSlOnT1(close, events, ptSl, molecule):    
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)

    if ptSl[0] > 0:
        pt = ptSl[0] * events_['trgt']
    else:
        pt = pd.Series(index=events.index) # NaNs

    if ptSl[1] > 0:
        sl = - ptSl[1] * events_['trgt']
    else:
        sl = pd.Series(index=events.index) # 'mo NaNs

    for loc, t1 in events_['t1'].fillna(close.index[-1]).iteritems():
        
        df0 = close[loc:t1] # path prices   loc :이벤트 시작위치 : 일별 변동성이 평균 변동성을 넘었을때  
                                            # t1 : 이벤트가 시작하고 하루 지난시점
         # df0  이벤트 발생 부터 하루후까지의 모든 dollar bar 데이터
        
        df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side'] # path returns # 모든 기간에 대한 수익률을 구함
        #  (loc :1 ) (loc :2) (loc :3) ... 
        out.loc[loc, 'sl'] = df0[df0<sl[loc]].index.min() # earliest stop loss # 수익률이 stop 지점에 도달한 것중 가장 가까운값
        out.loc[loc, 'pt'] = df0[df0>pt[loc]].index.min() # earliest profit take # 수익률이 profit 지점에 도달한 것중 가장 가까운값
        
    return out


def barrier_touched(out_df, events):

    store = []
    for date_time, values in out_df.iterrows():
        ret = values['ret']
        target = values['trgt']

        pt_level_reached = ret > target * events.loc[date_time, 'pt']
        sl_level_reached = ret < -target * events.loc[date_time, 'sl']

        if ret > 0.0 and pt_level_reached:
            # Top barrier reached
            store.append(1)
        elif ret < 0.0 and sl_level_reached:
            # Bottom barrier reached
            store.append(-1)
        else:
            # Vertical barrier reached
            store.append(0)

    # Save to 'bin' column and return
    out_df['bin'] = store
    return out_df


def getVerticalBarriers(close, tEvents, numDays):
    t1 = close.index.searchsorted(tEvents+pd.Timedelta(days=numDays))
    t1 = t1[t1 < close.shape[0]]
    t1 = pd.Series(close.index[t1], index=tEvents[:t1.shape[0]]) # NaNs at the end
    return t1


def getBins(events, close):

    # 1) prices aligned with events
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')
    
    # 2) create out object
    out = pd.DataFrame(index=events_.index)

    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1
    if 'side' in events_:
        out['ret'] *= events_['side']  # meta-labeling

    out['trgt'] = events_['trgt']
    out = barrier_touched(out, events)

    if 'side' in events_:
        out.loc[out['ret'] <= 0, 'bin'] = 0
        
    if 'side' in events_:
        out['side'] = events['side']
    return out


def run(df:pd.DataFrame, expected_imbalance_window:int=100, num_prev_bars:int=100,
        expected_num_ticks:int=100, expected_num_ticks_min_max:list=[80, 200],
        run_type:str='tick')->pd.DataFrame:
    """런 바를 구한다
    df:                                   틱 데이터의 pandas.DataFrame 객체 입력
    expected_imbalance_window:            기대 불균형의 최대 윈도우 크기
    num_prev_bars:                        E[T]의 지수가중평균을 구할 때의 window 및 span 크기
    expected_num_ticks:
    expected_num_ticks_min_max:           제한을 두지 않을 경우는 [0, np.inf]로 설정
    run_type:                             tick->틱 불균형바, volume->거래량 불균형바, dollar->달러(원) 불균형바
    """

    print(f'(*) {run_type} run bar를 생성합니다.')
    
    # 바 추출 타입 체크
    # assert run_type in ('tick', 'volume', 'dollar'), 'wrong run_type'
    _run_type = ('tick', 'volume', 'dollar').index(run_type)

    # 바 추출 이후 초기화 되지 않을 변수들
    signs_sell = []
    signs_buy = []
    list_bars = []
    num_ticks_bar = []
    tick_num = 0
    prv_sign = 0
    expected_imbalance_buy = expected_imbalance_sell = None
    buy_ticks_proportion = []
    
    # 바 추출 이후 초기화될 변수들
    price_open = price_close = prv_price = None
    exp_buy_ticks_proportion = None
    exp_sell_ticks_proportion = None
    price_high, price_low = -np.inf, np.inf
    cum_theta = cum_tick = cum_dollar = cum_volume = cum_theta_buy = cum_theta_sell = buy_tick_num = 0
    
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
        # 단위 시간 애 중복이 있는 경우 tick 순서대로 microsecond 단위로 유니크한 값을 기록하도록 함
        if prv_date_time == _date_time:
            same_time_idx += 1
            date_time += '.{:06d}'.format(same_time_idx)
        else:
            same_time_idx = 0
            date_time += '.{:06d}'.format(same_time_idx)
        prv_date_time = _date_time
        price = d[1]
        volume = d[2]
        dollar = price * volume
        ## make ohlc ##
        if price_open is None: price_open = price       # open
        if price > price_high: price_high = price       # high
        if price < price_low: price_low = price         # low
        price_close = price                             # close
        
        ## 누적 tick / dollar / volume
        cum_tick += 1
        cum_dollar += dollar
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
            imbalance = _sign * dollar
        
        
        if imbalance > 0:
            cum_theta_buy += imbalance
            buy_tick_num += 1
            signs_buy.append(imbalance)         # 매수 불균형 바 list
        elif imbalance < 0:
            _imbalance = abs(imbalance)
            cum_theta_sell += _imbalance
            signs_sell.append(_imbalance)         # 매도 불균형 바 list

        prv_price = price
        prv_sign = _sign

        ############# 초기 기대 불균형 값 세팅 #################
        if expected_imbalance_buy is None or expected_imbalance_sell is None:
            expected_imbalance_buy = _get_expected_imbalance(signs_buy, expected_num_ticks, expected_imbalance_window, warm_up=True)
            expected_imbalance_sell = _get_expected_imbalance(signs_sell, expected_num_ticks, expected_imbalance_window, warm_up=True)

            if expected_imbalance_buy is not None and expected_imbalance_sell is not None:
                exp_buy_ticks_proportion = buy_tick_num / cum_tick
                exp_sell_ticks_proportion = (1 - exp_buy_ticks_proportion)
        
        if exp_buy_ticks_proportion is None: max_proportion = None
        else:
            max_proportion = max(
                expected_imbalance_buy * exp_buy_ticks_proportion,
                expected_imbalance_sell * exp_sell_ticks_proportion
            )
        
        max_theta = max(cum_theta_buy, cum_theta_sell)

        ############# bar 추출 #############
        if max_proportion is not None and max_theta > expected_num_ticks * max_proportion:
            #### bar 생성 ####
            bar_info = dict(date_time=date_time, tick_num=tick_num, open=price_open, high= price_high, low=price_low, close=price_close, 
            cum_vol=cum_volume, cum_dallar=cum_dollar)

            # # 관측 값이 과도하게 크거나 작을 경우 이상치로 판단하여 무시함
            # if _run_type:
            #     if imbalance > 0 and is_outlier(signs_buy[:-1], signs_buy[-1]):
            #         _ = signs_buy.pop()          # 이미 입력된 이상치 제거
            #         continue
            #     elif imbalance < 0 and is_outlier(signs_sell[:-1], signs_sell[-1]):
            #         _ = signs_sell.pop()         # 이미 입력된 이상치 제거
            #         continue
             
            list_bars.append(bar_info)
            num_ticks_bar.append(cum_tick)
            buy_ticks_proportion.append(buy_tick_num / cum_tick)
            
            # 기대  buy ticks proportion based on formed bars
            exp_buy_ticks_proportion = ewm_mean(buy_ticks_proportion[-num_prev_bars:], num_prev_bars)[-1]
            exp_sell_ticks_proportion = (1 - exp_buy_ticks_proportion)

            #### 기존 값 초기화 ####
            expected_num_ticks = _get_exp_num_ticks(num_ticks_bar, num_prev_bars, expected_num_ticks_min_max)                   # E[T]의 기대 크기
            expected_imbalance_buy = _get_expected_imbalance(signs_buy, expected_num_ticks, expected_imbalance_window)          # 기대 불균형
            expected_imbalance_sell = _get_expected_imbalance(signs_sell, expected_num_ticks, expected_imbalance_window)        # 기대 불균형

            # 바 추출 이후 초기화될 변수들
            price_open = price_close = None
            price_high, price_low = -np.inf, np.inf
            cum_theta_buy = cum_theta_sell = cum_tick = cum_dollar = cum_volume = buy_tick_num = 0
    

    bar.finish()
    df = pd.DataFrame(list_bars)
    df['date_time'] = pd.to_datetime(df['date_time'])
    return df

    
