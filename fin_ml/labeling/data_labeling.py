import pandas as pd
import numpy as np


def apply_pt_sl_on_t1(close, events, pt_sl):
    # events_ = events.loc[molecule]
    events_ = events
    out = events_[['t1']].copy(deep=True)
    profit_taking_multiple = pt_sl[0]
    stop_loss_multiple = pt_sl[1]
    # Profit taking 활성화
    # pt = Profit taking 배수 * 일일변동률
    if profit_taking_multiple > 0:
        profit_taking = profit_taking_multiple * events_['trgt']
    else:
        profit_taking = pd.Series(index=events.index)  # NaNs(빈 값 시리즈)

    # Stop loss active
    # sl = Stop loss 배수 * 일일변동률
    if stop_loss_multiple > 0:
        stop_loss = -stop_loss_multiple * events_['trgt']
    else:
        stop_loss = pd.Series(index=events.index)  # NaNs(빈 값 시리즈)
    
    # pt, sl 초기화
    out['pt'] = pd.Series(dtype=events.index.dtype)
    out['sl'] = pd.Series(dtype=events.index.dtype)
    
    # Get events
    for loc, vertical_barrier in events_['t1'].fillna(close.index[-1]).iteritems():
        closing_prices = close[loc: vertical_barrier]  # Path prices for a given trade
        cum_returns = (closing_prices / close[loc] - 1) * events_.at[loc, 'side']  # Path returns
        out.at[loc, 'sl'] = cum_returns[cum_returns < stop_loss[loc]].index.min()  # 최초 stop loss 날짜
        out.at[loc, 'pt'] = cum_returns[cum_returns > profit_taking[loc]].index.min()  # 최초 profit taking 날짜
    return out

def add_vertical_barrier(t_events, close, timedelta):
    # t_events + timedelta 와 가장 가까운 time index 값을 찾는다.
    nearest_index = close.index.searchsorted(t_events + timedelta)
    nearest_index = nearest_index[nearest_index < close.shape[0]]
    nearest_timestamp = close.index[nearest_index]
    return pd.Series(data=nearest_timestamp, index=t_events[:nearest_index.shape[0]])

def get_events(close, t_events, pt_sl, target, min_ret, num_threads, vertical_barrier_times=False, side=None):
    # t_events 날짜의 일일 변동성 설정 => target
    target = target.reindex(t_events)       # t_events날짜의 일일 변동성
    target = target[target > min_ret]       # 일일 변동성 필터(>min_ret)
    
    # Vertical Barrier(최대보유기간) 설정
    if vertical_barrier_times is False:
        vertical_barrier_times = pd.Series(pd.NaT, index=t_events, dtype=t_events.dtype)
    
    if side is None:
        side_ = pd.Series(1.0, index=target.index)      # 기본 1.0 값으로 세팅
        pt_sl_ = [pt_sl[0], pt_sl[1]]
    else:
        side_ = side.reindex(target.index)              # side(베팅 값)이 있는 경우 값을 입력
        pt_sl_ = pt_sl[:2]
    
    # Create a new df with [v_barrier, target, side] and drop rows that are NA in target
    events = pd.concat({'t1': vertical_barrier_times, 'trgt': target, 'side': side_}, axis=1)
    events = events.dropna(subset=['trgt'])

    # Apply Triple Barrier
    first_touch_dates = apply_pt_sl_on_t1(close=close, events=events, pt_sl=pt_sl_)
    for idx in events.index:
        events.at[idx, 't1'] = first_touch_dates.loc[idx, :].dropna().min()
    if side is None:
        events = events.drop('side', axis=1)
    
    # Add profit taking and stop loss multiples for vertical barrier calculations
    events['pt'] = pt_sl[0]
    events['sl'] = pt_sl[1]
    return events

def barrier_touched(out_df, events):
    store = []
    '''
    [out_df]
                                    ret      trgt
    2009-10-14 15:33:02.000000 -0.010239  0.013817
    2009-10-28 12:42:05.000001  0.018047  0.016389
    2009-11-09 15:49:45.000000  0.001342  0.018590
    2010-01-07 14:23:33.000003 -0.005319  0.014963
    2010-01-27 14:17:42.000000  0.007669  0.014890
    ...                              ...       ...
    2020-04-28 09:30:55.000000 -0.015748  0.031094
    2020-05-01 13:21:29.000000  0.017099  0.031826
    2020-05-18 16:00:00.000000  0.035839  0.030562
    2020-06-03 12:33:51.000000  0.013644  0.030437
    2020-06-11 12:54:01.000000  0.004858  0.029992
    '''
    for date_time, values in out_df.iterrows():
        ret = values['ret']
        target = values['trgt']
        pt_level_reached = ret > np.log(1 + target) * events.loc[date_time, 'pt']
        sl_level_reached = ret < -np.log(1 + target) * events.loc[date_time, 'sl']
        if ret > 0.0 and pt_level_reached:
            # 수익 중인 경우 타겟 값을 초과했는지 확인 => 터치한 경우 1
            store.append(1)
        elif ret < 0.0 and sl_level_reached:
            # 손실 중인 경우 타겟 값을 미달했는지 확인 => 터치한 경우 -1
            store.append(-1)
        else:
            # 그렇지 않은 경우 => 버티컬 배리어 터치 0
            store.append(0)

    # store은 -1, 0, 1 값의 연속으로 구성된다.
    out_df['bin'] = store
    return out_df

def get_bins(triple_barrier_events, close):
    # close 데이터를 트리플 배리어의 모든 시간(event time + t1 time) 정보 기준으로 재인덱싱한다.
    events_ = triple_barrier_events.dropna(subset=['t1'])       # t1이 NaN 인 값을 제거한다.
    idx1 = events_.index                        # 트리플 배리어 날짜
    t1_date = events_['t1'].array               # 버티컬 배리어 날짜
    all_dates = idx1.union(other=t1_date).drop_duplicates()     # 트리플 + 버티컬 날짜를 모두 합치고 중복을 제거한다.
    prices = close.reindex(all_dates, method='bfill')       # close를 all_dates 기준으로 인덱스를 다시 만든다.

    # out 데이터프레임 생성
    out_df = pd.DataFrame(index=events_.index)

    # log 수익률 계산(t1 - current)
    out_df['ret'] = np.log(prices.loc[events_['t1'].array].array) - np.log(prices.loc[events_.index])
    out_df['trgt'] = events_['trgt']        # 변동성

    # Meta labeling(side) 필드가 있는 경우 세팅된 배율만큼 ret 값 조정
    if 'side' in events_:
        out_df['ret'] = out_df['ret'] * events_['side']  # meta-labeling

    # 트리플 베리어 결과를 구한다(-1, 0, 1)
    out_df = barrier_touched(out_df, triple_barrier_events)

    # Meta labeling: label incorrect events with a 0
    if 'side' in events_:
        out_df.loc[out_df['ret'] <= 0, 'bin'] = 0

    # 로그 수익률을 일반 수익률로 변환
    out_df['ret'] = np.exp(out_df['ret']) - 1

    # Add the side to the output. This is useful for when a meta label model must be fit
    tb_cols = triple_barrier_events.columns
    if 'side' in tb_cols:
        out_df['side'] = triple_barrier_events['side']

    return out_df

def drop_labels(events, min_pct=.05):
    # Apply weights, drop labels with insufficient examples
    while True:
        df0 = events['bin'].value_counts(normalize=True)
        if df0.min() > min_pct or df0.shape[0] < 3:
            break
        print('dropped label: ', df0.idxmin(), df0.min())
        events = events[events['bin'] != df0.idxmin()]
    return events