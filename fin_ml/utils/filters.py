import numpy as np
import pandas as pd


def cusum(g_raw:pd.Series, threshold:float):
    neg_events = []
    pos_events = []
    t_events = []
    s_pos = s_neg = 0

    # DataFrame 변환
    g_raw = pd.DataFrame(g_raw)
    g_raw.columns = ['price']
    # log 리턴 변환
    g_raw['log_ret'] = g_raw.price.apply(np.log).diff()
    g_raw['threshold'] = threshold
    g_raw = g_raw.iloc[1:]          # 첫 번째 값 날림

    # Get event time stamps for the entire series
    for tup in g_raw.itertuples():
        thresh = tup.threshold
        pos = float(s_pos + tup.log_ret)
        neg = float(s_neg + tup.log_ret)
        s_pos = max(0.0, pos)           # 누적된 값이 음수가 되는 것을 방지한다.
        s_neg = min(0.0, neg)           # 누적된 값이 양수가 되는 것을 방지한다.
        if s_neg < -thresh:
            s_neg = 0
            neg_events.append(tup.Index)
            t_events.append(tup.Index)
        elif s_pos > thresh:
            s_pos = 0
            pos_events.append(tup.Index)
            t_events.append(tup.Index)

    return neg_events, pos_events, pd.Series(t_events)
