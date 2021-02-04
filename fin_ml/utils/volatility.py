import pandas as pd
from fin_ml.utils.ewm import ewm_var


def get_daily_vol(close, span=100):    
    """일별 변동성을 계산한다
    """
    # 일별 거래량, 종가에 따라 재인덱싱
    df = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df = df[df > 0]
    df = (pd.Series(close.index[df - 1], index=close.index[close.shape[0] - df.shape[0]:]))

    df = close.loc[df.index] / close.loc[df.array].array - 1  # daily returns
    # df = ewm_var(df, span=span, std=True)
    df = df.ewm(span=span).std()
    return df

