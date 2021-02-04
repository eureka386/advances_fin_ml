import numpy as np

#pd.ewm().mean()
def ewm_mean(data:list, span:int):
    """pandas.ewm의 adjust=True 옵션 방식 구현
    (참고 : https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html)
    
    data: [x_0, x_1, ..., x_t]
    span: Specify decay in terms of span, α=2/(span+1)
    """
    # data 초기화
    data_len = len(data)
    _ewma = []
    alpha = 2 / (span + 1)
    weight = 1
    ewma_old = data[0]      
    alpha_ratio = (1 - alpha)
    _ewma.append(ewma_old)
    # 윈도우 계산
    for i in range(1, data_len):
        weight += alpha_ratio**i
        ewma_old = ewma_old * alpha_ratio + data[i]
        _ewma.append(ewma_old / weight)

    return _ewma



#pd.ewm().std()
def ewm_var(data:list, span:int, std:bool=False):
    """코드 참고 : https://stackoverflow.com/questions/58809786/pandas-ewm-var-and-std
    """
    # Initialize 
    alpha = 2./(1 + span) # Alpha
    
    # Initialize variable
    varcalc=[]

    # Calculate exponential moving variance
    for i in range(0,len(data)):
        # Get window
        z = np.array(data[0:i+1])

        # Get weights: w
        n = len(z)
        w = (1-alpha)**np.arange(n-1, -1, -1) # This is reverse order to match Series order

        # Calculate exponential moving average
        ewma = np.sum(w * z) / np.sum(w)
        _base = (np.sum(w)**2 - np.sum(w**2))
        # if _base == 0:
        #     continue
        bias = np.sum(w)**2 / _base
        ewmvar = bias * np.sum(w * (z - ewma)**2) / np.sum(w)

        # select output type 
        if std:
            varcalc.append(ewmvar**0.5)
        else:
            varcalc.append(ewmvar)

    return varcalc

