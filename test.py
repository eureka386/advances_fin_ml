import cProfile
import sys
import fin_ml
from fin_ml.tests import test_get_imbalance
from multiprocessing import Process
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    sys.path        # <-참조하게될 패키시 경로 순위 참고
    p_list = []
    for _run_type in ('dollar', 'volume', 'tick'):
        test_get_imbalance.test_main(_run_type)
    # cProfile.run("test_get_imbalance.test_main('dollar')")
    
    # # 멀티프로세스 진행 
    # for _run_type in ('dollar', 'volume', 'tick'):
    #     p = Process(target=test_get_imbalance.test_main, args=(_run_type, ))
    #     p.start()
    #     p_list.append(p)

    # for p in p_list:
    #     p.join()
