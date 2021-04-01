import cProfile
import sys
import fin_ml
from fin_ml.tests import test_get_run_bar, test_get_imbalance, test_get_labeling
from multiprocessing import Process
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    sys.path        # <-참조하게될 패키시 경로 순위 참고
    p_list = []
    _run_type = 'dollar'
    # sample_bar = test_get_imbalance.test_main(_run_type)        # 불균형 바를 구한다.
    sample_bar = test_get_run_bar.test_main(_run_type)          # 런 바를 구한다.
    labeled_data = test_get_labeling.test_main(data=sample_bar)   # label 데이터를 구한다.
    print(labeled_data.head(10))