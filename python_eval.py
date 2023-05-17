import pandas as pd

from python_utils import compute_all_metrics

df = pd.read_csv("./dataset/Python/test.csv")
ref_list = df['code'].tolist()

df = pd.read_csv("./result/TruduckenGen_list.csv", header=None)
TruduckenGen_list = df[0].tolist()

compute_all_metrics(TruduckenGen_list, ref_list)
