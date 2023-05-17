import pandas as pd

from java_utils import compute_all_metrics

df = pd.read_csv("./dataset/Java/valid.csv")
ref_list = df['code'].tolist()

df = pd.read_csv("./result/TruduckenGen.csv", header=None)
TruduckenGen_list = df[0].tolist()

compute_all_metrics(TruduckenGen_list, ref_list)


# from tqdm import tqdm
# from java_compile import check_java_code
# df = pd.read_csv("dataset/Java/Pisces-java-test.csv")
# ids = df['id'].tolist()
# codes = df['Code'].tolist()
# datas = []
# for i in tqdm(range(len(ids))):
#     id = ids[i]
#     code = codes[i]
#     if check_java_code(code) == False:
#         print(id)
#         datas.append(id)
# df = pd.DataFrame(datas)
# df.to_csv("error.csv", index=False, header=False)
