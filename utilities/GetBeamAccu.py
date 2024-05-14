"""
calculate the accuracy of the beam search
"""


import pandas as pd

df_py = pd.read_csv('data/predict/output/v4_0_0_1/beam_search/test_result_expr_finial.csv')

# beam i = beam 0 + beam 1 + ... + beam i
for i in range(1, 6):
    df_py['beam ' + str(i)] = df_py['beam ' + str(i)] + df_py['beam ' + str(i - 1)]

df_py = df_py > 0

# count the number of True in each column
accuracy = df_py.sum()/len(df_py)
print(accuracy)

