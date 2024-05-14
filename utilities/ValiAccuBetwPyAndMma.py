import pandas as pd

df_py = pd.read_csv('data/predict/output/v4_0_0_0/beam_search/test_result_expr_finial.csv')
df_mma = pd.read_csv('data/predict/output/v4_0_0_0/beam_search/sameRowsExpr.csv')

df_py = df_py['beam 0']
df_mma = df_mma['idx']
# the locate where df_py['beam 0'] == 1
df_py_success = df_py.loc[df_py == 1]

# the element in df_mma means the row that should be deleted in df_success
df_mma = df_mma - 1
df_py_success = df_py_success.drop(df_mma)

print(df_py_success)
