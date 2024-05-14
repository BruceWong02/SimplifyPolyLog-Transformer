import pandas as pd
import sympy as sp

x = sp.symbols("x")

# file_name = "data/predict/output/v4_0_0_1/beam_search/Predict_result_finial_a_merged.csv"
# file_name = "data/predict/output/fixed-2/beam_search/Predict_result_finial_a_merged.csv"
file_name = "data/predict/output/long_length/beam_search/Predict_result_finial_a_merged.csv"

df = pd.read_csv(file_name)

# df['predict_mma'] = df['predict_mma'].apply(sp.simplify)
# df['tgt_mma'] = df['tgt_mma'].apply(sp.simplify)

# # compare the two columns
# df['equal'] = df['predict_mma'] == df['tgt_mma']

# # count the number of True
# print(df['equal'].value_counts())


# # ---------------------------------
# # check the accuracy of predict 0
# zero_tgt_mma = df.loc[df['tgt_mma'] == '0']

# result = zero_tgt_mma['predict_mma'].eq('0')

# false_count = result.value_counts().get(False, 0)
# print(false_count)

# print(len(result))


# ---------------------------------
# check the probability of predict non expression
result = df['predict_mma'].eq('y')
true_count = result.value_counts().get(True, 0)
print(true_count)

print(len(result))

# ---------------------------------
# check the probability of timeout
result = df['predict_mma'].eq('z')
true_count = result.value_counts().get(True, 0)
print(true_count)

print(len(result))
