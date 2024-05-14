"""
用Pandas加载data并检测有哪些重复的数据。返回数据所在行数。
"""

import pandas as pd

# df = pd.read_csv("test_num.csv")
df = pd.read_csv("data/predict/input/Test_tokens_finial.csv")


def find_duplicates_in_string_columns(df, column1, column2):
    indices = {}
    for index, row in df.iterrows():
        value_tuple = (row[column1], row[column2])
        if value_tuple in indices:
            indices[value_tuple].append(index)
        else:
            indices[value_tuple] = [index]

    return [v for v in indices.values() if len(v) > 1]


duplicate_indices_string_columns = find_duplicates_in_string_columns(df, "src", "tgt")
length = len(duplicate_indices_string_columns)

print(length)

print(
    duplicate_indices_string_columns[:4]
    if length > 4
    else duplicate_indices_string_columns
)
