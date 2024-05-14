"""
This script is used to transform the data from csv to json format for Alpaca.

['src', 'tgt', src_mma', 'tgt_mma', 'ns', 'nt', 'n_scr']
  -->
[
  {
    "instruction": "用户指令（必填）",
    "input": "用户输入（选填）",
    "output": "模型回答（必填）",
    "system": "系统提示词（选填）",
    "history": [
      ["第一轮指令（选填）", "第一轮回答（选填）"],
      ["第二轮指令（选填）", "第二轮回答（选填）"]
    ]
  }
]
"""


import pandas as pd
import json
from tqdm import tqdm

# read the csv file
df = pd.read_csv('data/train/Train_tokens_finial.csv')

# initialize a list to store the data
alpaca_data = []

# iterate through the dataframe
# for index, row in df.iterrows():
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    item = {
        "instruction": row['src_mma'],
        "input": "",
        "output": row['tgt_mma'],
        "system": "Simplify the GPL function.",
    }
    alpaca_data.append(item)

# write the data to a json file
with open('polylog_train-alpaca.json', 'w', encoding='utf-8') as f:
    json.dump(alpaca_data, f, ensure_ascii=False, indent=2)