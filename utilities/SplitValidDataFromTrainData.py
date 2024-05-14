import pandas as pd
import numpy as np

train_df = pd.read_csv("data/train/Train_tokens_finial_a.csv")
# random choose 20% data
validate_df = train_df.sample(frac=0.2)
validate_df.to_csv("data/train/Validate_tokens_finial_a.csv", index=False)

# delete data
train_df = train_df.drop(validate_df.index)

# update train data
train_df.to_csv("data/train/Train_tokens_finial_a.csv", index=False)