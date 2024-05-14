import pandas as pd


pred_file = "data/predict/output/v4_0_0_1/beam_search/Predict_result_finial_a.csv"
tgt_file = "data/predict/input/Test_tokens_finial_a.csv"
# pred_single_beam_file = "data/predict/output/v4_0_0_0/Predict_result_finial_a_single_beam.csv"
output_file = "data/predict/output/v4_0_0_1/beam_search/Predict_result_finial_a_merged.csv"


df_pred = pd.read_csv(pred_file, encoding='utf-8')
df_tgt = pd.read_csv(tgt_file, encoding='utf-8')

# for beam search
df_pred_single_beam = df_pred.iloc[::6]
df_pred_single_beam.reset_index(drop=True, inplace=True)

# for greedy search
# df_pred_single_beam = df_pred

# df_pred_single_beam.to_csv(pred_single_beam_file, index=False, encoding='utf-8')

assert df_pred_single_beam.shape[0] == df_tgt.shape[0], "The number of predictions and targets are not equal!"

df_output = pd.concat([df_pred_single_beam['predict_mma'], df_tgt['tgt_mma']], axis=1)
df_output.to_csv(output_file, index=False, encoding='utf-8')
print("Merge finished! Output file: ", output_file)
