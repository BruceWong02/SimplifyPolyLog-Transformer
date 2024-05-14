import pandas as pd
import sympy as sp
from alive_progress import alive_bar

from TokenToExpr import tokens_to_expr_ExprWay_w_check

# from Functions.BinaryFunctions.TokenToExpr import tokens_to_expr_ExprWay_w_check

predict_filename = "data/predict/input/Test_tokens_finial.csv"
result_filename = "data/predict/output/v3_2_0_0/Predict_result_finial.csv"
num_beam_hyps_to_keep = 6

df_pre = pd.read_csv(predict_filename)
tgt_pns = df_pre["tgt"]

# ["predict", "predict_mma"]
df_re = pd.read_csv(result_filename)
pred_sps = df_re["predict"]

validation_result = {f"beam {i}": [] for i in range(num_beam_hyps_to_keep)}

with alive_bar(len(tgt_pns), title="Validating") as bar:
    for id, pred_sp in enumerate(pred_sps):
        # check if is in a new batch id
        beam_id = id % num_beam_hyps_to_keep
        if 0 == beam_id:
            batch_id = id // num_beam_hyps_to_keep
            tgt_pn = tgt_pns[batch_id]  # string
            tgt_tokens = tgt_pn.split()
            tgt_sp = tokens_to_expr_ExprWay_w_check(tgt_tokens)
            bar()

        pred_sp = pred_sps[id]
        pred_sp = sp.sympify(pred_sp)

        if 0 == sp.simplify(tgt_sp - pred_sp):
            validation_result[f"beam {beam_id}"].append(1)
        else:
            validation_result[f"beam {beam_id}"].append(0)

df = pd.DataFrame(validation_result)
# calculate accuracy
num_data = len(df)
for i in range(len(df.columns)):
    count_ones = df.iloc[:, : i + 1].sum().sum()  # count number of 1
    accuracy = count_ones / num_data
    accuracy = round(accuracy, 4)
    print(f"accuracy of head {i+1} beams: {accuracy}")
