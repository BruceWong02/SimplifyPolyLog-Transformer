# version: 4.1.1.0

import Transformer as tf

# from TokenToExpr import tokens_to_expr_ExprWay_w_check
from TokenToExpr import tokens_to_expr_ExprWay_w_check_and_timelimit as tokens_to_expr_ExprWay_w_check

# from Functions.BinaryFunctions.TokenToExpr import tokens_to_expr_ExprWay_w_check


import torch
from torch.utils.data import DataLoader
import sympy as sp
import pandas as pd
import csv
from alive_progress import alive_bar
import os
import logging
import time
import datetime

PRED_METHOD = "beam_search"
# PRED_METHOD = "greedy"
# PRED_METHOD = "top-p"

PREDICT_batch_size = 16
BEAM_size = 6
TOP_p = 0.8
NUM_beam_hyps_to_keep = 6
MAX_LENGTH = 512
predict_filename = "data/predict/input/Test_tokens_finial_a.csv"
checkpoint_path = "checkpoints/20240414/checkpoint10_20240414_06.pth.tar"
result_filename = "data/predict/output/Predict_result_finial_a.csv"
test_result_filename = "data/predict/output/test_result_expr_finial.csv"


def Pred4Test(
    model,
    predict_dataloader: DataLoader,
    SOS_idx,
    EOS_idx,
    iVocab,
    result_filename,
    pred_method='beam_search', # 'greedy' or 'beam_search'
    beam_size=1,
    top_p=0.9,
    length_penalty=0.75,
    num_beam_hyps_to_keep=1,
):
    """
    Generate predicted functions and save them
    """

    y = sp.symbols("y")
    z = sp.symbols("z")

    model.eval()

    if pred_method == 'beam_search':
        validation_result = {f"beam {i}": [] for i in range(num_beam_hyps_to_keep)}
    else:
        validation_result = {"beam 0": []}

    with open(result_filename, "w", newline="") as result_csvfile:
        result_writer = csv.writer(result_csvfile)
        result_writer.writerow(["predict", "predict_mma"])

        with alive_bar(len(predict_dataloader), title="Predicting") as bar:
            with torch.no_grad():
                # counter = 0
                for batch in predict_dataloader:
                    # src and tgt shape: (batch size, sequence_length)
                    src = batch["src"].to(tf.device)
                    tgt_sp = batch["tgt_sp"]

                    # Predict
                    if pred_method == 'greedy':
                        num_beam_hyps_to_keep = 1
                        pred_idx_tokens_batch = model.Generate_greedy(
                            src, 
                            SOS_idx, 
                            EOS_idx,
                            max_length=MAX_LENGTH,
                        ).tolist()
                        # list(sequence_length)
                    elif pred_method == 'top-p':
                        num_beam_hyps_to_keep = 1
                        pred_idx_tokens_batch = model.Generate_topp(
                            src,
                            SOS_idx,
                            EOS_idx,
                            top_p=top_p,
                            max_length=MAX_LENGTH,
                        ).tolist()
                    else:
                        pred_idx_tokens_batch = model.Generate_beam(
                            src,
                            SOS_idx,
                            EOS_idx,
                            num_beams=beam_size,  # beam size
                            length_penalty=length_penalty,
                            num_beam_hyps_to_keep=num_beam_hyps_to_keep,
                            max_length=MAX_LENGTH,
                        ).tolist()

                    # Convert into expressions
                    for id, pred_idx_tokens in enumerate(pred_idx_tokens_batch):
                        # check if is in a new batch id
                        beam_id = id % num_beam_hyps_to_keep
                        if 0 == beam_id:
                            batch_id = id // num_beam_hyps_to_keep
                            tgt_sp_expr = tgt_sp[batch_id]  # string
                            tgt_sp_expr = sp.sympify(tgt_sp_expr)

                        # idx -> string tokens
                        tokens = [
                            iVocab[idx]
                            for idx in pred_idx_tokens
                            if idx != model.tgt_pad_idx
                        ]
                        tokens = tokens[1:-1]

                        # string tokens -> Mma syntax
                        try:
                            # mathematica_expr = tokens_to_mma_ExprWay_w_check(tokens)
                            expression = tokens_to_expr_ExprWay_w_check(tokens)
                            mathematica_expr = sp.mathematica_code(expression)
                        except (IndexError, ValueError, AssertionError):
                            # return y when error occurred
                            expression = y
                            mathematica_expr = "y"
                        except (TimeoutError):
                            # return z when error occurred
                            expression = z
                            mathematica_expr = "z"

                        # Validate
                        if 0 == sp.simplify(tgt_sp_expr - expression):
                            validation_result[f"beam {beam_id}"].append(1)
                        else:
                            validation_result[f"beam {beam_id}"].append(0)

                        # Save
                        result_writer.writerow([expression, mathematica_expr])

                    # Update progress bar
                    # counter += 1
                    bar()

    df = pd.DataFrame(validation_result)
    # save test result
    df.to_csv(test_result_filename, index=False, encoding="utf-8")
    # calculate accuracy
    accuracies = []
    num_data = len(df)
    assert num_data == len(predict_dataloader.dataset), "add data error"
    for i in range(len(df.columns)):
        count_ones = df.iloc[:, : i + 1].sum().sum()  # count number of 1
        accuracy = count_ones / num_data
        accuracy = round(accuracy, 4)
        accuracies.append(accuracy)
        print(f"accuracy of head {i+1} beams: {accuracy}")

    return accuracies


Vocab = tf.Vocab
# InverseVocab = {str(value): key for key, value in Vocab.items()}
InverseVocab = {value: key for key, value in Vocab.items()}


# model
model = tf.model


# Load data
predict_dataset = tf.MyDataset(
    predict_filename, Vocab, if_add_tgt_mma=True
)  # (src, tgt, tgt_mma)
predict_dataloader = DataLoader(
    predict_dataset,
    batch_size=PREDICT_batch_size,
    shuffle=False,
    collate_fn=predict_dataset.pad_collate,
)

# load checkpoint
checkpoint = torch.load(checkpoint_path, map_location=torch.device(model.device))
model.Load_checkpoint(checkpoint)


# -------------------------------------------------------
# Start Predicting
start_time = time.time()

accus = Pred4Test(
    model,
    predict_dataloader,
    tf.sos_idx,
    tf.eos_idx,
    InverseVocab,
    result_filename,
    pred_method=PRED_METHOD,
    beam_size=BEAM_size,
    top_p=TOP_p,
    length_penalty=0.6,
    num_beam_hyps_to_keep=NUM_beam_hyps_to_keep,
)
print("Predict4Test finished.")


# logging
if not os.path.exists("runs/logs"):
    os.makedirs("runs/logs")
logging.basicConfig(filename="runs/logs/Predict.log", level=logging.INFO)

end_time = time.time()
runtime = end_time - start_time
now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

logging.info("[{}] Total time taken: {:.4f} seconds".format(timestamp, runtime))
logging.info(
    "\tpredicting dataset filename: {}, predict dataset length: {}".format(
        predict_filename, len(predict_dataset)
    )
)
logging.info("\tcheckpoint: {}".format(checkpoint_path))
logging.info("\tpredict method: {}".format(PRED_METHOD))
logging.info(
    "\tpredict batch size: {}, beam size: {}".format(PREDICT_batch_size, BEAM_size)
)
train_loss_list = checkpoint["loss_list"]
logging.info(
    "\tepoch trained: {}, Last training loss: {}".format(
        checkpoint["epoch"], train_loss_list[-1]
    )
)
for n, accu in enumerate(accus):
    logging.info(f"\taccuracy of head {n+1} beams: {accu}")
