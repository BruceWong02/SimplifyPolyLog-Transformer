# version: 4.0.0.0

import main.Transformer as tf

from Functions.BinaryFunctions.TokenToExpr import tokens_to_expr_ExprWay_w_check
from Functions.BinaryFunctions.FuncGener_general import sympy_to_prefix_notation

import torch
import sympy as sp



PRED_METHOD = "beam_search"  # "greedy" or "beam_search or top-p"
# PRED_METHOD = "greedy"
# PRED_METHOD = "top-p"
MMA_SYNTAX = False
BEAM_size = 6
TOP_p = 0.9
NUM_beam_hyps_to_keep = 3
LENGTH_PENALTY = 0.6
MAX_LENGTH = 512

checkpoint_path = "checkpoints/20240414/checkpoint10_20240414_06.pth.tar"


def process_prefix_notation(src_user, vocab):
    src_tokens = src_user.split()
    # Then each batch in Dataloader has size: (batch_size, sequence_length)
    src = torch.tensor([vocab[token] for token in src_tokens])
    # Add <sos> and <eos>
    src = torch.cat((torch.tensor([vocab["<sos>"]]), src, torch.tensor([vocab["<eos>"]])))
    # pack into a batch
    src = src.unsqueeze(0)
    return src

def process_and_print(pred_idx_tokens, iVocab, mma_syntax=False):
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
        if mma_syntax:
            mathematica_expr = sp.mathematica_code(expression)
            print(f"Mathematica code: {mathematica_expr}")
        else:
            print(f"Expression: {expression}")
    except (IndexError, ValueError, AssertionError):
        print(f"Failed to predict.")

def Predict(
    src,
    model,
    SOS_idx,
    EOS_idx,
    iVocab,
    pred_method="beam_search",  # "greedy" or "beam_search"
    mma_syntax=False,
    beam_size=1,
    top_p=0.9,
    length_penalty=0.75,
    num_beam_hyps_to_keep=1,
):
    """
    Generate predicted functions and print them
    """

    model.eval()
    with torch.no_grad():
        # src shape: (batch size, sequence_length)
        src = src.to(tf.device)

        # Predict
        if pred_method == "greedy":
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
                # batch_size=len(src),
                num_beams=beam_size,  # beam size
                length_penalty=length_penalty,
                num_beam_hyps_to_keep=num_beam_hyps_to_keep,
            ).tolist()

        # Convert into expressions
        for _, pred_idx_tokens in enumerate(pred_idx_tokens_batch):
            process_and_print(pred_idx_tokens, iVocab, mma_syntax)


Vocab = tf.Vocab
# InverseVocab = {str(value): key for key, value in Vocab.items()}
InverseVocab = {value: key for key, value in Vocab.items()}

# model
model = tf.model

# load checkpoint
checkpoint = torch.load(checkpoint_path, map_location=torch.device(model.device))
model.Load_checkpoint(checkpoint)

# -------------------------------------------------------
# Start Predicting

for _ in range(100):
    src_user = input("Input sympy expression ('exit' to exit): ")
    if src_user == "exit":
        break
    src_user = sp.sympify(src_user)
    # check if the input is a valid expression
    if not isinstance(src_user, sp.Expr):
        print("Invalid expression.")
        continue
    src_user = sympy_to_prefix_notation(src_user)
    src_user = process_prefix_notation(src_user, Vocab)

    Predict(
        src_user,
        model,
        tf.sos_idx,
        tf.eos_idx,
        InverseVocab,
        pred_method=PRED_METHOD,
        mma_syntax=MMA_SYNTAX,
        beam_size=BEAM_size,
        top_p=TOP_p,
        length_penalty=LENGTH_PENALTY,
        num_beam_hyps_to_keep=NUM_beam_hyps_to_keep,
    )

