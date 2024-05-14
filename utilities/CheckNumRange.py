import csv
import pandas as pd


def check_num_in_csv(file_name):
    with open(file_name, "r") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header
        for row in csv_reader:
            for item in row:
                try:
                    if int(item) >= 10:
                        return True
                except ValueError:
                    continue
    return False


def check(pns):
    max_value = 0
    idx = None

    for i, pn in enumerate(pns):
        tokens = pn.split()
        for token in tokens:
            if token.isdigit() and int(token) > max_value:
                # return token, i
                max_value = int(token)
                idx = i

    # return None, None
    return max_value, idx


# df = pd.read_csv("data/predict/input/Test_tokens_finial.csv")
df = pd.read_csv("data/train/Train_tokens_finial.csv")

src_pns = df["src"]
tgt_pns = df["tgt"]

result_src, idx_src = check(src_pns)
result_tgt, idx_tgt = check(tgt_pns)

# if result_src is not None:
if result_src > 0:
    print(f"src has number greater than 20: {result_src}, idx: {idx_src}")
# if result_tgt is not None:
if result_tgt > 0:
    print(f"tgt has number greater than 20: {result_tgt}, idx: {idx_tgt}")

print("over")
