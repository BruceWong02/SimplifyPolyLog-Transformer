import pandas as pd


def check_if_number_after_number(token_list):
    """check if thie prefix notation looks like ['Add', '1', '20', 'Mul']"""

    next_token_list = token_list[1:] + [""]
    for token, next_token in zip(token_list, next_token_list):
        if token.isdigit() and next_token.isdigit():
            print(f"Found: {token} {next_token}")
            return True
    return False


if __name__ == "__main__":
    filename = "data/predict/input/Test_tokens_finial.csv"
    df = pd.read_csv(filename)

    for label in ["src", "tgt"]:
        for i in range(len(df)):
            pn = df.loc[i, label]
            tokens = pn.split()
            if check_if_number_after_number(tokens):
                print(f"found: {label} row {i}: {pn}")
                exit(0)

    print(f"No number after number in {filename}")