import pandas as pd
import time


def split_multi_digit_tokens_a(token_list):
    """['Add', '1', '20', '3'] -> ['Add', '1', '2', '0', '3']"""

    token_list_new = []
    for i, token in enumerate(token_list):
        if token.isdigit() and int(token) > 9:
            token_new = list(token)
            token_list_new += token_new
        else:
            token_list_new.append(token)
    return token_list_new


def split_multi_digit_tokens_b(token_list):
    """
    ['Add', '253', 'x'] ->
    ['Add', '1', 'Add', '3', 'Add', 'Mul', '5', '10', 'Mul', '2', 'Pow', '10', '2', 'x']
    """

    ...


if __name__ == "__main__":
    start_time = time.time()

    split_multi_digit_tokens = split_multi_digit_tokens_a

    filename = "data/predict/input/Test_tokens_finial.csv"
    df = pd.read_csv(filename)

    df["src"] = df["src"].str.split().apply(split_multi_digit_tokens).str.join(" ")
    df["tgt"] = df["tgt"].str.split().apply(split_multi_digit_tokens).str.join(" ")

    new_filename = "data/predict/input/Test_tokens_finial_a.csv"
    df.to_csv(new_filename, index=False)
    print(f"Saved to {new_filename}")

    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time:.2f} seconds.")
    print("finished")
