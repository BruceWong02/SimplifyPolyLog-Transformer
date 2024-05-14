import pandas as pd
import time

import sympy as sp
from FuncGener_general import PolyLog_multinomial_generator, sympy_to_prefix_notation


def generate_funcs():
    ns_range = [0, 3]
    nt_range = [0, 3]
    n_scr_max = 10
    coefficient_range = [-2, 2]
    constant_range = [1, 8]
    max_len = 512

    # Generate datas
    generator = PolyLog_multinomial_generator(
        ns_range,
        nt_range,
        n_scr_max,
        coefficient_range,
        constant_range,
        max_len,
    )
    # generate sympy expressions
    simple_expression, scrambled_expression = generator.generate()
    scrambled_prefix_notation = sympy_to_prefix_notation(scrambled_expression)
    simple_prefix_notation = sympy_to_prefix_notation(simple_expression)
    scrambled_expression_mma = sp.mathematica_code(scrambled_expression)
    simple_expression_mma = sp.mathematica_code(simple_expression)

    return [
        scrambled_prefix_notation,
        simple_prefix_notation,
        scrambled_expression,
        simple_expression,
        scrambled_expression_mma,
        simple_expression_mma,
        generator.ns,
        generator.nt,
        generator.n_scr,
    ]


if __name__ == "__main__":
    start_time = time.time()

    filename = "data/predict/input/Test_tokens_finial_a.csv"
    # filename = "data/train/Train_tokens_finial_a.csv"
    df = pd.read_csv(filename)

    max_src_len = df["src"].str.split().apply(len).max()
    max_tgt_len = df["tgt"].str.split().apply(len).max()

    print(f"Max length of 'src': {max_src_len}")
    print(f"Max length of 'tgt': {max_tgt_len}")

    threshold = 512
    if max_src_len > threshold:
        large_rows = df[df["src"].str.split().apply(len) > threshold]
        large_rows_src = large_rows.index.tolist()
        print(f"Rows with 'src' length > {threshold}: {large_rows_src}")
        # Generate new data for rows with 'src' length > 512
        for row in large_rows_src:
            new_data = generate_funcs()
            df.loc[row] = new_data

        df.to_csv(filename, index=False)
        print(f"Updated file saved to {filename}")

    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time:.2f} seconds.")
    print("finished")
