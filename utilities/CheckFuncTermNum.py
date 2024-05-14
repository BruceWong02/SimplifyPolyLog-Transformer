import pandas as pd
import sympy as sp
import time
from multiprocessing import Pool
import numpy as np


def parse_and_count(expr):
    a = sp.parse_expr(expr)
    return len(sp.Add.make_args(a))


def parallelize_dataframe(df, func, n_cores=40):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def apply_parse_and_count(df):
    df["count_src"] = df["src_sp"].apply(parse_and_count)
    df["count_tgt"] = df["tgt_sp"].apply(parse_and_count)

    return df


if __name__ == "__main__":
    start_time = time.time()

    filename = "data/predict/input/Test_tokens_finial.csv"
    df = pd.read_csv(filename)[["src_sp", "tgt_sp"]]

    df = parallelize_dataframe(df, apply_parse_and_count, n_cores=40)
    src_term_max = df["count_src"].max()
    tgt_term_max = df["count_tgt"].max()

    print(f"max number of src terms: {src_term_max}")
    print(f"max number of tgt terms: {tgt_term_max}")

    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time:.2f} seconds.")
    print("finished")

    print("done")
