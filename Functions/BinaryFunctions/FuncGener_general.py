# !/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------
# @copyright (C), 2024, Jinxin Wang, All rights reserved.
# @File Name   : FuncGener_general.py
# @Author      : Jinxin Wang
# @Version     : 1.3.0
# @Date        : 2024-05-05 10:36:29
# @Description :
#
# Generate weight 2 classical polylog function expressions.
# Using indentities to make expressions more complicated.
# Tokenize expressions, check the length, combine them to prefix notations and save them.
# Convert the expressions to Mathematica code and save them
# Save some parameters.
# create log file and record date, time, runtime and number of pairs generated.
#
# Note:
# 1. We ignore the case that scrambled expression and simple expression are both 0
#   here in order to avoid generating this expression pair repeatedly. We can add this case manually.
# 2. The structure of prefix notation is binary tree here.
#
# the data in csv file is 2*prefix notation (str), 2*mathematica code (str), ns, nt, n_scr.
#
# In this file, we use multiprocessing to accelerate the generating,
#   and use multithreading to accelerate the merging.
# 
# Support generating long length data for long length test.
#
# ----------------------------------------------------------------

VERSION = "1.3.0"

import sympy as sp
import random
import csv
import logging
import time
import datetime
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor


# print("%"*40)
class PolyLog_multinomial_generator:
    def __init__(
        self,
        ns_range: list[int],
        nt_range: list[int],
        n_scr_max: int,
        coefficient_range: list[int],
        constant_range: list[int],
        max_len: int = 512,
        min_len: int = 0,
    ):
        # self.ns = 0  # number of distinct dilogarithm terms
        # self.nt = 0  # number of 0 terms
        # self.nums = nt + ns  # total number of terms
        # self.n_scr = 0  # total number of scrambles
        self.ns_range = ns_range  # range of ns
        self.nt_range = nt_range  # range of nt
        self.n_scr_max = n_scr_max  # maximum of n_scr range
        self.coefficient_range = coefficient_range  # range of coefficients of h_i(x)
        self.constant_range = constant_range  # range of integer constants c
        self.max_len = max_len
        self.min_len = min_len
        self.expression = None

        self.check_input()

    def check_range(self, range_values, range_name):
        assert (
            len(range_values) == 2
        ), f"{range_name} should be a list with length 2. (e.g. [min, max])"
        assert (
            range_values[0] <= range_values[1]
        ), f"the minimum of the {range_name} > the maximum"

    def check_input(self):
        self.check_range(self.ns_range, "ns range")
        self.check_range(self.nt_range, "nt range")
        self.check_range(self.coefficient_range, "coefficient range")
        self.check_range(self.constant_range, "constant range")

    def sample_para_n(self):
        ns_min = self.ns_range[0]
        ns_max = self.ns_range[1]
        self.ns = random.randint(ns_min, ns_max)  # number of distinct dilogarithm terms

        nt_min = self.nt_range[0]
        nt_max = self.nt_range[1]
        self.nt = random.randint(nt_min, nt_max)  # number of 0 terms

        self.nums = self.nt + self.ns  # total number of terms

        n_scr_min = self.nt
        n_scr_max = self.n_scr_max
        self.n_scr = random.randint(n_scr_min, n_scr_max)  # total number of scrambles

        assert (
            self.n_scr >= self.nt
        ), "(n_scr < nt) Total number of scrambles is less than the number of 0 terms. Each zero term need to be acted upon at least once."

    def sample_polynomial_function(self):
        """generate polynomial function while limit the degree to be at most 2."""
        x = sp.symbols("x")
        coeff_min = self.coefficient_range[0]
        coeff_max = self.coefficient_range[1]
        coeffs = [random.randint(coeff_min, coeff_max) for _ in range(3)]
        return sum(c * x**i for i, c in enumerate(coeffs))

    def sample_rational_function(self):
        """generate rational function except constant"""
        denominator = self.sample_polynomial_function()
        # avoid denominator being 0
        while 0 == denominator:
            denominator = self.sample_polynomial_function()

        fraction = self.sample_polynomial_function() / denominator
        fraction = sp.cancel(fraction)
        # note: simplify is much slower than cancel
        # fraction = sp.simplify(fraction)

        # avoid constant (we do not consider constant here)
        while fraction.is_constant():
            fraction = self.sample_polynomial_function() / denominator
            fraction = sp.cancel(fraction)
            # fraction = sp.simplify(fraction)

        # In case for bugs
        if fraction.is_constant():
            print(fraction)
            raise TypeError("The result of the rational function is a constant.")

        return fraction

    def distribute_scrambles_num(self):
        """
        Randomly distribute the number of indentities applied on which term.
        Making sure that each zero term would be acted upon by at least one identity.
        if scrambles number of the term chosen is on the top, then let the right one to be chosen.
        """
        total_scrambles_num = self.n_scr
        num_scrambles = self.ns
        num_zero_terms = self.nt
        num_terms = self.nums

        extra_scrambles = total_scrambles_num - num_zero_terms
        # incase
        need_regenerate = False
        if extra_scrambles < 0:
            raise ValueError("ERROR: the total scrambles is less than the zero terms.")
        elif total_scrambles_num > 3 * num_terms:
            # avoid one identity being applied twice
            need_regenerate = True
            return 0, need_regenerate

        scrambles_num_list = [0] * num_scrambles + [1] * num_zero_terms
        for _ in range(extra_scrambles):
            idx = random.randint(0, num_terms - 1)
            while 3 == scrambles_num_list[idx]:
                idx = (1 + idx) % num_terms
            scrambles_num_list[idx] += 1
        return scrambles_num_list, need_regenerate

    @staticmethod
    def identity(coeff, h_term, half_func_term, id):
        """Apply identities to terms"""
        # inversion
        coeff_new = -coeff
        if 0 == id:
            h_term_new = sp.cancel(1 / h_term)
            half_func_term_new = half_func_term + coeff * (
                -sp.polylog(2, h_term) - sp.polylog(2, h_term_new)
            )
        # reflection
        elif 1 == id:
            h_term_new = sp.cancel(1 - h_term)
            half_func_term_new = half_func_term + coeff * (
                -sp.polylog(2, h_term) - sp.polylog(2, h_term_new)
            )
        # duplication
        elif 2 == id:
            h_term_new = sp.cancel(-h_term)
            h_term_new_2 = sp.cancel(h_term**2)
            half_func_term_new = half_func_term + coeff * (
                -sp.polylog(2, h_term)
                - sp.polylog(2, h_term_new)
                + sp.Rational(1, 2) * sp.polylog(2, h_term_new_2)
            )
        else:
            raise ValueError("ERROR: identity index error")
        return coeff_new, h_term_new, half_func_term_new

    def check_length(self, scrambled_expr):
        can_pass = False
        scrambled_tokens = tokenize_sympy_expr(scrambled_expr)
        seq_len = len(scrambled_tokens)
        if seq_len <= self.max_len and seq_len >= self.min_len:
            # Expression length passed
            can_pass = True
        return can_pass

    def generate(self):
        """
        generate simple expression and scrambled expression.
        """

        # init
        simple_expr = None
        scrambled_expr = None

        for timer in range(30):
            # sample parameters ns, nt and n_scr
            self.sample_para_n()
            ns = self.ns
            nums = self.nums

            # Sample rational functions h_i(x) and g_i(x) --- [(g_i)s, (h_i)s]
            h_terms = [self.sample_rational_function() for _ in range(nums)]
            # Sample integer constants c
            const_min = self.constant_range[0]
            const_max = self.constant_range[1]
            constants = [random.randint(const_min, const_max) for _ in range(nums)]

            # Distribute scrambles amongst terms --- [ns..., nt...]
            scrambles_num_list, need_regenerate = self.distribute_scrambles_num()
            # avoid one identity being applied twice
            if need_regenerate:
                continue

            # [ns..., nt..., -nt...]
            func_terms = [
                constants[i] * sp.polylog(2, h_terms[i]) for i in range(nums)
            ] + [-constants[i] * sp.polylog(2, h_terms[i]) for i in range(ns, nums)]
            # combine terms of simple expression
            simple_expr = sum(term for term in func_terms[:ns])

            # apply identity
            for i in range(nums):
                coeff = constants[i]
                h_term = h_terms[i]
                func_term = func_terms[i]

                identities = [0, 1, 2]
                random.shuffle(identities)
                for j in range(scrambles_num_list[i]):
                    coeff, h_term, func_term = self.identity(
                        coeff, h_term, func_term, identities[j]
                    )
                func_terms[i] = func_term

            # combine terms of scrambled expression
            scrambled_expr = sum(term for term in func_terms)
            # ignore 0
            if 0 == scrambled_expr:
                continue

            # print_result(L_max, expression, simp_expression, n_scr, ns, nt, timer)
            can_pass = self.check_length(scrambled_expr)
            if can_pass:
                break

        if simple_expr is None or scrambled_expr is None:
            raise TimeoutError(f"Did not generate expressions. Looped {timer} times.")

        # simple_expr = sp.simplify(simple_expr)
        # scrambled_expr = sp.simplify(scrambled_expr)
        return simple_expr, scrambled_expr

    def print_result(self, scrambled_expr, simple_expr):
        tokens = tokenize_sympy_expr(scrambled_expr)
        seq_len = len(tokens)

        print(f"original simple expression f: {simple_expr}\n")
        print(f"finial scrambled expression F: {scrambled_expr}\n")
        print(f"scrambled expression tokens: {tokens}\n")
        print(f"length of sequence: {seq_len}")
        print(f"n_scr: {self.n_scr}\t n_s: {self.ns}\t n_t: {self.nt}")

        # print(f"loop times: {timer+1}")


def handle_binary_op(expr, op_name, convert_func):
    """transfer a binary operation into a tree node"""
    node_left = convert_func(expr.args[0])
    right_expr = expr.func(*expr.args[1:]) if len(expr.args) > 2 else expr.args[1]
    node_right = convert_func(right_expr)
    return [op_name] + node_left + node_right


def handle_int(num):
    """
    Separate negative signs, add positive signs.
    Also split the number into digits. (['23'] -> ['2', '3'])
    We don't handle 0 here.
    """
    num = int(num)
    num_abs = abs(num)
    num_token = str(num_abs)
    if 0 == num:
        return ["0"]
    else:
        sign_token = "+" if num > 0 else "-"
        return (
            [sign_token] + list(num_token) if num_abs > 9 else [sign_token, num_token]
        )


def tokenize_sympy_expr(expr) -> list[str]:
    """
    Convert sympy expression to token list of prefix notation directly with all elements as strings,
    where prefix notation has a binary tree structure.
    e.g.: x**3 + sp.polylog(2, (x - 10)) --> ['Add', 'Pow', 'x', '+', '3', 'polylog', '+', '2', 'Add', '-', '1', '0', 'x']
    """

    op_handlers = {
        sp.Add: lambda e: handle_binary_op(e, "Add", tokenize_sympy_expr),
        sp.Mul: lambda e: handle_binary_op(e, "Mul", tokenize_sympy_expr),
        sp.Pow: lambda e: handle_binary_op(e, "Pow", tokenize_sympy_expr),
        sp.polylog: lambda e: handle_binary_op(e, "polylog", tokenize_sympy_expr),
        sp.Symbol: lambda e: [str(e)],
        (int, float, sp.Integer): lambda e: handle_int(e),
        sp.Rational: lambda e: ["Rational"] + handle_int(e.p) + handle_int(e.q),
    }

    for expr_type, handler in op_handlers.items():
        if isinstance(expr, expr_type):
            return handler(expr)

    raise ValueError(f"Unexpected expression type: {type(expr)}")


def tokens_to_prefix_notation(tokens: list[str]) -> str:
    prefix_notation = " ".join(tokens)
    return prefix_notation


def sympy_to_prefix_notation(expr) -> str:
    tokens = tokenize_sympy_expr(expr)
    prefix_notation = tokens_to_prefix_notation(tokens)
    return prefix_notation


def GenerateAndWriteData(task_id):
    filepath = f"data/generated/tokens_{task_id}.csv"
    num_functions = NUM_FUNCTIONS

    with open(filepath, "w", newline="") as csvfile:
        # create writer
        writer = csv.writer(csvfile)

        # sample parameters
        # ns # number of distinct dilogarithm terms
        # nt # number of 0 terms
        # n_scr # total number of scrambles
        if LONG_LENGHTH_DATA:
            ns_range = [0, 5]
            nt_range = [3, 6]
            n_scr_max = 18
            coefficient_range = [-20, 20]
            constant_range = [1, 20]
            max_len = 998
            min_len = 513
        else:
            ns_range = [0, 3]
            nt_range = [0, 3]
            n_scr_max = 10
            coefficient_range = [-2, 2]
            constant_range = [1, 8]
            max_len = 512
            min_len = 0

        # Generate datas
        for func_id in range(num_functions):
            # ---- Generate simple expressions and scrambled expressions ----

            # create generator
            generator = PolyLog_multinomial_generator(
                ns_range,
                nt_range,
                n_scr_max,
                coefficient_range,
                constant_range,
                max_len,
                min_len,
            )
            # generate sympy expressions
            simple_expression, scrambled_expression = generator.generate()

            # ---- Tokenize and save data ----

            # convert to prefix notation
            scrambled_prefix_notation = sympy_to_prefix_notation(scrambled_expression)
            simple_prefix_notation = sympy_to_prefix_notation(simple_expression)
            # convert to Mathemtica syntax
            scrambled_expression_mma = sp.mathematica_code(scrambled_expression)
            simple_expression_mma = sp.mathematica_code(simple_expression)

            # write data and information to CSV file
            writer.writerow(
                [
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
            )

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] process {task_id} finished.")


def merge_files(file_group, output_folder):
    group_index = int(file_group[0].split("_")[1].split(".")[0]) // 5  # 获取组的序号
    output_file = os.path.join(output_folder, f"tokens_{group_index}.csv")

    with open(output_file, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        for file in file_group:
            with open(file, "r", newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    writer.writerow(row)

    print(f"group {group_index} finished")


NUM_FUNCTIONS = 100
LONG_LENGHTH_DATA = True

if __name__ == "__main__":
    num_tasks = 50
    num_functions = NUM_FUNCTIONS  # remember to sync the num_functions in GenerateAndWriteData
    total_num_functions = num_functions * num_tasks

    if not os.path.exists("runs/logs/"):
        os.makedirs("runs/logs/")
    logging.basicConfig(filename="runs/logs/FuncGener_general.log", level=logging.INFO)

    data_folder = "data/generated/"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    start_time = time.time()

    with multiprocessing.Pool() as pool:
        print(f"Number of processes: {pool._processes}")
        pool.map(GenerateAndWriteData, range(num_tasks))

    end_time = time.time()
    runtime = end_time - start_time
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    # Logging and print information
    logging.info("[{}] Binary tree structure. Code version: {}".format(timestamp, VERSION))
    logging.info(
        "Total time taken for genetaring functions: {:.4f} seconds".format(runtime)
    )
    logging.info(
        "Total number of functions: {}, file numbers: {}".format(
            total_num_functions, num_tasks
        )
    )
    logging.info("\tnumber of CPU cores: {}, ".format(os.cpu_count()))

    print(
        "Total time taken: {:.4f} seconds, total number of functions: {}".format(
            runtime, total_num_functions
        )
    )

    # Merge files
    output_folder = os.path.join(data_folder, "merged_files/")
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # acquire all files
    all_files = sorted(
        [
            f
            for f in os.listdir(data_folder)
            if f.startswith("tokens_") and f.endswith(".csv")
        ]
    )
    all_files = [os.path.join(data_folder, f) for f in all_files]
    # group files
    file_groups = [all_files[i : i + 5] for i in range(0, len(all_files), 5)]

    start_time = time.time()
    # use thread pool
    max_workers = 120
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(merge_files, file_group, output_folder)
            for file_group in file_groups
        ]

        # make sure all tasks were done
        for future in futures:
            future.result()
    end_time = time.time()
    runtime = end_time - start_time
    logging.info(
        "\tTotal time taken for merging files: {:.4f} seconds, max thread number: {}".format(
            runtime, max_workers
        )
    )

    print("Main task finished.")

    # print("%"*40)
