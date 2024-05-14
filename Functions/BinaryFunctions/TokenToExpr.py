# !/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------
# @copyright (C), 2023, Jinxin Wang, All rights reserved.
# @File Name   : TokenToExpr.py
# @Author      : Jinxin Wang
# @Version     :
# @Date        : 2024-05-07 22:11:12
# @Description :
#
# Provide two functions to convert tokens back to expressions.
# Recommend to use ExprWay, which is more efficient.
# Also can convert expressions to Mathematica syntax
# Provide functions with result check and time limit
#
# ----------------------------------------------------------------


import sympy as sp
import multiprocessing


def tokens_to_expr_StrWay(tokens):
    """
    Convert tokens into a sympy expression. Supports single variable 'x'.
    Process string first, then use sp.simplify to transfer to sympy expression here.
    e.g.: ['Add', 'Pow', 'x', '+', '3', 'polylog', '+', '2', 'Add', '-', '1', 'x'] --> x**3 + sp.polylog(2, (x - 1))
    """
    stack = []
    for token in reversed(tokens):  # Reverse the list for correct parsing
        if token.isdigit():
            stack.append(token)
        elif token == "x":
            stack.append("x")
        elif token in {"-", "+"}:
            stack[-1] = f"{token}{stack[-1]}"
        elif token in {"polylog", "Add", "Mul", "Pow", "Rational"}:
            operand2, operand1 = stack.pop(), stack.pop()
            stack.append(f"{token}({operand1}, {operand2})")

    if len(stack) == 1:
        return sp.simplify(stack[0])
    else:
        raise IndexError("Error in tokens_to_expr_StrWay: Invalid token processing.")


def tokens_to_expr_ExprWay(tokens):
    """
    Convert tokens into a sympy expression. Supports single variable 'x'.
    transfer to sympy function ASAP, then combine to expression here.
    Note: this function does not check the output.
    e.g.: ['Add', 'Pow', 'x', '+', '3', 'polylog', '+', '2', 'Add', '-', '1', 'x'] --> x**3 + sp.polylog(2, (x - 1))
    """
    if not tokens:
        raise ValueError("Empty token list")

    token = tokens.pop(0)
    if token.isdigit():
        return sp.Integer(int(token))
    elif token == "x":
        return sp.symbols("x")
    elif token in {"+", "-"}:
        operand1 = tokens_to_expr_ExprWay(tokens)
        unit_op_map = {
            "+": lambda I: +I,
            "-": lambda I: -I,
        }
        assert operand1.is_number, f"non-number ({operand1}) after a sign"
        return unit_op_map[token](operand1)
    elif token in {"Add", "Mul", "Pow", "polylog", "Rational"}:
        operand1 = tokens_to_expr_ExprWay(tokens)
        operand2 = tokens_to_expr_ExprWay(tokens)
        binary_op_map = {
            "Add": sp.Add,
            "Mul": sp.Mul,
            "Pow": sp.Pow,
            "polylog": sp.polylog,
            "Rational": sp.Rational,
        }
        return binary_op_map[token](operand1, operand2)
    # Add more operations as needed
    else:
        raise ValueError(f"Unsupported token: {token}")


def tokens_to_expr_ExprWay_w_check(tokens):
    expression = tokens_to_expr_ExprWay(tokens)
    if 0 == len(tokens):
        expression = sp.simplify(expression)
        # expression = sp.cancel(expression)
        # Note: it is already the result after sp.simplify.
        #   If use cancel, it will change the structure.
        return expression
    else:
        raise IndexError(
            "Error in tokens_to_expr_ExprWay_w_check: Invalid token processing."
        )


def tokens_to_mma_ExprWay_w_check(tokens):
    expression = tokens_to_expr_ExprWay(tokens)
    if 0 == len(tokens):
        expression = sp.simplify(expression)
        # expression = sp.cancel(expression)
        # Note: it is already the result after sp.simplify.
        #   If use cancel, it will change the structure.
        mma_expression = sp.mathematica_code(expression)
        return mma_expression
    else:
        raise IndexError(
            "Error in tokens_to_mma_ExprWay_w_check: Invalid token processing."
        )



def worker(tokens, return_dict):
    try:
        expression = tokens_to_expr_ExprWay(tokens)
        if 0 == len(tokens):
            expression = sp.simplify(expression)
            # expression = sp.cancel(expression)
            # Note: it is already the result after sp.simplify.
            #   If use cancel, it will change the structure.
            return_dict['value'] = expression
        else:
            return_dict['value'] = None
    except (ValueError, AssertionError, TypeError):
        return_dict['value'] = None

def tokens_to_expr_ExprWay_w_check_and_timelimit(tokens):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    p = multiprocessing.Process(target=worker, args=(tokens, return_dict))
    p.start()

    # set time limit to 5 seconds
    time_limit = 5
    p.join(timeout=time_limit)

    if p.is_alive():
        p.terminate()
        raise TimeoutError(f"Execution of tokens_to_expr_ExprWay exceeded {time_limit} seconds.")
    else:
        expression = return_dict['value']
        if expression is None:
            raise IndexError(
                "Error in tokens_to_expr_ExprWay_w_check_and_timelimit: Invalid token processing."
            )
        else:
            return expression



if __name__ == "__main__":
    # convert tokens to expressoin in Sympy syntax

    # "y"
    # tokens = ['Add', '2', '1', 'x']

    # "z"
    tokens = ['Add', 'Mul', '+', '4', 'polylog', '+', '2', 'Mul', 'Pow', 'Add', 'Add', '2', 'Add', 'x', 'Pow', 'x', '+', '2', '-', '1', 'Add', '+', '2', 'Mul', '+', '2', 'Pow', 'x', '+', '2', 'polylog', '+', '2', 'Mul', 'Rational', '+', '1', '+', '2', 'Mul', 'Pow', 'x', '-', '2', 'Add', '+', '1', 'Add', 'Mul', '-', '2', 'Pow', 'x', '+', '2', 'Mul', '+', '2', 'x']
    
    # "y"
    # tokens = ['Mul', '2', '6', 'polylog', '2', 'Mul', '3', 'x', 'Pow', 'Add', '2',
    #           '-', '2', 'x', '-', '1', 'Add', '2', '1', 'Mul', '2', '-', '2', 'x']
    
    
    try:
        expr = tokens_to_expr_ExprWay_w_check_and_timelimit(tokens)
    except IndexError:
        expr = "y"
    except TimeoutError:
        expr = "z"

    print(expr)

    # expr = tokens_to_expr(tokens)
    # print(expr)
    print()

    # convert to Mathematica syntax
    mathematica_expr = sp.mathematica_code(expr)
    print(mathematica_expr)
