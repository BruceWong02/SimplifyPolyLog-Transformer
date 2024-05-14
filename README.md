# SimplyPolyLog-Transformer

## 介绍

我们尝试训练了一个 Transformer 模型来简化 weight-2 polylogarithms。这个仓库中包含模型文件、训练代码、测试代码和各类工具代码。

此外，这里我们还提供了以下相关的文件：
- 生成数据集的脚本 `generate_dataset.py`。
- 一个 checkpoints，可以达到超过 98% 的 anti-symbol 准确率。
- 一个 Mathematica 代码来检验模型的表达式准确率和 anti-symbol 准确率。
> 注意：计算 symbol 需要提前安装 PolyLogTools 包。

### 准确率定义
| 表达式准确率 | anti-symbol 准确率 |
| - | - |
| 直接比较数学表达式 | 比较 symbol 的反对易部分 |



## Introduction

We tried to train a Transformer model to simplify the weight-2 polylogarithms. This repository contains the model files, training code, testing code, and utilities.

In addition, we also provide the following related files:
- A script `generate_dataset.py` to generate the dataset.
- A checkpoint that can achieve an accuracy of over 98%.
- A Mathematica code to test the expression accuracy and anti-symbol accuracy of the model.
> Notice: Computing the symbol requires the PolyLogTools package to be installed in advance.

### Accuracy Definition
| Expression Accuracy | Anti-Symbol Accuracy |
| - | - |
| Directly comparing mathematical expressions | Comparing the anti-symbol part of the symbol |
