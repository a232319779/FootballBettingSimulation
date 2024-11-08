# -*- coding: utf-8 -*-
# @Time     : 2024/11/08 13:42:29
# @Author   : ddvv
# @Site     : https://ddvvmmzz.github.io
# @File     : main2.py
# @Software : Visual Studio Code
# @WeChat   : NextB

import argparse
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.linear_model import LinearRegression

__doc__ = """
当前使用的评估指标：
夏普比率 (Sharpe Ratio)：衡量策略的收益率相对于风险的比率。
波动率 (Volatility)：衡量策略的波动性，即收益率的标准差。
最大回撤 (Maximum Drawdown)：衡量策略的最大亏损幅度。
破产次数 (Bankruptcy Count)：衡量策略的最大亏损次数。

其他可以参考的指标：
信息比率 (Information Ratio, IR)：衡量超过基准的收益（即阿尔法收益）相对于跟踪误差的比例。与夏普比率类似，但使用基准索引而不是无风险利率。
卡玛比率 (Calmar Ratio)：衡量年化收益率与最大回撤之间的比率。该比率越高，风险调整后的收益越好。
索提诺比率 (Sortino Ratio)：类似于夏普比率，但只考虑下行风险。用来更好地评估策略在不对称风险下的表现。
净值增长率 (Compound Annual Growth Rate, CAGR)：表示投资随着时间的年复合增长率，是长周期表现的衡量标准。
特雷诺比率 (Treynor Ratio)：类似于夏普比率，但使用贝塔来评估（即考虑系统性风险）。
胜率 (Win Rate)：计算盈交易占总交易的比例，反映策略成功的频率。
平均盈亏比 (Average Profit/Loss Ratio)：比较平均盈利交易与平均亏损交易的比率，帮助评估交易的质量。
权益曲线倾斜度 (Equity Curve Slope)：衡量权益曲线的上升斜率，反映赚钱的速度。
投资回收期 (Payback Period)：衡量收回最初投资花费的时间。
"""


# 投注策略模拟
def simulate_strategy(
    base_amount,
    strategy,
    outcomes_length,
    win_rate,
    odds_range,
    max_loss_reset,
    max_capital,
):
    capital = 0
    bet_amount = base_amount
    odds = np.random.uniform(*odds_range, outcomes_length)
    outcomes = np.random.choice(
        [1, 0], size=outcomes_length, p=[win_rate, 1 - win_rate]
    )

    capital_history = []
    loss_streak = 0

    fibonacci_sequence = [base_amount, base_amount]
    fibonacci_index = 0

    for i in range(outcomes_length):
        if capital + max_capital <= 0:
            # 补0到剩余长度
            capital_history.extend([0] * (outcomes_length - i))
            break

        if strategy == "均注法":
            bet_amount = base_amount
        elif strategy == "倍投法" and outcomes[i] == 0:
            bet_amount *= 2
        elif strategy == "斐波那契法" and outcomes[i] == 0:
            fibonacci_index += 1
            if fibonacci_index >= len(fibonacci_sequence):
                fibonacci_sequence.append(
                    fibonacci_sequence[-1] + fibonacci_sequence[-2]
                )
            bet_amount = fibonacci_sequence[fibonacci_index]

        bet_amount = min(bet_amount, capital + max_capital)

        if outcomes[i] == 1:
            capital += bet_amount * (odds[i] - 1)
            if strategy != "均注法":
                bet_amount = base_amount
                fibonacci_index = 0
                fibonacci_sequence = [base_amount, base_amount]
                loss_streak = 0
        else:
            capital -= bet_amount
            loss_streak += 1
            if loss_streak >= max_loss_reset:
                bet_amount = base_amount
                fibonacci_index = 0
                fibonacci_sequence = [base_amount, base_amount]
                loss_streak = 0

        capital_history.append(capital + max_capital)

    return capital_history


# 评估指标
def evaluate_strategy(capital_history, benchmark_returns, outcomes_length):
    capital_history = np.array(capital_history)

    # 创建一个收益率数组，并用非零资本计算收益率
    returns = np.zeros(len(capital_history) - 1)
    non_zero_indices = capital_history[:-1] != 0

    returns[non_zero_indices] = (
        np.diff(capital_history)[non_zero_indices]
        / capital_history[:-1][non_zero_indices]
    )

    # 填充破产状态时的收益率
    returns[~non_zero_indices] = 0  # 或者-1等，根据业务需求

    # 将数组长度调整为一致
    min_length = min(len(returns), len(benchmark_returns))
    returns = returns[:min_length]
    benchmark_returns = benchmark_returns[:min_length]

    # 处理 NaN 和无穷大值
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    benchmark_returns = np.nan_to_num(
        benchmark_returns, nan=0.0, posinf=0.0, neginf=0.0
    )

    total_return = capital_history[-1] - capital_history[0]
    annualized_return = np.mean(returns) * outcomes_length if len(returns) > 0 else 0
    volatility = np.std(returns) * np.sqrt(outcomes_length) if len(returns) > 0 else 0
    sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
    max_drawdown = np.max(np.maximum.accumulate(capital_history) - capital_history)

    # 计算阿尔法和贝塔
    if len(returns) > 0:
        model = LinearRegression().fit(benchmark_returns.reshape(-1, 1), returns)
        alpha = model.intercept_
        beta = model.coef_[0]
    else:
        alpha = 0
        beta = 0

    return {
        "最终资本": capital_history[-1],
        "总收益": total_return,
        "年化收益率": annualized_return,
        "波动率": volatility,
        "夏普比率": sharpe_ratio,
        "最大回撤": max_drawdown,
        "破产次数": np.sum(capital_history <= 0),
        "阿尔法": alpha,
        "贝塔": beta,
    }


def generate_benchmark_returns(
    base_amount, outcomes_length, win_rate, odds, max_capital
):
    capital = 0
    outcomes = np.random.choice(
        [1, 0], size=outcomes_length, p=[win_rate, 1 - win_rate]
    )

    capital_history = [capital + max_capital]
    bet_amount = base_amount

    for i in range(outcomes_length):
        if outcomes[i] == 1:
            capital += bet_amount * (odds - 1)
        else:
            capital -= bet_amount

        capital_history.append(capital + max_capital)

    capital_history = np.array(capital_history)
    returns = np.zeros(len(capital_history) - 1)

    # Calculate returns only where previous capital is not zero
    non_zero_indices = capital_history[:-1] != 0
    returns[non_zero_indices] = (
        np.diff(capital_history)[non_zero_indices]
        / capital_history[:-1][non_zero_indices]
    )

    # Fill with a value (like 0 or -1) for cases where previous capital was zero
    returns[~non_zero_indices] = -1  # or 0, depending on how you want to handle it

    return returns


def main():
    parser = argparse.ArgumentParser(description="竞彩足球投注模拟程序.")

    parser.add_argument(
        "--base_amount", type=float, default=100, help="初始投注金额, 默认为100"
    )
    parser.add_argument(
        "--outcomes_length",
        type=int,
        default=365,
        help="模拟单次投注数量, 默认365，一天一单，模拟一年",
    )
    parser.add_argument("--win_rate", type=float, default=0.5, help="胜率, 默认为0.5")
    parser.add_argument(
        "--odds_range",
        type=float,
        nargs=2,
        default=[1.93, 2.0],
        help="赔率范围,默认为(1.93, 2.0)",
    )
    parser.add_argument(
        "--max_loss_reset",
        type=int,
        default=6,
        help="最大连输场数后重置, 默认为6, 用于倍投法和斐波那契法",
    )
    parser.add_argument(
        "--simulations", type=int, default=100, help="模拟次数, 默认100次"
    )
    parser.add_argument(
        "--max_capital", type=float, default=10000, help="最大本金, 默认为10000"
    )
    parser.add_argument(
        "--base_rate",
        type=float,
        default=0.5,
        help="基准胜率, 默认为0.5, 以均注法的收益生成基准收益, 计算阿尔法",
    )
    parser.add_argument(
        "--base_odds",
        type=float,
        default=2.0,
        help="基准赔率, 默认为2.0, 以均注法的收益生成基准收益, 计算贝塔",
    )

    args = parser.parse_args()

    # 参数设置
    base_amount = args.base_amount  # 初始投注金额
    outcomes_length = args.outcomes_length  # 模拟单次投注数量
    win_rate = args.win_rate  # 胜率
    odds_range = args.odds_range  # 赔率范围
    max_loss_reset = args.max_loss_reset  # 最大连输场数后重置
    simulations = args.simulations  # 模拟次数
    max_capital = args.max_capital  # 最大本金
    base_rate = args.base_rate  # 基准胜率
    base_odds = args.base_odds  # 基准赔率

    strategies = ["均注法", "倍投法", "斐波那契法"]
    print(
        f"模拟参数设置如下:\n\t最大本金max_capital: {max_capital}\n\t初始投注金额base_amount: {base_amount}\n\t单次投注量outcomes_length: {outcomes_length}\n\t胜率win_rate: {win_rate}\n\t赔率范围odds_range: {odds_range}\n\t最大连输场数后重置max_loss_reset: {max_loss_reset}\n\t模拟次数simulations: {simulations}\n\t基准收益: 由基准胜率base_rate为{base_rate}, 基准赔率base_odds为{base_odds}的均注法生成"
    )

    # 生成基准收益
    benchmark_returns = generate_benchmark_returns(
        base_amount, outcomes_length, base_rate, base_odds, max_capital
    )

    # 仿真多个策略并显示结果
    results_table = []

    for strategy in strategies:
        results = []
        bankrupt_count = 0

        for _ in range(simulations):
            simulation_results = simulate_strategy(
                base_amount,
                strategy,
                outcomes_length,
                win_rate,
                odds_range,
                max_loss_reset,
                max_capital,
            )
            evaluation = evaluate_strategy(
                simulation_results, benchmark_returns, outcomes_length
            )
            if evaluation["破产次数"]:
                bankrupt_count += 1
            results.append(evaluation)

        df_results = pd.DataFrame(results)
        average_results = df_results.mean(numeric_only=True).round(4)
        average_results["破产次数"] = bankrupt_count
        results_table.append([strategy] + list(average_results.values))

    # 表格输出
    headers = [
        "策略",
        "最终资产",
        "总收益",
        f"{outcomes_length}单收益率",
        "波动率",
        "夏普比率",
        "最大回撤",
        f"破产次数/{simulations}模拟",
        "阿尔法",
        "贝塔",
    ]
    print(tabulate(results_table, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    main()
