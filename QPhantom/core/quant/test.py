import time
import numpy as np
import pandas as pd
from numba import jit
from collections import OrderedDict

@jit
def _back_test(
    n_code,
    top_k,
    max_k,
    unit_max_k,
    skip_rate,
    min_cost,
    cycle_starts,
    cycle_ends,
    buy_price,
    sell_price,
    period_col,
    buy_flag_col,
    price_col,
    score_col,
    code_col,
    funds,
    score_threshold
    ):
    n_cycle = cycle_starts.shape[0]
    price_tb = np.zeros(n_code, dtype=np.float64)
    buy_price_tb = np.zeros(n_code, dtype=np.float64)
    sell_price_tb = np.zeros(n_code, dtype=np.float64)
    period_tb = np.zeros(n_code, dtype=np.int32)
    cost_tb = np.zeros(n_code, dtype=np.float64)
    value_all = np.zeros((2, n_cycle), dtype=np.float64)
    trade_count = np.zeros((2, n_cycle), dtype=np.int32)
    # 0: "open", 1: "high", 2: "low", 3: "close", 4: "max_drawdown"
    hold_log = np.zeros((5, n_cycle * max_k), dtype=np.float64)
    # 0: "code", 1: "start", 2: "end", 3: "period"
    hold_indices = np.zeros((4, n_cycle * max_k), dtype=np.int32)
    hold_size = 0
    current_hold_code_index = np.zeros(n_code, dtype=np.int32)
    hold_flag = np.zeros(n_code, dtype=np.int32)
    current_k = 0
    try_count = 0
    for d_i in range(n_cycle):
        limited_funds = 0.0
        i_start, i_end = cycle_starts[d_i], cycle_ends[d_i]
        hold_flag[:] = period_tb[:]
        # update price
        # sell
        for i in range(i_start, i_end):
            current_code = code_col[i]
            price_tb[current_code] = price_col[i]
            if period_tb[current_code] > 0:
                period_tb[current_code] -= 1
                current_price = price_tb[current_code] if period_tb[current_code] > 0 else sell_price_tb[current_code]
                current_value = cost_tb[current_code] * current_price / buy_price_tb[current_code]
                hold_i = current_hold_code_index[current_code]
                hold_log[4, hold_i] = max(hold_log[4, hold_i], 1.0 - current_value / hold_log[1, hold_i])
                hold_log[1, hold_i] = max(hold_log[1, hold_i], current_value)
                hold_log[2, hold_i] = min(hold_log[2, hold_i], current_value)
                hold_log[3, hold_i] = current_value
                hold_indices[3, hold_i] += 1
                hold_indices[2, hold_i] = d_i
                if period_tb[current_code] == 0:
                    limited_funds += current_value
                    cost_tb[current_code] = 0
                    sell_price_tb[current_code] = 0
                    buy_price_tb[current_code] = 0
                    current_k -= 1
                    trade_count[1, d_i] += 1
                    current_hold_code_index[current_code] = -1

        # 尚未卖出的code算当天持有的价值
        for code in range(n_code):
            if period_tb[code] > 0:
                # print(cost_tb[code], price_tb[code], buy_price_tb[code])
                value_all[0, d_i] += cost_tb[code] * price_tb[code] / buy_price_tb[code]
        # buy
        cost_each = max(funds / (max_k - current_k), min_cost) if max_k > current_k else 0.0
        real_k_left = min(max_k - current_k, unit_max_k, int(funds / cost_each) if cost_each > 1e-6 else unit_max_k)
        for i in range(i_start, i_end):
            if real_k_left <= 0:
                break
            if i >= i_start + top_k:
                break
            if score_col[i] < score_threshold:
                break
            current_code = code_col[i]
            if hold_flag[current_code] > 0:
                continue
            #random skip some stock
            if np.random.random() < skip_rate:
                continue
            real_k_left -= 1
            try_count += 1
            if buy_flag_col[i] == True:
                current_k += 1
                trade_count[0, d_i] += 1
                period_tb[current_code] = period_col[i] - 1
                cost_tb[current_code] = cost_each
                price_tb[current_code] = price_col[i]
                sell_price_tb[current_code] = sell_price[i]
                buy_price_tb[current_code] = buy_price[i]
                funds -= cost_each
                # 计算买入的股票在当天的价值
                current_value = cost_each
                hold_i = hold_size
                hold_size += 1
                current_hold_code_index[current_code] = hold_i
                hold_indices[0, hold_i] = current_code
                hold_indices[1, hold_i] = d_i
                hold_indices[2, hold_i] = d_i
                hold_indices[3, hold_i] = 1
                hold_log[0, hold_i] = cost_each
                hold_log[1, hold_i] = cost_each
                hold_log[2, hold_i] = cost_each
                hold_log[3, hold_i] = cost_each
                hold_log[4, hold_i] = 0.0
                value_all[0, d_i] += cost_each
        # print(funds, limited_funds, value_all[d_i])
        funds += limited_funds
        # 把剩余的资金加入到统计中
        value_all[1, d_i] = funds
    return value_all, trade_count, hold_log[:, :hold_size], hold_indices[:, :hold_size], try_count


def back_test(
    funds,
    top_k,
    score_threshold,
    max_k,
    unit_max_k,
    col_code,
    col_time,
    col_price,
    col_benchmark,
    col_score,
    col_period,
    col_buy_flag,
    col_buy_price,
    col_sell_price,
    min_cost=5000,
    skip_rate=0.3,
    verbose=False
    ):
    col_i_code, uniq_codes = pd.factorize(col_code, sort=True)
    col_i_time, uniq_times = pd.factorize(col_time, sort=True)
    base_df = pd.DataFrame({
        "code":col_i_code,
        "time": col_i_time,
        "score": np.array(col_score) if col_score is not None else 1.0,
        "price": np.array(col_price),
        "buy_price": np.array(col_buy_price),
        "sell_price": np.array(col_sell_price),
        "period": np.array(col_period),
        "buy_flag": np.array(col_buy_flag),
        "benchmark": np.array(col_benchmark)
    })
    if col_score is None:
        base_df.sort_values(["time"])
        gb = base_df.groupby("time")
        base_df = pd.concat([gb.get_group(group).sample(frac=1.0, replace=False) for group in gb.groups])
    else:
        base_df.sort_values(["time", "score"], ascending=[True, False], inplace=True)
    base_df.reset_index(inplace=True, drop=True)
    group_sizes = np.array(base_df.groupby("time")["code"].count(), dtype=np.int32)
    end_indices = np.array(np.add.accumulate(group_sizes))
    start_indices = np.roll(end_indices, 1)
    start_indices[0] = 0
    benchmark = np.array(base_df.groupby("time")["benchmark"].mean())
    #TODO: try_count has no use, we could calculate buy success rate using this count
    t_st = time.time()
    value_all, buy_sell_count, hold_log, hold_indices, try_count = _back_test(
        n_code=uniq_codes.shape[0],
        top_k=top_k,
        max_k=max_k,
        unit_max_k=unit_max_k,
        skip_rate=skip_rate,
        min_cost=min_cost,
        cycle_starts=start_indices,
        cycle_ends=end_indices,
        buy_price=np.array(base_df["buy_price"]),
        sell_price=np.array(base_df["sell_price"]),
        period_col=np.array(base_df["period"]),
        buy_flag_col=np.array(base_df["buy_flag"]),
        price_col=np.array(base_df["price"]),
        code_col=np.array(base_df["code"]),
        score_col=np.array(base_df["score"]),
        funds=funds,
        score_threshold=score_threshold
    )
    t_ed = time.time()
    if verbose is True:
        print(f"RAW BACKTEST TIME: {t_ed - t_st:.3f}s")
    hold_df = pd.DataFrame(OrderedDict([
        ("code", uniq_codes[hold_indices[0]]),
        ("start", uniq_times[hold_indices[1]]),
        ("end", uniq_times[hold_indices[2]]),
        ("open", hold_log[0]),
        ("high", hold_log[1]),
        ("low", hold_log[2]),
        ("close", hold_log[3]),
        ("max_drawdown", hold_log[4]),
        ("period", hold_indices[3])
    ]))
    res_time = uniq_times[np.arange(uniq_times.shape[0])]
    trade_df = pd.DataFrame(OrderedDict([
        ("index", res_time),
        ("position", value_all[0]),
        ("funds", value_all[1]),
        ("benchmark", benchmark),
        ("buy_count", buy_sell_count[0]),
        ("sell_count", buy_sell_count[1])
    ]))
    return trade_df, hold_df
