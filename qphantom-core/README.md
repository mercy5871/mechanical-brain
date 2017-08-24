## Q-Phantom Core

Q-Phantom的公共核心库

#### Install

```bash
# install editmode
pip install -e git+ssh://gitlab@git.q-phantom.com/QPhantom/core.git#egg=QPhantom-core
```

#### Update

```bash
# cd src/qpahantom-core
git pull
```

#### Notifier

###### Shell

```bash
python -m QPhantom.exec.guard --to caojianfeng@ppmoney.com --project USHIS_DATA --label Daily_sync $CMD
```

###### Python API

```python
from QPhantom.core.notify import MailNotifier

notifier = MailNotifier.default(to='you@emailhost.com', project='Test Project')

logger = notifier.logger

# notifier.info, notifier.warning, notifier.error类似
notifier.warning("Warning Label", "Warning Message")

# with statement
with notifier.guardian("Guard Label"):
  logger.info("fetch_start")
  data = fetch_data()
  data.to_hdf()
  logger.info("fetch_end")

# function wrapper

@notifier.guard("Guad Label")
def fetch_data():
  logger.info("fetch_start")
  fetch_data()
  data.to_hdf()
  logger.info("fetch_end")

fetch_data()

```

## Trigger交易策略使用指南

Trigger模块用于支持特定卖出规则下的要本生成，不包含买入选股策略。买入选股策略可以基于trigger的结果进行进一步处理（比如用模型在每个周期取top-k)

```python
def gen_label(builder, min_period, triggers):
    builder.do_init()
    buy_at = 1.005
    extra_cost_rate = 0.0025
    builder.label("trigger", {
        #买入挂单价格，如果是未来价格（比如当天的均价），这里相当于滑点
        "buy_at": buy_at,
        #购买的条件，这里可以对涨停的情况进行过滤
        "buy_cond": (builder["w_low"] < builder["w_high"]) & (builder["low"] < builder["w_open"]),
        #额外的成本，包括手续费，印花税等。这个费用是双向均摊的，买入和卖出都会计算
        "extra_cost_rate": extra_cost_rate,
        #买入定价基准，也是默认的卖出定价基准
        "base_col": "w_open",
        #数据的high列，主要用于判断买入卖出能否成功
        "high": "w_high",
        #数据的low列
        "low": "w_low",
        #对于最后都没有卖出的股票，使用该列计算其价值
        "fallback": "w_close",
        #最少持有的时间周期，不包括买入时的周期，实际对资金的占用周期至少是min_period + 1
        "min_period": min_period,
        #卖出需要给股票留出的空间，在目标初期的least-time个周期内不会发生卖出
        "least_time": 12,
        #实际卖出的触发规则
        "trigger": triggers,
        #卖出和买入能够发生的先决条件，这里是成交额上要有保证
        "trade_cond": builder["amount"] > 1e7
    }, key="trigger")

    builder.eval()
    return [v["trigger"] for v in builder.get_label(do_init=True)]

df_y = gen_label(builder, min_period=min_period, triggers=[
    {
        # 卖出挂单价格
        "sell_at": 1.05,
        # 卖出条件，可以使一个数组，表示每天是否达到卖出条件，该列可以由模型生成，由模型判断每天是否卖出
        "flag": True,
        # 卖出价格的基准，如果不设置，会采用上面的base_col，注意，该列不要使用未来信息，比如当前周期的均价或者close
        "sell_on": "w_open"
    }
    # trigger可以有多个，触发过程是第一个的flag匹配则尝试用第一个条件卖，否则进入后一个trigger，一旦触发，则不论该价格能否卖出成功，都不会再触发新的trigger。
])
```

#### 基于Trigger结果的回测

```python
def do_test(base_df, base_y, window_size=10, score_col=None, score_threshold=0.5, unit_max_k=5):
    '''
    Args:
        base_df: 用于提取价格和股票等信息，可以通过buidler.get_df()接口获得
        base_y: Trigger生成的数据，由上面的gen_label接口生成
        window_size: 展示交易次数和持仓比例的滑动窗口大小（平滑曲线用）
        score_col: 模型选股的分值列，回测会按该列取top-k
        score_threshold: 对于score的阈值
        unit_max_k: 每天最多买入的股票数量
    '''
    with measureTime("test total"):
        trade_log = back_test(
            #股票代码列
            col_code=base_df["stock"],
            #时间周期列
            col_time=base_df["date"],
            #该列用于评估持有中的股票的价值
            col_price=base_df["w_close"],
            # 以下列都是trigger生成的，一般按下面的填就行，当然，你也可以不用trigger接口，自己算
            col_period=base_y["trigger_period"],
            col_buy_flag=base_y["trigger_buy_flag"],
            col_buy_price=base_y["trigger_buy_price"],
            col_sell_price=base_y["trigger_sell_price"],
            # benchmark列
            col_benchmark=base_df["iclose"],
            # 模型预测的分值，None表示随机
            col_score=score_col,
            # 初始资金数额
            funds=500000,
            # 在每个周期对score取top-k
            top_k=10,
            # 在score上的阈值
            score_threshold=score_threshold,
            # 最多持有多少支
            max_k=20,
            # 每个周期最大买入支数
            unit_max_k=unit_max_k,
            # 每支股票最少使用资金
            min_cost=5000,
            # 随机跳过股票，增加随机性，用于测试策略的稳定性
            skip_rate=0.4
        )
    M = Metrics(size=(10, 10), fontsize=12)
    M.plot_trade_log(trade_log, window_size)
```

##### 随机选股的回测

```python
# df_y是由上面gen_label生成的
def test_on_random(df_y, unit_max_k=4):
    plot_dist(df_y, [True, True, True])
    df_y_train, df_y_val, df_y_test=df_y
    df_train, df_val, df_test=builder.get_df()

    do_test(base_df = df_train, base_y = df_y_train, unit_max_k=unit_max_k)
    do_test(base_df = df_val, base_y = df_y_val, unit_max_k=unit_max_k)
    do_test(base_df = df_test, base_y = df_y_test, unit_max_k=unit_max_k)
```

## FAQ

##### 安装提示没有clone权限

请把ssh公钥添加到工程deploy_key中，这样就可以clone了
