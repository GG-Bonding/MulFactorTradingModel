# Gold Realtime Signal Engine V0

最小可运行黄金实时信号：新闻提出假设，价格做最终裁判。

输出：`BUY` / `SELL` / `HOLD`。

## 如何申请 / 配置 Token

1. 打开 [金十数据智能开放平台](https://mcp.jin10.com/app/)
2. 登录后点击「立即体验」或「管理TOKEN」，复制 Bearer Token
3. 在本目录执行：

```bash
cp .env.example .env
```

4. 编辑 `.env`：

```text
JIN10_TOKEN=你的token
```

不要把 Token 写进代码或提交到 Git。Replay 模式不需要 Token。

## 安装

需要 Python 3.11+。

```bash
cd gold-signal
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

## Replay

不连金十，用 `tests/fixtures/` 里的 4 个场景验证逻辑：

```bash
cd gold-signal
source .venv/bin/activate
export PYTHONPATH=src
python -m gold_signal.main --mode replay
pytest
```

预期：

- Case 1 利多新闻 + 金银欧上涨 → `BUY`
- Case 2 利空新闻 + 金银欧下跌 → `SELL`
- Case 3 利多新闻 + 黄金下跌 → `SELL` 或 `HOLD`，不能 `BUY`
- Case 4 市场互相冲突 → `HOLD`

## Live

```bash
cd gold-signal
source .venv/bin/activate
export PYTHONPATH=src
python -m gold_signal.main --mode live
```

可选：

```bash
python -m gold_signal.main --mode live --interval 8 --ticks 3
```

Live 会调用金十官方 MCP：`list_flash`、`get_quote`、`get_kline`。工具名以 `tools/list` 实际返回为准。

## 当前规则

判断顺序：新闻 → 黄金有没有反应 → 白银/EURUSD 确认 → 信号。

| 项 | 分数 |
| --- | --- |
| 强利多 / 强利空新闻 | +2 / -2 |
| XAUUSD 1m 明显涨/跌 | +2 / -2 |
| XAUUSD 3m 明显涨/跌 | +1 / -1 |
| XAGUSD 1m 明显涨/跌 | +1 / -1 |
| EURUSD 1m 涨/跌（美元弱/强） | +1 / -1 |

阈值集中在 `src/gold_signal/models.py` 的 `Thresholds`：

- 明显移动：`|return| >= 0.08%`（1m）或 `|z| >= 1.5`
- `score >= +5` 且黄金 1m 为正 → `BUY`
- `score <= -5` 且黄金 1m 为负 → `SELL`
- 其他 → `HOLD`

价格优先：利多新闻但黄金 1m/3m 下跌 → 新闻分数清零，不能 BUY。利空新闻但黄金上涨同理。

新闻衰减：0-2 分钟 100%，2-5 分钟 70%，5-10 分钟 30%，超过 10 分钟为 0。同一新闻只发一次主信号。

信号全部写入 `data/signals.jsonl`，包括 HOLD。BUY/SELL 会尽量补上之后 1/5/15/30 分钟收益。

## 已知限制

- 新闻分类是关键词 + 公布值/预期规则，不是 LLM
- 没有仓位、止损、MT5 下单
- Live 依赖金十 MCP 对 XAUUSD / XAGUSD / EURUSD 的行情权限
- 若 `get_kline` 不可用，会用报价缓存凑 1 分钟收益，启动后第一分钟可能没有信号
- 今天下午若没有重大新闻，Live 会长时间 `HOLD`，这是正确行为

## 下一步

- 用 `signals.jsonl` 统计 BUY/SELL 后的 1m/5m/15m/30m 胜率
- 只在胜率站得住之后再考虑接入交易
