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

周日黄金休市时，可用 BTC 测同一套信号（行情来自 Binance 公开接口，新闻仍是金十）：

```bash
python -m gold_signal.main --mode live --book btc
```

每个产品单独一条信号，不再把所有新闻都打到黄金上：

```bash
python -m gold_signal.main --mode live --book all --ticks 1
```

| 产品 | 新闻怎么映射 | 价格 |
| --- | --- | --- |
| XAUUSD / XAGUSD | 沿用黄金规则 | 金十 1 分钟 |
| EURUSD | 美国宽松偏多欧元；战争和法国国债收益率上升偏空欧元 | 金十 |
| USDJPY | 只跟美国利率，方向与黄金相反；战争不猜 | 金十 |
| USOIL / UKOIL | 只跟原油供给和战争冲击；CPI 不会自动打成原油 | 金十 |
| 纳指 `NQ=F` | 美国利率与黄金同向于宽松；战争偏空股指；法国国债不映射 | Yahoo 1 分钟 |
| BTCUSDT | 只跟比特币自己的涨跌和监管；宏观不直接映射 | Binance |

确认腿必须和该产品同向。价格不确认就 HOLD。没有 1 分钟 K 线的产品本轮跳过，不编信号。美股个股仍只在 `--mode tape` 报价。

可选：

```bash
python -m gold_signal.main --mode live --interval 8 --ticks 3
```

Live 会调用金十官方 MCP：`list_flash`、`get_quote`、`get_kline`。工具名以 `tools/list` 实际返回为准。

## 当前规则

判断顺序：新闻提出方向 → 只看新闻时刻之后的价格 → 确认腿同窗口 → 信号。

1 分钟和 3 分钟收益是 \(P(t_{news}+\Delta)/P(t_{news})-1\)，不是当前价相对 3 分钟前的涨跌。新闻之前的行情不算反应。+15 秒、+30 秒、+1 分钟、+3 分钟、+5 分钟写进 `data/events.jsonl`。没到时间就显示 waiting，不补数字。

终端里的 `Strength 82/100` 是规则强度，不是上涨概率。

各国国债收益率按持有成本处理：收益率/实际利率上升偏空黄金，回落偏多黄金。仍要等金价确认，新闻本身不会直接 BUY/SELL。

法国 OAT / Bund / 意大利国债 **不是** 同一条黄金空头。系统用 `--mode europe` 分开看：Bund = 政策利率，OAT–Bund = 法国信用，Italy–Bund = 欧元区碎片化。缺数据就标 missing，不编数字，也不会单独把黄金打成 SELL。

```bash
python -m gold_signal.main --mode europe --ticks 1
```

阶段（阈值在 `Thresholds`）：

- `INSUFFICIENT`：OAT 或 Bund 拿不到
- `POLICY_HAWKISH`：Bund 上行且利差仍窄
- `FRANCE_REPRICING`：OAT–Bund ≥ 80bp
- `FRANCE_STRESS`：OAT–Bund ≥ 100bp 且法国银行弱于 CAC
- `EZ_FRAGMENTATION`：OAT–Bund ≥ 150bp **并且** Italy–Bund ≥ 250bp、银行承压、EUR/GBP 下跌

数据源：CNBC 国债报价、Yahoo 法国银行/EURGBP/CAC、金十 EURUSD/GBPUSD（有 token 时）。快照写入 `data/europe.jsonl`。

## 纳指 / 石油 / 外汇 / 美股报价

只报价，不进黄金 BUY/SELL。

```bash
python -m gold_signal.main --mode tape --ticks 1
```

| 组 | 产品 | 来源 |
| --- | --- | --- |
| 纳指 | 纳指100期货 `NQ=F` | Yahoo。金十 `quote://codes` 里没有纳指 |
| 石油 | WTI `USOIL`、布伦特 `UKOIL` | 金十 |
| 外汇 | EURUSD、GBPUSD、USDJPY、AUDUSD、USDCNH、USDCHF、NZDUSD、USDCAD | 金十 |
| 美股 | NVDA、AAPL、MSFT、AMZN、GOOGL、META、TSLA | Yahoo。这是固定流动性篮子，不是当日热度排名 |

金十快讯不是推送流。Live 每 8 秒拉一次 `list_flash`（可带 `cursor` 翻更早的页）。MCP 响应里的 SSE 只是这一次 HTTP 调用的返回格式。

## Polymarket

先用公开 Gamma 搜索找到合约，再用公开 CLOB 订单簿推送更新 Yes 价，不需要 token，也不进买卖分。推送断了就留着上一笔，并在面板上标成 gamma 而不是 clob。

```bash
python -m gold_signal.main --mode polymarket --ticks 1
```

固定搜索：`Fed decision`、`US recession`、`CPI`、`gold`、`crude oil`。优先用标题对得上的未关闭事件，再取成交量最高的几个合约。Yes 价缺失就标 missing。黄金 Live 大约每 5 分钟附带一条，失败不影响信号。

## 当前规则
| --- | --- |
| 强利多 / 强利空新闻 | +2 / -2 |
| XAUUSD 1m 明显涨/跌 | +2 / -2 |
| XAUUSD 3m 明显涨/跌 | +1 / -1 |
| XAGUSD 1m 明显涨/跌 | +1 / -1 |
| EURUSD 1m 涨/跌（美元弱/强） | +1 / -1 |

阈值集中在 `src/gold_signal/models.py` 的 `Thresholds`：

- 明显移动：新闻后的 `|return| >= 0.08%`（1 分钟）或 `|return| >= 0.12%`（3 分钟）
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
- 欧洲信用模块没有官方逐日 OAT/Bund API；CNBC/Yahoo 失败时只报 missing，不回退到过时的 FRED 月度数据
- CAC40 不代表法国内需股；银行相对 CAC 才是主权-银行循环的温度计

## 下一步

- 用 `signals.jsonl` 统计 BUY/SELL 后的 1m/5m/15m/30m 胜率
- 只在胜率站得住之后再考虑接入交易
