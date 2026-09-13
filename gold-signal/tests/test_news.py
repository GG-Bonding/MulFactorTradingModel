from datetime import datetime, timedelta, timezone

from gold_signal.models import NewsImpact
from gold_signal.news import classify_news, news_age_seconds, news_score, news_weight
from gold_signal.jin10 import make_event_id
from gold_signal.models import FlashNews


def test_bullish_nfp_miss():
    impact = classify_news("美国8月非农就业人数低于预期")
    assert impact.direction == 1
    assert impact.importance >= 2
    assert news_score(impact, 30) == 2


def test_bullish_actual_vs_expected():
    impact = classify_news("美国非农 公布值:11.4 预期:16.5")
    assert impact.direction == 1


def test_bearish_cpi_beat():
    impact = classify_news("美国CPI超预期升温")
    assert impact.direction == -1
    assert impact.importance >= 2
    assert news_score(impact, 30) == -2


def test_bearish_hawkish_fed():
    impact = classify_news("美联储宣布加息，鲍威尔讲话偏鹰派")
    assert impact.direction == -1


def test_ordinary_news_ignored():
    impact = classify_news("某科技公司发布新款手机并召开发布会")
    assert impact.direction == 0
    assert impact.importance == 0
    assert impact.reason == "普通新闻忽略"


def test_war_is_bullish_gold():
    impact = classify_news("中东军事冲突升级，多地传出空袭")
    assert impact.direction == 1
    assert impact.importance == 3


def test_hormuz_attack_is_bullish_gold():
    impact = classify_news("英国海事贸易组织：地方当局正在撤离在霍尔木兹海峡遭袭的船员。")
    assert impact.direction == 1
    assert impact.importance >= 2


def test_bitcoin_pump_is_relevant_bullish():
    impact = classify_news("比特币突破前高并持续上涨")
    assert impact.direction == 1
    assert impact.importance >= 2


def test_news_weight_decay():
    assert news_weight(60) == 1.0
    assert news_weight(180) == 0.7
    assert news_weight(400) == 0.3
    assert news_weight(601) == 0.0


def test_news_expired_score_is_zero():
    impact = NewsImpact(direction=1, importance=3, confidence=0.9, reason="nfp")
    assert news_score(impact, 11 * 60) == 0


def test_event_id_stable():
    ts = datetime(2026, 9, 13, 14, 34, tzinfo=timezone.utc)
    a = make_event_id(ts, "美国8月非农就业人数低于预期")
    b = make_event_id(ts, "美国8月非农就业人数低于预期")
    assert a == b


def test_news_age_seconds():
    now = datetime(2026, 9, 13, 14, 35, 12, tzinfo=timezone.utc)
    news = FlashNews(
        event_id="x",
        title="nfp",
        content="美国8月非农就业人数低于预期",
        published_at=now - timedelta(seconds=83),
    )
    assert 80 <= news_age_seconds(news, now) <= 85
