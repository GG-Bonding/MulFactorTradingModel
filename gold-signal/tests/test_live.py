from gold_signal.main import main


def test_live_requires_token(monkeypatch, tmp_path):
    monkeypatch.delenv("JIN10_TOKEN", raising=False)
    monkeypatch.delenv("JIN10_BEARER_TOKEN", raising=False)
    code = main(["--mode", "live", "--ticks", "1", "--output", str(tmp_path / "signals.jsonl")])
    assert code == 1
