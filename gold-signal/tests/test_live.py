from gold_signal.main import main


def test_live_requires_token(monkeypatch, tmp_path):
    monkeypatch.setenv("JIN10_TOKEN", "")
    monkeypatch.setenv("JIN10_BEARER_TOKEN", "")
    monkeypatch.setattr("gold_signal.main.load_dotenv", lambda *args, **kwargs: None)
    code = main(["--mode", "live", "--ticks", "1", "--output", str(tmp_path / "signals.jsonl")])
    assert code == 1
