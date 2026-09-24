from gold_signal.models import PolymarketBoard, PolymarketContract
from gold_signal.polymarket import apply_market_message, contracts_from_search


def test_yes_price_comes_from_outcome_not_from_a_guess():
    rows = contracts_from_search(
        {
            "events": [
                {
                    "title": "Fed Decision in October?",
                    "slug": "fed-decision-in-october",
                    "closed": False,
                    "markets": [
                        {
                            "question": "Will the Fed decrease interest rates by 50+ bps after the October 2026 meeting?",
                            "slug": "fed-50bps",
                            "outcomes": "[\"Yes\", \"No\"]",
                            "outcomePrices": "[\"0.0025\", \"0.9975\"]",
                            "bestBid": 0.002,
                            "bestAsk": 0.003,
                            "volume": "100",
                            "endDate": "2026-10-29T03:59:00Z",
                            "updatedAt": "2026-09-24T02:52:18Z",
                            "closed": False,
                        },
                        {
                            "question": "Will the Fed cut 25 bps?",
                            "slug": "fed-25bps",
                            "outcomes": ["Yes", "No"],
                            "outcomePrices": ["0.61", "0.39"],
                            "volume": "900",
                            "closed": False,
                        },
                    ],
                }
            ]
        },
        "Fed",
        limit=4,
    )
    by_slug = {row.slug: row for row in rows}
    assert by_slug["fed-25bps"].yes == 0.61
    assert by_slug["fed-50bps"].yes == 0.0025
    assert by_slug["fed-50bps"].source == "Polymarket gamma API"
    assert rows[0].slug == "fed-25bps"


def test_closed_event_and_missing_yes_are_not_invented():
    assert contracts_from_search({"events": [{"title": "old", "closed": True, "markets": []}]}, "Fed") == []
    rows = contracts_from_search(
        {
            "events": [
                {
                    "title": "Gold",
                    "closed": False,
                    "markets": [
                        {
                            "question": "Gold above 5000?",
                            "slug": "gold-5000",
                            "outcomes": "[\"Yes\", \"No\"]",
                            "outcomePrices": "not-json",
                            "closed": False,
                        }
                    ],
                }
            ]
        },
        "gold",
    )
    assert rows[0].yes is None
    assert rows[0].missing == "Yes price missing"


def test_hint_skips_an_unrelated_open_event():
    rows = contracts_from_search(
        {
            "events": [
                {
                    "title": "Japan recession in 2026?",
                    "closed": False,
                    "markets": [
                        {
                            "question": "Japan recession in 2026?",
                            "slug": "japan",
                            "outcomes": ["Yes", "No"],
                            "outcomePrices": ["0.045", "0.955"],
                            "volume": "10",
                            "closed": False,
                        }
                    ],
                },
                {
                    "title": "US recession by end of 2026?",
                    "closed": False,
                    "markets": [
                        {
                            "question": "US recession by end of 2026?",
                            "slug": "us",
                            "outcomes": ["Yes", "No"],
                            "outcomePrices": ["0.22", "0.78"],
                            "volume": "10",
                            "closed": False,
                        }
                    ],
                },
            ]
        },
        "recession",
        hint="US recession",
    )
    assert rows[0].slug == "us"
    assert rows[0].yes == 0.22


def _board() -> PolymarketBoard:
    return PolymarketBoard(
        as_of=__import__("datetime").datetime(2026, 9, 24, tzinfo=__import__("datetime").timezone.utc),
        contracts=[
            PolymarketContract(
                topic="Fed",
                event="Fed Decision",
                question="No change?",
                slug="no-change",
                token_id="yes-token",
                yes=0.30,
                source="Polymarket gamma API",
            )
        ],
    )


def test_clob_quote_replaces_gamma_snapshot():
    board = _board()
    assert apply_market_message(
        board,
        {
            "event_type": "best_bid_ask",
            "asset_id": "yes-token",
            "best_bid": "0.33",
            "best_ask": "0.34",
            "timestamp": "1782753357257",
        },
    )
    row = board.contracts[0]
    assert row.bid == 0.33
    assert row.ask == 0.34
    assert abs(row.yes - 0.335) < 1e-12
    assert row.source == "Polymarket CLOB websocket"
    assert not apply_market_message(board, {"event_type": "best_bid_ask", "asset_id": "other", "best_bid": "0.1", "best_ask": "0.2"})
    assert row.yes == 0.335


def test_book_uses_best_bid_and_ask_not_the_first_level():
    board = _board()
    apply_market_message(
        board,
        {
            "event_type": "book",
            "asset_id": "yes-token",
            "bids": [{"price": "0.08"}, {"price": "0.09"}],
            "asks": [{"price": "0.99"}, {"price": "0.98"}],
            "timestamp": "1782753357257",
        },
    )
    assert board.contracts[0].bid == 0.09
    assert board.contracts[0].ask == 0.98


def test_yes_token_is_kept_for_the_socket():
    rows = contracts_from_search(
        {
            "events": [
                {
                    "title": "Fed decision",
                    "closed": False,
                    "markets": [
                        {
                            "question": "No change?",
                            "slug": "hold",
                            "outcomes": ["Yes", "No"],
                            "outcomePrices": ["0.33", "0.67"],
                            "clobTokenIds": ["yes-id", "no-id"],
                            "volume": "5",
                            "closed": False,
                        }
                    ],
                }
            ]
        },
        "Fed",
        hint="Fed decision",
    )
    assert rows[0].token_id == "yes-id"


def test_empty_search_returns_no_contracts():
    assert contracts_from_search({}, "recession") == []
    assert contracts_from_search({"events": []}, "oil") == []
