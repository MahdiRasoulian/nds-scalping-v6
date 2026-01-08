"""Self-test checks for execution contracts and trade reconciliation."""

from datetime import datetime, timedelta
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.trading_bot.contracts import normalize_position
from src.trading_bot.trade_tracker import TradeTracker


def test_normalize_position():
    raw = {
        "ticket": 123,
        "symbol": "XAUUSD",
        "type": "BUY",
        "volume": 0.2,
        "entry_price": 2000.0,
        "current_price": 2001.0,
        "sl": 1995.0,
        "tp": 2010.0,
        "profit": 5.0,
        "magic": 202401,
        "comment": "NDS_TEST",
        "time": datetime.now(),
        "time_update": datetime.now(),
    }
    normalized = normalize_position(raw)
    assert normalized["position_ticket"] == 123
    assert normalized["side"] == "BUY"
    assert normalized["symbol"] == "XAUUSD"


def test_trade_reconcile():
    tracker = TradeTracker()
    now = datetime.now()

    open_event = {
        "event_type": "OPEN",
        "event_time": now,
        "symbol": "XAUUSD",
        "order_ticket": 987,
        "position_ticket": None,
        "side": "BUY",
        "volume": 0.1,
        "entry_price": 2000.0,
        "exit_price": None,
        "sl": 1995.0,
        "tp": 2010.0,
        "profit": None,
        "pips": None,
        "reason": None,
        "metadata": {"magic": 202401, "comment": "NDS_TEST"},
    }
    tracker.add_trade_open(open_event)

    open_positions = [
        normalize_position(
            {
                "ticket": 555,
                "symbol": "XAUUSD",
                "type": "BUY",
                "volume": 0.1,
                "entry_price": 2000.0,
                "current_price": 2002.0,
                "sl": 1995.0,
                "tp": 2010.0,
                "profit": 10.0,
                "magic": 202401,
                "comment": "NDS_TEST",
                "time": now - timedelta(minutes=1),
                "time_update": now,
            }
        )
    ]

    added, updated, closed_candidates = tracker.reconcile_with_open_positions(open_positions)
    assert updated >= 1
    assert not closed_candidates
    assert tracker.get_active_trades_count() == 1


if __name__ == "__main__":
    test_normalize_position()
    test_trade_reconcile()
    print("✅ Self-test passed")
