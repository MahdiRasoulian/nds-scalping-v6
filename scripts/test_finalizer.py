from config.settings import config
from src.trading_bot.nds.models import AnalysisResult
from src.trading_bot.risk_manager import create_scalping_risk_manager


class ConfigWrapper(dict):
    def get_full_config(self):
        return config.get_full_config()


def run_test(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"{status}: {name}")
    return condition


def main():
    overrides = ConfigWrapper(
        {
            "risk_settings": {
                "RISK_AMOUNT_USD": 10000.0,
                "LIMIT_ORDER_MIN_CONFIDENCE": 74.0,
            },
            "risk_manager_config": {
                "MAX_LOT_SIZE": 0.02,
            },
        }
    )
    risk_manager = create_scalping_risk_manager(config=overrides)

    analysis = AnalysisResult(
        signal="BUY",
        confidence=80.0,
        score=75.0,
        entry_price=2000.0,
        stop_loss=1995.0,
        take_profit=2010.0,
        reasons=["Test entry idea"],
        context={"market_metrics": {"atr": 6.0}},
        timestamp="2025-01-01T00:00:00Z",
        timeframe="M5",
        current_price=2000.0,
    )

    live_snapshot = {"bid": 2000.5, "ask": 2001.0, "spread": 0.5}
    finalized = risk_manager.finalize_order(analysis, live_snapshot)

    results = []
    results.append(
        run_test(
            "Finalizer picks MARKET when deviation within max",
            finalized is not None and finalized.order_type == "MARKET",
        )
    )
    results.append(
        run_test(
            "Lot clamp handled in RiskManager",
            finalized is not None
            and any("Volume clamped" in reason for reason in finalized.reasons),
        )
    )
    results.append(
        run_test(
            "Finalized payload includes SL/TP adjustments",
            finalized is not None
            and finalized.sl is not None
            and finalized.tp is not None,
        )
    )

    if all(results):
        print("✅ All finalizer checks passed.")
    else:
        print("❌ One or more finalizer checks failed.")


if __name__ == "__main__":
    main()
