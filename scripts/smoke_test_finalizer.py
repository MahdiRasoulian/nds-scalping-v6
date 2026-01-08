import copy
import importlib.util
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from config.settings import config as config_manager
from src.trading_bot.risk_manager import create_scalping_risk_manager

models_path = project_root / "src" / "trading_bot" / "nds" / "models.py"
spec = importlib.util.spec_from_file_location("nds_models", models_path)
nds_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nds_models)

AnalysisResult = nds_models.AnalysisResult
LivePriceSnapshot = nds_models.LivePriceSnapshot


def build_analysis(signal: str, entry: float, sl: float, tp: float, confidence: float) -> AnalysisResult:
    return AnalysisResult(
        signal=signal,
        confidence=confidence,
        score=1.0,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
        reasons=["smoke-test"],
        context={"market_metrics": {"atr": 3.0, "volatility_ratio": 1.0}},
        timestamp="2024-01-01T00:00:00Z",
        timeframe="M5",
        current_price=entry
    )


def run_case(name, risk_manager, config_payload, analysis, live):
    result = risk_manager.finalize_order(
        analysis=analysis,
        live=live,
        symbol=config_payload["trading_settings"]["SYMBOL"],
        config=config_payload
    )

    print(f"[{name}]")
    print(f"  allowed={result.is_trade_allowed}")
    print(f"  order_type={result.order_type}")
    print(f"  lot={result.lot_size}")
    print(f"  rr_ratio={result.rr_ratio}")
    print(f"  deviation_pips={result.deviation_pips}")
    print(f"  reject_reason={result.reject_reason}")
    print(f"  decision_notes={result.decision_notes}")
    if not result.decision_notes:
        print("  ⚠️ decision_notes was empty")
    print("-" * 60)


def main():
    base_config = copy.deepcopy(config_manager.get_full_config())
    max_dev = base_config["risk_settings"]["MAX_PRICE_DEVIATION_PIPS"]
    limit_conf = base_config["risk_settings"]["LIMIT_ORDER_MIN_CONFIDENCE"]
    point_size = base_config["trading_settings"]["GOLD_SPECIFICATIONS"]["POINT"]

    entry = 2400.0
    sl = 2394.0
    tp = 2406.0

    analysis_buy = build_analysis("BUY", entry, sl, tp, confidence=limit_conf + 1)
    small_dev_live = LivePriceSnapshot(
        bid=entry - (max_dev * point_size * 0.25),
        ask=entry + (max_dev * point_size * 0.25)
    )
    large_dev_live = LivePriceSnapshot(
        bid=entry - (max_dev * point_size * 2.0),
        ask=entry + (max_dev * point_size * 2.0)
    )

    risk_manager = create_scalping_risk_manager()
    run_case("Base config", risk_manager, base_config, analysis_buy, small_dev_live)

    overrides = {"risk_manager_config": {"MAX_LOT_SIZE": 1.5}}
    override_config = copy.deepcopy(base_config)
    override_config["risk_manager_config"].update(overrides["risk_manager_config"])

    risk_manager_with_overrides = create_scalping_risk_manager(overrides=overrides)
    run_case("With overrides", risk_manager_with_overrides, override_config, analysis_buy, small_dev_live)


if __name__ == "__main__":
    main()
