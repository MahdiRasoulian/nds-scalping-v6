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
        reasons=["test-case"],
        context={"market_metrics": {"atr": 1.0, "volatility_ratio": 1.0}},
        timestamp="2024-01-01T00:00:00Z",
        timeframe="M5",
        current_price=entry
    )


def run_case(risk_manager, config_payload, name, analysis, live, expected_allowed, expected_type=None):
    result = risk_manager.finalize_order(
        analysis=analysis,
        live=live,
        symbol=config_payload["trading_settings"]["SYMBOL"],
        config=config_payload
    )

    allowed_ok = result.is_trade_allowed == expected_allowed
    type_ok = True
    if expected_type is not None:
        type_ok = result.order_type == expected_type

    status = "PASS" if allowed_ok and type_ok else "FAIL"
    print(f"[{status}] {name}")
    print(f"  allowed={result.is_trade_allowed} type={result.order_type} rr={result.rr_ratio:.2f}")
    print(f"  reject_reason={result.reject_reason}")
    print(f"  notes={result.decision_notes}")
    print("-" * 60)
    return status == "PASS"


def main():
    base_config = copy.deepcopy(config_manager.get_full_config())
    risk_manager = create_scalping_risk_manager(config=config_manager)

    max_dev = base_config["risk_settings"]["MAX_PRICE_DEVIATION_PIPS"]
    limit_conf = base_config["risk_settings"]["LIMIT_ORDER_MIN_CONFIDENCE"]
    min_rr = base_config["risk_manager_config"]["MIN_RR_RATIO"]
    point_size = base_config["trading_settings"]["GOLD_SPECIFICATIONS"]["POINT"]

    entry = 2400.0
    sl = 2390.0
    tp = 2420.0

    analysis_buy = build_analysis("BUY", entry, sl, tp, confidence=limit_conf + 1)
    analysis_sell = build_analysis("SELL", entry, entry + 10.0, entry - 20.0, confidence=limit_conf + 1)

    small_dev_live = LivePriceSnapshot(
        bid=entry - (max_dev * point_size * 0.25),
        ask=entry + (max_dev * point_size * 0.25)
    )
    large_dev_live = LivePriceSnapshot(
        bid=entry - (max_dev * point_size * 2.0),
        ask=entry + (max_dev * point_size * 2.0)
    )

    results = []
    results.append(run_case(
        risk_manager,
        base_config,
        "Small deviation allows market",
        analysis_buy,
        small_dev_live,
        expected_allowed=True,
        expected_type="MARKET"
    ))

    results.append(run_case(
        risk_manager,
        base_config,
        "Large deviation with high confidence selects limit",
        analysis_buy,
        large_dev_live,
        expected_allowed=True,
        expected_type="LIMIT"
    ))

    low_conf_analysis = build_analysis("BUY", entry, sl, tp, confidence=limit_conf - 1)
    results.append(run_case(
        risk_manager,
        base_config,
        "Large deviation with low confidence rejects",
        low_conf_analysis,
        large_dev_live,
        expected_allowed=False
    ))

    low_rr_tp = entry + (sl - entry) * -(min_rr * 0.5)
    low_rr_analysis = build_analysis("BUY", entry, sl, low_rr_tp, confidence=limit_conf + 1)
    results.append(run_case(
        risk_manager,
        base_config,
        "RR below minimum rejects",
        low_rr_analysis,
        small_dev_live,
        expected_allowed=False
    ))

    sell_case_live = LivePriceSnapshot(
        bid=entry - (max_dev * point_size * 0.25),
        ask=entry + (max_dev * point_size * 0.25)
    )
    results.append(run_case(
        risk_manager,
        base_config,
        "Sell signal uses risk manager",
        analysis_sell,
        sell_case_live,
        expected_allowed=True
    ))

    if all(results):
        print("ALL TESTS PASS")
    else:
        print("SOME TESTS FAILED")


if __name__ == "__main__":
    main()
