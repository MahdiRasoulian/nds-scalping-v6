# Refactor Map: Split Bot Orchestration vs. Components

## New files & responsibilities
- `src/trading_bot/realtime_price.py`
  - Owns real-time price monitoring (native monitor or legacy polling) and cached price retrieval.
- `src/trading_bot/trade_tracker.py`
  - Houses the `TradeTracker` class for tracking open/closed trades and daily stats.
- `src/trading_bot/user_controls.py`
  - Implements `get_user_action`, `handle_user_action`, `wait_with_controls`, and `print_status` helpers.
- `src/trading_bot/execution_reporting.py`
  - Generates JSON execution reports and optional full reports via `ReportGenerator`.

## Functions moved from `bot.py`
- Real-time monitoring + price cache
  - From `NDSBot._start_real_time_price_monitor`, `_start_legacy_price_monitor`, `get_current_price`
  - Now in `RealTimePriceMonitor.start()` and `RealTimePriceMonitor.get_current_price()`.
- Trade tracking
  - From inlined `TradeTracker` class
  - Now in `src/trading_bot/trade_tracker.py`.
- User controls
  - From `NDSBot.get_user_action`, `handle_user_action`, `wait_with_controls`, `print_status`
  - Now in `UserControls` helpers.
- Execution reporting
  - From inline reporting block in `NDSBot.execute_scalping_trade`
  - Now in `generate_execution_report()`.

## Behavior-neutral adjustments
- `NDSBot` now orchestrates by delegating to the new helper modules.
- Analyzer config flags (`REAL_TIME_ENABLED`, `USE_CURRENT_PRICE_FOR_ANALYSIS`) are applied via a copied settings dict instead of in-place mutation.
