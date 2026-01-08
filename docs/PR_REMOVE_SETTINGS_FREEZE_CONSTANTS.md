# PR: remove-settings-py-and-freeze-constants

## A) What changed
- `config/settings.py` now loads only `config/bot_config.json`, validates required keys, and exposes read-only getters without embedded defaults.
- `src/trading_bot/nds/constants.py` is reduced to non-behavioral identifiers (session mapping + allowed analyzer keys).
- Analyzer/SMC/Risk Manager/Bot/CLI now read behavioral values exclusively from `bot_config.json`.
- `config/bot_config.json` now owns all previously hardcoded constants and session/timeframe metadata.

## B) settings.py: removed vs converted (and why)
- **Converted** into a minimal loader wrapper because it is imported across runtime modules (`bot.py`, `risk_manager.py`, `cli.py`, `mt5_client.py`). It now only loads `config/bot_config.json` and fails loudly on missing keys.

## C) constants.py: what remained (non-behavioral only)
- `SESSION_MAPPING`
- `ANALYSIS_CONFIG_KEYS` (string whitelist used by the analyzer)

## D) Exact keys moved from constants.py → bot_config.json
- `technical_settings.TIMEFRAME_SPECIFICS`
- `technical_settings.SWING_PERIOD_MAP`
- `technical_settings.FALLBACK_ATR_PCT`
- `technical_settings.VOLATILITY_STATES`
- `technical_settings.VOLUME_ZONES`
- `technical_settings.MARKET_TREND_THRESHOLDS`
- `technical_settings.DAILY_ATR_FACTOR`
- `technical_settings.ATR_MULTIPLIER`
- `technical_settings.FVG_VOLUME_CONFIRMATION`
- `sessions_config.BASE_TRADING_SESSIONS` (from `TRADING_SESSIONS`)
- `sessions_config.SCALPING_SESSIONS`
- `sessions_config.SCALPING_SESSION_ADJUSTMENT` (from `SCALPING_SESSION_MULTIPLIERS`)
- `sessions_config.SCALPING_HOLDING_TIMES`

## E) Files modified
- `config/settings.py`
- `config/bot_config.json`
- `src/trading_bot/nds/constants.py`
- `src/trading_bot/nds/analyzer.py`
- `src/trading_bot/nds/smc.py`
- `src/trading_bot/nds/__init__.py`
- `src/trading_bot/risk_manager.py`
- `src/trading_bot/bot.py`
- `src/trading_bot/mt5_client.py`
- `src/ui/cli.py`
- `main.py`

## F) Breaking references (if any) and follow-up PR notes
- No known breaking references; all previous constant lookups now read from `bot_config.json`.
- If new config keys are added later, they must be added to `bot_config.json` to satisfy strict validation in `config/settings.py`.

## G) Sanity checks performed (JSON validity, import smoke tests) and their results
- JSON validation: `python -m json.tool config/bot_config.json` ✅
- Import smoke tests:
  - `python -c "from src.trading_bot.nds.analyzer import analyze_gold_market; print('ok-analyzer')"` ❌ (`pandas` not installed)
  - `python -c "from src.trading_bot.nds.constants import SESSION_MAPPING, ANALYSIS_CONFIG_KEYS; print('ok-constants')"` ❌ (`pandas` not installed via package import)
