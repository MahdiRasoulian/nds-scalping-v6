# Config boundary and schema

ConfigManager is only used at the edges of the system; internal modules accept dict overrides.

## Ownership

- `config/settings.py` owns the persisted schema from `config/bot_config.json`.
- `src/trading_bot/bot.py` and `main.py` should only pass overrides that mirror top-level sections.
- `src/trading_bot/risk_manager.py` loads the base config via `ConfigManager` and applies overrides.
- `src/trading_bot/nds/analyzer.py` accepts a dict with `technical_settings` or `ANALYZER_SETTINGS`,
  plus `sessions_config`/`TRADING_SESSIONS`.

## Canonical paths used

**RiskManager expects these top-level sections in overrides:**
- `risk_settings.*` (e.g. `MAX_PRICE_DEVIATION_PIPS`, `LIMIT_ORDER_MIN_CONFIDENCE`, `MAX_ENTRY_ATR_DEVIATION`)
- `technical_settings.*` (e.g. `ATR_WINDOW`, `SCALPING_MIN_CONFIDENCE`, `MAX_SL_DISTANCE`, `MIN_SL_DISTANCE`)
- `risk_manager_config.*` (e.g. `MIN_RR_RATIO`, `MAX_LOT_SIZE`, `POSITION_TIMEOUT_MINUTES`)
- `sessions_config.*` (e.g. `SCALPING_SESSION_ADJUSTMENT`, `SCALPING_HOLDING_TIMES`, `MIN_SESSION_WEIGHT`)
- `trading_settings.GOLD_SPECIFICATIONS.*`

**Analyzer accepts either:**
- `ANALYZER_SETTINGS.*` (preferred)
- or `technical_settings.*` (fallback)
- `TRADING_SESSIONS.*` or `sessions_config.TRADING_SESSIONS.*`

## GOLD_SPECIFICATIONS compatibility

`GOLD_SPECIFICATIONS` supports uppercase keys from `bot_config.json`:
`TICK_VALUE_PER_LOT`, `POINT`, `MIN_LOT`, `MAX_LOT`, `LOT_STEP`, `CONTRACT_SIZE`, `DIGITS`.
RiskManager normalizes these into lowercase internal keys:
`tick_value_per_lot`, `point`, `min_lot`, `max_lot`, `lot_step`, `contract_size`, `digits`.

## Override shape example

```python
overrides = {
    "risk_manager_config": {"MAX_LOT_SIZE": 1.5},
    "risk_settings": {"MAX_PRICE_DEVIATION_PIPS": 40.0},
    "trading_rules": {"MIN_CANDLES_BETWEEN": 2},
}
```

## Consistency notes

- Do not flatten `risk_manager_config` into the top level; keep it nested so RiskManager mapping works.
- Use `risk_settings.MAX_PRICE_DEVIATION_PIPS` (not `MAX_PRICE_DEVIATION`) to match the base schema.
