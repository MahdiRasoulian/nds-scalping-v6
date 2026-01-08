## Execution Hardening Notes

### Position Contract
`mt5_client.get_open_positions()` now returns normalized records with these keys:
- `position_ticket`, `symbol`, `side`, `volume`, `entry_price`, `current_price`
- `sl`, `tp`, `profit`, `magic`, `comment`, `open_time`, `update_time`

### Closure Verification
Missing positions are **not** treated as closed immediately. The bot now:
1. Reconciles open positions every cycle.
2. For missing positions, queries `mt5_client.get_position_history(...)`.
3. Only when history confirms closure does it generate a CLOSE event and notify.

If history is not found, the trade is marked `UNKNOWN` and retried next cycle.

### Reporting Locations
Execution events are written as JSON Lines:
- `reports/YYYY-MM-DD/trades/{position_ticket}/events.jsonl`
- `reports/YYYY-MM-DD/trades/{position_ticket}/summary.json`

Legacy execution reports remain at:
- `trade_reports/scalping_executions/`
- `trade_reports/scalping_reports/` (when full report generation is available)

### Config Updates (Optional)
Telegram can now read credentials from environment variables:
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
