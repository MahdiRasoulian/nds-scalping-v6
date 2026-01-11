"""
NDS Backtest Engine (XAUUSD scalping)
------------------------------------
Design goals:
- Works with your existing NDS analyzer (analyze_gold_market) and ScalpingRiskManager.finalize_order
- Candle-by-candle simulation (walk-forward) with reproducible records for analysis
- Parameter override friendly (grid search / random search)
- Produces a detailed trades ledger + equity curve + per-cycle diagnostics
- **Anti-lookahead / anti-leakage**:
    * Analysis at bar i only sees candles up to i (inclusive).
    * Any new order decided at bar i is **executed no earlier than bar i+1**.
    * SL/TP evaluation is done using subsequent bars only.
    * For LIMIT orders: order is placed at i, then filled only if future bars reach the limit price.

Expected OHLCV input (from loader; Excel/CSV):
time, open, high, low, close, volume (or tick_volume)
Time must be parseable by pandas.to_datetime.

Notes:
- Bid/Ask are synthesized using spread (in price units, e.g., 0.25$). You may pass a fixed spread.
- SL/TP are evaluated intra-bar using high/low:
    BUY: SL hits if low <= SL; TP hits if high >= TP
    SELL: SL hits if high >= SL; TP hits if low <= TP
  If both hit in same bar, we apply a conservative rule: SL first (worst-case).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import math

import pandas as pd
import numpy as np


# -----------------------------
# Import your project modules
# -----------------------------
def _import_analyzer():
    # Prefer your package layout
    try:
        from src.trading_bot.nds.analyzer import analyze_gold_market
        return analyze_gold_market
    except Exception:
        pass

    # Fallback: if running standalone next to analyzer.py
    try:
        from analyzer import analyze_gold_market  # type: ignore
        return analyze_gold_market
    except Exception as e:
        raise ImportError(
            "Could not import analyze_gold_market. Place this backtest tool inside your repo "
            "or ensure analyzer.py is importable."
        ) from e


def _import_risk_manager():
    try:
        from src.trading_bot.risk_manager import ScalpingRiskManager
        return ScalpingRiskManager
    except Exception:
        pass

    try:
        from risk_manager import ScalpingRiskManager  # type: ignore
        return ScalpingRiskManager
    except Exception as e:
        raise ImportError(
            "Could not import ScalpingRiskManager. Place this backtest tool inside your repo "
            "or ensure risk_manager.py is importable."
        ) from e


analyze_gold_market = _import_analyzer()
ScalpingRiskManager = _import_risk_manager()


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Position:
    id: str
    side: str  # BUY / SELL
    symbol: str
    open_time: pd.Timestamp
    open_bar_index: int

    entry_price: float
    stop_loss: float
    take_profit: float
    lot: float

    order_type: str  # MARKET/LIMIT (simulated)

    confidence: float
    score: float
    rr: float
    session: str
    deviation_pips: float
    notes: List[str]

    close_time: Optional[pd.Timestamp] = None
    close_bar_index: Optional[int] = None
    close_price: Optional[float] = None
    exit_reason: Optional[str] = None  # TP/SL/TIMEOUT/EOD/MANUAL
    pnl_usd: Optional[float] = None
    duration_bars: Optional[int] = None


@dataclass
class PendingOrder:
    """
    Pending order that is created at decision bar i and can only fill on bars > i.
    For MARKET orders, we force-fill on i+1 open (worst-case: spread + slippage).
    For LIMIT orders, we fill when the limit price is touched by future bar high/low.
    """
    id: str
    created_time: pd.Timestamp
    created_bar_index: int

    side: str
    symbol: str
    order_type: str  # MARKET / LIMIT
    limit_price: Optional[float]  # for LIMIT, the target price
    stop_loss: float
    take_profit: float
    lot: float

    confidence: float
    score: float
    rr: float
    session: str
    deviation_pips: float
    notes: List[str]

    expires_bar_index: int  # inclusive expiry; if current bar index > expires -> cancel
    reject_reason: Optional[str] = None


@dataclass
class BacktestConfig:
    symbol: str = "XAUUSD!"
    timeframe: str = "M15"
    bars_to_fetch: int = 3500

    # simulation
    warmup_bars: int = 300  # minimum bars before allowing trades
    spread: float = 0.25    # price units ($)
    slippage: float = 0.10  # price units ($) - applied on entry/exit
    commission_per_lot: float = 0.0  # USD per lot, per side

    # governance
    allow_multiple_positions: bool = True
    max_positions: int = 5
    min_candles_between_trades: int = 10
    min_time_between_trades_minutes: int = 150
    daily_max_trades: int = 40

    # risk
    starting_equity: float = 1000.0
    max_daily_risk_percent: float = 6.0

    # pending orders
    market_fill_next_bar_only: bool = True  # enforce anti-lookahead
    limit_expiry_bars: int = 6              # how many bars a LIMIT can wait before cancel (conservative)

    # meta
    random_seed: int = 7

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BacktestResult:
    config: Dict[str, Any]
    overrides: Dict[str, Any]
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    cycle_log: pd.DataFrame
    metrics: Dict[str, Any]


# -----------------------------
# Helpers
# -----------------------------
def deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _get_cfg(cfg: Dict[str, Any], dotted: str, default=None):
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _normalize_signal(sig: Any) -> str:
    s = str(sig or "NONE").upper().strip()
    if s in ("BUY", "SELL"):
        return s
    return "NONE"


def _pnl_usd_gold(side: str, entry: float, exit_: float, lot: float, contract_size: float = 100.0) -> float:
    """
    For spot gold CFD/FX, common approximation:
      PnL = (exit - entry) * contract_size * lot   for BUY
      PnL = (entry - exit) * contract_size * lot   for SELL
    """
    if side == "BUY":
        return (exit_ - entry) * contract_size * lot
    return (entry - exit_) * contract_size * lot


def _apply_slippage(price: float, side: str, is_entry: bool, slippage: float) -> float:
    """
    Conservative slippage:
      Entry: BUY worse -> +slip ; SELL worse -> -slip
      Exit : BUY worse -> -slip (sell) ; SELL worse -> +slip (buy to cover)
    """
    if slippage <= 0:
        return price
    if is_entry:
        return price + slippage if side == "BUY" else price - slippage
    return price - slippage if side == "BUY" else price + slippage


def _bar_hit(side: str, h: float, l: float, sl: float, tp: float) -> Tuple[Optional[str], Optional[float]]:
    """
    Determine if TP/SL is hit inside a bar. Worst-case ordering when both hit in same bar.
    """
    if side == "BUY":
        sl_hit = l <= sl
        tp_hit = h >= tp
        if sl_hit and tp_hit:
            return "SL", sl  # worst-case
        if sl_hit:
            return "SL", sl
        if tp_hit:
            return "TP", tp
        return None, None
    else:
        sl_hit = h >= sl
        tp_hit = l <= tp
        if sl_hit and tp_hit:
            return "SL", sl
        if sl_hit:
            return "SL", sl
        if tp_hit:
            return "TP", tp
        return None, None


def _limit_touched(side: str, h: float, l: float, limit_price: float) -> bool:
    """
    Conservative limit fill check:
      BUY LIMIT fills if low <= limit_price
      SELL LIMIT fills if high >= limit_price
    """
    if side == "BUY":
        return l <= limit_price
    return h >= limit_price


# -----------------------------
# Main engine
# -----------------------------
class NDSBacktester:
    def __init__(
        self,
        bot_config: Dict[str, Any],
        bt_cfg: Optional[BacktestConfig] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ):
        self.base_bot_config = dict(bot_config or {})
        self.overrides = overrides or {}
        self.bot_config = deep_update(self.base_bot_config, self.overrides)

        self.bt = bt_cfg or BacktestConfig(
            symbol=_get_cfg(self.bot_config, "trading_settings.SYMBOL", "XAUUSD!"),
            timeframe=_get_cfg(self.bot_config, "trading_settings.TIMEFRAME", "M15"),
            bars_to_fetch=int(_get_cfg(self.bot_config, "trading_settings.BARS_TO_FETCH", 3500) or 3500),
            starting_equity=float(self.bot_config.get("ACCOUNT_BALANCE", 1000.0) or 1000.0),
            max_positions=int(_get_cfg(self.bot_config, "trading_rules.MAX_POSITIONS", 5) or 5),
            allow_multiple_positions=bool(_get_cfg(self.bot_config, "trading_rules.ALLOW_MULTIPLE_POSITIONS", True)),
            min_candles_between_trades=int(_get_cfg(self.bot_config, "trading_rules.MIN_CANDLES_BETWEEN_TRADES", 10) or 10),
            min_time_between_trades_minutes=int(_get_cfg(self.bot_config, "trading_rules.MIN_TIME_BETWEEN_TRADES_MINUTES", 150) or 150),
            daily_max_trades=int(_get_cfg(self.bot_config, "trading_rules.DAILY_MAX_TRADES", 40) or 40),
            max_daily_risk_percent=float(_get_cfg(self.bot_config, "risk_settings.MAX_DAILY_RISK_PERCENT", 6.0) or 6.0),
            spread=float(_get_cfg(self.bot_config, "trading_settings.GOLD_SPECIFICATIONS.TYPICAL_SPREAD", 0.25) or 0.25),
            slippage=float(_get_cfg(self.bot_config, "trading_settings.GOLD_SPECIFICATIONS.TYPICAL_SLIPPAGE", 0.10) or 0.10),
        )

        self.rng = np.random.default_rng(self.bt.random_seed)

        # RiskManager uses internal config.settings; finalize_order also accepts config dict
        self.risk_manager = ScalpingRiskManager(overrides=None)

        self.positions: List[Position] = []
        self.pending: List[PendingOrder] = []

        self.last_trade_time: Optional[pd.Timestamp] = None
        self.last_trade_bar_index: Optional[int] = None

        self.equity = float(self.bt.starting_equity)
        self.balance = float(self.bt.starting_equity)

        self._daily_trades = 0
        self._daily_pnl = 0.0
        self._daily_risk_used = 0.0  # approximate using risk_amount_usd
        self._current_day: Optional[pd.Timestamp] = None

        # caches for outputs
        self._equity_rows: List[Dict[str, Any]] = []
        self._cycle_rows: List[Dict[str, Any]] = []
        self._trade_rows: List[Dict[str, Any]] = []

        # gold specs
        self.contract_size = float(_get_cfg(self.bot_config, "trading_settings.GOLD_SPECIFICATIONS.CONTRACT_SIZE", 100) or 100)

    def _reset_daily(self, day: pd.Timestamp):
        self._daily_trades = 0
        self._daily_pnl = 0.0
        self._daily_risk_used = 0.0
        self._current_day = day

    def _can_open_new(self, now: pd.Timestamp, bar_index: int) -> Tuple[bool, str]:
        # Note: pending orders count as "intent" but do not consume max_positions until filled.
        if not self.bt.allow_multiple_positions and (self.positions or self.pending):
            return False, "WAIT_FOR_CLOSE_BEFORE_NEW_TRADE"

        if len(self.positions) >= self.bt.max_positions:
            return False, "MAX_POSITIONS"

        if self._daily_trades >= self.bt.daily_max_trades:
            return False, "DAILY_MAX_TRADES"

        if self.last_trade_bar_index is not None:
            if (bar_index - self.last_trade_bar_index) < self.bt.min_candles_between_trades:
                return False, "MIN_CANDLES_BETWEEN_TRADES"

        if self.last_trade_time is not None:
            delta_min = (now - self.last_trade_time).total_seconds() / 60.0
            if delta_min < self.bt.min_time_between_trades_minutes:
                return False, "MIN_TIME_BETWEEN_TRADES_MINUTES"

        # daily risk cap (approx using planned risk)
        if self.bt.max_daily_risk_percent is not None and self.bt.max_daily_risk_percent > 0:
            cap = self.balance * (self.bt.max_daily_risk_percent / 100.0)
            if self._daily_risk_used >= cap:
                return False, "MAX_DAILY_RISK_PERCENT"

        return True, "OK"

    def _synthesize_live_from_close(self, close_price: float, now: pd.Timestamp) -> Dict[str, Any]:
        """
        Live snapshot used by RiskManager. For anti-lookahead:
        - In backtest we treat this as the snapshot at the *end* of decision bar i.
        - Execution (fill) happens on bar i+1.
        """
        half = float(self.bt.spread) / 2.0
        bid = float(close_price) - half
        ask = float(close_price) + half
        return {
            "bid": bid,
            "ask": ask,
            "last": float(close_price),
            "spread": float(self.bt.spread),
            "source": "backtest_synth",
            "timestamp": pd.Timestamp(now).to_pydatetime().isoformat(),
        }

    def _close_position(self, pos: Position, now: pd.Timestamp, bar_index: int, exit_price: float, reason: str):
        exit_exec = _apply_slippage(exit_price, pos.side, is_entry=False, slippage=self.bt.slippage)
        pnl = _pnl_usd_gold(pos.side, pos.entry_price, exit_exec, pos.lot, contract_size=self.contract_size)

        # commissions (close side; open side was applied at fill time)
        pnl -= float(self.bt.commission_per_lot) * float(pos.lot)

        pos.close_time = now
        pos.close_bar_index = bar_index
        pos.close_price = float(exit_exec)
        pos.exit_reason = reason
        pos.pnl_usd = float(pnl)
        pos.duration_bars = int(bar_index - pos.open_bar_index)

        self.balance += float(pnl)
        self.equity = self.balance

        self._daily_pnl += float(pnl)

        self._trade_rows.append(asdict(pos))

    def _fill_market_order_next_bar(self, po: PendingOrder, now: pd.Timestamp, bar_index: int, bar_open: float):
        """
        Force market fill on bar_open (i+1). Conservative:
        - BUY fills at ask (open + half spread) + slippage
        - SELL fills at bid (open - half spread) - slippage
        """
        half = float(self.bt.spread) / 2.0
        if po.side == "BUY":
            raw_entry = float(bar_open) + half
        else:
            raw_entry = float(bar_open) - half

        entry_exec = _apply_slippage(raw_entry, po.side, is_entry=True, slippage=self.bt.slippage)

        # apply open-side commission
        open_comm = float(self.bt.commission_per_lot) * float(po.lot)

        # Create position
        pos = Position(
            id=po.id,
            side=po.side,
            symbol=po.symbol,
            open_time=now,
            open_bar_index=bar_index,
            entry_price=float(entry_exec),
            stop_loss=float(po.stop_loss),
            take_profit=float(po.take_profit),
            lot=float(po.lot),
            order_type="MARKET",
            confidence=float(po.confidence),
            score=float(po.score),
            rr=float(po.rr),
            session=str(po.session),
            deviation_pips=float(po.deviation_pips),
            notes=list(po.notes) + [f"filled_market_on_bar_open idx={bar_index}"],
        )
        self.positions.append(pos)

        # debit open commission immediately
        self.balance -= open_comm
        self.equity = self.balance

        self.last_trade_time = now
        self.last_trade_bar_index = bar_index
        self._daily_trades += 1

    def _try_fill_limit_order(self, po: PendingOrder, now: pd.Timestamp, bar_index: int, h: float, l: float) -> bool:
        """
        Fill LIMIT if touched by current bar.
        Conservative: fill at limit price with adverse slippage.
        """
        if po.limit_price is None:
            return False

        if not _limit_touched(po.side, h=h, l=l, limit_price=float(po.limit_price)):
            return False

        entry_exec = _apply_slippage(float(po.limit_price), po.side, is_entry=True, slippage=self.bt.slippage)

        # apply open-side commission
        open_comm = float(self.bt.commission_per_lot) * float(po.lot)

        pos = Position(
            id=po.id,
            side=po.side,
            symbol=po.symbol,
            open_time=now,
            open_bar_index=bar_index,
            entry_price=float(entry_exec),
            stop_loss=float(po.stop_loss),
            take_profit=float(po.take_profit),
            lot=float(po.lot),
            order_type="LIMIT",
            confidence=float(po.confidence),
            score=float(po.score),
            rr=float(po.rr),
            session=str(po.session),
            deviation_pips=float(po.deviation_pips),
            notes=list(po.notes) + [f"filled_limit idx={bar_index}"],
        )
        self.positions.append(pos)

        self.balance -= open_comm
        self.equity = self.balance

        self.last_trade_time = now
        self.last_trade_bar_index = bar_index
        self._daily_trades += 1
        return True

    def run(self, df: pd.DataFrame) -> BacktestResult:
        """
        Anti-lookahead implementation notes:
        - We assume df is time-sorted ascending.
        - We evaluate exits on current bar for positions opened in prior bars.
        - We decide new orders on bar i using history up to i (inclusive).
        - We execute MARKET orders on bar i+1 open.
        - We place LIMIT orders on bar i and fill them only if touched by bars > i.
        """
        if df is None or df.empty:
            raise ValueError("OHLCV dataframe is empty.")

        df = df.copy()

        # Normalize time index
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            df = df.sort_values("time").reset_index(drop=True)
            df = df.set_index("time")
        else:
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

        # Normalize volume
        if "volume" not in df.columns:
            if "tick_volume" in df.columns:
                df["volume"] = df["tick_volume"]
            else:
                df["volume"] = 0

        required = {"open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        times = df.index.to_list()
        n = len(df)

        # We need at least (warmup + 2) bars because decisions execute on i+1
        if n < (self.bt.warmup_bars + 2):
            raise ValueError(f"Not enough bars for warmup+execution. bars={n}, warmup={self.bt.warmup_bars}")

        # Walk-forward loop: i is decision bar, i+1 is execution bar
        # We also process exits & pending fills on each bar.
        for i in range(0, n):
            now = times[i]

            # daily reset
            day = pd.Timestamp(now.date())
            if self._current_day is None or day != self._current_day:
                self._reset_daily(day)

            # current bar OHLC
            o = float(df["open"].iloc[i])
            h = float(df["high"].iloc[i])
            l = float(df["low"].iloc[i])
            c = float(df["close"].iloc[i])

            # 1) First, process fills of pending LIMITs on this bar (but only if bar_index > created_bar_index)
            still_pending: List[PendingOrder] = []
            for po in self.pending:
                if i <= po.created_bar_index:
                    still_pending.append(po)
                    continue

                # expiry
                if i > po.expires_bar_index:
                    # cancel silently; optionally log it in cycle_log
                    continue

                if po.order_type == "LIMIT":
                    filled = self._try_fill_limit_order(po, now=now, bar_index=i, h=h, l=l)
                    if not filled:
                        still_pending.append(po)
                else:
                    # MARKET orders are forced fill only on next bar open; handled below.
                    still_pending.append(po)
            self.pending = still_pending

            # 2) Process exits for open positions using this bar's high/low (intra-bar)
            still_open: List[Position] = []
            for pos in self.positions:
                hit, level = _bar_hit(pos.side, h=h, l=l, sl=pos.stop_loss, tp=pos.take_profit)
                if hit:
                    self._close_position(pos, now, i, float(level), hit)
                    continue

                # optional timeout
                timeout_min = _get_cfg(self.bot_config, "risk_manager_config.POSITION_TIMEOUT_MINUTES", None)
                if timeout_min:
                    bars_per_min = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60}
                    tf = str(self.bt.timeframe).upper()
                    bar_minutes = bars_per_min.get(tf, 15)
                    max_bars = int(math.ceil(float(timeout_min) / float(bar_minutes)))
                    if (i - pos.open_bar_index) >= max_bars:
                        self._close_position(pos, now, i, c, "TIMEOUT")
                        continue

                still_open.append(pos)
            self.positions = still_open

            # 3) Force-fill MARKET pending orders on THIS bar open, but only if created at i-1
            # (anti-lookahead: no same-bar fills, no skipping)
            market_pending_keep: List[PendingOrder] = []
            for po in self.pending:
                if po.order_type != "MARKET":
                    market_pending_keep.append(po)
                    continue

                # fill only if this bar is exactly next bar after creation
                if i == po.created_bar_index + 1:
                    self._fill_market_order_next_bar(po, now=now, bar_index=i, bar_open=o)
                else:
                    # If missed the next bar (shouldn't happen), cancel to avoid hidden leakage.
                    # Conservative: cancel.
                    continue
            self.pending = market_pending_keep

            # 4) Record equity at each bar
            self._equity_rows.append(
                {
                    "time": now,
                    "equity": self.equity,
                    "balance": self.balance,
                    "open_positions": len(self.positions),
                    "pending_orders": len(self.pending),
                    "daily_trades": self._daily_trades,
                    "daily_pnl": self._daily_pnl,
                }
            )

            # 5) If not enough bars for warmup, skip decision
            if i < self.bt.warmup_bars:
                continue

            # 6) If i is last bar, we cannot create decisions (no i+1 to execute)
            if i >= n - 1:
                break

            # 7) Analysis window up to current bar (inclusive) ONLY
            window = df.iloc[max(0, i - self.bt.bars_to_fetch + 1): i + 1].copy()

            # Analyzer expects a "time" column in some paths; we provide reset_index.
            # Important: This reset_index contains no future rows beyond i.
            entry_factor = float(_get_cfg(self.bot_config, "technical_settings.ENTRY_FACTOR", 0.25) or 0.25)

            try:
                raw = analyze_gold_market(
                    dataframe=window.reset_index(),
                    timeframe=self.bt.timeframe,
                    entry_factor=entry_factor,
                    config=self.bot_config,
                    scalping_mode=True,
                    analysis_only=True,  # important: analyzer should not rely on execution side-effects
                )
                result = raw if isinstance(raw, dict) else (raw.__dict__ if hasattr(raw, "__dict__") else {})
            except Exception as e:
                result = {
                    "signal": "NONE",
                    "confidence": 0.0,
                    "score": 50.0,
                    "error": True,
                    "reasons": [str(e)],
                    "context": {},
                }

            analyzer_signal = _normalize_signal(result.get("signal"))
            score = float(result.get("score", 0.0) or 0.0)
            confidence = float(result.get("confidence", 0.0) or 0.0)
            min_conf = float(_get_cfg(self.bot_config, "technical_settings.SCALPING_MIN_CONFIDENCE", 38) or 38)

            session = str(result.get("context", {}).get("session_analysis", {}).get("current_session", "UNKNOWN"))

            # live snapshot from CLOSE of current bar i (known at decision time)
            live = self._synthesize_live_from_close(c, now=now)

            decision: Dict[str, Any] = {
                "cycle": i,
                "time": now,
                "analyzer_signal": analyzer_signal,
                "score": score,
                "confidence": confidence,
                "min_conf": min_conf,
                "close": c,
                "open_positions": len(self.positions),
                "pending_orders": len(self.pending),
            }

            # Bot-level gating similar to bot.py
            if analyzer_signal not in ("BUY", "SELL"):
                decision["final_signal"] = "NONE"
                decision["reject_reason"] = "ANALYZER_NONE"
                self._cycle_rows.append(decision)
                continue

            if confidence < min_conf:
                decision["final_signal"] = "NONE"
                decision["reject_reason"] = "CONF_TOO_LOW"
                self._cycle_rows.append(decision)
                continue

            can_open, gate_reason = self._can_open_new(now, i)
            if not can_open:
                decision["final_signal"] = "NONE"
                decision["reject_reason"] = gate_reason
                self._cycle_rows.append(decision)
                continue

            # finalize order (market/limit, deviation, rr, lot sizing) using your RiskManager
            order = self.risk_manager.finalize_order(
                analysis=result,
                live=live,
                symbol=self.bt.symbol,
                config=self.bot_config,
            )

            order_d = order if isinstance(order, dict) else (order.__dict__ if hasattr(order, "__dict__") else {})

            # RiskManager fields vary by version; support common keys
            allowed = bool(order_d.get("is_trade_allowed", order_d.get("allowed", False)))

            final_signal = _normalize_signal(order_d.get("signal", analyzer_signal))
            order_type = str(order_d.get("order_type", "MARKET")).upper()

            decision["final_signal"] = final_signal
            decision["order_type"] = order_type
            decision["is_trade_allowed"] = allowed
            decision["reject_reason"] = order_d.get("reject_reason")
            decision["deviation_pips"] = float(order_d.get("deviation_pips", 0.0) or 0.0)
            decision["rr_ratio"] = float(order_d.get("rr_ratio", 0.0) or 0.0)
            self._cycle_rows.append(decision)

            if not allowed or final_signal not in ("BUY", "SELL"):
                continue

            # Extract order levels (support alternative key names)
            entry_price = float(order_d.get("entry_price", order_d.get("planned_entry", c)) or c)
            sl = float(order_d.get("stop_loss", order_d.get("sl_price", order_d.get("sl", 0.0))) or 0.0)
            tp = float(order_d.get("take_profit", order_d.get("tp_price", order_d.get("tp", 0.0))) or 0.0)
            lot = float(order_d.get("lot_size", order_d.get("lot", 0.0)) or 0.0)

            # Guard against invalid orders
            if lot <= 0 or sl <= 0 or tp <= 0:
                continue

            notes = list(order_d.get("decision_notes") or [])

            po_id = f"BT_{now.strftime('%Y%m%d_%H%M%S')}_{i}"
            expires = i + int(self.bt.limit_expiry_bars)

            # MARKET: must fill on i+1 open
            # LIMIT : wait until touched, up to expiry bars
            if order_type.startswith("LIMIT"):
                pending = PendingOrder(
                    id=po_id,
                    created_time=now,
                    created_bar_index=i,
                    side=final_signal,
                    symbol=self.bt.symbol,
                    order_type="LIMIT",
                    limit_price=float(entry_price),
                    stop_loss=float(sl),
                    take_profit=float(tp),
                    lot=float(lot),
                    confidence=float(confidence),
                    score=float(score),
                    rr=float(order_d.get("rr_ratio", 0.0) or 0.0),
                    session=session,
                    deviation_pips=float(order_d.get("deviation_pips", 0.0) or 0.0),
                    notes=notes + ["placed_limit_from_decision_bar"],
                    expires_bar_index=expires,
                )
                self.pending.append(pending)
            else:
                pending = PendingOrder(
                    id=po_id,
                    created_time=now,
                    created_bar_index=i,
                    side=final_signal,
                    symbol=self.bt.symbol,
                    order_type="MARKET",
                    limit_price=None,
                    stop_loss=float(sl),
                    take_profit=float(tp),
                    lot=float(lot),
                    confidence=float(confidence),
                    score=float(score),
                    rr=float(order_d.get("rr_ratio", 0.0) or 0.0),
                    session=session,
                    deviation_pips=float(order_d.get("deviation_pips", 0.0) or 0.0),
                    notes=notes + ["placed_market_from_decision_bar", "fill_on_next_bar_open_only"],
                    expires_bar_index=i + 1,  # must fill on next bar only; else cancel
                )
                self.pending.append(pending)

            # approximate daily risk used = risk_amount_usd (if provided)
            risk_used = float(order_d.get("risk_amount_usd", _get_cfg(self.bot_config, "risk_settings.RISK_AMOUNT_USD", 0.0) or 0.0) or 0.0)
            self._daily_risk_used += max(0.0, risk_used)

        trades_df = pd.DataFrame(self._trade_rows)
        equity_df = pd.DataFrame(self._equity_rows).set_index("time")
        cycle_df = pd.DataFrame(self._cycle_rows).set_index("time") if self._cycle_rows else pd.DataFrame()

        metrics = compute_metrics(trades_df, equity_df, self.bt.starting_equity)

        return BacktestResult(
            config=self.bot_config,
            overrides=self.overrides,
            trades=trades_df,
            equity_curve=equity_df,
            cycle_log=cycle_df,
            metrics=metrics,
        )


# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(trades: pd.DataFrame, equity: pd.DataFrame, starting_equity: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "starting_equity": float(starting_equity),
        "ending_equity": float(equity["equity"].iloc[-1]) if not equity.empty else float(starting_equity),
        "net_pnl": 0.0,
        "total_trades": int(len(trades)) if trades is not None else 0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "max_drawdown": 0.0,
        "avg_trade_pnl": 0.0,
        "median_trade_pnl": 0.0,
        "expectancy": 0.0,
    }

    if equity is not None and not equity.empty:
        out["net_pnl"] = out["ending_equity"] - float(starting_equity)

        eq = equity["equity"].astype(float)
        roll_max = eq.cummax()
        dd = (eq - roll_max) / roll_max.replace(0, np.nan)
        out["max_drawdown"] = float(dd.min() * 100.0) if len(dd) else 0.0  # percent

    if trades is None or trades.empty:
        return out

    pnls = trades["pnl_usd"].astype(float)
    out["avg_trade_pnl"] = float(pnls.mean())
    out["median_trade_pnl"] = float(pnls.median())

    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    out["win_rate"] = float(len(wins) / len(pnls) * 100.0) if len(pnls) else 0.0

    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) else 0.0
    out["profit_factor"] = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

    # expectancy per trade
    p_win = len(wins) / len(pnls) if len(pnls) else 0.0
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(abs(losses.mean())) if len(losses) else 0.0
    out["expectancy"] = float(p_win * avg_win - (1 - p_win) * avg_loss)

    return out
