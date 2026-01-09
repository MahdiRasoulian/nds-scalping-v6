# src/trading_bot/nds/analyzer.py
"""
آنالایزر اصلی NDS برای طلا - نسخه بازنویسی شده با منطق امتیازدهی سازگار

Config keys (whitelisted via ANALYSIS_CONFIG_KEYS):
- ATR_WINDOW, ADX_WINDOW
- SCALPING_MIN_CONFIDENCE, MIN_CONFIDENCE
- MIN_RVOL_SCALPING, MIN_SESSION_WEIGHT, MIN_STRUCTURE_SCORE
- DEFAULT_TIMEFRAME, MIN_RR, ATR_BUFFER_MULTIPLIER
- ADX_OVERRIDE_THRESHOLD, ADX_OVERRIDE_PERSISTENCE_BARS, ADX_OVERRIDE_REQUIRE_BOS
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .constants import (
    ANALYSIS_CONFIG_KEYS,
    SESSION_MAPPING,
)
from .models import (
    AnalysisResult, SessionAnalysis,
    OrderBlock, FVG, MarketStructure, MarketTrend
)
from .indicators import IndicatorCalculator
from .smc import SMCAnalyzer

logger = logging.getLogger(__name__)


class GoldNDSAnalyzer:
    """
    نسخه ماژولار و بهینه‌شده آنالایزر طلا (XAUUSD) - نسخه 6.0
    """

    def __init__(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None):
        if df.empty:
            raise ValueError("DataFrame is empty. Cannot initialize analyzer.")

        self.df = df.copy()
        self.config = config or {}
        self.debug_analyzer = False

        technical_settings = (self.config or {}).get('technical_settings', {})
        sessions_config = (self.config or {}).get('sessions_config', {})
        self.GOLD_SETTINGS = technical_settings.copy()
        self.TRADING_SESSIONS = sessions_config.get('BASE_TRADING_SESSIONS', {}).copy()
        self.timeframe_specifics = technical_settings.get('TIMEFRAME_SPECIFICS', {})
        self.swing_period_map = technical_settings.get('SWING_PERIOD_MAP', {})

        self.atr: Optional[float] = None

        self._apply_custom_config()
        self._validate_dataframe()
        self.timeframe = self._detect_timeframe()
        self._apply_timeframe_settings()

        self._log_info(
            "[NDS][INIT] initialized candles=%s timeframe=%s",
            len(self.df),
            self.timeframe,
        )

    def _apply_custom_config(self) -> None:
        """اعمال تنظیمات سفارشی از config خارجی"""
        if not self.config:
            return

        analyzer_config, sessions_config = self._extract_config_payload()
        if analyzer_config:
            self._apply_analyzer_settings(analyzer_config)

        if sessions_config:
            self._apply_sessions_config(sessions_config)

    def _extract_config_payload(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """دریافت تنظیمات مجاز آنالایزر و سشن‌ها از کانفیگ اصلی"""
        analyzer_config: Dict[str, Any] = {}
        sessions_config: Dict[str, Any] = {}

        if 'ANALYZER_SETTINGS' in self.config:
            analyzer_config = self.config.get('ANALYZER_SETTINGS', {}) or {}
        elif 'technical_settings' in self.config:
            analyzer_config = self.config.get('technical_settings', {}) or {}

        if 'DEBUG_ANALYZER' in self.config and 'DEBUG_ANALYZER' not in analyzer_config:
            analyzer_config = {**analyzer_config, 'DEBUG_ANALYZER': self.config.get('DEBUG_ANALYZER')}

        if 'TRADING_SESSIONS' in self.config:
            sessions_config = self.config.get('TRADING_SESSIONS', {}) or {}
        else:
            sessions_config = self.config.get('sessions_config', {}).get('TRADING_SESSIONS', {}) or {}

        return analyzer_config, sessions_config

    def _apply_analyzer_settings(self, analyzer_config: Dict[str, Any]) -> None:
        """اعمال تنظیمات تحلیل با وایت‌لیست دقیق"""
        self.debug_analyzer = bool(analyzer_config.get('DEBUG_ANALYZER', self.debug_analyzer))

        type_map = {
            'ATR_WINDOW': int,
            'ADX_WINDOW': int,
            'SCALPING_MIN_CONFIDENCE': float,
            'MIN_CONFIDENCE': float,
            'MIN_RVOL_SCALPING': float,
            'MIN_SESSION_WEIGHT': float,
            'MIN_STRUCTURE_SCORE': float,
            'DEFAULT_TIMEFRAME': str,
            'MIN_RR': float,
            'ATR_BUFFER_MULTIPLIER': float,
            'ADX_OVERRIDE_THRESHOLD': float,
            'ADX_OVERRIDE_PERSISTENCE_BARS': int,
            'ADX_OVERRIDE_REQUIRE_BOS': bool,
        }

        validated_config: Dict[str, Any] = {}
        ignored_keys: List[str] = []

        for key, value in analyzer_config.items():
            if key == 'DEBUG_ANALYZER':
                continue
            if key not in ANALYSIS_CONFIG_KEYS:
                ignored_keys.append(key)
                continue

            target_type = type_map.get(key)
            parsed_value = None
            if target_type is None:
                if isinstance(value, (int, float, bool, str)):
                    parsed_value = value
            elif target_type is bool:
                if isinstance(value, bool):
                    parsed_value = value
                elif isinstance(value, str):
                    parsed_value = value.strip().lower() in {'1', 'true', 'yes', 'y'}
            else:
                try:
                    parsed_value = target_type(value)
                except (TypeError, ValueError):
                    parsed_value = None

            if parsed_value is None:
                self._log_debug("[NDS][INIT] ignored setting %s=%s", key, value)
                continue

            validated_config[key] = parsed_value

        if validated_config:
            self.GOLD_SETTINGS.update(validated_config)
            self._log_debug("[NDS][INIT] applied analyzer settings=%s", len(validated_config))

        if ignored_keys:
            self._log_debug("[NDS][INIT] ignored non-analysis keys=%s", sorted(set(ignored_keys)))

    def _apply_sessions_config(self, sessions_config: Dict[str, Any]) -> None:
        """اعمال تنظیمات سشن‌ها از config متمرکز"""
        for session_name, session_data in sessions_config.items():
            if not isinstance(session_data, dict):
                continue

            converted_session = {
                'start': session_data.get('start', 0),
                'end': session_data.get('end', 0),
                'weight': session_data.get('weight', 0.5)
            }
            standard_name = SESSION_MAPPING.get(session_name, session_name)

            if standard_name in self.TRADING_SESSIONS:
                self.TRADING_SESSIONS[standard_name].update(converted_session)
            else:
                self.TRADING_SESSIONS[standard_name] = converted_session

            self._log_debug("[NDS][SESSIONS] applied config %s=%s", standard_name, converted_session)

    def _log_debug(self, message: str, *args: Any) -> None:
        if self.debug_analyzer:
            logger.debug(message, *args)

    def _log_info(self, message: str, *args: Any) -> None:
        logger.info(message, *args)

    def _normalize_volatility_state(self, volatility_state: Optional[str]) -> str:
        if not volatility_state:
            return "MODERATE_VOLATILITY"
        state = str(volatility_state).upper()
        mapping = {
            "HIGH": "HIGH_VOLATILITY",
            "LOW": "LOW_VOLATILITY",
            "MODERATE": "MODERATE_VOLATILITY",
        }
        if state in mapping:
            return mapping[state]
        if state.endswith("_VOLATILITY"):
            return state
        return "MODERATE_VOLATILITY"

    def _append_reason(self, reasons: List[str], reason: str) -> None:
        reasons.append(reason)
        self._log_debug("[NDS][REASONS] %s", reason)

    def _validate_dataframe(self) -> None:
        """اعتبارسنجی DataFrame ورودی"""
        required_columns = ['time', 'open', 'high', 'low', 'close']
        missing = [col for col in required_columns if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.df['time'] = pd.to_datetime(self.df['time'], utc=True)

        if len(self.df) > 1 and self.df['time'].iloc[0] > self.df['time'].iloc[-1]:
            logger.warning("[NDS][INIT] DataFrame not sorted chronologically. Sorting...")
            self.df = self.df.sort_values('time').reset_index(drop=True)

        if 'volume' not in self.df.columns:
            self.df['volume'] = 1.0

    def _detect_timeframe(self) -> str:
        """شناسایی تایم‌فریم با استفاده از میانه اختلاف زمانی"""
        default_tf = self.GOLD_SETTINGS.get('DEFAULT_TIMEFRAME', 'M15')

        if len(self.df) < 2:
            return default_tf

        deltas = self.df['time'].diff().dt.total_seconds().dropna()
        sample = deltas.head(200)
        if sample.empty:
            return default_tf

        median = float(sample.median())
        if median <= 0:
            return default_tf

        filtered = sample[sample <= median * 3]
        median = float(filtered.median()) if not filtered.empty else median

        tf_map = {
            'M1': 60,
            'M5': 300,
            'M15': 900,
            'M30': 1800,
            'H1': 3600,
            'H4': 14400,
            'D1': 86400,
        }
        closest_tf = None
        closest_diff = None
        for tf_name, seconds in tf_map.items():
            diff = abs(median - seconds)
            if closest_diff is None or diff < closest_diff:
                closest_tf = tf_name
                closest_diff = diff

        if closest_tf and closest_diff is not None and closest_diff <= tf_map[closest_tf] * 0.1:
            return closest_tf

        return default_tf

    def _apply_timeframe_settings(self) -> None:
        """اعمال تنظیمات اختصاصی تایم‌فریم"""
        tf_settings = self.timeframe_specifics.get(self.timeframe)
        if not tf_settings:
            self._log_debug("[NDS][INIT] missing TIMEFRAME_SPECIFICS timeframe=%s", self.timeframe)
            return
        self.GOLD_SETTINGS.update(tf_settings)

    def _analyze_trading_sessions(self, volume_analysis: Dict[str, Any]) -> SessionAnalysis:
        """تحلیل جامع سشن‌های معاملاتی

        سیاست جدید:
        - current_session/weight/activity فقط کیفیت هستند.
        - is_active_session فقط وقتی False می‌شود که واقعاً untradable باشیم (market closed, data ناقص, spread غیرعادی, ...).
        """
        last_time = self.df['time'].iloc[-1]
        hour = last_time.hour

        session_info = self._is_valid_trading_session(last_time)

        # ---- RVOL (NaN-safe) ----
        rvol = volume_analysis.get('rvol', 1.0)
        try:
            rvol = float(rvol)
        except Exception:
            rvol = 1.0
        if pd.isna(rvol):
            rvol = 1.0

        volume_trend = str(volume_analysis.get('volume_trend', 'NEUTRAL') or 'NEUTRAL').upper()

        # ---- Activity (Quality only) ----
        if rvol > 1.2 or volume_trend == 'INCREASING':
            session_activity = 'HIGH'
        elif rvol < 0.8 and volume_trend == 'DECREASING':
            session_activity = 'LOW'
        else:
            session_activity = 'NORMAL'

        # ---- Determine "untradable" (Active/Inactive واقعی) ----
        # این کلیدها ممکن است در پروژه نباشند؛ اگر نباشند هیچ مشکلی نیست.
        market_status = str(volume_analysis.get("market_status", "") or "").upper()  # e.g. OPEN/CLOSED/HALTED
        data_ok = volume_analysis.get("data_ok", None)  # True/False if available

        spread = volume_analysis.get("spread", None)       # numeric if available
        max_spread = volume_analysis.get("max_spread", None)  # numeric if available

        untradable_reasons = []
        untradable = False

        # 1) invalid/parse failure from session_info (خیلی مهم)
        if not bool(session_info.get('is_valid', True)):
            untradable = True
            untradable_reasons.append("invalid_time")

        # 2) market status (optional)
        if market_status in ("CLOSED", "HALTED"):
            untradable = True
            untradable_reasons.append(f"market_status={market_status}")

        # 3) data ok flag (optional)
        if data_ok is False:
            untradable = True
            untradable_reasons.append("data_ok=False")

        # 4) spread sanity (optional)
        if spread is not None and max_spread is not None:
            try:
                if float(spread) > float(max_spread):
                    untradable = True
                    untradable_reasons.append(f"spread={float(spread):.4f}>max={float(max_spread):.4f}")
            except Exception:
                # اگر قابل تبدیل نبود، تصمیم‌گیری را به این معیار وابسته نکن
                pass

        is_active_session = (not untradable)

        analysis = SessionAnalysis(
            current_session=session_info.get('session', 'OTHER'),
            session_weight=session_info.get('weight', 0.5),
            weight=session_info.get('weight', 0.5),
            gmt_hour=hour,
            # سیاست جدید: active فقط برای untradable false می‌شود
            is_active_session=is_active_session,
            is_overlap=session_info.get('is_overlap', False),
            session_activity=session_activity,
            optimal_trading=session_info.get(
                'optimal_trading',
                session_info.get('weight', 0.5) >= 1.2
            )
        )

        # لاگ ارتقا یافته: active و دلیل untradable هم چاپ می‌شود
        try:
            reasons_str = ",".join(untradable_reasons) if untradable_reasons else "-"
            self._log_info(
                "[NDS][SESSIONS] current=%s weight=%.2f activity=%s overlap=%s active=%s untradable=%s reasons=%s rvol=%.2f trend=%s",
                analysis.current_session,
                float(analysis.weight),
                analysis.session_activity,
                bool(analysis.is_overlap),
                bool(analysis.is_active_session),
                bool(untradable),
                reasons_str,
                float(rvol),
                volume_trend,
            )
        except Exception:
            # لاگ نباید تحلیل را fail کند
            pass

        return analysis


    def _hour_in_session(self, hour: int, start: int, end: int) -> bool:
        """بررسی ساعت داخل بازه [start, end) با پشتیبانی از عبور از نیمه‌شب."""
        if start == end:
            return False
        if start < end:
            return start <= hour < end
        return hour >= start or hour < end


    def _is_valid_trading_session(self, check_time: datetime) -> Dict[str, Any]:
        """بررسی سشن معاملاتی با پشتیبانی از نام‌های جدید و قدیم

        سیاست جدید:
        - is_valid دیگر به معنی "primary session" نیست.
        - is_valid فقط یعنی timestamp معتبر/قابل‌تحلیل است (نه تعطیلی بازار).
        - تشخیص untradable در _analyze_trading_sessions انجام می‌شود (با data_ok/spread/market_status).
        """
        if not isinstance(check_time, datetime):
            try:
                check_time = pd.to_datetime(check_time)
            except (ValueError, TypeError):
                return {
                    'is_valid': False, 'is_overlap': False, 'session': 'INVALID',
                    'weight': 0.0, 'optimal_trading': False
                }

        hour = check_time.hour
        raw_sessions = self.TRADING_SESSIONS

        sessions = {}
        for config_name, data in raw_sessions.items():
            standard_name = SESSION_MAPPING.get(config_name, config_name)
            if standard_name not in sessions or data.get('weight', 0) > sessions[standard_name].get('weight', 0):
                sessions[standard_name] = data

        session_name = 'OTHER'
        session_weight = 0.5
        is_overlap = False

        def check_in_session(name: str) -> Tuple[bool, float]:
            session = sessions.get(name)
            if session and self._hour_in_session(hour, session.get('start', 0), session.get('end', 0)):
                return True, session.get('weight', 0.5)
            return False, 0.0

        in_overlap, weight = check_in_session('OVERLAP')
        if in_overlap:
            session_name = 'OVERLAP'
            session_weight = weight
            is_overlap = True
        else:
            in_london, weight = check_in_session('LONDON')
            if in_london:
                session_name = 'LONDON'
                session_weight = weight
            else:
                in_ny, weight = check_in_session('NEW_YORK')
                if in_ny:
                    session_name = 'NEW_YORK'
                    session_weight = weight
                else:
                    in_asia, weight = check_in_session('ASIA')
                    if in_asia:
                        session_name = 'ASIA'
                        session_weight = weight

        optimal_trading = session_weight >= 1.0

        return {
            # سیاست جدید: معتبر بودن timestamp
            'is_valid': True,
            'is_overlap': is_overlap,
            'session': session_name,
            'weight': session_weight,
            'hour': hour,
            'optimal_trading': optimal_trading
        }


    def generate_trading_signal(
        self,
        timeframe: str = 'M15',
        entry_factor: float = 0.5,
        scalping_mode: bool = True,
    ) -> AnalysisResult:
        """
        تولید سیگنال نهایی با ادغام سیستم تشخیص ساختار الترا پرو و تاییدیه ADX/Volume
        نسخه تحلیل‌محور: خروجی فقط ایده معاملاتی (بدون منطق اجرا/ریسک).
        """
        mode = "Scalping" if scalping_mode else "Regular"
        self._log_info("[NDS][INIT] start analysis mode=%s analysis_only=true", mode)

        try:
            atr_window = self.GOLD_SETTINGS.get('ATR_WINDOW', 14)
            self.df, atr_v = IndicatorCalculator.calculate_atr(self.df, atr_window)
            self.atr = atr_v

            current_close = float(self.df['close'].iloc[-1])

            adx_window = self.GOLD_SETTINGS.get('ADX_WINDOW', 14)
            self.df, adx_v, plus_di, minus_di, di_trend = IndicatorCalculator.calculate_adx(self.df, adx_window)

            self._log_info(
                "[NDS][INDICATORS] atr=%.2f adx=%.2f plus_di=%.2f minus_di=%.2f di_trend=%s",
                atr_v,
                adx_v,
                plus_di,
                minus_di,
                di_trend,
            )

            if scalping_mode:
                atr_short_df, _ = IndicatorCalculator.calculate_atr(self.df.copy(), 7)
                atr_short_value = float(atr_short_df['atr_7'].iloc[-1])
            else:
                atr_short_value = None

            atr_for_scoring = atr_short_value if (scalping_mode and atr_short_value) else atr_v

            volume_analysis = IndicatorCalculator.analyze_volume(self.df, 5 if scalping_mode else 20)
            volatility_state = self._normalize_volatility_state(self._determine_volatility(atr_v, atr_for_scoring))
            session_analysis = self._analyze_trading_sessions(volume_analysis)

            last_candle = self.df.iloc[-1]
            self._log_debug(
                "[NDS][INDICATORS] last_candle time=%s open=%.2f high=%.2f low=%.2f close=%.2f rvol=%.2f",
                last_candle['time'],
                last_candle['open'],
                last_candle['high'],
                last_candle['low'],
                last_candle['close'],
                float(volume_analysis.get('rvol', 1.0)),
            )

            smc = SMCAnalyzer(self.df, self.atr, self.GOLD_SETTINGS)
            swings = smc.detect_swings(timeframe)
            fvgs = smc.detect_fvgs()
            order_blocks = smc.detect_order_blocks()
            sweeps = smc.detect_liquidity_sweeps(swings)

            structure = smc.determine_market_structure(
                swings=swings,
                lookback_swings=4,
                volume_analysis=volume_analysis,
                volatility_state=volatility_state,
                adx_value=adx_v,
            )

            self._log_info("[NDS][SMC][STRUCTURE] %s", structure)

            final_structure = self._apply_adx_override(structure, adx_v, plus_di, minus_di)

            score, reasons, score_breakdown = self._calculate_scoring_system(
                structure=final_structure,
                adx_value=adx_v,
                volume_analysis=volume_analysis,
                fvgs=fvgs,
                sweeps=sweeps,
                order_blocks=order_blocks,
                current_price=current_close,
                swings=swings,
                atr_value=atr_for_scoring,
                session_analysis=session_analysis,
            )

            confidence = self._calculate_confidence(
                score,
                volatility_state,
                session_analysis,
                volume_analysis,
                scalping_mode,
                sweeps=sweeps,
            )

            signal = self._determine_signal(score, confidence, volatility_state, scalping_mode)

            self._log_info(
                "[NDS][RESULT] score=%.1f confidence=%.1f signal=%s volatility=%s",
                score,
                confidence,
                signal,
                volatility_state,
            )

            result_payload = self._build_initial_result(
                signal=signal,
                confidence=confidence,
                score=score,
                reasons=reasons,
                structure=final_structure,
                atr_value=atr_v,
                atr_short_value=atr_short_value,
                adx_value=adx_v,
                plus_di=plus_di,
                minus_di=minus_di,
                volume_analysis=volume_analysis,
                recent_range=self._calculate_recent_range(scalping_mode),
                recent_position=self._calculate_recent_position(current_close, scalping_mode),
                volatility_state=volatility_state,
                session_analysis=session_analysis,
                current_price=current_close,
                timeframe=timeframe,
                score_breakdown=score_breakdown,
                scalping_mode=scalping_mode,
            )

            result_payload = self._apply_final_filters(result_payload, scalping_mode)
            signal = result_payload.get('signal')
            reasons = result_payload.get('reasons', reasons)

            entry_price = None
            stop_loss = None
            take_profit = None
            if signal in ["BUY", "SELL"]:
                entry_idea = self._build_entry_idea(
                    signal=signal,
                    fvgs=fvgs,
                    order_blocks=order_blocks,
                    structure=final_structure,
                    atr_value=atr_for_scoring,
                    entry_factor=entry_factor,
                    current_price=current_close,
                    adx_value=adx_v,
                )
                entry_price = entry_idea.get("entry_price")
                stop_loss = entry_idea.get("stop_loss")
                take_profit = entry_idea.get("take_profit")
                if entry_idea.get("reason"):
                    reasons.append(entry_idea["reason"])

                # --- تغییر کلیدی: اگر Entry Zone نداریم، معامله را کامل کنسل کن ---
                if entry_price is None or stop_loss is None or take_profit is None:
                    self._log_info(
                        "[NDS][ENTRY_IDEA] trade skipped: missing zone (entry=%s stop=%s tp=%s) -> signal NONE",
                        entry_price,
                        stop_loss,
                        take_profit,
                    )
                    signal = "NONE"
                    entry_price = None
                    stop_loss = None
                    take_profit = None
                    reasons.append("Trade skipped: no valid entry zone (FVG/OB) and fallback not allowed.")
                    result_payload["signal"] = "NONE"

            return self._build_analysis_result(
                signal=signal,
                confidence=confidence,
                score=score,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reasons=reasons,
                context=result_payload,
                timeframe=timeframe,
                current_price=current_close,
            )

        except Exception as e:
            logger.error("[NDS][RESULT] analysis failed: %s", str(e), exc_info=True)
            return self._create_error_result(str(e), timeframe, current_close=None)

    def _calculate_recent_range(self, scalping_mode: bool) -> float:
        """محاسبه محدوده قیمت اخیر"""
        lookback_bars = 20 if scalping_mode else 96
        recent_high = float(self.df['high'].tail(lookback_bars).max())
        recent_low = float(self.df['low'].tail(lookback_bars).min())
        return recent_high - recent_low

    def _calculate_recent_position(self, current_price: float, scalping_mode: bool) -> float:
        """محاسبه موقعیت قیمت در محدوده اخیر"""
        lookback_bars = 20 if scalping_mode else 96
        recent_high = float(self.df['high'].tail(lookback_bars).max())
        recent_low = float(self.df['low'].tail(lookback_bars).min())
        recent_range = recent_high - recent_low
        return (current_price - recent_low) / recent_range if recent_range > 0 else 0.5

    def _normalize_structure_score(self, raw_score: Optional[float]) -> float:
        """نرمال‌سازی structure_score به بازه 0..100"""
        if raw_score is None:
            return 0.0
        score = float(raw_score)
        normalized = score
        if score <= 1.05:
            normalized = score * 100
        elif score <= 10:
            normalized = score * 10
        else:
            normalized = score
        normalized = max(0.0, min(100.0, float(normalized)))
        self._log_debug(
            "[NDS][SCORE] structure_score raw=%.2f normalized=%.2f",
            score,
            normalized,
        )
        return normalized

    def _bounded(self, value: float, minimum: float = -1.0, maximum: float = 1.0) -> float:
        return max(minimum, min(maximum, value))

    def _min_stop_distance(self, atr_value: float) -> float:
        """Minimum logical SL distance from entry (in price units, e.g., USD for XAUUSD).

        This is a safety guardrail to prevent inverted/degenerate SL placement due to
        swing-anchor selection or noisy structure levels.
        """
        try:
            k = float(self.GOLD_SETTINGS.get("MIN_STOP_ATR_MULT", 0.35))
        except Exception:
            k = 0.35
        return max(0.01, float(atr_value) * k)


    def _apply_adx_override(
        self,
        structure: MarketStructure,
        adx_value: float,
        plus_di: float,
        minus_di: float,
    ) -> MarketStructure:
        """بهبود منطق override ADX با تاییدیه اضافی"""
        threshold = float(self.GOLD_SETTINGS.get('ADX_OVERRIDE_THRESHOLD', 30.0))
        persistence_bars = int(self.GOLD_SETTINGS.get('ADX_OVERRIDE_PERSISTENCE_BARS', 3))
        require_bos = bool(self.GOLD_SETTINGS.get('ADX_OVERRIDE_REQUIRE_BOS', True))

        if adx_value <= threshold or structure.trend.value != "RANGING":
            return structure

        dominance = None
        if plus_di > minus_di:
            dominance = "BULLISH"
        elif minus_di > plus_di:
            dominance = "BEARISH"

        if dominance is None:
            return structure

        di_persist = False
        if 'plus_di' in self.df.columns and 'minus_di' in self.df.columns:
            recent = self.df[['plus_di', 'minus_di']].tail(max(persistence_bars, 1))
            if dominance == "BULLISH":
                di_persist = bool((recent['plus_di'] > recent['minus_di']).all())
            else:
                di_persist = bool((recent['minus_di'] > recent['plus_di']).all())

        bos_confirmed = structure.bos in {"BULLISH_BOS", "BEARISH_BOS"}
        if require_bos and not bos_confirmed:
            self._log_debug("[NDS][FILTER] ADX override blocked: BOS required")
            return structure

        if not di_persist:
            self._log_debug("[NDS][FILTER] ADX override blocked: DI persistence not met")
            return structure

        new_trend = MarketTrend.UPTREND if dominance == "BULLISH" else MarketTrend.DOWNTREND
        self._log_info("[NDS][FILTER] ADX override applied trend=%s", new_trend.value)
        return MarketStructure(
            trend=new_trend,
            bos=structure.bos,
            choch=structure.choch,
            last_high=structure.last_high,
            last_low=structure.last_low,
            current_price=structure.current_price,
            range_width=structure.range_width,
            range_mid=structure.range_mid,
            bos_choch_confidence=structure.bos_choch_confidence,
            volume_analysis=structure.volume_analysis,
            volatility_state=structure.volatility_state,
            adx_value=adx_value,
            structure_score=structure.structure_score,
        )

    def _determine_volatility(self, atr_long: float, atr_short: float) -> str:
        """تعیین وضعیت نوسان"""
        volatility_ratio = atr_short / atr_long if atr_long > 0 else 1.0

        if volatility_ratio > 1.3:
            return "HIGH_VOLATILITY"
        if volatility_ratio > 0.8:
            return "MODERATE_VOLATILITY"
        return "LOW_VOLATILITY"

    def _calculate_scoring_system(
        self,
        structure: MarketStructure,
        adx_value: float,
        volume_analysis: Dict[str, Any],
        fvgs: List[FVG],
        sweeps: List[Any],
        order_blocks: List[OrderBlock],
        current_price: float,
        swings: List[Any],
        atr_value: float,
        session_analysis: SessionAnalysis,
    ) -> Tuple[float, List[str], Dict[str, Any]]:
        """سیستم امتیازدهی سازگار با وزن‌های مشخص"""
        settings = self.GOLD_SETTINGS
        weights = {
            'structure': 30,
            'trend': 20,
            'fvg': 15,
            'sweeps': 10,
            'order_blocks': 10,
            'volume_session': 15,
        }

        reasons: List[str] = []
        breakdown: Dict[str, Any] = {
            'weights': weights,
            'sub_scores': {},
            'raw_signals': {},
            'modifiers': {},
        }

        structure_score = self._normalize_structure_score(getattr(structure, 'structure_score', 0.0))
        bos_component = 0.0
        choch_component = 0.0
        if structure.bos == "BULLISH_BOS":
            bos_component = 1.0
            self._append_reason(reasons, "✅ Bullish BOS")
        elif structure.bos == "BEARISH_BOS":
            bos_component = -1.0
            self._append_reason(reasons, "🔻 Bearish BOS")

        if structure.choch == "BULLISH_CHOCH":
            choch_component = 1.0
            self._append_reason(reasons, "✅ Bullish CHOCH")
        elif structure.choch == "BEARISH_CHOCH":
            choch_component = -1.0
            self._append_reason(reasons, "🔻 Bearish CHOCH")

        structure_component = self._bounded((structure_score - 50.0) / 50.0)
        structure_sub = self._bounded(
            0.45 * bos_component + 0.35 * choch_component + 0.2 * structure_component
        )
        breakdown['sub_scores']['structure'] = structure_sub
        breakdown['raw_signals']['structure_score'] = structure_score

        trend_dir = 0.0
        if structure.trend.value == "UPTREND":
            trend_dir = 1.0
            self._append_reason(reasons, f"📈 Uptrend (ADX: {adx_value:.1f})")
        elif structure.trend.value == "DOWNTREND":
            trend_dir = -1.0
            self._append_reason(reasons, f"📉 Downtrend (ADX: {adx_value:.1f})")

        trend_strength = min(1.0, max(0.0, adx_value / 40.0))
        rvol = float(volume_analysis.get('rvol', 1.0))
        rvol_strength = min(1.0, max(0.0, (rvol - 0.8) / 1.2))
        trend_sub = self._bounded(trend_dir * (0.7 * trend_strength + 0.3 * rvol_strength))
        breakdown['sub_scores']['trend'] = trend_sub
        breakdown['raw_signals']['adx'] = adx_value
        breakdown['raw_signals']['rvol'] = rvol

        fvg_sub = 0.0
        valid_fvgs = [f for f in fvgs if not f.filled]
        recent_fvgs = [f for f in valid_fvgs if (len(self.df) - 1 - f.index) <= 10]
        fvg_values: List[float] = []
        for fvg in recent_fvgs:
            if fvg.bottom <= current_price <= fvg.top:
                size_ratio = fvg.size / atr_value if atr_value > 0 else 0.0
                size_score = min(1.0, size_ratio / 2.0)
                strength_score = min(1.0, fvg.strength / 2.0)
                base_score = min(1.0, 0.5 * size_score + 0.5 * strength_score)
                sign = 1.0 if fvg.type.value == "BULLISH_FVG" else -1.0
                alignment = 1.0
                if (sign > 0 and structure.trend.value == "DOWNTREND") or (
                    sign < 0 and structure.trend.value == "UPTREND"
                ):
                    alignment = 0.6
                fvg_value = self._bounded(sign * base_score * alignment)
                fvg_values.append(fvg_value)
                breakdown['raw_signals'][f"fvg_{fvg.index}"] = {
                    'type': fvg.type.value,
                    'size': fvg.size,
                    'strength': fvg.strength,
                    'score': fvg_value,
                }
                self._append_reason(
                    reasons,
                    f"{'🟢' if sign > 0 else '🔴'} {fvg.type.value} (Size: ${fvg.size:.2f})",
                )
            else:
                self._log_debug(
                    "[NDS][SMC][FVG] skipped index=%s price=%.2f range=(%.2f,%.2f)",
                    fvg.index,
                    current_price,
                    fvg.bottom,
                    fvg.top,
                )

        if fvg_values:
            fvg_sub = self._bounded(sum(fvg_values) / len(fvg_values))
        breakdown['sub_scores']['fvg'] = fvg_sub

        sweep_values: List[float] = []
        for idx, sweep in enumerate(sweeps[-3:]):
            sweep_type = getattr(sweep, 'type', 'UNKNOWN')
            sweep_penetration = float(getattr(sweep, 'penetration', 0.0))
            sweep_strength = float(getattr(sweep, 'strength', 1.0))
            sign = 1.0 if sweep_type == 'BULLISH_SWEEP' else -1.0
            penetration_ratio = sweep_penetration / atr_value if atr_value > 0 else 0.0
            penetration_score = min(1.0, penetration_ratio / 1.5)
            strength_score = min(1.0, sweep_strength / 2.0)
            sweep_value = self._bounded(sign * (0.6 * penetration_score + 0.4 * strength_score))
            sweep_values.append(sweep_value)
            breakdown['raw_signals'][f"sweep_{idx}"] = {
                'type': sweep_type,
                'penetration': sweep_penetration,
                'strength': sweep_strength,
                'score': sweep_value,
            }
            self._append_reason(
                reasons,
                f"{'✅' if sign > 0 else '🔻'} {sweep_type} (Penetration: ${sweep_penetration:.2f})",
            )

        sweep_sub = self._bounded(sum(sweep_values) / len(sweep_values)) if sweep_values else 0.0
        breakdown['sub_scores']['sweeps'] = sweep_sub

        ob_values: List[float] = []
        recent_obs = order_blocks[-5:] if order_blocks else []
        for idx, ob in enumerate(recent_obs):
            ob_mid = getattr(ob, 'mid', (ob.high + ob.low) / 2)
            distance_atr = abs(current_price - ob_mid) / atr_value if atr_value > 0 else 999.0
            if distance_atr > 1.0:
                self._log_debug(
                    "[NDS][SMC][OB] skipped type=%s distance_atr=%.2f",
                    ob.type,
                    distance_atr,
                )
                continue
            sign = 1.0 if ob.type == 'BULLISH_OB' else -1.0
            distance_score = max(0.0, 1.0 - distance_atr)
            strength_score = min(1.0, ob.strength / 2.0)
            ob_value = self._bounded(sign * (0.6 * strength_score + 0.4 * distance_score))
            ob_values.append(ob_value)
            breakdown['raw_signals'][f"ob_{idx}"] = {
                'type': ob.type,
                'strength': ob.strength,
                'distance_atr': distance_atr,
                'score': ob_value,
            }
            self._append_reason(
                reasons,
                f"{'🟢' if sign > 0 else '🔴'} {ob.type} (Strength: {ob.strength:.1f})",
                )

        ob_sub = self._bounded(sum(ob_values) / len(ob_values)) if ob_values else 0.0
        breakdown['sub_scores']['order_blocks'] = ob_sub

        session_weight = float(session_analysis.weight)
        rvol_component = self._bounded((rvol - 1.0) / 1.0)
        session_component = self._bounded((session_weight - 0.8) / 0.4)
        volume_session_sub = self._bounded(0.6 * rvol_component + 0.4 * session_component)
        breakdown['sub_scores']['volume_session'] = volume_session_sub
        breakdown['modifiers']['session_weight'] = session_weight

        total_weighted = sum(
            weights[name] * breakdown['sub_scores'][name]
            for name in weights
        )

        # --- Structure sanity dampening (prevents fake trends / weak confirmations) ---
        # اگر هیچ BOS/CHOCH نداریم و ADX هم پایین است، اجازه ندهیم امتیاز به ناحیه سیگنال نزدیک شود.
        try:
            sanity_adx_max = float(settings.get('SANITY_ADX_MAX', 18.0))
            sanity_damp = float(settings.get('SANITY_NO_BOS_CHOCH_DAMP', 0.55))
            sanity_min_structure = float(settings.get('SANITY_MIN_STRUCTURE_SCORE', 35.0))
        except Exception:
            sanity_adx_max, sanity_damp, sanity_min_structure = 18.0, 0.55, 35.0

        no_confirm = (getattr(structure, "bos", "NONE") == "NONE" and getattr(structure, "choch", "NONE") == "NONE")
        weak_adx = adx_value < sanity_adx_max
        weak_struct = structure_score < sanity_min_structure

        if no_confirm and weak_adx:
            applied_damp = sanity_damp * (0.8 if weak_struct else 1.0)  # اگر ساختار هم ضعیف است، کمی سخت‌تر
            total_weighted *= applied_damp
            breakdown.setdefault('modifiers', {})
            breakdown['modifiers']['sanity_no_bos_choch'] = {
                'applied': True,
                'adx': round(adx_value, 2),
                'structure_score': round(structure_score, 2),
                'damp': round(applied_damp, 3),
                'reason': 'no BOS/CHOCH with low ADX',
            }
            self._append_reason(
                reasons,
                f"⚠️ Weak confirmation (no BOS/CHOCH, ADX: {adx_value:.1f}) → score dampened"
            )
            self._log_debug(
                "[NDS][SANITY] dampened total_weighted by %.3f (no_confirm=%s weak_adx=%s weak_struct=%s)",
                applied_damp,
                no_confirm,
                weak_adx,
                weak_struct,
            )
        else:
            breakdown.setdefault('modifiers', {})
            breakdown['modifiers']['sanity_no_bos_choch'] = {
                'applied': False,
                'adx': round(adx_value, 2),
                'structure_score': round(structure_score, 2),
            }

        score = 50.0 + 0.5 * total_weighted

        score = max(0.0, min(100.0, score))
        breakdown['summary'] = {
            'total_weighted': total_weighted,
            'score': score,
            'structure_score': structure_score,
        }

        self._log_debug(
            "[NDS][SCORE] total_weighted=%.2f score=%.2f sub_scores=%s",
            total_weighted,
            score,
            breakdown['sub_scores'],
        )

        # INFO-level scoring trace (single-line, parse-friendly)
        try:
            contribs = {k: float(weights[k]) * float(breakdown['sub_scores'][k]) for k in weights}
            self._log_info(
                "[NDS][SCORE_BREAKDOWN] structure_sub=%.4f (w=%.2f c=%.4f) "
                "trend_sub=%.4f (w=%.2f c=%.4f) "
                "fvg_sub=%.4f (w=%.2f c=%.4f) "
                "sweeps_sub=%.4f (w=%.2f c=%.4f) "
                "ob_sub=%.4f (w=%.2f c=%.4f) "
                "volume_session_sub=%.4f (w=%.2f c=%.4f) "
                "-> total_weighted=%.4f formula=50+0.5*total clamp(0..100) score=%.2f",
                float(breakdown['sub_scores']['structure']), float(weights['structure']), float(contribs['structure']),
                float(breakdown['sub_scores']['trend']), float(weights['trend']), float(contribs['trend']),
                float(breakdown['sub_scores']['fvg']), float(weights['fvg']), float(contribs['fvg']),
                float(breakdown['sub_scores']['sweeps']), float(weights['sweeps']), float(contribs['sweeps']),
                float(breakdown['sub_scores']['order_blocks']), float(weights['order_blocks']), float(contribs['order_blocks']),
                float(breakdown['sub_scores']['volume_session']), float(weights['volume_session']), float(contribs['volume_session']),
                float(total_weighted), float(score),
            )
        except Exception:
            pass



        return score, reasons, breakdown

    def _calculate_confidence(
        self,
        normalized_score: float,
        volatility_state: str,
        session_analysis: SessionAnalysis,
        volume_analysis: Dict[str, Any],
        scalping_mode: bool = True,
        sweeps: Optional[list] = None,
    ) -> float:
        """محاسبه اعتماد سیگنال

        نکته عملیاتی:
        - این تابع باید کاملاً قابل ردیابی (traceable) باشد تا هر ناسازگاری بین SCORE و CONFIDENCE
        در لاگ‌ها سریعاً قابل کشف باشد.

        سیاست جدید:
        - session_weight و activity فقط کیفیت هستند.
        - inactive penalty (×0.8) فقط برای untradable واقعی اعمال می‌شود.
        """
        # --- Base from score distance to neutral (50) ---
        base_confidence = abs(normalized_score - 50) * 2.4

        # --- Volatility adjustment ---
        volatility_state = self._normalize_volatility_state(volatility_state)
        vol_mult = 1.0
        if volatility_state == 'HIGH_VOLATILITY':
            vol_mult = 1.1 if scalping_mode else 0.8
        elif volatility_state == 'LOW_VOLATILITY':
            vol_mult = 0.9 if scalping_mode else 1.05
        base_confidence *= vol_mult

        # --- Session adjustment ---
        session_mult = 1.0
        session_name = str(getattr(session_analysis, "current_session", "UNKNOWN") or "UNKNOWN")
        session_weight = float(getattr(session_analysis, "session_weight", 1.0) or 1.0)

        # upstream flag (بعد از اصلاح upstream باید معنی درست داشته باشد)
        upstream_active = bool(getattr(session_analysis, "is_active_session", True))

        # activity (کیفیت)
        session_activity = str(
            getattr(session_analysis, "session_activity",
                    getattr(session_analysis, "activity", "UNKNOWN"))
            or "UNKNOWN"
        ).upper()

        # strong_signal guard
        strong_signal = abs(normalized_score - 50) > 15

        # 1) Quality penalties/bonuses
        if session_weight < 0.6:
            session_mult *= 0.75

        if strong_signal and session_weight > 0.8:
            session_mult *= 1.15

        # 2) Determine "effective_active" (فقط untradable خاموش می‌کند)
        # untradable signals are optional and may not exist
        market_status = str(volume_analysis.get("market_status", "") or "").upper()
        data_ok = volume_analysis.get("data_ok", None)
        spread = volume_analysis.get("spread", None)
        max_spread = volume_analysis.get("max_spread", None)

        untradable = False
        untradable_reasons = []

        if market_status in ("CLOSED", "HALTED"):
            untradable = True
            untradable_reasons.append(f"market_status={market_status}")

        if data_ok is False:
            untradable = True
            untradable_reasons.append("data_ok=False")

        if spread is not None and max_spread is not None:
            try:
                if float(spread) > float(max_spread):
                    untradable = True
                    untradable_reasons.append(f"spread={float(spread):.4f}>max={float(max_spread):.4f}")
            except Exception:
                pass

        # effective_active: اگر upstream false بود ولی هیچ untradable نداریم، override به true
        effective_active = upstream_active
        if (not upstream_active) and (not untradable):
            effective_active = True

        # inactive penalty فقط برای untradable واقعی
        if not effective_active:
            session_mult *= 0.8

        base_confidence *= session_mult

        # --- RVOL adjustment ---
        rvol_mult = 1.0
        current_rvol = volume_analysis.get('rvol', 1.0)
        try:
            current_rvol = float(current_rvol)
        except Exception:
            current_rvol = 1.0

        if current_rvol > 1.2:
            rvol_mult *= 1.1
        elif current_rvol < 0.8:
            rvol_mult *= 0.9
        base_confidence *= rvol_mult

        # --- Sweep bonus (contextual) ---
        sweep_mult = 1.0
        sweeps_count = len(sweeps) if sweeps else 0
        if sweeps_count > 0 and strong_signal:
            sweep_mult *= 1.05
            base_confidence *= sweep_mult

        # --- Range compression penalty (if near-neutral) ---
        range_penalty_mult = 1.0
        if 42 <= normalized_score <= 58:
            range_penalty_mult = 0.5
            base_confidence *= range_penalty_mult

        # --- Clamp ---
        confidence = min(95, base_confidence * 1.1)
        confidence = max(10, confidence)

        # --- Instrumentation (single-line, deterministic, parse-friendly) ---
        try:
            reasons_str = ",".join(untradable_reasons) if untradable_reasons else "-"
            self._log_info(
                "[NDS][CONF] score=%.2f base=%.2f vol=%s vol_mult=%.3f "
                "session=%s weight=%.2f activity=%s upstream_active=%s effective_active=%s untradable=%s reasons=%s strong=%s session_mult=%.3f "
                "rvol=%.2f rvol_mult=%.3f sweeps=%d sweep_mult=%.3f "
                "range_mult=%.3f -> conf=%.2f",
                float(normalized_score),
                float(abs(normalized_score - 50) * 2.4),
                volatility_state,
                float(vol_mult),
                session_name,
                float(session_weight),
                session_activity,
                bool(upstream_active),
                bool(effective_active),
                bool(untradable),
                reasons_str,
                bool(strong_signal),
                float(session_mult),
                float(current_rvol),
                float(rvol_mult),
                int(sweeps_count),
                float(sweep_mult),
                float(range_penalty_mult),
                float(confidence),
            )
        except Exception:
            # لاگ نباید باعث fail شدن آنالیز شود
            pass

        return round(confidence, 1)


    def _determine_signal(
        self,
        normalized_score: float,
        confidence: float,
        volatility_state: str,
        scalping_mode: bool = True,
    ) -> str:
        """تعیین سیگنال با آستانه‌های پویا (بدون فیلتر اعتماد)"""
        if scalping_mode:
            volatility_state = self._normalize_volatility_state(volatility_state)
            if volatility_state == 'HIGH_VOLATILITY':
                buy_threshold = 60
                sell_threshold = 40
            elif volatility_state == 'LOW_VOLATILITY':
                buy_threshold = 65
                sell_threshold = 35
            else:
                buy_threshold = 55
                sell_threshold = 45
        else:
            buy_threshold = 65
            sell_threshold = 35

        if normalized_score >= buy_threshold:
            return "BUY"
        if normalized_score <= sell_threshold:
            return "SELL"
        return "NONE"

    def _build_initial_result(
        self,
        signal: str,
        confidence: float,
        score: float,
        reasons: List[str],
        structure: MarketStructure,
        atr_value: float,
        atr_short_value: Optional[float],
        adx_value: float,
        plus_di: float,
        minus_di: float,
        volume_analysis: Dict[str, Any],
        recent_range: float,
        recent_position: float,
        volatility_state: str,
        session_analysis: SessionAnalysis,
        current_price: float,
        timeframe: str,
        score_breakdown: Dict[str, Any],
        scalping_mode: bool = True,
    ) -> Dict[str, Any]:
        """ساخت ساختار نتیجه اولیه"""
        normalized_structure_score = self._normalize_structure_score(
            getattr(structure, 'structure_score', 0.0)
        )

        current_rvol = float(volume_analysis.get('rvol', 1.0))
        if pd.isna(current_rvol):
            current_rvol = 1.0

        result = {
            "signal": signal,
            "confidence": confidence,
            "score": round(score, 1),
            "reasons": reasons[:8],
            "structure": {
                "trend": structure.trend.value,
                "bos": structure.bos,
                "choch": structure.choch,
                "last_high": structure.last_high.price if structure.last_high else None,
                "last_low": structure.last_low.price if structure.last_low else None,
                "range_width": round(structure.range_width, 2) if structure.range_width else 0,
                "range_mid": round(structure.range_mid, 2) if structure.range_mid else 0,
                "structure_score": normalized_structure_score,
            },
            "market_metrics": {
                "atr": round(atr_value, 2),
                "adx": round(adx_value, 1),
                "plus_di": round(plus_di, 1),
                "minus_di": round(minus_di, 1),
                "recent_range": round(recent_range, 2),
                "recent_position": round(recent_position, 2),
                "volatility_state": volatility_state,
                "current_rvol": round(current_rvol, 2),
            },
            "analysis_data": {
                "volume_analysis": volume_analysis,
                "score_breakdown": score_breakdown,
            },
            "session_analysis": {
                "current_session": session_analysis.current_session,
                "session_weight": session_analysis.session_weight,
                "is_active_session": session_analysis.is_active_session,
                "optimal_trading": session_analysis.optimal_trading,
                "weight": session_analysis.weight,
            },
            "timestamp": datetime.now().isoformat(),
            "current_price": current_price,
            "timeframe": timeframe,
            "scalping_mode": scalping_mode,
        }

        if scalping_mode and atr_short_value:
            result["market_metrics"]["atr_short"] = round(atr_short_value, 2)

        return result

    def _adaptive_min_rvol(self, base_min_rvol: float, structure_score: float) -> float:
        if structure_score >= 90.0:
            return base_min_rvol * 0.5
        if structure_score >= 70.0:
            return base_min_rvol * 0.75
        return base_min_rvol

    def _apply_final_filters(self, analysis_result: Dict[str, Any], scalping_mode: bool = True) -> Dict[str, Any]:
        """
        اعمال فیلترهای نهایی با اتصال به تنظیمات مرکزی
        """
        original_signal = analysis_result.get('signal', 'NONE')
        reasons = analysis_result.get('reasons', [])

        self._log_debug("[NDS][FILTER] start signal=%s", original_signal)

        if original_signal != 'NONE':
            settings = self.GOLD_SETTINGS

            if scalping_mode:
                base_min_rvol = settings.get('MIN_RVOL_SCALPING', 0.75)
                current_rvol = analysis_result.get('market_metrics', {}).get('current_rvol', 1.0)
                structure = analysis_result.get('structure', {})
                structure_score = float(structure.get('structure_score', 0.0))
                adaptive_min_rvol = self._adaptive_min_rvol(base_min_rvol, structure_score)

                self._log_debug(
                    "[NDS][FILTER] rvol base=%.2f current=%.2f structure_score=%.1f adaptive=%.2f",
                    base_min_rvol,
                    current_rvol,
                    structure_score,
                    adaptive_min_rvol,
                )

                if current_rvol < adaptive_min_rvol:
                    analysis_result['signal'] = 'NONE'
                    self._append_reason(
                        reasons,
                        f"Volume too low (RVOL: {current_rvol:.2f} < {adaptive_min_rvol:.2f})"
                    )
                elif adaptive_min_rvol < base_min_rvol:
                    self._append_reason(
                        reasons,
                        f"Volume accepted due to high structure score ({structure_score:.1f})"
                    )

            current_confidence = analysis_result.get('confidence', 0.0)
            if scalping_mode:
                min_confidence = settings.get('SCALPING_MIN_CONFIDENCE', 45.0)
                confidence_type = "SCALPING"
            else:
                min_confidence = settings.get('MIN_CONFIDENCE', 50.0)
                confidence_type = "NORMAL"

            self._log_debug(
                "[NDS][FILTER] confidence current=%.1f%% %s_min=%.1f%%",
                current_confidence,
                confidence_type,
                min_confidence,
            )

            if current_confidence < min_confidence:
                analysis_result['signal'] = 'NONE'
                self._append_reason(
                    reasons,
                    f"Confidence too low ({current_confidence:.1f}% < {min_confidence}%)"
                )

            session_analysis = analysis_result.get('session_analysis', {})
            session_weight = float(session_analysis.get('weight', 0.5))
            min_session_weight = settings.get('MIN_SESSION_WEIGHT', 0.3)

            self._log_debug(
                "[NDS][FILTER] session weight=%.2f min=%.2f",
                session_weight,
                min_session_weight,
            )

            if session_weight < min_session_weight:
                analysis_result['signal'] = 'NONE'
                self._append_reason(
                    reasons,
                    f"Low session weight ({session_weight:.2f} < {min_session_weight})"
                )

            structure = analysis_result.get('structure', {})
            structure_score = float(structure.get('structure_score', 0.0))
            min_structure_score = settings.get('MIN_STRUCTURE_SCORE', 20.0)


            # --- Hard structure sanity gate (scalping) ---
            # اگر BOS/CHOCH نداریم و ADX پایین است، حتی با score متوسط اجازه سیگنال نده.
            market_metrics = analysis_result.get('market_metrics', {})
            adx_v = float(market_metrics.get('adx', 0.0) or 0.0)
            bos_v = structure.get('bos', 'NONE')
            choch_v = structure.get('choch', 'NONE')

            sanity_reject_adx_max = float(settings.get('SANITY_ADX_REJECT_MAX', 18.0))
            sanity_reject_structure = float(settings.get('SANITY_STRUCTURE_REJECT_SCORE', 40.0))

            if scalping_mode and analysis_result.get('signal', 'NONE') != 'NONE':
                if bos_v == 'NONE' and choch_v == 'NONE' and adx_v < sanity_reject_adx_max and structure_score < sanity_reject_structure:
                    analysis_result['signal'] = 'NONE'
                    self._append_reason(
                        reasons,
                        f"Rejected: no BOS/CHOCH with low ADX (ADX: {adx_v:.1f}, structure: {structure_score:.1f})"
                    )
                    self._log_debug(
                        "[NDS][FILTER][SANITY] reject no_confirm adx=%.2f structure=%.2f (max_adx=%.2f min_struct=%.2f)",
                        adx_v,
                        structure_score,
                        sanity_reject_adx_max,
                        sanity_reject_structure,
                    )

            self._log_debug(
                "[NDS][FILTER] structure score=%.1f min=%.1f",
                structure_score,
                min_structure_score,
            )

            if structure_score < min_structure_score:
                analysis_result['signal'] = 'NONE'
                self._append_reason(
                    reasons,
                    f"Weak market structure (Score: {structure_score:.1f} < {min_structure_score})"
                )

        analysis_result['reasons'] = reasons
        final_signal = analysis_result.get('signal', 'NONE')

        if original_signal != final_signal:
            self._log_debug(
                "[NDS][FILTER] changed signal original=%s final=%s reasons=%s",
                original_signal,
                final_signal,
                reasons,
            )
        else:
            self._log_debug("[NDS][FILTER] result signal=%s", final_signal)

        return analysis_result

    def _select_swing_anchor(self, structure: MarketStructure, signal: str) -> Optional[float]:
        if signal == "BUY" and structure.last_low:
            return structure.last_low.price
        if signal == "SELL" and structure.last_high:
            return structure.last_high.price
        return None

    def _build_entry_idea(
        self,
        signal: str,
        fvgs: List[FVG],
        order_blocks: List[OrderBlock],
        structure: MarketStructure,
        atr_value: float,
        entry_factor: float,
        current_price: float,
        adx_value: float,
    ) -> Dict[str, Optional[float]]:
        """ساخت ایده ورود/خروج بدون منطق اجرا یا مدیریت ریسک"""
        idea = {
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "reason": None,
        }

        self._log_debug(
            "[NDS][ENTRY_IDEA] start signal=%s atr=%.2f entry_factor=%.2f price=%.2f",
            signal,
            atr_value,
            entry_factor,
            current_price,
        )

        if atr_value is None or atr_value <= 0:
            idea["reason"] = "Invalid ATR for entry idea"
            return idea

        buffer_mult = float(self.GOLD_SETTINGS.get('ATR_BUFFER_MULTIPLIER', 0.5))
        min_rr = float(self.GOLD_SETTINGS.get('MIN_RR', 1.5))
        tp_multiplier = 1.5
        if adx_value is not None and adx_value > 40:
            tp_multiplier = 2.0
        elif adx_value is not None and adx_value > 25:
            tp_multiplier = 1.7

        valid_fvgs = [f for f in fvgs if not f.filled]

        # --- Fallback gating: فقط در ساختار/روند بسیار قوی اجازه fallback ---
        fallback_min_adx = float(self.GOLD_SETTINGS.get('FALLBACK_MIN_ADX', 30.0))
        fallback_min_structure = float(self.GOLD_SETTINGS.get('FALLBACK_MIN_STRUCTURE_SCORE', 80.0))
        normalized_structure_score = self._normalize_structure_score(getattr(structure, 'structure_score', 0.0))
        trend_value = structure.trend.value if getattr(structure, 'trend', None) else "RANGING"
        safe_adx = float(adx_value) if adx_value is not None else 0.0

        allow_fallback = (
            trend_value in {"UPTREND", "DOWNTREND"}
            and safe_adx >= fallback_min_adx
            and normalized_structure_score >= fallback_min_structure
        )

        # لاگ دقیق علت مجاز/غیرمجاز شدن fallback (هم debug و هم info برای visibility)
        self._log_debug(
            "[NDS][ENTRY_IDEA][FALLBACK] allow=%s trend=%s adx=%.1f(>=%.1f) structure_score=%.1f(>=%.1f)",
            allow_fallback,
            trend_value,
            safe_adx,
            fallback_min_adx,
            normalized_structure_score,
            fallback_min_structure,
        )
        self._log_info(
            "[NDS][ENTRY_IDEA][FALLBACK] allow=%s trend=%s adx=%.1f structure_score=%.1f thresholds(adx>=%.1f score>=%.1f)",
            allow_fallback,
            trend_value,
            safe_adx,
            normalized_structure_score,
            fallback_min_adx,
            fallback_min_structure,
        )
        def finalize_trade(entry: float, stop: float, target: Optional[float], reason: str) -> Dict[str, Optional[float]]:
            """Finalize and validate trade idea geometry.

            Hard guarantees:
              BUY:  stop < entry < target
              SELL: target < entry < stop

            If SL is on the wrong side due to swing-anchor/structure issues, we correct it to a
            minimum ATR-based distance. If geometry still invalid, we invalidate the idea.
            """
            if entry is None or stop is None:
                idea["reason"] = "Invalid entry/stop (None)"
                self._log_info("[NDS][ENTRY_IDEA][INVALID] %s", idea["reason"])
                return idea

            entry = float(entry)
            stop = float(stop)

            # Minimum stop distance (guardrail)
            min_stop = self._min_stop_distance(atr_value)

            # Enforce SL side (correct if needed)
            if signal == "BUY":
                if stop >= entry - 1e-9:
                    corrected = entry - min_stop
                    self._log_info(
                        "[NDS][ENTRY_IDEA][FIX] BUY stop was not below entry: stop=%.2f entry=%.2f -> stop=%.2f (min_stop=%.2f)",
                        stop,
                        entry,
                        corrected,
                        min_stop,
                    )
                    stop = corrected
            else:  # SELL
                if stop <= entry + 1e-9:
                    corrected = entry + min_stop
                    self._log_info(
                        "[NDS][ENTRY_IDEA][FIX] SELL stop was not above entry: stop=%.2f entry=%.2f -> stop=%.2f (min_stop=%.2f)",
                        stop,
                        entry,
                        corrected,
                        min_stop,
                    )
                    stop = corrected

            risk = abs(entry - stop)
            if risk <= 0:
                idea["reason"] = f"Invalid risk: entry={entry:.2f} stop={stop:.2f}"
                self._log_info("[NDS][ENTRY_IDEA][INVALID] %s", idea["reason"])
                return idea

            # Build/adjust TP
            if target is None:
                if signal == "BUY":
                    target = entry + risk * tp_multiplier
                else:
                    target = entry - risk * tp_multiplier

            target = float(target)

            # Enforce minimum RR
            rr_target = entry + risk * min_rr if signal == "BUY" else entry - risk * min_rr
            if signal == "BUY":
                target = max(target, rr_target)
            else:
                target = min(target, rr_target)

            # Final geometry validation
            if signal == "BUY":
                valid = (stop < entry < target)
            else:
                valid = (target < entry < stop)

            if not valid:
                idea["reason"] = f"Invalid geometry for {signal}: tp={target:.2f} entry={entry:.2f} sl={stop:.2f}"
                self._log_info("[NDS][ENTRY_IDEA][INVALID] %s", idea["reason"])
                return idea

            idea["entry_price"] = entry
            idea["stop_loss"] = stop
            idea["take_profit"] = target
            idea["reason"] = reason

            self._log_info(
                "[NDS][ENTRY_IDEA] finalized signal=%s entry=%.2f sl=%.2f tp=%.2f risk=%.2f rr=%.2f reason=%s",
                signal,
                entry,
                stop,
                target,
                risk,
                (abs(target - entry) / risk) if risk > 0 else 0.0,
                reason,
            )
            return idea

        swing_anchor = self._select_swing_anchor(structure, signal)

        if signal == "BUY":
            target_fvgs = [
                f for f in valid_fvgs
                if f.type.value == "BULLISH_FVG"
                and f.top < current_price
                and f.size >= (atr_value * 0.1)
            ]
            best_fvg = max(target_fvgs, key=lambda x: x.strength) if target_fvgs else None
            if best_fvg:
                self._log_debug(
                    "[NDS][ENTRY_IDEA] bullish FVG selected index=%s top=%.2f bottom=%.2f strength=%.2f",
                    best_fvg.index,
                    best_fvg.top,
                    best_fvg.bottom,
                    best_fvg.strength,
                )
                # --- تغییر کلیدی: حذف ریسک کرش best_fvg.height ---
                fvg_height = abs(float(best_fvg.top) - float(best_fvg.bottom))
                entry = float(best_fvg.top) - (fvg_height * entry_factor)
                stop = (swing_anchor - atr_value * buffer_mult) if swing_anchor else float(best_fvg.bottom) - (atr_value * 0.5)
                target = float(best_fvg.top) + (fvg_height * tp_multiplier)
                return finalize_trade(entry, stop, target, f"Bullish FVG idea (strength: {best_fvg.strength:.1f})")

            bullish_obs = [ob for ob in order_blocks if ob.type == 'BULLISH_OB']
            if bullish_obs:
                best_ob = max(bullish_obs, key=lambda x: x.strength)
                self._log_debug(
                    "[NDS][ENTRY_IDEA] bullish OB selected index=%s high=%.2f low=%.2f strength=%.2f",
                    best_ob.index,
                    best_ob.high,
                    best_ob.low,
                    best_ob.strength,
                )
                entry = float(best_ob.low) + (float(best_ob.high) - float(best_ob.low)) * 0.3
                stop = (swing_anchor - atr_value * buffer_mult) if swing_anchor else float(best_ob.low) - (atr_value * 0.5)
                target = float(best_ob.high) + (float(best_ob.high) - float(best_ob.low)) * tp_multiplier
                return finalize_trade(entry, stop, target, f"Bullish OB idea (strength: {best_ob.strength:.1f})")

            # --- Fallback BUY فقط وقتی allow_fallback=True ---
            if not allow_fallback:
                idea["reason"] = (
                    f"No valid FVG/OB for BUY; fallback disabled "
                    f"(ADX={safe_adx:.1f}<{fallback_min_adx:.1f} or Structure={normalized_structure_score:.1f}<{fallback_min_structure:.1f} or Trend={trend_value})"
                )
                self._log_debug("[NDS][ENTRY_IDEA] bullish fallback blocked reason=%s", idea["reason"])
                return idea

            fallback_entry = current_price - (atr_value * 0.3)
            stop = (swing_anchor - atr_value * buffer_mult) if swing_anchor else (current_price - (atr_value * 1.2))
            self._log_debug("[NDS][ENTRY_IDEA] bullish fallback entry=%.2f stop=%.2f", fallback_entry, stop)
            return finalize_trade(fallback_entry, stop, None, "Fallback bullish idea (only allowed in strong structure)")

        if signal == "SELL":
            target_fvgs = [
                f for f in valid_fvgs
                if f.type.value == "BEARISH_FVG"
                and f.bottom > current_price
                and f.size >= (atr_value * 0.1)
            ]
            best_fvg = max(target_fvgs, key=lambda x: x.strength) if target_fvgs else None
            if best_fvg:
                self._log_debug(
                    "[NDS][ENTRY_IDEA] bearish FVG selected index=%s top=%.2f bottom=%.2f strength=%.2f",
                    best_fvg.index,
                    best_fvg.top,
                    best_fvg.bottom,
                    best_fvg.strength,
                )
                # --- تغییر کلیدی: حذف ریسک کرش best_fvg.height ---
                fvg_height = abs(float(best_fvg.top) - float(best_fvg.bottom))
                entry = float(best_fvg.bottom) + (fvg_height * entry_factor)
                stop = (swing_anchor + atr_value * buffer_mult) if swing_anchor else float(best_fvg.top) + (atr_value * 0.5)
                target = float(best_fvg.bottom) - (fvg_height * tp_multiplier)
                return finalize_trade(entry, stop, target, f"Bearish FVG idea (strength: {best_fvg.strength:.1f})")

            bearish_obs = [ob for ob in order_blocks if ob.type == 'BEARISH_OB']
            if bearish_obs:
                best_ob = max(bearish_obs, key=lambda x: x.strength)
                self._log_debug(
                    "[NDS][ENTRY_IDEA] bearish OB selected index=%s high=%.2f low=%.2f strength=%.2f",
                    best_ob.index,
                    best_ob.high,
                    best_ob.low,
                    best_ob.strength,
                )
                entry = float(best_ob.high) - (float(best_ob.high) - float(best_ob.low)) * 0.3
                stop = (swing_anchor + atr_value * buffer_mult) if swing_anchor else float(best_ob.high) + (atr_value * 0.5)
                target = float(best_ob.low) - (float(best_ob.high) - float(best_ob.low)) * tp_multiplier
                return finalize_trade(entry, stop, target, f"Bearish OB idea (strength: {best_ob.strength:.1f})")

            # --- Fallback SELL فقط وقتی allow_fallback=True ---
            if not allow_fallback:
                idea["reason"] = (
                    f"No valid FVG/OB for SELL; fallback disabled "
                    f"(ADX={safe_adx:.1f}<{fallback_min_adx:.1f} or Structure={normalized_structure_score:.1f}<{fallback_min_structure:.1f} or Trend={trend_value})"
                )
                self._log_debug("[NDS][ENTRY_IDEA] bearish fallback blocked reason=%s", idea["reason"])
                return idea

            fallback_entry = current_price + (atr_value * 0.3)
            stop = (swing_anchor + atr_value * buffer_mult) if swing_anchor else (current_price + (atr_value * 1.2))
            self._log_debug("[NDS][ENTRY_IDEA] bearish fallback entry=%.2f stop=%.2f", fallback_entry, stop)
            return finalize_trade(fallback_entry, stop, None, "Fallback bearish idea (only allowed in strong structure)")

        return idea

    def _build_analysis_result(
        self,
        signal: str,
        confidence: float,
        score: float,
        entry_price: Optional[float],
        stop_loss: Optional[float],
        take_profit: Optional[float],
        reasons: List[str],
        context: Dict[str, Any],
        timeframe: str,
        current_price: float,
    ) -> AnalysisResult:
        """ساخت خروجی نهایی تحلیل‌محور"""
        return AnalysisResult(
            signal=signal,
            confidence=round(confidence, 1),
            score=round(score, 1),
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasons=reasons[:12],
            context=context,
            timestamp=datetime.now().isoformat(),
            timeframe=timeframe,
            current_price=current_price,
        )

    def _create_error_result(
        self,
        error_message: str,
        timeframe: str,
        current_close: Optional[float],
    ) -> AnalysisResult:
        """ایجاد نتیجه خطا"""
        return AnalysisResult(
            signal="NONE",
            confidence=0.0,
            score=50.0,
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            reasons=[f"Error: {error_message}"],
            context={"error": True},
            timestamp=datetime.now().isoformat(),
            timeframe=timeframe,
            current_price=current_close or 0.0,
        )


def analyze_gold_market(
    dataframe: pd.DataFrame,
    timeframe: str = 'M15',
    entry_factor: float = 0.25,
    config: Optional[Dict[str, Any]] = None,
    scalping_mode: bool = True,
) -> AnalysisResult:
    """
    تابع اصلی برای تحلیل بازار طلا و ایجاد confirming signal
    """
    if dataframe is None or dataframe.empty:
        return AnalysisResult(
            signal="NONE",
            confidence=0.0,
            score=50.0,
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            reasons=["DataFrame is empty"],
            context={"error": True},
            timestamp=datetime.now().isoformat(),
            timeframe=timeframe,
            current_price=0.0,
        )

    try:
        mode = "Scalping" if scalping_mode else "Regular"
        logger.info(
            "[NDS][INIT] create analyzer mode=%s timeframe=%s candles=%s",
            mode,
            timeframe,
            len(dataframe),
        )

        analyzer = GoldNDSAnalyzer(dataframe, config=config)
        result = analyzer.generate_trading_signal(timeframe, entry_factor, scalping_mode)

        return result

    except Exception as e:
        logger.error("[NDS][RESULT] analysis failed: %s", str(e), exc_info=True)
        return AnalysisResult(
            signal="NONE",
            confidence=0.0,
            score=50.0,
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            reasons=[f"Analysis error: {str(e)}"],
            context={"error": True},
            timestamp=datetime.now().isoformat(),
            timeframe=timeframe,
            current_price=0.0,
        )
