# src/trading_bot/nds/analyzer.py
"""
آنالایزر اصلی NDS برای طلا - نسخه بهبودیافته با رفع تضاد SMC-ADX
"""
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional

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
    نسخه ماژولار و بهینه‌شده آنالایزر طلا (XAUUSD) - نسخه 5.3
    """
    
    def __init__(self, df: pd.DataFrame, config: Dict = None):
        if df.empty:
            raise ValueError("DataFrame is empty. Cannot initialize analyzer.")
        
        self.df = df.copy()
        self.config = config or {}
        self.debug_analyzer = False
        
        # تنظیمات پایه
        technical_settings = (self.config or {}).get('technical_settings', {})
        sessions_config = (self.config or {}).get('sessions_config', {})
        self.GOLD_SETTINGS = technical_settings.copy()
        self.TRADING_SESSIONS = sessions_config.get('BASE_TRADING_SESSIONS', {}).copy()
        self.timeframe_specifics = technical_settings.get('TIMEFRAME_SPECIFICS', {})
        self.swing_period_map = technical_settings.get('SWING_PERIOD_MAP', {})
        
        self.atr = None
        
        # اعمال تنظیمات سفارشی از config متمرکز
        self._apply_custom_config()
        
        # اعتبارسنجی داده‌ها
        self._validate_dataframe()
        
        # شناسایی تایم‌فریم
        self.timeframe = self._detect_timeframe()
        
        # اعمال تنظیمات تایم‌فریم
        self._apply_timeframe_settings()
        
        logger.info(f"✅ NDS Analyzer v5.3 initialized with {len(self.df)} candles | Timeframe: {self.timeframe}")

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

        validated_config: Dict[str, Any] = {}
        ignored_keys: List[str] = []

        for key, value in analyzer_config.items():
            if key == 'DEBUG_ANALYZER':
                continue
            if key not in ANALYSIS_CONFIG_KEYS:
                ignored_keys.append(key)
                continue

            if isinstance(value, (int, float, bool)):
                validated_config[key] = value
            elif isinstance(value, str):
                try:
                    validated_config[key] = float(value)
                except ValueError:
                    validated_config[key] = value
            else:
                self._log_debug(f"Ignored non-scalar analyzer setting: {key}={value}")

        if validated_config:
            self.GOLD_SETTINGS.update(validated_config)
            self._log_debug(f"Applied {len(validated_config)} analyzer settings from config")

        if ignored_keys:
            self._log_debug(f"Ignored non-analysis config keys: {sorted(set(ignored_keys))}")

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

            self._log_debug(f"Applied session config: {standard_name} = {converted_session}")

    def _log_debug(self, message: str, *args: Any) -> None:
        if self.debug_analyzer:
            logger.debug(message, *args)

    def _validate_dataframe(self) -> None:
        """اعتبارسنجی DataFrame ورودی"""
        required_columns = ['time', 'open', 'high', 'low', 'close']
        missing = [col for col in required_columns if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        self.df['time'] = pd.to_datetime(self.df['time'], utc=True)
        
        if len(self.df) > 1 and self.df['time'].iloc[0] > self.df['time'].iloc[-1]:
            logger.warning("DataFrame not sorted chronologically. Sorting...")
            self.df = self.df.sort_values('time').reset_index(drop=True)
        
        if 'volume' not in self.df.columns:
            self.df['volume'] = 1.0

    def _detect_timeframe(self) -> str:
        """شناسایی تایم‌فریم از روی داده‌ها"""
        if len(self.df) > 1:
            time_diff = (self.df['time'].iloc[1] - self.df['time'].iloc[0]).total_seconds()
            if time_diff == 900:
                return "M15"
            elif time_diff == 3600:
                return "H1"
            elif time_diff == 60:
                return "M1"
            elif time_diff == 300:
                return "M5"
            elif time_diff == 1800:
                return "M30"
            elif time_diff == 14400:
                return "H4"
            elif time_diff == 86400:
                return "D1"
        
        return "M15"

    def _apply_timeframe_settings(self):
        """اعمال تنظیمات اختصاصی تایم‌فریم"""
        tf_settings = self.timeframe_specifics.get(self.timeframe)
        if not tf_settings:
            raise KeyError(f"Missing TIMEFRAME_SPECIFICS for timeframe: {self.timeframe}")
        self.GOLD_SETTINGS.update(tf_settings)

    def _analyze_trading_sessions(self) -> SessionAnalysis:
        """تحلیل جامع سشن‌های معاملاتی"""
        last_time = self.df['time'].iloc[-1]
        hour = last_time.hour
        
        session_info = self._is_valid_trading_session(last_time)
        
        # محاسبه فعالیت سشن بر اساس حجم اخیر
        session_activity = 'UNKNOWN'
        recent_data = self.df.tail(20)
        if 'rvol' in recent_data.columns:
            avg_rvol = recent_data['rvol'].mean()
            session_activity = 'HIGH' if avg_rvol > 1.2 else 'LOW'
        
        analysis = SessionAnalysis(
            current_session=session_info.get('session', 'OTHER'),
            session_weight=session_info.get('weight', 0.5),
            weight=session_info.get('weight', 0.5),
            gmt_hour=hour,
            is_active_session=session_info.get('is_valid', False),
            is_overlap=session_info.get('is_overlap', False),
            session_activity=session_activity,
            optimal_trading=session_info.get('optimal_trading', session_info.get('weight', 0.5) >= 1.2)
        )
        
        logger.info(f"🏛️ Session Analysis: {analysis.current_session} (Weight: {analysis.weight:.1f})")
        return analysis

    def _is_valid_trading_session(self, check_time: datetime) -> Dict[str, Any]:
        """بررسی سشن معاملاتی با پشتیبانی از نام‌های جدید و قدیم"""
        if not isinstance(check_time, datetime):
            try:
                check_time = pd.to_datetime(check_time)
            except:
                return {
                    'is_valid': False, 'is_overlap': False, 'session': 'INVALID', 
                    'weight': 0.0, 'optimal_trading': False
                }
        
        hour = check_time.hour
        raw_sessions = self.TRADING_SESSIONS
        
        # 1. نگاشت سشن‌های جدید از کانفیگ به ساختار مورد انتظار آنالایزر
        # این بخش باعث می‌شود KeyError رفع شود
        sessions = {}
        # ایجاد یک دیکشنری استاندارد (LONDON, NEW_YORK, OVERLAP, ASIA)
        for config_name, data in raw_sessions.items():
            standard_name = SESSION_MAPPING.get(config_name, config_name)
            # اگر چند سشن به یک نام استاندارد نگاشت شوند (مثل LONDON_OPEN و LONDON_CORE)، 
            # آنالایزر بازه بزرگتر یا وزن بیشتر را در نظر می‌گیرد
            if standard_name not in sessions or data.get('weight', 0) > sessions[standard_name].get('weight', 0):
                sessions[standard_name] = data

        # 2. حالا با خیال راحت از نام‌های قدیمی استفاده کنید
        session_name = 'OTHER'
        session_weight = 0.5
        is_overlap = False
        
        # استفاده از .get() برای امنیت ۱۰۰ درصد در برابر کرش
        def check_in_session(name):
            s = sessions.get(name)
            if s and s['start'] <= hour <= s['end']:
                return True, s['weight']
            return False, 0

        # بررسی اورلپ
        in_overlap, weight = check_in_session('OVERLAP')
        if in_overlap:
            session_name = 'OVERLAP'
            session_weight = weight
            is_overlap = True
        else:
            # بررسی لندن
            in_london, weight = check_in_session('LONDON')
            if in_london:
                session_name = 'LONDON'
                session_weight = weight
            else:
                # بررسی نیویورک
                in_ny, weight = check_in_session('NEW_YORK')
                if in_ny:
                    session_name = 'NEW_YORK'
                    session_weight = weight
                else:
                    # بررسی آسیا
                    in_asia, weight = check_in_session('ASIA')
                    if in_asia:
                        session_name = 'ASIA'
                        session_weight = weight

        optimal_trading = session_weight >= 1.0
        
        return {
            'is_valid': session_weight >= 1.0,
            'is_overlap': is_overlap,
            'session': session_name,
            'weight': session_weight,
            'hour': hour,
            'optimal_trading': optimal_trading
        }

    def generate_trading_signal(self, timeframe: str = 'M15', entry_factor: float = 0.5,
                                scalping_mode: bool = True) -> AnalysisResult:
        """
        تولید سیگنال نهایی با ادغام سیستم تشخیص ساختار الترا پرو و تاییدیه ADX/Volume
        نسخه تحلیل‌محور: خروجی فقط ایده معاملاتی (بدون منطق اجرا/ریسک).
        """
        mode = "Scalping" if scalping_mode else "Regular"
        logger.info(f"🎯 Starting Gold NDS {mode} Analysis v5.6 (Analysis Only)...")

        try:
            # ۱. آماده‌سازی اندیکاتورهای پایه
            atr_window = self.GOLD_SETTINGS.get('ATR_WINDOW', 14)
            self.df, atr_v = IndicatorCalculator.calculate_atr(self.df, atr_window)
            self.atr = atr_v

            current_close = float(self.df['close'].iloc[-1])

            # محاسبه ADX
            adx_window = self.GOLD_SETTINGS.get('ADX_WINDOW', 14)
            self.df, adx_v, plus_di, minus_di, di_trend = IndicatorCalculator.calculate_adx(self.df, adx_window)

            # محاسبه ATR کوتاه‌مدت
            if scalping_mode:
                atr_short_df, _ = IndicatorCalculator.calculate_atr(self.df.copy(), 7)
                atr_short_value = atr_short_df['atr_7'].iloc[-1]
            else:
                atr_short_value = atr_v

            # ۲. تحلیل حجم و ۳. نوسان و ۴. سشن
            volume_analysis = IndicatorCalculator.analyze_volume(self.df, 5 if scalping_mode else 20)
            volatility_state = self._determine_volatility(atr_v, atr_short_value if scalping_mode else atr_v)
            session_analysis = self._analyze_trading_sessions()

            # ۵. تحلیل SMC پیشرفته
            smc = SMCAnalyzer(self.df, self.atr, self.GOLD_SETTINGS)
            swings = smc.detect_swings(timeframe)
            fvgs = smc.detect_fvgs()
            order_blocks = smc.detect_order_blocks()
            sweeps = smc.detect_liquidity_sweeps(swings)

            structure = smc.determine_market_structure(
                swings=swings, lookback_swings=4, volume_analysis=volume_analysis,
                volatility_state=volatility_state, adx_value=adx_v
            )

            # ۶. اعتبارسنجی ساختار
            logger.info(f"✅ ساختار شناسایی شده: {structure}")

            # ۷. اصلاح وضعیت روند با ADX Override
            final_structure = structure
            if adx_v > 30 and structure.trend.value == "RANGING":
                final_structure = MarketStructure(
                    trend=MarketTrend.UPTREND if plus_di > minus_di else MarketTrend.DOWNTREND,
                    bos=structure.bos, choch=structure.choch,
                    last_high=structure.last_high, last_low=structure.last_low,
                    current_price=structure.current_price, range_width=structure.range_width,
                    range_mid=structure.range_mid, bos_choch_confidence=structure.bos_choch_confidence,
                    volume_analysis=structure.volume_analysis, volatility_state=structure.volatility_state,
                    adx_value=adx_v, structure_score=structure.structure_score
                )
                logger.info(f"🚀 ADX Override applied: {final_structure.trend.value}")

            # ۸. سیستم امتیازدهی
            score, reasons, score_breakdown = self._calculate_scoring_system(
                final_structure, adx_v, volume_analysis['rvol'], fvgs, sweeps, order_blocks,
                current_close, swings, atr_short_value if scalping_mode else atr_v
            )

            # ۹. تنظیم وزن سشن
            if scalping_mode:
                if session_analysis.weight > 0.8:
                    score *= 1.1
                elif session_analysis.weight < 0.5:
                    score *= 0.9
            score = max(0, min(100, score))

            # ۱۰. بونوس ساختار
            if structure.is_valid_structure():
                score += (structure.structure_score / 10)
                score = max(0, min(100, score))

            # ۱۱. محاسبه اعتماد
            confidence = self._calculate_confidence(
                score, volatility_state, session_analysis,
                volume_analysis['rvol'], volume_analysis['volume_trend'], scalping_mode,
                sweeps=sweeps
            )

            # ۱۲. تعیین سیگنال اولیه
            signal = self._determine_signal(score, confidence, volatility_state, scalping_mode)

            result_payload = self._build_initial_result(
                signal=signal,
                confidence=confidence,
                score=score,
                reasons=reasons,
                structure=final_structure,
                atr_value=atr_v,
                atr_short_value=atr_short_value if scalping_mode else None,
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
                    atr_value=atr_short_value if scalping_mode else atr_v,
                    entry_factor=entry_factor,
                    current_price=current_close,
                    adx_value=adx_v,
                )
                entry_price = entry_idea.get("entry_price")
                stop_loss = entry_idea.get("stop_loss")
                take_profit = entry_idea.get("take_profit")
                if entry_idea.get("reason"):
                    reasons.append(entry_idea["reason"])

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
            logger.error(f"❌ Analysis failed: {str(e)}", exc_info=True)
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

        if atr_value is None or atr_value <= 0:
            idea["reason"] = "Invalid ATR for entry idea"
            return idea

        valid_fvgs = [f for f in fvgs if not f.filled]
        tp_multiplier = 1.5
        if adx_value is not None and adx_value > 40:
            tp_multiplier = 2.0
        elif adx_value is not None and adx_value > 25:
            tp_multiplier = 1.7

        if signal == "BUY":
            target_fvgs = [
                f for f in valid_fvgs
                if f.type.value == "BULLISH_FVG"
                and f.top < current_price
                and f.size >= (atr_value * 0.1)
            ]
            best_fvg = max(target_fvgs, key=lambda x: x.strength) if target_fvgs else None

            if best_fvg:
                fvg_height = best_fvg.height
                idea["entry_price"] = best_fvg.top - (fvg_height * entry_factor)
                idea["stop_loss"] = best_fvg.bottom - (atr_value * 0.5)
                idea["take_profit"] = best_fvg.top + (fvg_height * tp_multiplier)
                idea["reason"] = f"Bullish FVG idea (strength: {best_fvg.strength:.1f})"
                return idea

            bullish_obs = [ob for ob in order_blocks if ob.type == 'BULLISH_OB']
            if bullish_obs:
                best_ob = max(bullish_obs, key=lambda x: x.strength)
                entry_price = best_ob.low + (best_ob.high - best_ob.low) * 0.3
                idea["entry_price"] = entry_price
                idea["stop_loss"] = best_ob.low - (atr_value * 0.5)
                idea["take_profit"] = best_ob.high + (best_ob.high - best_ob.low) * tp_multiplier
                idea["reason"] = f"Bullish OB idea (strength: {best_ob.strength:.1f})"
                return idea

            fallback_entry = current_price - (atr_value * 0.3)
            structure_low = structure.last_low.price if structure.last_low else None
            idea["entry_price"] = fallback_entry
            idea["stop_loss"] = (structure_low - (atr_value * 0.2)) if structure_low else (current_price - (atr_value * 1.2))
            idea["take_profit"] = fallback_entry + (atr_value * tp_multiplier)
            idea["reason"] = "Fallback bullish idea"
            return idea

        if signal == "SELL":
            target_fvgs = [
                f for f in valid_fvgs
                if f.type.value == "BEARISH_FVG"
                and f.bottom > current_price
                and f.size >= (atr_value * 0.1)
            ]
            best_fvg = max(target_fvgs, key=lambda x: x.strength) if target_fvgs else None

            if best_fvg:
                fvg_height = best_fvg.height
                idea["entry_price"] = best_fvg.bottom + (fvg_height * entry_factor)
                idea["stop_loss"] = best_fvg.top + (atr_value * 0.5)
                idea["take_profit"] = best_fvg.bottom - (fvg_height * tp_multiplier)
                idea["reason"] = f"Bearish FVG idea (strength: {best_fvg.strength:.1f})"
                return idea

            bearish_obs = [ob for ob in order_blocks if ob.type == 'BEARISH_OB']
            if bearish_obs:
                best_ob = max(bearish_obs, key=lambda x: x.strength)
                entry_price = best_ob.high - (best_ob.high - best_ob.low) * 0.3
                idea["entry_price"] = entry_price
                idea["stop_loss"] = best_ob.high + (atr_value * 0.5)
                idea["take_profit"] = best_ob.low - (best_ob.high - best_ob.low) * tp_multiplier
                idea["reason"] = f"Bearish OB idea (strength: {best_ob.strength:.1f})"
                return idea

            fallback_entry = current_price + (atr_value * 0.3)
            structure_high = structure.last_high.price if structure.last_high else None
            idea["entry_price"] = fallback_entry
            idea["stop_loss"] = (structure_high + (atr_value * 0.2)) if structure_high else (current_price + (atr_value * 1.2))
            idea["take_profit"] = fallback_entry - (atr_value * tp_multiplier)
            idea["reason"] = "Fallback bearish idea"
            return idea

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

    def _create_error_result(self, error_message: str, timeframe: str, current_close: Optional[float]) -> AnalysisResult:
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

    def _determine_volatility(self, atr_long, atr_short):
        """تعیین وضعیت نوسان"""
        volatility_ratio = atr_short / atr_long if atr_long > 0 else 1.0
        
        if volatility_ratio > 1.3:
            return "HIGH_VOLATILITY"
        elif volatility_ratio > 0.8:
            return "MODERATE_VOLATILITY"
        else:
            return "LOW_VOLATILITY"

    def _calculate_scoring_system(self, structure, adx_value, current_rvol, fvgs, sweeps, order_blocks,
                                    current_price, swings, atr_value) -> Tuple[float, List, Dict]:
            """سیستم امتیازدهی پیشرفته NDS"""
            score = 50
            reasons = []
            score_breakdown = {}
            
            # 1. ساختار بازار (35%)
            structure_score = 0
            
            if structure.bos == "BULLISH_BOS":
                bos_strength = min(1.5, adx_value / 25)
                structure_score += 20 * bos_strength
                reasons.append(f"✅ Bullish BOS (ADX: {adx_value:.1f}, Strength: {bos_strength:.2f})")
                score_breakdown['bullish_bos'] = 20 * bos_strength
            
            elif structure.bos == "BEARISH_BOS":
                bos_strength = min(1.5, adx_value / 25)
                structure_score -= 20 * bos_strength
                reasons.append(f"🔻 Bearish BOS (ADX: {adx_value:.1f}, Strength: {bos_strength:.2f})")
                score_breakdown['bearish_bos'] = -20 * bos_strength
            
            if structure.choch == "BULLISH_CHoCH":
                choch_weight = 1.3 if current_rvol > 1.5 else 1.0
                structure_score += 25 * choch_weight
                reasons.append(f"✅ Bullish CHoCH (RVOL: {current_rvol:.1f}x)")
                score_breakdown['bullish_choch'] = 25 * choch_weight
            
            elif structure.choch == "BEARISH_CHoCH":
                choch_weight = 1.3 if current_rvol > 1.5 else 1.0
                structure_score -= 25 * choch_weight
                reasons.append(f"🔻 Bearish CHoCH (RVOL: {current_rvol:.1f}x)")
                score_breakdown['bearish_choch'] = -25 * choch_weight
            
            score += structure_score
            score_breakdown['structure_total'] = structure_score
            
            # 2. روند (25%)
            trend_score = 0
            if structure.trend.value == "UPTREND":
                trend_strength = min(2.0, adx_value / 20)
                volume_boost = 1.2 if current_rvol > 1.2 else 1.0
                trend_score += 15 * trend_strength * volume_boost
                reasons.append(f"📈 Uptrend (ADX: {adx_value:.1f}, RVOL: {current_rvol:.1f}x)")
                score_breakdown['uptrend'] = 15 * trend_strength * volume_boost
            
            elif structure.trend.value == "DOWNTREND":
                trend_strength = min(2.0, adx_value / 20)
                volume_boost = 1.2 if current_rvol > 1.2 else 1.0
                trend_score -= 15 * trend_strength * volume_boost
                reasons.append(f"📉 Downtrend (ADX: {adx_value:.1f}, RVOL: {current_rvol:.1f}x)")
                score_breakdown['downtrend'] = -15 * trend_strength * volume_boost
            
            score += trend_score
            score_breakdown['trend_total'] = trend_score
            
            # 3. FVGها (20%)
            fvg_score = 0
            unfilled_fvgs = [f for f in fvgs if not f.filled]
            recent_fvgs = [f for f in unfilled_fvgs if (len(self.df) - 1 - f.index) <= 10]
            
            for fvg in recent_fvgs:
                if fvg.bottom <= current_price <= fvg.top:
                    size_ratio = fvg.size / atr_value if atr_value > 0 else 1
                    size_score = min(15, size_ratio * 8)
                    volume_boost = 1.5 if fvg.strength > 1.2 else 1.0
                    
                    trend_alignment = 1.3 if (
                        (fvg.type.value == "BULLISH_FVG" and structure.trend.value == "UPTREND") or
                        (fvg.type.value == "BEARISH_FVG" and structure.trend.value == "DOWNTREND")
                    ) else 1.0
                    
                    if fvg.type.value == "BULLISH_FVG":
                        fvg_score += size_score * volume_boost * trend_alignment
                        reasons.append(f"🟢 Bullish FVG (Size: ${fvg.size:.2f}, Strength: {fvg.strength:.1f})")
                        score_breakdown[f'bullish_fvg_{fvg.index}'] = size_score * volume_boost * trend_alignment
                    else:
                        fvg_score -= size_score * volume_boost * trend_alignment
                        reasons.append(f"🔴 Bearish FVG (Size: ${fvg.size:.2f}, Strength: {fvg.strength:.1f})")
                        score_breakdown[f'bearish_fvg_{fvg.index}'] = -size_score * volume_boost * trend_alignment
            
            score += fvg_score
            score_breakdown['fvg_total'] = fvg_score
            
            # 4. سوئیپ‌ها (15%) - نسخه اصلاح شده NDS
            sweep_score = 0
            sweep_types = set()
            
            is_uptrend = structure.trend.value == "UPTREND"
            is_downtrend = structure.trend.value == "DOWNTREND"
            
            for sweep in sweeps[-3:]:
                if sweep.type not in sweep_types:
                    penetration_ratio = sweep.penetration / atr_value if atr_value > 0 else 1
                    sweep_power = min(12, penetration_ratio * 10) * sweep.strength
                    
                    if is_downtrend and sweep.type == 'BULLISH_SWEEP':
                        sweep_power *= 0.3
                        reasons.append(f"⚠️ Weak Bullish Sweep in Downtrend (Penetration: ${sweep.penetration:.2f})")
                    elif is_uptrend and sweep.type == 'BEARISH_SWEEP':
                        sweep_power *= 0.3
                        reasons.append(f"⚠️ Weak Bearish Sweep in Uptrend (Penetration: ${sweep.penetration:.2f})")
                    
                    if sweep.type == 'BEARISH_SWEEP':
                        sweep_score -= sweep_power
                        if not is_uptrend:
                            reasons.append(f"🔻 Bearish Sweep (Penetration: ${sweep.penetration:.2f})")
                        score_breakdown['bearish_sweep'] = -sweep_power
                        
                    elif sweep.type == 'BULLISH_SWEEP':
                        sweep_score += sweep_power
                        if not is_downtrend:
                            reasons.append(f"✅ Bullish Sweep (Penetration: ${sweep.penetration:.2f})")
                        score_breakdown['bullish_sweep'] = sweep_power
                    
                    sweep_types.add(sweep.type)
            
            score += sweep_score
            score_breakdown['sweep_total'] = sweep_score
            
            # 5. Order Blocks (10%)
            ob_score = 0
            recent_obs = order_blocks[-5:] if order_blocks else []
            
            for ob in recent_obs:
                distance_atr = abs(current_price - ob.mid) / atr_value if atr_value > 0 else 1000
                
                if distance_atr < 0.3:
                    ob_power = 8 * ob.strength
                    volume_confirmation = 1.2 if current_rvol > 1.5 else 1.0
                    
                    if ob.type == 'BULLISH_OB' and current_price > ob.low:
                        ob_score += ob_power * volume_confirmation
                        reasons.append(f"🟢 Bullish OB (Strength: {ob.strength:.1f}, Distance: {distance_atr:.2f}ATR)")
                        score_breakdown['bullish_ob'] = ob_power * volume_confirmation
                    elif ob.type == 'BEARISH_OB' and current_price < ob.high:
                        ob_score -= ob_power * volume_confirmation
                        reasons.append(f"🔴 Bearish OB (Strength: {ob.strength:.1f}, Distance: {distance_atr:.2f}ATR)")
                        score_breakdown['bearish_ob'] = -ob_power * volume_confirmation
            
            score += ob_score
            score_breakdown['ob_total'] = ob_score
            
            # نرمال‌سازی
            normalized_score = max(0, min(100, score))
            
            return normalized_score, reasons, score_breakdown

    def _calculate_confidence(self, normalized_score: float, volatility_state: str, 
                                session_analysis: SessionAnalysis, current_rvol: float, 
                                volume_trend: str, scalping_mode: bool = True,
                                sweeps: list = None) -> float:
        """محاسبه اعتماد سیگنال - بهینه شده برای اسکلپ طلا در آسیا"""
        
        # ۱. محاسبه پایه (قدرت فاصله از ۵۰)
        base_confidence = abs(normalized_score - 50) * 2.4
        
        # ۲. اصلاح تاثیر نوسان (باگ فیکس: عدم جریمه سنگین نوسان کم در اسکلپ)
        if volatility_state in ['HIGH_VOLATILITY', 'HIGH']:
            # نوسان بالا برای اسکلپ خوب است، اما نه انفجاری
            base_confidence *= 1.1 if scalping_mode else 0.8
        elif volatility_state in ['LOW_VOLATILITY', 'LOW']:
            # تغییر مهم: در اسکلپ، نوسان کم را فقط ۱۰٪ جریمه کن (نه ۳۰٪) چون در آسیا طبیعی است
            base_confidence *= 0.9 if scalping_mode else 1.2
        
        # ۳. تاثیر سشن (هوشمند)
        if session_analysis.optimal_trading:
            base_confidence *= 1.1
        elif session_analysis.weight < 0.8:
            # اگر سیگنال خیلی قوی است (چه خرید چه فروش)، سشن ضعیف را نادیده بگیر
            is_strong_signal = normalized_score > 85 or normalized_score < 15
            session_penalty = 0.95 if is_strong_signal else 0.75
            base_confidence *= session_penalty
        
        # ۴. تاثیر حجم (تعدیل شده برای RVOLهای معمولی)
        if current_rvol > 2.0:
            base_confidence *= 1.25
        elif current_rvol < 0.5: # حد پایین را از ۰.۴ به ۰.۵ رساندم
            base_confidence *= 0.6
        elif current_rvol < 0.8: # حجم‌های بین ۰.۵ تا ۰.۸ در آسیا قابل قبول‌اند
            base_confidence *= 0.9 # جریمه سبک (قبلاً ۰.۸ بود)
            
        # ۵. تاثیر سوئیپ‌های نقدینگی
        if sweeps:
            for sweep in sweeps:
                # سیگنال خرید (Score > 55) + سوئیپ نزولی (Bearish) = خطر!
                if normalized_score > 55 and sweep.type == 'BEARISH_SWEEP':
                    base_confidence *= 0.8 
                # سیگنال خرید + سوئیپ صعودی (Bullish) = تاییدیه عالی
                elif normalized_score > 55 and sweep.type == 'BULLISH_SWEEP':
                    base_confidence *= 1.15
                
                # اضافه شده: منطق برای سیگنال‌های فروش (Score < 45)
                elif normalized_score < 45 and sweep.type == 'BULLISH_SWEEP':
                     base_confidence *= 0.8 # خطر بازگشت به بالا
                elif normalized_score < 45 and sweep.type == 'BEARISH_SWEEP':
                     base_confidence *= 1.15 # تاییدیه ریزش
        
        # ۶. روند حجم
        if volume_trend == 'INCREASING' and current_rvol > 1.0: # کمی حد را پایین آوردم
            base_confidence = min(95, base_confidence * 1.1)
        
        # ۷. محدودسازی
        confidence = min(95, max(10, base_confidence)) # حداقل ۱۰ باشد
        
        # ۸. ناحیه رنج (Dead Zone)
        if 42 <= normalized_score <= 58:
            confidence *= 0.5
        
        return round(confidence, 1)

    def _determine_signal(self, normalized_score: float, confidence: float, 
                                volatility_state: str, scalping_mode: bool = True) -> str:
        """تعیین سیگنال با آستانه‌های پویا (فیکس شده برای پوزیشن‌های SELL)"""
        
        # ۱. تنظیمات پایه
        if scalping_mode:
            min_conf = self.GOLD_SETTINGS.get('SCALPING_MIN_CONFIDENCE', 35.0)
        else:
            min_conf = self.GOLD_SETTINGS.get('MIN_CONFIDENCE', 40.0)
            
        # --- باگ فیکس حیاتی: اعمال تخفیف برای سیگنال‌های قوی فروش ---
        effective_min_conf = min_conf
        
        # اگر امتیاز > ۸۵ (خرید قوی) یا < ۱۵ (فروش قوی) بود، حد نصاب را پایین بیاور
        if normalized_score > 85 or normalized_score < 15: 
            effective_min_conf = min_conf * 0.7
        
        # بررسی اعتماد
        if confidence < effective_min_conf:
            return "NONE"
        
        # ۲. تعیین آستانه‌ها
        if scalping_mode:
            if volatility_state in ['HIGH_VOLATILITY', 'HIGH']:
                buy_threshold = 60
                sell_threshold = 40
            elif volatility_state in ['LOW_VOLATILITY', 'LOW']:
                buy_threshold = 65
                sell_threshold = 35
            else: 
                buy_threshold = 55  
                sell_threshold = 45 
        else:
            buy_threshold = 65
            sell_threshold = 35
        
        # ۳. صدور سیگنال
        if normalized_score >= buy_threshold:
            return "BUY"
        elif normalized_score <= sell_threshold:
            return "SELL"
        else:
            return "NONE"

    def _build_initial_result(self, signal: str, confidence: float, score: float,
                                reasons: List, structure, atr_value: float, 
                                atr_short_value: Optional[float], adx_value: float, 
                                plus_di: float, minus_di: float,
                                volume_analysis: Dict, recent_range: float, recent_position: float,
                                volatility_state: str, session_analysis: SessionAnalysis,
                                current_price: float, timeframe: str, score_breakdown: Dict,
                                scalping_mode: bool = True) -> Dict:
        """ساخت ساختار نتیجه اولیه"""
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
                # 🔴 FIXED: اضافه کردن structure_score به دیکشنری structure
                "structure_score": structure.structure_score if hasattr(structure, 'structure_score') else 0.0
            },
            "market_metrics": {
                "atr": round(atr_value, 2),
                "adx": round(adx_value, 1),
                "plus_di": round(plus_di, 1),
                "minus_di": round(minus_di, 1),
                "recent_range": round(recent_range, 2),
                "recent_position": round(recent_position, 2),
                "volatility_state": volatility_state,
                "current_rvol": round(volume_analysis.get('rvol', 1), 2)
            },
            "analysis_data": {
                "volume_analysis": volume_analysis,
                "score_breakdown": score_breakdown
            },
            "session_analysis": {  # 🔴 FIXED: اضافه کردن session_analysis
                "current_session": session_analysis.current_session,
                "session_weight": session_analysis.session_weight,
                "is_active_session": session_analysis.is_active_session,
                "optimal_trading": session_analysis.optimal_trading,
                "weight": session_analysis.weight
            },
            "timestamp": datetime.now().isoformat(),
            "current_price": current_price,
            "timeframe": timeframe,
            "scalping_mode": scalping_mode
        }
        
        if scalping_mode and atr_short_value:
            result["market_metrics"]["atr_short"] = round(atr_short_value, 2)
        
        return result

    def _apply_final_filters(self, analysis_result: Dict, scalping_mode: bool = True) -> Dict:
        """
        اعمال فیلترهای نهایی با اتصال به تنظیمات مرکزی (نسخه اصلاح شده)
        """
        original_signal = analysis_result.get('signal', 'NONE')
        reasons = analysis_result.get('reasons', [])

        self._log_debug(f"Final filter start | signal={original_signal}")

        if original_signal != 'NONE':
            settings = self.GOLD_SETTINGS 

            self._log_debug(
                "Filter settings | MIN_RVOL_SCALPING=%s SCALPING_MIN_CONFIDENCE=%s "
                "MIN_SESSION_WEIGHT=%s MIN_STRUCTURE_SCORE=%s",
                settings.get('MIN_RVOL_SCALPING', 'NOT SET'),
                settings.get('SCALPING_MIN_CONFIDENCE', 'NOT SET'),
                settings.get('MIN_SESSION_WEIGHT', 'NOT SET'),
                settings.get('MIN_STRUCTURE_SCORE', 'NOT SET'),
            )

            # ======= ۱. فیلتر RVOL =======
            if scalping_mode:
                base_min_rvol = settings.get('MIN_RVOL_SCALPING', 0.75)
                current_rvol = analysis_result.get('market_metrics', {}).get('current_rvol', 1.0)
                
                # بررسی ساختار برای تنظیم هوشمند
                structure = analysis_result.get('structure', {})
                structure_score = structure.get('structure_score', 0.0)
                
                # منطق تعدیل
                if structure_score >= 90.0:
                    adaptive_min_rvol = base_min_rvol * 0.5
                    adjustment_note = " (Reduced due to STRONG Structure)"
                elif structure_score >= 70.0:
                    adaptive_min_rvol = base_min_rvol * 0.75
                    adjustment_note = " (Slightly reduced due to Good Structure)"
                else:
                    adaptive_min_rvol = base_min_rvol
                    adjustment_note = ""

                self._log_debug(
                    "RVOL filter | base=%.2f current=%.2f structure_score=%.1f adaptive=%.2f%s",
                    base_min_rvol,
                    current_rvol,
                    structure_score,
                    adaptive_min_rvol,
                    adjustment_note,
                )

                if current_rvol < adaptive_min_rvol:
                    analysis_result['signal'] = 'NONE'
                    reason_msg = f"Volume too low (RVOL: {current_rvol:.2f} < {adaptive_min_rvol:.2f})"
                    reasons.append(reason_msg)
                else:
                    if adjustment_note:
                        reasons.append(f"Volume accepted due to high structure score ({structure_score})")
            
            # ======= ۲. فیلتر Confidence =======
            current_confidence = analysis_result.get('confidence', 0.0)
            
            if scalping_mode:
                min_confidence = settings.get('SCALPING_MIN_CONFIDENCE', 45.0)
                confidence_type = "SCALPING"
            else:
                min_confidence = settings.get('MIN_CONFIDENCE', 50.0)
                confidence_type = "NORMAL"

            self._log_debug(
                "Confidence filter | current=%.1f%% %s_min=%.1f%%",
                current_confidence,
                confidence_type,
                min_confidence,
            )

            if current_confidence < min_confidence:
                analysis_result['signal'] = 'NONE'
                reason_msg = f"Confidence too low ({current_confidence:.1f}% < {min_confidence}%)"
                reasons.append(reason_msg)
            
            # ======= ۳. فیلتر وزن سشن =======
            session_analysis = analysis_result.get('session_analysis', {})
            session_weight = session_analysis.get('weight', 0.5)
            min_session_weight = settings.get('MIN_SESSION_WEIGHT', 0.3)

            self._log_debug(
                "Session filter | weight=%.2f min_weight=%.2f",
                session_weight,
                min_session_weight,
            )

            if session_weight < min_session_weight:
                analysis_result['signal'] = 'NONE'
                reason_msg = f"Low session weight ({session_weight:.2f} < {min_session_weight})"
                reasons.append(reason_msg)
            
            # ======= ۴. فیلتر امتیاز ساختار =======
            structure = analysis_result.get('structure', {})
            structure_score = structure.get('structure_score', 0.0)
            min_structure_score = settings.get('MIN_STRUCTURE_SCORE', 20.0)

            self._log_debug(
                "Structure filter | score=%.1f min_score=%.1f",
                structure_score,
                min_structure_score,
            )

            if structure_score < min_structure_score:
                analysis_result['signal'] = 'NONE'
                reason_msg = f"Weak market structure (Score: {structure_score:.1f} < {min_structure_score})"
                reasons.append(reason_msg)
        
        # ثبت دلایل و وضعیت نهایی
        analysis_result['reasons'] = reasons
        final_signal = analysis_result.get('signal', 'NONE')

        if original_signal != final_signal:
            self._log_debug(
                "Final filter changed signal | original=%s final=%s reasons=%s",
                original_signal,
                final_signal,
                reasons,
            )
        else:
            self._log_debug("Final filter result | signal=%s", final_signal)

        return analysis_result

def analyze_gold_market(dataframe: pd.DataFrame, timeframe: str = 'M15',
                        entry_factor: float = 0.25,
                        config: Dict = None, scalping_mode: bool = True) -> AnalysisResult:
    """
    تابع اصلی برای تحلیل بازار طلا و ایجاد سیگنال
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
        logger.info(f"🔄 Creating Gold Analyzer for {mode} timeframe {timeframe} ({len(dataframe)} candles)")
        
        analyzer = GoldNDSAnalyzer(dataframe, config=config)
        result = analyzer.generate_trading_signal(timeframe, entry_factor, scalping_mode)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {str(e)}", exc_info=True)
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
