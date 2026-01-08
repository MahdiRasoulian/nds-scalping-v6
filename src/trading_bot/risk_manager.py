"""
مدیریت ریسک اسکلپینگ حرفه‌ای برای طلا (XAUUSD) - نسخه اسکلپینگ
بهینه‌شده برای معاملات M1-M5 با استراتژی اسکلپینگ NDS
نسخه یکپارچه با bot_config.json
"""

import logging
import numpy as np
from typing import Dict, Optional, Any, Tuple, List, TYPE_CHECKING, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, time, timedelta, timezone
import math

from config.settings import config

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.trading_bot.nds.models import AnalysisResult

@dataclass
class FinalizedOrderParams:
    symbol: str
    signal: str
    order_type: str
    volume: float
    planned_entry: float
    final_entry: float
    sl: float
    tp: float
    risk_usd: float
    rr_ratio: float
    deviation_pips: float
    deviation_ok: bool
    reasons: List[str] = field(default_factory=list)
    risk_details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ScalpingRiskParameters:
    """پارامترهای ریسک محاسبه‌شده برای اسکلپینگ"""
    lot_size: float
    risk_amount: float
    risk_percent: float
    actual_risk_percent: float
    position_value: float
    margin_required: float
    leverage_used: float
    validation_passed: bool
    warnings: list
    notes: list
    calculation_details: Dict[str, Any]
    scalping_specific: Dict[str, Any]  # پارامترهای خاص اسکلپینگ
    
    def __str__(self):
        return (f"Lot: {self.lot_size:.3f}, "
                f"Risk: ${self.risk_amount:.2f} ({self.actual_risk_percent:.3f}%), "
                f"SL Distance: {self.scalping_specific.get('sl_distance', 0):.2f}$, "
                f"Valid: {self.validation_passed}")


class ScalpingRiskManager:
    """
    مدیر ریسک اسکلپینگ حرفه‌ای برای معاملات طلا
    با پشتیبانی کامل از اسکلپینگ با تایم‌فریم کوتاه
    """
    
    GOLD_SPECS = {}
    
# ==================== تنظیمات پیش‌فرض اسکلپینگ ====================
    
    @property
    def DEFAULT_SCALPING_CONFIG(self):
        """تنظیمات اسکلپینگ مبتنی بر bot_config.json"""
        if hasattr(self, 'settings'):
            return self.settings.copy()

        full_config = config.get_full_config()
        return self._merge_with_config(full_config, {})
    
    def __init__(self, config: Dict = None, logger: logging.Logger = None):
        """
        مقداردهی مدیر ریسک اسکلپینگ با ساختار یکپارچه و حرفه‌ای.
        
        Args:
            config: دیکشنری تنظیمات خام (معمولاً از فایل JSON یا خروجی ConfigManager)
            logger: آبجکت لاگر برای ثبت وقایع
        """
        full_config = config.get_full_config()
        if config:
            for key, value in config.items():
                if isinstance(value, dict) and isinstance(full_config.get(key), dict):
                    full_config[key].update(value)
                else:
                    full_config[key] = value
        merged_config = self._merge_with_config(full_config, {})
        
        # ۲. مقداردهی لاگر
        self._logger = logger or logging.getLogger(__name__)
        
        self._logger.info("🔄 bot_config.json merged into RiskManager.")

        # ۴. ذخیره تنظیمات نهایی در self.settings (منبع واحد حقیقت)
        self.settings = merged_config
        
        # جهت سازگاری با کدهای قدیمی که ممکن است از self.config استفاده کنند
        self.config = self.settings 

        trading_settings = full_config.get('trading_settings', {})
        self.GOLD_SPECS = trading_settings.get('GOLD_SPECIFICATIONS', {})

        # ۵. وضعیت ردیابی ریسک اسکلپینگ (بدون تغییر)
        self.daily_risk_used = 0.0
        self.daily_profit_loss = 0.0
        self.active_positions = 0
        self.consecutive_losses = 0
        self.trades_today = 0
        self.scalping_positions = []  # لیست پوزیشن‌های اسکلپینگ فعال
        
        # ۶. آمار اسکلپینگ (بدون تغییر)
        self.scalping_stats = {
            'total_scalps': 0,
            'winning_scalps': 0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'avg_duration': 0.0,
            'best_scalp': 0.0,
            'worst_scalp': 0.0,
        }
        
        self.last_update = datetime.now()
        
        # ۷. لاگ‌های نهایی برای تأیید صحت بارگذاری
        self._logger.info("✅ Scalping Risk Manager Initialized - Gold Scalping Optimized")
        self._logger.info(f"📊 Total parameters loaded: {len(self.settings)}")
        
        # نمایش مقادیر کلیدی در لاگ برای اطمینان از صحت Merge
        # استفاده از نام‌های داخلی که در Mapping تعریف کردیم
        min_conf = self.settings.get('SCALPING_MIN_CONFIDENCE', 'N/A')
        max_sl = self.settings.get('MAX_SL_DISTANCE', 'N/A')
        risk_usd = self.settings.get('SCALPING_RISK_USD', 'N/A')
        
        self._logger.info(f"📝 Key settings: Conf={min_conf}%, MaxSL={max_sl}$, Risk={risk_usd}$")
    
    def _merge_with_config(self, config: Dict, merged_config: Dict) -> Dict:
        """
        نسخه حرفه‌ای و یکپارچه ادغام تنظیمات با استفاده از Mapping داینامیک.
        مطابق با ساختار bot_config.json نسخه جدید.
        """
        
        # ۱. تعریف نگاشت (Mapping): {نام در فایل JSON : نام داخلی در RiskManager}
        mapping = {
            'risk_settings': {
                'MIN_RISK_DOLLARS': 'MIN_RISK_DOLLARS',
                'MIN_RISK_REWARD': 'MIN_RISK_REWARD',
                'MAX_RISK_REWARD': 'MAX_RISK_REWARD',
                'DEFAULT_RISK_REWARD': 'DEFAULT_RISK_REWARD',
                'RISK_AMOUNT_USD': 'SCALPING_RISK_USD',
                'MIN_CONFIDENCE': 'MIN_CONFIDENCE',
                'MAX_PRICE_DEVIATION_PIPS': 'MAX_PRICE_DEVIATION_PIPS',
                'MAX_ENTRY_ATR_DEVIATION': 'MAX_ENTRY_ATR_DEVIATION',
                'LIMIT_ORDER_MIN_CONFIDENCE': 'LIMIT_ORDER_MIN_CONFIDENCE'
            },
            'technical_settings': {
                'ATR_WINDOW': 'ATR_WINDOW',
                'SWING_PERIOD': 'SWING_PERIOD',
                'ADX_WINDOW': 'ADX_WINDOW',
                'FVG_MIN_SIZE_MULTIPLIER': 'FVG_MIN_SIZE_MULTIPLIER',
                'MIN_ATR_DISTANCE_MULTIPLIER': 'MIN_ATR_DISTANCE_MULTIPLIER',
                'ENTRY_FACTOR': 'ENTRY_FACTOR',
                'FIXED_BUFFER': 'FIXED_BUFFER',
                'RANGE_TOLERANCE': 'RANGE_TOLERANCE',
                'MAX_SL_DISTANCE': 'MAX_SL_DISTANCE',
                'MIN_SL_DISTANCE': 'MIN_SL_DISTANCE',
                'SCALPING_MIN_CONFIDENCE': 'SCALPING_MIN_CONFIDENCE',
                'SCALPING_MAX_BARS_BACK': 'SCALPING_MAX_BARS_BACK',
                'SCALPING_MAX_DISTANCE_ATR': 'SCALPING_MAX_DISTANCE_ATR',
                'SCALPING_MIN_FVG_SIZE_ATR': 'SCALPING_MIN_FVG_SIZE_ATR',
                'MIN_RVOL_SCALPING': 'RVOL_THRESHOLD',  # تطبیق با نام RVOL_THRESHOLD در ثابت‌ها
                'ATR_SL_MULTIPLIER': 'ATR_SL_MULTIPLIER'
            },
            'risk_manager_config': {
                'MAX_RISK_PERCENT': 'MAX_RISK_PERCENT',
                'MIN_RISK_PERCENT': 'MIN_RISK_PERCENT',
                'MAX_DAILY_RISK_PERCENT': 'MAX_DAILY_RISK_PERCENT',
                'MAX_POSITIONS': 'MAX_POSITIONS',
                'MAX_DAILY_TRADES': 'MAX_DAILY_TRADES',
                'HIGH_CONFIDENCE': 'HIGH_CONFIDENCE',
                'MIN_RR_RATIO': 'MIN_RR_RATIO',
                'TARGET_RR_RATIO': 'TARGET_RR_RATIO',
                'MAX_LEVERAGE': 'MAX_LEVERAGE',
                'MAX_LOT': 'MAX_LOT_SIZE', # مپ کردن MAX_LOT به نام مورد استفاده در محاسبات لات
                'MAX_LOT_SIZE': 'MAX_LOT_SIZE',
                'POSITION_TIMEOUT_MINUTES': 'POSITION_TIMEOUT_MINUTES'
            }
        }

        # ۲. چرخه ادغام هوشمند (Smart Merge)
        for section_name, fields in mapping.items():
            # بررسی وجود بخش (مثلاً risk_settings) در کانفیگ ورودی
            if section_name in config:
                config_section = config[section_name]
                for json_key, internal_key in fields.items():
                    # اگر کلید در کانفیگ بود، مقدار را جایگزین کن
                    if json_key in config_section:
                        merged_config[internal_key] = config_section[json_key]

        # ۳. مدیریت بخش سشن‌ها (به دلیل ساختار دیکشنری تودرتو)
        if 'sessions_config' in config:
            s_config = config['sessions_config']
            
            # ادغام ضرایب اسکلپینگ سشن‌ها
            if 'SCALPING_SESSION_ADJUSTMENT' in s_config:
                merged_config['SCALPING_SESSION_MULTIPLIERS'] = s_config['SCALPING_SESSION_ADJUSTMENT']

            if 'SCALPING_HOLDING_TIMES' in s_config:
                merged_config['SCALPING_HOLDING_TIMES'] = s_config['SCALPING_HOLDING_TIMES']
            
            # ادغام حداقل وزن سشن
            if 'MIN_SESSION_WEIGHT' in s_config:
                merged_config['MIN_SESSION_WEIGHT'] = s_config['MIN_SESSION_WEIGHT']

        return merged_config
    
    # ==================== متدهای کمکی سشن‌ها ====================
    @staticmethod
    def get_current_scalping_session(dt: datetime = None) -> str:
        """
        Detect current scalping session based on LOCAL trading time (UTC+3).
        This avoids false DEAD_ZONE detection caused by UTC mismatch.
        """

        # ===============================
        # 1. Define trading timezone offset
        # ===============================
        TRADING_UTC_OFFSET = 3  # Iraq / Middle East

        if dt is None:
            dt = datetime.utcnow() + timedelta(hours=TRADING_UTC_OFFSET)

        current_time = dt.time()

        sessions = config.get('sessions_config.SCALPING_SESSIONS', {})

        for session_name, session_data in sessions.items():
            start_hour = session_data.get('start', 0)
            end_hour = session_data.get('end', 0)

            start_time = time(start_hour, 0)
            end_time = time(end_hour, 0)

            # ===============================
            # Normal session (same day)
            # ===============================
            if start_time <= end_time:
                if start_time <= current_time < end_time:
                    return session_name

            # ===============================
            # Overnight session (e.g. 22 → 01)
            # ===============================
            else:
                if current_time >= start_time or current_time < end_time:
                    return session_name

        # ===============================
        # Fallback (safety)
        # ===============================
        return 'DEAD_ZONE'



            
    
    @staticmethod
    def is_scalping_friendly_session(session: str) -> bool:
        """
        بررسی مناسب بودن سشن برای اسکلپینگ.

        - این متد فقط «سازگاری پایه» سشن را بررسی می‌کند
        - تصمیم‌گیری نهایی (مانند DEAD_ZONE override) در can_scalp انجام می‌شود
        """

        # DEAD_ZONE به صورت پیش‌فرض مسدود نمی‌شود
        # منطق اجازه/عدم اجازه آن در can_scalp و بر اساس کیفیت سیگنال است
        if session == 'DEAD_ZONE':
            return True

        session_multiplier = config.get('sessions_config.SCALPING_SESSION_ADJUSTMENT', {}).get(session, 0)

        # سشن‌هایی با ضریب مناسب برای اسکلپینگ
        return session_multiplier >= 0.7

    
    def get_scalping_multiplier(self, session: str) -> float:
        """
        دریافت ضریب ریسک برای اسکلپینگ از منبع واحد تنظیمات.
        """
        # اولویت با تنظیمات داینامیک در self.settings است که در Init لود شده
        multipliers = self.settings.get('SCALPING_SESSION_MULTIPLIERS', {})
        
        # مقادیر از 0.1 (Dead Zone) تا 1.0 (Overlap) متغیر هستند
        multiplier = multipliers.get(session, 0.5)
        
        self._logger.debug(f"🔍 Scalping Session Multiplier for {session}: {multiplier}")
        return multiplier
    
    def get_max_holding_time(self, session: str) -> int:
        """دریافت حداکثر زمان نگهداری بر اساس سشن (دقیقه)."""
        holding_configs = self.settings.get('SCALPING_HOLDING_TIMES', {})
        
        # بازگشت مقدار (پیش‌فرض 60 دقیقه اگر سشن یافت نشد)
        # نکته: مقادیر باید از bot_config.json تامین شوند
        return holding_configs.get(session, 60)
    
    # ==================== متدهای اصلی ====================
    
    def calculate_scalping_position_size(self, 
                                       account_equity: float,
                                       entry_price: float,
                                       stop_loss: float,
                                       take_profit: float,
                                       signal_confidence: float,
                                       atr_value: float = None,
                                       market_volatility: float = 1.0,
                                       session: str = None,
                                       max_risk_usd: float = None) -> 'ScalpingRiskParameters':
        """
        محاسبه حجم معامله اسکلپینگ با پارامترهای بهینه شده و تنظیمات یکپارچه
        """
        # مقداردهی اولیه
        params = ScalpingRiskParameters(
            lot_size=0.0,
            risk_amount=0.0,
            risk_percent=0.0,
            actual_risk_percent=0.0,
            position_value=0.0,
            margin_required=0.0,
            leverage_used=0.0,
            validation_passed=False,
            warnings=[],
            notes=[],
            calculation_details={},
            scalping_specific={}
        )
        
        # دسترسی به تنظیمات یکپارچه شده
        s = self.settings
        
        # 1. اعتبارسنجی اولیه برای اسکلپینگ
        if not self._validate_scalping_parameters(entry_price, stop_loss, take_profit, 
                                                 signal_confidence, atr_value, params):
            return params
        
        # 2. محاسبه فاصله استاپ و اعتبارسنجی با ATR
        sl_distance = abs(entry_price - stop_loss)
        atr_multiplier = s.get('ATR_SL_MULTIPLIER', 1.5)
        
        if atr_value:
            # تطبیق استاپ با ATR
            optimal_sl_distance = atr_value * atr_multiplier
            if sl_distance > optimal_sl_distance * 1.5:
                params.warnings.append(f"SL distance ({sl_distance:.2f}$) > 1.5x optimal ATR-based SL ({optimal_sl_distance:.2f}$)")
        
        # 3. تعیین حداکثر ریسک دلاری برای اسکلپینگ
        if max_risk_usd is None:
            max_risk_usd = self._get_max_scalping_risk_usd(account_equity)
        
        # 4. محاسبه درصد ریسک بر اساس اعتماد
        base_risk_percent = self._calculate_scalping_risk_percent(signal_confidence, account_equity)
        
        # 5. تنظیم بر اساس سشن اسکلپینگ
        if session is None:
            session = self.get_current_scalping_session()
        session_multiplier = self.get_scalping_multiplier(session)
        
        # 6. تنظیم بر اساس نوسان بازار
        volatility_multiplier = self._calculate_scalping_volatility_multiplier(market_volatility)
        
        # 7. تنظیم بر اساس سابقه اسکلپینگ
        history_multiplier = self._calculate_scalping_history_multiplier()
        
        # 8. محاسبه ریسک نهایی اسکلپینگ
        final_risk_percent = base_risk_percent * session_multiplier * \
                             volatility_multiplier * history_multiplier
        
        # محدودیت‌های ریسک اسکلپینگ
        final_risk_percent = self._apply_scalping_risk_limits(final_risk_percent, account_equity, max_risk_usd)
        
        # 9. محاسبه ریسک دلاری
        risk_amount = min((account_equity * final_risk_percent) / 100, max_risk_usd)
        
        # 10. محاسبه حجم معامله اسکلپینگ
        lot_size = self._calculate_scalping_lot_size(entry_price, stop_loss, risk_amount, sl_distance)
        
        # 11. محاسبات مالی
        position_value = lot_size * self.GOLD_SPECS['contract_size'] * entry_price
        margin_required = self._calculate_scalping_margin(lot_size, entry_price)
        actual_risk = self._calculate_actual_scalping_risk(lot_size, entry_price, stop_loss)
        actual_risk_percent = (actual_risk / account_equity) * 100
        
        # 12. محاسبه RR
        rr_ratio = abs(take_profit - entry_price) / sl_distance if sl_distance > 0 else 0
        
        # 13. پر کردن پارامترهای اسکلپینگ
        params.lot_size = lot_size
        params.risk_amount = risk_amount
        params.risk_percent = final_risk_percent
        params.actual_risk_percent = actual_risk_percent
        params.position_value = position_value
        params.margin_required = margin_required
        params.leverage_used = position_value / account_equity
        params.validation_passed = True
        
        # 14. اطلاعات خاص اسکلپینگ
        max_holding = self.get_max_holding_time(session)
        params.scalping_specific = {
            'sl_distance': sl_distance,
            'rr_ratio': rr_ratio,
            'session': session,
            'max_holding_minutes': max_holding,
            'optimal_exit_time': (datetime.now() + timedelta(minutes=max_holding * 0.7)).isoformat(),
            'atr_based': atr_value is not None,
            'atr_value': atr_value,
            'position_id': f"SCLP_{int(datetime.now().timestamp())}",
            'scalping_grade': self._calculate_scalping_grade(rr_ratio, sl_distance, signal_confidence)
        }
        
        # 15. جزئیات محاسبات
        params.calculation_details = {
            'base_risk_percent': base_risk_percent,
            'session_multiplier': session_multiplier,
            'volatility_multiplier': volatility_multiplier,
            'history_multiplier': history_multiplier,
            'final_risk_usd': risk_amount,
            'max_allowed_risk_usd': max_risk_usd,
            'stop_distance': sl_distance,
            'risk_reward_ratio': rr_ratio,
            'account_equity': account_equity,
            'timestamp': datetime.now().isoformat(),
            'scalping_mode': True
        }
        
        self._logger.info(f"📊 Scalping position calculated: {params}")
        return params

    def _normalize_analysis_payload(self, analysis: Union['AnalysisResult', Dict[str, Any]]) -> Dict[str, Any]:
        """Normalize AnalysisResult/dataclass payloads to a dict."""
        if analysis is None:
            return {}
        if isinstance(analysis, dict):
            return analysis
        if hasattr(analysis, "__dataclass_fields__"):
            return asdict(analysis)
        if hasattr(analysis, "__dict__"):
            return dict(analysis.__dict__)
        return {}

    def finalize_order(
        self,
        analysis: Union['AnalysisResult', Dict[str, Any]],
        live_snapshot: Dict[str, float]
    ) -> Optional[FinalizedOrderParams]:
        """
        Finalize an order decision using live market snapshot and unified risk settings.
        """
        analysis_payload = self._normalize_analysis_payload(analysis)
        signal = analysis_payload.get('signal')
        if not signal or signal in ['NONE', 'NEUTRAL']:
            return None

        if not live_snapshot:
            self._logger.warning("❌ Live snapshot missing, cannot finalize order.")
            return None

        bid = live_snapshot.get('bid')
        ask = live_snapshot.get('ask')
        spread = live_snapshot.get('spread')
        if bid is None or ask is None:
            self._logger.warning("❌ Live snapshot missing bid/ask, cannot finalize order.")
            return None

        symbol = analysis_payload.get('symbol') or config.get('trading_settings.SYMBOL')
        planned_entry = analysis_payload.get('entry_price')
        stop_loss = analysis_payload.get('stop_loss')
        take_profit = analysis_payload.get('take_profit')
        confidence = analysis_payload.get('confidence', 0)
        reasons = list(analysis_payload.get('reasons', []))

        if stop_loss is None or take_profit is None:
            self._logger.warning("❌ Missing SL/TP in analysis result, cannot finalize order.")
            return None

        market_entry = ask if signal == 'BUY' else bid
        if planned_entry is None:
            planned_entry = market_entry
            reasons.append("No planned entry from analysis; using market snapshot.")

        deviation = abs(planned_entry - market_entry)
        deviation_pips = deviation * 10
        max_deviation_pips = self.settings.get('MAX_PRICE_DEVIATION_PIPS', 0.0)
        limit_min_confidence = self.settings.get('LIMIT_ORDER_MIN_CONFIDENCE', 74.0)

        order_type = "MARKET"
        final_entry = market_entry
        deviation_ok = True

        if deviation_pips > max_deviation_pips:
            if confidence >= limit_min_confidence:
                order_type = "LIMIT"
                final_entry = planned_entry
                reasons.append(
                    f"Deviation {deviation_pips:.1f} pips > max {max_deviation_pips:.1f}: using LIMIT."
                )
            else:
                reasons.append(
                    f"Deviation {deviation_pips:.1f} pips > max {max_deviation_pips:.1f} with low confidence."
                )
                self._logger.warning("❌ Price deviation exceeded without confidence; order rejected.")
                return None
        else:
            reasons.append(
                f"Deviation {deviation_pips:.1f} pips <= max {max_deviation_pips:.1f}: using MARKET."
            )

        analysis_context = analysis_payload.get('context', {}) or {}
        market_metrics = analysis_payload.get('market_metrics') or analysis_context.get('market_metrics', {})
        atr_value = market_metrics.get('atr_short') or market_metrics.get('atr')
        max_entry_atr_deviation = self.settings.get('MAX_ENTRY_ATR_DEVIATION', None)

        if atr_value and max_entry_atr_deviation:
            atr_deviation = deviation / atr_value if atr_value > 0 else 0
            if atr_deviation > max_entry_atr_deviation:
                reasons.append(
                    f"Entry deviation {atr_deviation:.2f} ATR > max {max_entry_atr_deviation:.2f}."
                )
                deviation_ok = False
                self._logger.warning("❌ Entry deviation exceeds ATR limit; order rejected.")
                return None

        if final_entry != planned_entry:
            entry_delta = final_entry - planned_entry
            stop_loss = stop_loss + entry_delta
            take_profit = take_profit + entry_delta
            reasons.append("Adjusted SL/TP to preserve distances after entry update.")

        sl_distance = abs(final_entry - stop_loss)
        rr_ratio = abs(take_profit - final_entry) / sl_distance if sl_distance > 0 else 0

        account_equity = config.get('ACCOUNT_BALANCE')
        max_risk_usd = self.settings.get('SCALPING_RISK_USD', config.get('risk_settings.RISK_AMOUNT_USD'))
        current_session = self.get_current_scalping_session()
        risk_params = self.calculate_scalping_position_size(
            account_equity=account_equity,
            entry_price=final_entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_confidence=confidence,
            atr_value=atr_value,
            market_volatility=market_metrics.get('volatility_ratio', 1.0),
            session=current_session,
            max_risk_usd=max_risk_usd
        )

        if not risk_params.validation_passed:
            self._logger.warning(f"❌ Risk validation failed: {risk_params.warnings}")
            return None

        min_lot = self.GOLD_SPECS.get('min_lot') or self.GOLD_SPECS.get('MIN_LOT')
        max_lot_spec = self.GOLD_SPECS.get('max_lot') or self.GOLD_SPECS.get('MAX_LOT')
        max_lot_limit = self.settings.get('MAX_LOT_SIZE', max_lot_spec)
        max_lot = min(max_lot_spec, max_lot_limit) if max_lot_spec else max_lot_limit

        if min_lot and risk_params.lot_size <= min_lot:
            reasons.append(f"Volume clamped to min lot {min_lot}.")
        if max_lot and risk_params.lot_size >= max_lot:
            reasons.append(f"Volume clamped to max lot {max_lot}.")

        return FinalizedOrderParams(
            symbol=symbol,
            signal=signal,
            order_type=order_type,
            volume=risk_params.lot_size,
            planned_entry=planned_entry,
            final_entry=final_entry,
            sl=stop_loss,
            tp=take_profit,
            risk_usd=risk_params.risk_amount,
            rr_ratio=rr_ratio,
            deviation_pips=deviation_pips,
            deviation_ok=deviation_ok,
            reasons=reasons,
            risk_details={
                'risk_percent': risk_params.risk_percent,
                'actual_risk_percent': risk_params.actual_risk_percent,
                'scalping_specific': risk_params.scalping_specific,
                'warnings': risk_params.warnings,
            }
        )

    def _validate_scalping_parameters(self, entry: float, sl: float, tp: float,
                                    confidence: float, atr_value: float, 
                                    params: ScalpingRiskParameters) -> bool:
        """اعتبارسنجی پارامترهای اسکلپینگ با استفاده از settings یکپارچه"""
        errors = []
        s = self.settings
        
        # بررسی قیمت‌ها
        if entry <= 0 or sl <= 0 or tp <= 0:
            errors.append("Prices must be positive")
        
        # بررسی جهت SL/TP
        sl_distance = abs(entry - sl)
        is_valid_buy = (sl < entry) and (tp > entry)
        is_valid_sell = (sl > entry) and (tp < entry)
        
        if not (is_valid_buy or is_valid_sell):
            errors.append(f"Invalid SL/TP direction | Entry: {entry}, SL: {sl}, TP: {tp}")
        
        # بررسی اعتماد سیگنال برای اسکلپینگ
        min_confidence = s.get('SCALPING_MIN_CONFIDENCE', 55)
        if confidence < min_confidence:
            errors.append(f"Signal confidence ({confidence}%) below minimum ({min_confidence}%)")
        
        # بررسی فاصله استاپ برای اسکلپینگ
        min_sl_distance = s.get('MIN_SL_DISTANCE', 2.0)
        max_sl_distance = s.get('MAX_SL_DISTANCE', 10.0)
        
        if sl_distance < min_sl_distance:
            errors.append(f"Stop distance ({sl_distance:.2f}$) too small (min: {min_sl_distance}$)")
        
        if sl_distance > max_sl_distance:
            errors.append(f"Stop distance ({sl_distance:.2f}$) too large (max: {max_sl_distance}$)")
        
        # بررسی نسبت ریسک/پاداش برای اسکلپینگ
        rr_ratio = abs(tp - entry) / sl_distance if sl_distance > 0 else 0
        min_rr_ratio = s.get('MIN_RISK_REWARD', 1.0)
        
        if rr_ratio < min_rr_ratio:
            errors.append(f"Risk/Reward ratio ({rr_ratio:.2f}) below minimum ({min_rr_ratio})")
        
        # بررسی با ATR
        if atr_value and atr_value > 0:
            atr_multiplier = s.get('ATR_SL_MULTIPLIER', 1.5)
            optimal_sl = atr_value * atr_multiplier
            if sl_distance > optimal_sl * 2.0:
                errors.append(f"Stop distance ({sl_distance:.2f}$) > 2x ATR-based stop")
        
        if errors:
            params.warnings.extend(errors)
            self._logger.warning(f"❌ Scalping validation failed: {errors[:3]}")
            return False
        
        return True

    def _get_max_scalping_risk_usd(self, account_equity: float) -> float:
        """دریافت حداکثر ریسک دلاری برای اسکلپینگ"""
        s = self.settings
        max_risk_percent = s.get('MAX_RISK_PERCENT', 0.5)
        max_risk_usd = (account_equity * max_risk_percent) / 100
        
        # محدودیت مطلق اسکلپینگ از نگاشت جدید
        scalping_risk_limit = s.get('SCALPING_RISK_USD', 50.0)
        return min(max_risk_usd, scalping_risk_limit)

    def _calculate_scalping_risk_percent(self, confidence: float, account_equity: float) -> float:
        """محاسبه درصد ریسک برای اسکلپینگ بر اساس اعتماد"""
        s = self.settings
        min_confidence = s.get('SCALPING_MIN_CONFIDENCE', 55)
        high_confidence = s.get('HIGH_CONFIDENCE', 85)
        
        if confidence >= high_confidence:
            base_risk = 0.5
        elif confidence >= min_confidence:
            range_confidence = high_confidence - min_confidence
            normalized = (confidence - min_confidence) / range_confidence
            base_risk = 0.1 + (0.4 * normalized)
        else:
            base_risk = 0.0
        
        # اعمال حداقل ریسک دلاری
        min_risk_dollars = s.get('MIN_RISK_DOLLARS', 0.5)
        min_risk_percent = (min_risk_dollars / account_equity) * 100
        return max(base_risk, min_risk_percent)

    def _calculate_scalping_volatility_multiplier(self, volatility: float) -> float:
        """محاسبه ضریب نوسان برای اسکلپینگ بر اساس VOLATILITY_STATES"""
        v_thresholds = config.get('technical_settings.VOLATILITY_STATES', {})
        
        if volatility < v_thresholds.get('MODERATE_VOLATILITY', {}).get('threshold', 0.8):
            return 0.7
        elif volatility > v_thresholds.get('HIGH_VOLATILITY', {}).get('threshold', 1.3):
            return 0.6
        elif 0.9 <= volatility <= 1.1:
            return 1.0
        else:
            return 0.8

    def _calculate_scalping_history_multiplier(self) -> float:
        """محاسبه ضریب بر اساس سابقه اسکلپینگ و تنظیمات یکپارچه"""
        s = self.settings
        multiplier = 1.0
        
        if self.consecutive_losses >= 2:
            multiplier *= 0.5
            self._logger.warning(f"Consecutive scalping losses: {self.consecutive_losses}")
        
        max_trades_per_day = s.get('MAX_DAILY_TRADES', 20)
        if self.trades_today >= max_trades_per_day * 0.8:
            reduction = 1.0 - (self.trades_today / max_trades_per_day)
            multiplier *= max(0.3, reduction)
        
        if self.scalping_stats['total_scalps'] > 10:
            win_rate = self.scalping_stats['winning_scalps'] / self.scalping_stats['total_scalps']
            if win_rate < 0.5:
                multiplier *= 0.7
        
        return max(0.2, multiplier)

    def _apply_scalping_risk_limits(self, risk_percent: float, account_equity: float, 
                                   max_risk_usd: float) -> float:
        """اعمال محدودیت‌های ریسک با استفاده از تنظیمات یکپارچه"""
        s = self.settings
        
        min_risk_dollars = s.get('MIN_RISK_DOLLARS', 0.5)
        min_risk_percent = (min_risk_dollars / account_equity) * 100
        risk_percent = max(risk_percent, min_risk_percent)
        
        max_daily_percent = s.get('MAX_DAILY_RISK_PERCENT', 1.0)
        daily_risk_left = max_daily_percent - ((self.daily_risk_used / account_equity) * 100)
        risk_percent = min(risk_percent, max(0, daily_risk_left))
        
        max_risk_percent_from_usd = (max_risk_usd / account_equity) * 100
        risk_percent = min(risk_percent, max_risk_percent_from_usd)
        
        return risk_percent

    def _calculate_scalping_lot_size(self, entry_price: float, stop_loss: float, 
                                    risk_amount: float, sl_distance: float) -> float:
        """محاسبه حجم اسکلپینگ با دقت بالا"""
        risk_per_standard_lot = sl_distance * self.GOLD_SPECS['tick_value_per_lot']
        
        if risk_per_standard_lot <= 0:
            return self.GOLD_SPECS['min_lot']
        
        raw_lot = risk_amount / risk_per_standard_lot
        lot_step = self.GOLD_SPECS['lot_step']
        
        if lot_step > 0:
            steps = round(raw_lot / lot_step)
            calculated_lot = steps * lot_step
        else:
            calculated_lot = raw_lot
        
        min_lot = self.GOLD_SPECS['min_lot']
        # استفاده از تنظیمات مپ شده برای حداکثر حجم
        max_lot_limit = self.settings.get('MAX_LOT_SIZE', 2.0)
        max_lot = min(self.GOLD_SPECS['max_lot'], max_lot_limit)
        
        if calculated_lot > max_lot * 0.5:
            calculated_lot = max_lot * 0.5
        
        final_lot = max(min_lot, min(calculated_lot, max_lot))
        return round(final_lot, 3)

    def _calculate_scalping_margin(self, lot_size: float, entry_price: float) -> float:
        """محاسبه مارجین برای اسکلپینگ"""
        contract_value = lot_size * self.GOLD_SPECS['contract_size'] * entry_price
        leverage = self.settings.get('MAX_LEVERAGE', 50)
        margin = contract_value / leverage
        return margin * 1.05

    def _calculate_actual_scalping_risk(self, lot_size: float, entry_price: float, 
                                        stop_loss: float) -> float:
        """محاسبه ریسک واقعی اسکلپینگ"""
        sl_distance = abs(entry_price - stop_loss)
        risk_per_tick = lot_size * self.GOLD_SPECS['tick_value_per_lot']
        return sl_distance * risk_per_tick

    def _calculate_scalping_grade(self, rr_ratio: float, sl_distance: float, 
                               confidence: float) -> str:
        """محاسبه گرید کیفی اسکلپینگ با تنظیمات مپ شده"""
        score = 0
        s = self.settings
        
        # امتیاز RR
        min_rr = s.get('MIN_RISK_REWARD', 1.0)
        target_rr = s.get('DEFAULT_RISK_REWARD', 1.2)
        
        if rr_ratio >= target_rr * 1.25: score += 3
        elif rr_ratio >= target_rr: score += 2
        elif rr_ratio >= min_rr: score += 1
        
        # امتیاز SL distance
        max_sl = s.get('MAX_SL_DISTANCE', 10.0)
        if sl_distance <= max_sl * 0.5: score += 3
        elif sl_distance <= max_sl * 0.7: score += 2
        elif sl_distance <= max_sl: score += 1
        
        # امتیاز اعتماد
        high_conf = s.get('HIGH_CONFIDENCE', 85)
        min_conf = s.get('SCALPING_MIN_CONFIDENCE', 55)
        
        if confidence >= high_conf: score += 3
        elif confidence >= (high_conf + min_conf) / 2: score += 2
        elif confidence >= min_conf: score += 1
        
        grades = {8: "A+", 6: "A", 4: "B", 2: "C", 0: "D"}
        for threshold, grade in grades.items():
            if score >= threshold: return grade
        return "D"

    def update_scalping_trade_result(self, profit_loss: float, position_size: float, 
                                    duration_minutes: float):
        """به‌روزرسانی وضعیت پس از بسته شدن معامله اسکلپینگ"""
        self.daily_profit_loss += profit_loss
        self.daily_risk_used += abs(profit_loss)
        self.scalping_stats['total_scalps'] += 1
        
        if profit_loss > 0:
            self.scalping_stats['winning_scalps'] += 1
            ws = self.scalping_stats['winning_scalps']
            self.scalping_stats['avg_win'] = ((self.scalping_stats['avg_win'] * (ws - 1) + profit_loss) / ws)
            self.consecutive_losses = 0
            if profit_loss > self.scalping_stats['best_scalp']:
                self.scalping_stats['best_scalp'] = profit_loss
        else:
            self.consecutive_losses += 1
            loss_count = self.scalping_stats['total_scalps'] - self.scalping_stats['winning_scalps']
            if loss_count > 0:
                self.scalping_stats['avg_loss'] = ((self.scalping_stats['avg_loss'] * (loss_count - 1) + abs(profit_loss)) / loss_count)
            if profit_loss < self.scalping_stats['worst_scalp']:
                self.scalping_stats['worst_scalp'] = profit_loss
        
        self.scalping_stats['avg_duration'] = ((self.scalping_stats['avg_duration'] * (self.scalping_stats['total_scalps'] - 1) + duration_minutes) / self.scalping_stats['total_scalps'])
        self.trades_today += 1
        self.active_positions = max(0, self.active_positions - 1)
        
        self._logger.info(f"Scalping trade result: PnL=${profit_loss:.2f}, Daily PnL=${self.daily_profit_loss:.2f}")

    def can_scalp(self, account_equity: float) -> Tuple[bool, str]:
        """
        بررسی امکان اسکلپینگ جدید با تنظیمات مپ شده
        - بدون حذف منطق‌های قبلی
        - با DEAD_ZONE override واقعی و enforce شده
        """
        reasons = []
        s = self.settings

        # ===============================
        # 1. Daily Risk Limit
        # ===============================
        max_daily_percent = s.get('MAX_DAILY_RISK_PERCENT', 1.0)
        daily_risk_used_percent = (
            (self.daily_risk_used / account_equity) * 100
            if account_equity > 0 else 0
        )

        if daily_risk_used_percent >= max_daily_percent:
            reasons.append(f"Daily risk limit reached ({daily_risk_used_percent:.1f}%)")

        # ===============================
        # 2. Consecutive Losses
        # ===============================
        if self.consecutive_losses >= 2:
            reasons.append(f"Consecutive losses: {self.consecutive_losses}")

        # ===============================
        # 3. Active Positions Limit
        # ===============================
        max_positions = s.get('MAX_POSITIONS', 4)
        if self.active_positions >= max_positions:
            reasons.append(f"Active positions: {self.active_positions}/{max_positions}")

        # ===============================
        # 4. Daily Trades Limit
        # ===============================
        max_trades = s.get('MAX_DAILY_TRADES', 20)
        if self.trades_today >= max_trades:
            reasons.append(f"Daily trade limit: {self.trades_today}/{max_trades}")

        # ===============================
        # 5. Scalping Session Handling (FIXED)
        # ===============================
        current_session = self.get_current_scalping_session()

        if not self.is_scalping_friendly_session(current_session):

            # ===== DEAD_ZONE OVERRIDE =====
            if current_session == 'DEAD_ZONE':
                confidence = getattr(self, 'last_signal_confidence', 0.0)
                adx = getattr(self, 'last_adx', 0.0)

                if confidence >= 65.0 and adx >= 20.0:
                    # ✅ اجازه معامله در DEAD_ZONE
                    self.session_risk_multiplier = 0.4

                    self.logger.info(
                        f"🔥 DEAD_ZONE override accepted | "
                        f"Confidence={confidence:.1f}% | ADX={adx:.1f}"
                    )
                else:
                    reasons.append(f"Non-optimal session: {current_session}")
            else:
                reasons.append(f"Non-optimal session: {current_session}")

        # ===============================
        # 6. Final Decision (CRITICAL FIX)
        # ===============================
        if reasons:
            return False, " | ".join(reasons)

        return True, "OK"

    
    def get_scalping_summary(self) -> Dict[str, Any]:
        """دریافت خلاصه وضعیت اسکلپینگ"""
        current_session = self.get_current_scalping_session()
        return {
            'daily_risk_used': self.daily_risk_used,
            'daily_profit_loss': self.daily_profit_loss,
            'active_positions': self.active_positions,
            'consecutive_losses': self.consecutive_losses,
            'trades_today': self.trades_today,
            'scalping_stats': self.scalping_stats,
            'last_update': self.last_update.isoformat(),
            'can_scalp': self.can_scalp(1000)[0],
            'current_session': current_session,
            'session_friendly': self.is_scalping_friendly_session(current_session),
            'session_multiplier': self.get_scalping_multiplier(current_session),
            'max_holding_minutes': self.get_max_holding_time(current_session)
        }


# تابع اصلی برای اسکلپینگ
def create_scalping_risk_manager(config: Dict = None, **kwargs) -> ScalpingRiskManager:
    """
    ایجاد مدیر ریسک اسکلپینگ
    
    Args:
        config: دیکشنری تنظیمات
        **kwargs: پارامترهای اضافی
    
    Returns:
        ScalpingRiskManager: نمونه ایجاد شده
    """
    return ScalpingRiskManager(config=config, **kwargs)


# تست عملکرد
if __name__ == "__main__":
    print("🧪 Testing Gold Scalping Risk Manager...")
    
    # استفاده از config متمرکز
    test_config = {
        'risk_manager_config': {
            'MAX_RISK_PERCENT': 0.5,
            'MIN_RISK_PERCENT': 0.05,
            'MAX_DAILY_RISK_PERCENT': 1.0,
            'MAX_POSITIONS': 3,
            'MAX_DAILY_TRADES': 20,
            'MIN_CONFIDENCE': 65,
            'HIGH_CONFIDENCE': 85,
            'MAX_SL_DISTANCE': 10.0,
            'MIN_SL_DISTANCE': 2.0,
            'ATR_SL_MULTIPLIER': 1.0,
            'MIN_RR_RATIO': 1.0,
            'TARGET_RR_RATIO': 1.2,
            'MAX_LEVERAGE': 50,
            'MAX_LOT_SIZE': 2.0,
            'MIN_RISK_USD': 5.0,
            'MAX_RISK_USD': 50.0,
            'POSITION_TIMEOUT_MINUTES': 60,
        }
    }
    
    # ایجاد مدیر ریسک اسکلپینگ
    srm = ScalpingRiskManager(config=test_config)
    
    # تست محاسبه حجم اسکلپینگ
    params = srm.calculate_scalping_position_size(
        account_equity=10000.0,
        entry_price=2150.0,
        stop_loss=2145.0,      # 5 دلار فاصله (اسکلپینگ)
        take_profit=2156.0,    # 6 دلار سود (RR=1.2)
        signal_confidence=80.0,
        atr_value=6.5,
        market_volatility=1.1,
        session='OVERLAP_PEAK',
        max_risk_usd=30.0
    )
    
    print(f"\n✅ Scalping Test Results:")
    print(f"   Lot Size: {params.lot_size:.3f}")
    print(f"   Risk Amount: ${params.risk_amount:.2f}")
    print(f"   Risk Percent: {params.risk_percent:.3f}%")
    print(f"   Actual Risk: {params.actual_risk_percent:.3f}%")
    print(f"   SL Distance: {params.scalping_specific.get('sl_distance', 0):.2f}$")
    print(f"   RR Ratio: {params.scalping_specific.get('rr_ratio', 0):.2f}")
    print(f"   Scalping Grade: {params.scalping_specific.get('scalping_grade', 'N/A')}")
    print(f"   Max Holding: {params.scalping_specific.get('max_holding_minutes', 0)}min")
    print(f"   Validation: {'PASS' if params.validation_passed else 'FAIL'}")
    
    if params.warnings:
        print(f"   Warnings: {params.warnings}")
    
    # تست بررسی امکان معامله
    can_scalp, reason = srm.can_scalp(10000.0)
    print(f"\n✅ Can Scalp: {can_scalp} - {reason}")
    
    # تست خلاصه وضعیت
    summary = srm.get_scalping_summary()
    print(f"\n✅ Current Session: {summary['current_session']}")
    print(f"   Session Friendly: {summary['session_friendly']}")
    print(f"   Session Multiplier: {summary['session_multiplier']:.2f}")
    
    print("\n✅ Gold Scalping Risk Manager test completed successfully!")
