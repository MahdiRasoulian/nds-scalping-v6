"""
تحلیل ساختار بازار و الگوهای SMC
"""
import logging
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd

from .models import (
    SwingPoint, SwingType, FVG, FVGType,
    OrderBlock, LiquiditySweep, MarketStructure, MarketTrend
)

logger = logging.getLogger(__name__)


class SMCAnalyzer:
    """
    تحلیل‌گر ساختار بازار و الگوهای Smart Money Concepts
    """

    def __init__(self, df: pd.DataFrame, atr_value: float, settings: dict = None):
        if settings is None:
            raise ValueError("SMCAnalyzer requires settings from bot_config.json")
        self.df = df
        self.atr = atr_value
        self.GOLD_SETTINGS = settings
        self.settings = self.GOLD_SETTINGS
        self.debug_smc = bool(self.settings.get("DEBUG_SMC", False))
        self._prepare_data()

    def _log_debug(self, message: str, *args: Any) -> None:
        if self.debug_smc:
            logger.debug(message, *args)

    def _log_info(self, message: str, *args: Any) -> None:
        logger.info(message, *args)

    def _log_verbose(self, message: str, *args: Any) -> None:
        """لاگ مرحله‌ای با سطح INFO (فقط وقتی DEBUG_SMC فعال باشد).

        در نسخه‌های قبلی پروژه، لاگ‌های مفصل در خروجی INFO نمایش داده می‌شد.
        این متد همان رفتار را (به صورت کنترل‌شده) بازمی‌گرداند تا بتوانید
        قدم‌به‌قدم رفتار ربات را رصد کنید.
        """
        if self.debug_smc:
            logger.info(message, *args)

    def _normalize_volatility_state(self, volatility_state: Optional[str]) -> Optional[str]:
        if not volatility_state:
            return None
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
        return state

    def _prepare_data(self) -> None:
        """آماده‌سازی داده‌های پایه"""
        self.df = self.df.copy()
        self.df['body'] = abs(self.df['close'] - self.df['open'])
        self.df['range'] = self.df['high'] - self.df['low']
        self.df['body_ratio'] = self.df['body'] / self.df['range'].replace(0, 0.001)
        self.df['mid_price'] = (self.df['high'] + self.df['low']) / 2

    def _get_swing_period(self, timeframe: str) -> int:
        """تعیین دوره سوینگ بر اساس تایم‌فریم"""
        swing_period_map = self.settings.get('SWING_PERIOD_MAP', {})
        if timeframe.upper() in swing_period_map:
            return swing_period_map[timeframe.upper()]

        raise KeyError(f"Missing SWING_PERIOD_MAP for timeframe: {timeframe}")

    def detect_swings(self, timeframe: str = 'M15') -> List[SwingPoint]:
        """
        نسخه نهایی و بهینه‌شده شناسایی سوینگ برای انس جهانی طلا
        تمرکز بر دقت در تایید ساختار (BOS/CHOCH)
        """
        period = self._get_swing_period(timeframe)
        df = self.df.reset_index(drop=True)

        if len(df) < period * 2 + 1:
            self._log_debug(
                "[NDS][SMC][SWINGS] insufficient data (have=%s need=%s)",
                len(df),
                period * 2 + 1,
            )
            return []

        high_series = df['high']
        low_series = df['low']

        high_rolling_max = high_series.rolling(window=2 * period + 1, center=True).max()
        low_rolling_min = low_series.rolling(window=2 * period + 1, center=True).min()

        valid_range = range(period, len(df) - period)
        high_indices = [i for i in high_series[high_series == high_rolling_max].index if i in valid_range]
        low_indices = [i for i in low_series[low_series == low_rolling_min].index if i in valid_range]

        self._log_verbose(
            "[NDS][SMC][SWINGS] فرکتال‌های اولیه - High: %s, Low: %s",
            len(high_indices),
            len(low_indices),
        )

        if not high_indices and not low_indices:
            self._log_debug("[NDS][SMC][SWINGS] no initial fractals found")

        min_distance = self.atr * self.settings.get('MIN_ATR_DISTANCE_MULTIPLIER', 1.2)
        min_vol_mult = self.settings.get('MIN_VOLUME_MULTIPLIER', 0.6)
        has_volume = 'volume' in df.columns

        high_swings = []
        last_h_price = None
        for idx in high_indices:
            price = float(df['high'].iloc[idx])
            volume_ok = True
            avg_vol = 1.0
            current_vol = None
            if has_volume:
                recent_vol = df['volume'].iloc[max(0, idx - period):idx]
                avg_vol = float(recent_vol.mean()) if not recent_vol.empty else 1.0
                if pd.isna(avg_vol):
                    avg_vol = 1.0
                current_vol = float(df['volume'].iloc[idx])
                if pd.isna(current_vol):
                    current_vol = avg_vol
                volume_ok = current_vol >= avg_vol * min_vol_mult

            if volume_ok and (last_h_price is None or abs(price - last_h_price) >= min_distance):
                high_swings.append(SwingPoint(
                    index=idx,
                    price=price,
                    time=df['time'].iloc[idx],
                    type=SwingType.HIGH,
                    side='HIGH'
                ))
                last_h_price = price
            self._log_debug(
                "[NDS][SMC][SWINGS] high idx=%s price=%.2f volume_ok=%s avg_vol=%.2f current_vol=%s",
                idx,
                price,
                volume_ok,
                avg_vol,
                f"{current_vol:.2f}" if current_vol is not None else "N/A",
            )

        low_swings = []
        last_l_price = None
        for idx in low_indices:
            price = float(df['low'].iloc[idx])
            volume_ok = True
            avg_vol = 1.0
            current_vol = None
            if has_volume:
                recent_vol = df['volume'].iloc[max(0, idx - period):idx]
                avg_vol = float(recent_vol.mean()) if not recent_vol.empty else 1.0
                if pd.isna(avg_vol):
                    avg_vol = 1.0
                current_vol = float(df['volume'].iloc[idx])
                if pd.isna(current_vol):
                    current_vol = avg_vol
                volume_ok = current_vol >= avg_vol * min_vol_mult

            if volume_ok and (last_l_price is None or abs(price - last_l_price) >= min_distance):
                low_swings.append(SwingPoint(
                    index=idx,
                    price=price,
                    time=df['time'].iloc[idx],
                    type=SwingType.LOW,
                    side='LOW'
                ))
                last_l_price = price
            self._log_debug(
                "[NDS][SMC][SWINGS] low idx=%s price=%.2f volume_ok=%s avg_vol=%.2f current_vol=%s",
                idx,
                price,
                volume_ok,
                avg_vol,
                f"{current_vol:.2f}" if current_vol is not None else "N/A",
            )

        self._log_verbose(
            "[NDS][SMC][SWINGS] سوینگ‌های اولیه - High: %s, Low: %s",
            len(high_swings),
            len(low_swings),
        )

        all_swings = sorted(high_swings + low_swings, key=lambda x: x.index)
        if not all_swings:
            self._log_debug("[NDS][SMC][SWINGS] no swings after filters")
            return []

        cleaned = self._clean_consecutive_swings(all_swings)
        if cleaned:
            all_swings = cleaned
            self._log_debug("[NDS][SMC][SWINGS] cleaning result=%s", len(all_swings))
        else:
            self._log_debug("[NDS][SMC][SWINGS] cleaning removed all swings")
            return []

        meaningful = self._filter_meaningful_swings(all_swings)
        if meaningful:
            all_swings = meaningful
            self._log_debug("[NDS][SMC][SWINGS] meaningful swings=%s", len(all_swings))
        else:
            self._log_debug("[NDS][SMC][SWINGS] meaningful filter removed all swings")
            return []

        last_h, last_l = None, None
        for swing in all_swings:
            if swing.side == 'HIGH':
                if last_h:
                    swing.type = SwingType.HH if swing.price > last_h.price else SwingType.LH
                else:
                    swing.type = SwingType.HIGH
                last_h = swing
            else:
                if last_l:
                    swing.type = SwingType.LL if swing.price < last_l.price else SwingType.HL
                else:
                    swing.type = SwingType.LOW
                last_l = swing

        self._log_debug("[NDS][SMC][SWINGS] final swings=%s", len(all_swings))
        self._log_info("[NDS][SMC][SWINGS] detected swings=%s", len(all_swings))

        if self.debug_smc and all_swings:
            # نمایش چند سوینگ آخر برای رصد سریع ساختار
            self._log_verbose(
                "[NDS][SMC][SWINGS] Swing Types: HH=%s, LH=%s, LL=%s, HL=%s",
                sum(1 for s in all_swings if getattr(s, 'type', None) == SwingType.HH),
                sum(1 for s in all_swings if getattr(s, 'type', None) == SwingType.LH),
                sum(1 for s in all_swings if getattr(s, 'type', None) == SwingType.LL),
                sum(1 for s in all_swings if getattr(s, 'type', None) == SwingType.HL),
            )
            tail = all_swings[-3:]
            for idx, s in enumerate(tail, start=max(1, len(all_swings)-2)):
                self._log_verbose(
                    "[NDS][SMC][SWINGS] Swing %s: %s@%.2f (%s) time=%s",
                    idx,
                    getattr(s, 'side', 'N/A'),
                    float(getattr(s, 'price', 0.0)),
                    getattr(getattr(s, 'type', None), 'value', str(getattr(s, 'type', 'N/A'))),
                    getattr(s, 'time', 'N/A'),
                )

        return all_swings

    def _clean_consecutive_swings(self, swings: List[SwingPoint]) -> List[SwingPoint]:
        """حذف سوینگ‌های تکراری در یک سمت برای به دست آوردن ساختار زیگزاگی تمیز"""
        if not swings:
            self._log_debug("[NDS][SMC][SWINGS] cleaning empty input")
            return []

        cleaned = []
        for s in swings:
            if not cleaned:
                cleaned.append(s)
                continue

            last = cleaned[-1]
            if last.side == s.side:
                if s.side == 'HIGH' and s.price > last.price:
                    cleaned[-1] = s
                    self._log_debug(
                        "[NDS][SMC][SWINGS] replace high %.2f -> %.2f",
                        last.price,
                        s.price,
                    )
                elif s.side == 'LOW' and s.price < last.price:
                    cleaned[-1] = s
                    self._log_debug(
                        "[NDS][SMC][SWINGS] replace low %.2f -> %.2f",
                        last.price,
                        s.price,
                    )
            else:
                cleaned.append(s)

        return cleaned

    def _filter_meaningful_swings(self, swings: List[SwingPoint]) -> List[SwingPoint]:
        """حذف نوسانات فرسایشی که حرکت قیمتی موثری ندارند"""
        if len(swings) < 3:
            self._log_debug("[NDS][SMC][SWINGS] meaningful short list (%s swings)", len(swings))
            return swings

        atr_threshold = self.atr * self.settings.get('MEANINGFUL_MOVE_MULT', 0.5)
        meaningful = []

        for i, s in enumerate(swings):
            if i == 0 or i == len(swings) - 1:
                meaningful.append(s)
                continue

            move_size = abs(s.price - swings[i - 1].price)
            if move_size >= atr_threshold:
                meaningful.append(s)
            elif i + 1 < len(swings):
                next_move_size = abs(swings[i + 1].price - s.price)
                if next_move_size >= atr_threshold:
                    meaningful.append(s)

        return meaningful


def detect_fvgs(self) -> List[FVG]:
    """شناسایی FVGها با تشخیص صحیح «باز بودن» (Unfilled) و لاگ مرحله‌ای.

    مشکل گزارش‌شده (تعداد زیاد FVG اما unfilled=0) معمولاً از اینجا می‌آید که
    منطق «پر شدن» خیلی سهل‌گیرانه باشد (مثلاً صرفاً لمس لبه‌ی بالایی/پایینی را filled
    حساب کند). در SMC برای *Mitigation کامل*، بهتر است عبور قیمت از مرز مقابل gap
    معیار filled باشد.

    تنظیمات مرتبط:
    - FVG_MIN_SIZE_MULTIPLIER (پیش‌فرض: 0.1)
    - FVG_LOOKAHEAD_BARS (پیش‌فرض: 80)
    - FVG_FILL_MODE: یکی از {"full", "touch"} (پیش‌فرض: "full")
        * full  : filled=True فقط وقتی کل گپ «میتگیِیت» شود (عبور از مرز مقابل).
        * touch : filled=True با اولین ورود/لمس ناحیه گپ.
    """
    df = self.df
    fvg_list: List[FVG] = []

    if len(df) < 3:
        return fvg_list

    min_fvg_size = float(self.atr * self.settings.get('FVG_MIN_SIZE_MULTIPLIER', 0.1))
    fill_mode = str(self.settings.get("FVG_FILL_MODE", "full")).strip().lower()
    if fill_mode not in ("full", "touch"):
        fill_mode = "full"

    self._log_verbose(
        "[NDS][SMC][FVG] start | candles=%s atr=%.4f min_size=%.4f fill_mode=%s",
        len(df),
        float(self.atr),
        min_fvg_size,
        fill_mode,
    )

    # --- 1) Detection (3-candle pattern) ---
    for i in range(2, len(df)):
        # Candle-2 (middle candle) is i-1
        c2_high = float(df['high'].iloc[i - 1])
        c2_low = float(df['low'].iloc[i - 1])
        c2_close = float(df['close'].iloc[i - 1])
        c2_open = float(df['open'].iloc[i - 1])

        c2_body = abs(c2_close - c2_open)
        c2_range = max(0.0000001, c2_high - c2_low)  # جلوگیری از تقسیم بر صفر

        # Candle-1 (left) and Candle-3 (right)
        c1_high = float(df['high'].iloc[i - 2])
        c1_low = float(df['low'].iloc[i - 2])
        c3_high = float(df['high'].iloc[i])
        c3_low = float(df['low'].iloc[i])

        # ---- Bullish FVG: c3_low > c1_high ----
        if c3_low > c1_high:
            gap_top = c3_low          # upper boundary
            gap_bottom = c1_high      # lower boundary
            gap_size = gap_top - gap_bottom

            body_condition = c2_close > c2_open
            body_size_condition = c2_body > (c2_range * 0.3)
            size_condition = gap_size >= min_fvg_size

            volume_condition = True
            if 'rvol' in df.columns:
                try:
                    volume_condition = float(df['rvol'].iloc[i - 1]) > 0.8
                except (TypeError, ValueError):
                    volume_condition = True

            if body_condition and body_size_condition and size_condition and volume_condition:
                strength = 1.0
                if c2_body > c2_range * 0.7:
                    strength = 1.5
                if 'rvol' in df.columns:
                    try:
                        if float(df['rvol'].iloc[i - 1]) > 1.5:
                            strength *= 1.2
                    except (TypeError, ValueError):
                        pass

                fvg_list.append(FVG(
                    type=FVGType.BULLISH,
                    top=float(gap_top),
                    bottom=float(gap_bottom),
                    mid=float((gap_top + gap_bottom) / 2.0),
                    time=df['time'].iloc[i - 1],
                    index=i - 1,
                    size=float(gap_size),
                    strength=float(strength),
                ))

        # ---- Bearish FVG: c1_low > c3_high ----
        if c1_low > c3_high:
            gap_top = c1_low          # upper boundary
            gap_bottom = c3_high      # lower boundary
            gap_size = gap_top - gap_bottom

            body_condition = c2_close < c2_open
            body_size_condition = c2_body > (c2_range * 0.3)
            size_condition = gap_size >= min_fvg_size

            volume_condition = True
            if 'rvol' in df.columns:
                try:
                    volume_condition = float(df['rvol'].iloc[i - 1]) > 0.8
                except (TypeError, ValueError):
                    volume_condition = True

            if body_condition and body_size_condition and size_condition and volume_condition:
                strength = 1.0
                if c2_body > c2_range * 0.7:
                    strength = 1.5
                if 'rvol' in df.columns:
                    try:
                        if float(df['rvol'].iloc[i - 1]) > 1.5:
                            strength *= 1.2
                    except (TypeError, ValueError):
                        pass

                fvg_list.append(FVG(
                    type=FVGType.BEARISH,
                    top=float(gap_top),
                    bottom=float(gap_bottom),
                    mid=float((gap_top + gap_bottom) / 2.0),
                    time=df['time'].iloc[i - 1],
                    index=i - 1,
                    size=float(gap_size),
                    strength=float(strength),
                ))

    if not fvg_list:
        self._log_info("[NDS][SMC][FVG] detected=0 unfilled=0")
        return fvg_list

    # --- 2) Fill / Unfilled evaluation ---
    lookahead = int(self.settings.get('FVG_LOOKAHEAD_BARS', 80))
    lookahead = max(3, min(lookahead, len(df)))  # حداقل معنادار

    touched_count = 0
    filled_count = 0

    for fvg in fvg_list:
        check_limit = min(fvg.index + lookahead, len(df) - 1)

        if check_limit <= fvg.index + 1:
            # کندل‌های کافی برای بررسی نداریم → برای اسکلپینگ بهتر است open فرض شود
            fvg.filled = False
            continue

        touched = False
        filled = False

        # منطقه گپ: [bottom, top]
        top = float(fvg.top)
        bottom = float(fvg.bottom)

        for j in range(fvg.index + 1, check_limit + 1):
            high_j = float(df['high'].iloc[j])
            low_j = float(df['low'].iloc[j])

            # ورود به ناحیه (overlap)
            if low_j <= top and high_j >= bottom:
                touched = True

            if fvg.type == FVGType.BULLISH:
                # Mitigation کامل: عبور low از مرز پایین گپ
                if low_j <= bottom:
                    filled = True
                    break
            else:  # BEARISH
                # Mitigation کامل: عبور high از مرز بالای گپ
                if high_j >= top:
                    filled = True
                    break

        if fill_mode == "touch":
            fvg.filled = touched
        else:
            fvg.filled = filled

        if touched:
            touched_count += 1
        if fvg.filled:
            filled_count += 1

    unfilled_count = len(fvg_list) - filled_count

    # --- 3) Logging ---
    self._log_info("[NDS][SMC][FVG] detected=%s unfilled=%s", len(fvg_list), unfilled_count)

    if self.debug_smc:
        # نمایش مختصر وضعیت، با تمرکز روی FVGهای باز نزدیک قیمت
        current_price = float(df['close'].iloc[-1])
        open_fvgs = [f for f in fvg_list if not getattr(f, "filled", False)]
        # مرتب‌سازی بر اساس فاصله تا قیمت جاری
        def _dist(f: FVG) -> float:
            # فاصله به نزدیک‌ترین لبه
            return min(abs(current_price - float(f.top)), abs(current_price - float(f.bottom)))

        open_fvgs_sorted = sorted(open_fvgs, key=_dist)[:10]
        self._log_verbose(
            "[NDS][SMC][FVG] debug | touched=%s filled=%s open=%s (showing up to 10 nearest)",
            touched_count,
            filled_count,
            len(open_fvgs),
        )
        for k, f in enumerate(open_fvgs_sorted, start=1):
            self._log_verbose(
                "[NDS][SMC][FVG] open #%s | %s | idx=%s time=%s zone=[%.2f-%.2f] size=%.2f strength=%.2f dist=%.2f",
                k,
                f.type.value if hasattr(f.type, "value") else str(f.type),
                getattr(f, "index", -1),
                getattr(f, "time", "N/A"),
                float(f.bottom),
                float(f.top),
                float(getattr(f, "size", 0.0)),
                float(getattr(f, "strength", 1.0)),
                _dist(f),
            )

    return fvg_list


def detect_order_blocks(self, lookback: int = 50) -> List[OrderBlock]:
    """شناسایی Order Block های معتبر به سبک SMC (با فیلتر Fresh/Mitigated) و لاگ مرحله‌ای.

    منطق پایه:
    - Bullish OB: آخرین کندل مخالف (قرمز) قبل از displacement صعودی.
    - Bearish OB: آخرین کندل مخالف (سبز) قبل از displacement نزولی.

    مشکل رایج در پروژه‌های اسکلپ: OB زیاد شناسایی می‌شود اما عملاً همه "میتگیِیت" شده‌اند،
    یا برعکس همه fresh فرض می‌شوند. برای رصد صحیح، این تابع علاوه بر raw detection،
    یک فیلتر "fresh" هم اعمال می‌کند.

    تنظیمات مرتبط:
    - OB_MIN_MOVE_ATR (پیش‌فرض: 1.0)
    - OB_LOOKAHEAD_BARS (پیش‌فرض: 120)  (برای تشخیص mitigated)
    - OB_RETURN_LIMIT (پیش‌فرض: 5)      (تعداد خروجی نهایی)
    """
    df = self.df
    order_blocks: List[OrderBlock] = []

    if len(df) < lookback + 5:
        return order_blocks

    atr = float(self.atr) if self.atr else 0.0
    min_move_mult = float(self.settings.get("OB_MIN_MOVE_ATR", 1.0))
    min_move_size = atr * min_move_mult

    self._log_verbose(
        "[NDS][SMC][OB] start | candles=%s atr=%.4f lookback=%s min_move=%.4f",
        len(df),
        atr,
        lookback,
        min_move_size,
    )

    # --- 1) Raw detection ---
    for i in range(lookback, len(df) - 3):
        candle_a = df.iloc[i]
        candle_b = df.iloc[i + 1]
        candle_c = df.iloc[i + 2]

        # Bullish OB: red candle then strong up displacement
        is_red = float(candle_a['close']) < float(candle_a['open'])
        move_up = float(candle_b['close']) - float(candle_a['high'])
        is_strong_up = (
            float(candle_b['close']) > float(candle_a['high'])
            and float(candle_b['close']) > float(candle_b['open'])
            and (move_up > min_move_size or (float(candle_b['close']) - float(candle_b['open'])) > atr * 0.8)
        )

        if is_red and is_strong_up:
            strength = 1.0
            if float(candle_c['close']) > float(candle_b['high']):
                strength += 0.5
            if 'rvol' in df.columns:
                try:
                    if float(df['rvol'].iloc[i + 1]) > 1.2:
                        strength += 0.5
                except (TypeError, ValueError):
                    pass

            order_blocks.append(OrderBlock(
                type='BULLISH_OB',
                high=float(candle_a['high']),
                low=float(candle_a['low']),
                time=candle_a['time'],
                index=i,
                strength=float(strength),
            ))

        # Bearish OB: green candle then strong down displacement
        is_green = float(candle_a['close']) > float(candle_a['open'])
        move_down = float(candle_a['low']) - float(candle_b['close'])
        is_strong_down = (
            float(candle_b['close']) < float(candle_a['low'])
            and float(candle_b['close']) < float(candle_b['open'])
            and (move_down > min_move_size or (float(candle_b['open']) - float(candle_b['close'])) > atr * 0.8)
        )

        if is_green and is_strong_down:
            strength = 1.0
            if float(candle_c['close']) < float(candle_b['low']):
                strength += 0.5
            if 'rvol' in df.columns:
                try:
                    if float(df['rvol'].iloc[i + 1]) > 1.2:
                        strength += 0.5
                except (TypeError, ValueError):
                    pass

            order_blocks.append(OrderBlock(
                type='BEARISH_OB',
                high=float(candle_a['high']),
                low=float(candle_a['low']),
                time=candle_a['time'],
                index=i,
                strength=float(strength),
            ))

    self._log_info("[NDS][SMC][OB] detected raw=%s", len(order_blocks))
    if not order_blocks:
        return order_blocks

    # --- 2) Fresh/Mitigated filter ---
    lookahead = int(self.settings.get("OB_LOOKAHEAD_BARS", 120))
    lookahead = max(10, min(lookahead, len(df)))
    return_limit = int(self.settings.get("OB_RETURN_LIMIT", 5))
    return_limit = max(1, min(return_limit, 20))

    fresh_blocks: List[OrderBlock] = []
    mitigated_count = 0

    for ob in order_blocks:
        start = int(ob.index) + 1
        end = min(int(ob.index) + lookahead, len(df) - 1)
        if start >= len(df):
            fresh_blocks.append(ob)
            continue

        ob_high = float(ob.high)
        ob_low = float(ob.low)

        mitigated = False
        for j in range(start, end + 1):
            high_j = float(df['high'].iloc[j])
            low_j = float(df['low'].iloc[j])

            # اگر کندل بعدی وارد محدوده OB شود → mitigated
            if low_j <= ob_high and high_j >= ob_low:
                mitigated = True
                break

        if mitigated:
            mitigated_count += 1
        else:
            fresh_blocks.append(ob)

    # اولویت با OBهای قوی‌تر و نزدیک‌تر (به انتهای دیتا)
    fresh_blocks_sorted = sorted(
        fresh_blocks,
        key=lambda x: (-(float(getattr(x, "strength", 1.0))), -int(getattr(x, "index", 0))),
    )

    selected = fresh_blocks_sorted[:return_limit]
    self._log_verbose(
        "[NDS][SMC][OB] raw=%s mitigated=%s fresh=%s selected=%s",
        len(order_blocks),
        mitigated_count,
        len(fresh_blocks),
        len(selected),
    )

    if self.debug_smc:
        for k, ob in enumerate(selected, start=1):
            self._log_verbose(
                "[NDS][SMC][OB] fresh #%s | %s | idx=%s time=%s range=[%.2f-%.2f] strength=%.2f",
                k,
                getattr(ob, "type", "OB"),
                int(getattr(ob, "index", -1)),
                getattr(ob, "time", "N/A"),
                float(getattr(ob, "low", 0.0)),
                float(getattr(ob, "high", 0.0)),
                float(getattr(ob, "strength", 1.0)),
            )

    # برای سازگاری با نسخه‌های قبل، خروجی نهایی محدود می‌شود
    return selected

    def detect_liquidity_sweeps(self, swings: List[SwingPoint], lookback_swings: int = 5) -> List[LiquiditySweep]:
        """
        شناسایی نفوذهای فیک (Liquidity Sweeps) با استانداردهای SMC
        """
        if not swings:
            return []

        sweeps = []
        recent_data = self.df.tail(40)
        recent_highs = [s for s in swings if s.side == 'HIGH'][-lookback_swings:]
        recent_lows = [s for s in swings if s.side == 'LOW'][-lookback_swings:]

        atr_value = self.atr
        min_penetration = atr_value * self.settings.get('MIN_SWEEP_PENETRATION_MULTIPLIER', 0.2)
        max_penetration = atr_value * 3.0  # کاهش سیگنال‌های پرنفوذ غیرواقعی

        for _, row in recent_data.iterrows():
            candle_range = row['high'] - row['low']
            if candle_range < (atr_value * 0.5):
                continue

            rvol_value = 1.0
            if 'rvol' in self.df.columns:
                try:
                    rvol_value = float(row['rvol'])
                except (TypeError, ValueError):
                    rvol_value = 1.0
                if pd.isna(rvol_value):
                    rvol_value = 1.0

            for swing in recent_highs:
                if row['time'] <= swing.time:
                    continue

                if row['high'] > swing.price and row['close'] < swing.price:
                    penetration = row['high'] - swing.price

                    if min_penetration <= penetration <= max_penetration:
                        upper_wick = row['high'] - max(row['open'], row['close'])
                        body_size = abs(row['close'] - row['open'])

                        is_valid_shape = (
                            (upper_wick > body_size)
                            or (row['close'] < row['open'])
                            or (upper_wick > candle_range * 0.4)
                        )

                        has_high_volume = rvol_value > 1.5

                        if is_valid_shape or has_high_volume:
                            strength = min(3.0, (penetration / atr_value) + (0.5 if has_high_volume else 0))

                            sweep = LiquiditySweep(
                                time=row['time'],
                                type='BEARISH_SWEEP',
                                level=swing.price,
                                penetration=penetration,
                                description=f"Bearish Sweep (RVOL: {rvol_value:.1f}x)",
                                strength=strength
                            )
                            sweeps.append(sweep)

            for swing in recent_lows:
                if row['time'] <= swing.time:
                    continue

                if row['low'] < swing.price and row['close'] > swing.price:
                    penetration = swing.price - row['low']

                    if min_penetration <= penetration <= max_penetration:
                        lower_wick = min(row['open'], row['close']) - row['low']
                        body_size = abs(row['close'] - row['open'])

                        is_valid_shape = (
                            (lower_wick > body_size)
                            or (row['close'] > row['open'])
                            or (lower_wick > candle_range * 0.4)
                        )

                        has_high_volume = rvol_value > 1.5

                        if is_valid_shape or has_high_volume:
                            strength = min(3.0, (penetration / atr_value) + (0.5 if has_high_volume else 0))

                            sweep = LiquiditySweep(
                                time=row['time'],
                                type='BULLISH_SWEEP',
                                level=swing.price,
                                penetration=penetration,
                                description=f"Bullish Sweep (RVOL: {rvol_value:.1f}x)",
                                strength=strength
                            )
                            sweeps.append(sweep)

        unique_sweeps = []
        seen = set()
        for sweep in reversed(sweeps):
            key = (sweep.time, sweep.type, round(sweep.level, 2))
            if key not in seen:
                seen.add(key)
                unique_sweeps.append(sweep)

        unique_sweeps.reverse()

        self._log_info("[NDS][SMC][SWEEPS] detected fresh=%s", len(unique_sweeps))
        return unique_sweeps

    def determine_market_structure(
        self,
        swings: List[SwingPoint],
        lookback_swings: int = 4,
        volume_analysis: Optional[Dict] = None,
        volatility_state: Optional[str] = None,
        adx_value: Optional[float] = None,
    ) -> MarketStructure:
        """
        تعیین ساختار بازار با منطق NDS (Nodal Displacement Sequencing)
        تمرکز بر جابجایی نودها (Displacement) و تقارن فرکتالی
        """
        normalized_volatility = self._normalize_volatility_state(volatility_state)

        if len(swings) < 3:
            current_price = float(self.df['close'].iloc[-1])
            return MarketStructure(
                trend=MarketTrend.RANGING,
                bos="NONE",
                choch="NONE",
                last_high=None,
                last_low=None,
                current_price=current_price,
                bos_choch_confidence=0.0,
                volume_analysis=volume_analysis,
                volatility_state=normalized_volatility,
                adx_value=adx_value,
                structure_score=8.0
            )

        # در اسکلپینگ XAUUSD، کاهش فیلتر ATR در نوسان بالا از حذف نودهای مفید جلوگیری می‌کند.
        if normalized_volatility == "HIGH_VOLATILITY":
            dynamic_multiplier = 0.75
        elif normalized_volatility == "LOW_VOLATILITY":
            dynamic_multiplier = 1.1
        else:
            dynamic_multiplier = 1.0
        min_swing_distance = self.atr * dynamic_multiplier

        major_swings = []
        last_high_p, last_low_p = None, None

        for swing in swings:
            if swing.side == 'HIGH':
                if last_high_p is None or abs(swing.price - last_high_p) >= min_swing_distance:
                    major_swings.append(swing)
                    last_high_p = swing.price
            else:
                if last_low_p is None or abs(swing.price - last_low_p) >= min_swing_distance:
                    major_swings.append(swing)
                    last_low_p = swing.price

        recent_swings = self._get_relevant_swings(major_swings, lookback_swings)
        last_high = next((s for s in reversed(recent_swings) if s.side == 'HIGH'), None)
        last_low = next((s for s in reversed(recent_swings) if s.side == 'LOW'), None)

        current_price = float(self.df['close'].iloc[-1])
        current_high = float(self.df['high'].iloc[-1])
        current_low = float(self.df['low'].iloc[-1])

        trend, trend_strength, trend_confidence = self._determine_trend_with_confidence(
            recent_swings,
            current_price,
            volume_analysis,
            normalized_volatility,
            adx_value,
        )

        nds_displacement = False
        if last_high and current_price > last_high.price:
            trend = MarketTrend.UPTREND
            nds_displacement = True
        elif last_low and current_price < last_low.price:
            trend = MarketTrend.DOWNTREND
            nds_displacement = True

        bos, choch, bos_choch_confidence = self._detect_bos_choch(
            last_high=last_high,
            last_low=last_low,
            current_high=current_high,
            current_low=current_low,
            current_close=current_price,
            trend=trend,
            trend_strength=trend_strength,
            volume_analysis=volume_analysis,
            volatility_state=normalized_volatility,
        )

        range_width, range_mid = None, None
        if last_high and last_low:
            range_width = abs(last_high.price - last_low.price)
            range_mid = (last_high.price + last_low.price) / 2

            min_range = self.atr * 0.5
            if range_width < min_range:
                range_width = None

        structure_score = self._calculate_structure_score(
            bos=bos,
            choch=choch,
            confidence=bos_choch_confidence,
            trend_strength=trend_strength,
            volume_analysis=volume_analysis,
            volatility_state=normalized_volatility,
            range_width=range_width,
            last_high=last_high,
            last_low=last_low,
            adx_value=adx_value,
        )

        if nds_displacement:
            # بونوس محدود برای جابجایی معتبر نودها، بدون تغییر BOS برای جلوگیری از سیگنال‌های کاذب.
            structure_score = min(100.0, structure_score + 10.0)

        structure_score = max(0.0, min(100.0, structure_score))
        volume_payload = dict(volume_analysis) if volume_analysis else None
        if volume_payload is not None:
            volume_payload.setdefault("nds_displacement", nds_displacement)

        structure = MarketStructure(
            trend=trend,
            bos=bos,
            choch=choch,
            last_high=last_high,
            last_low=last_low,
            current_price=current_price,
            range_width=range_width,
            range_mid=range_mid,
            bos_choch_confidence=bos_choch_confidence,
            volume_analysis=volume_payload,
            volatility_state=normalized_volatility,
            adx_value=adx_value,
            structure_score=structure_score,
        )

        self._log_info(
            "[NDS][SMC][STRUCTURE] Trend=%s BOS=%s CHOCH=%s Conf=%.1f%% Score=%.1f",
            trend.value,
            bos,
            choch,
            bos_choch_confidence * 100,
            structure_score,
        )

        return structure

    def _calculate_structure_score(
        self,
        bos: str,
        choch: str,
        confidence: float,
        trend_strength: float,
        volume_analysis: Optional[Dict],
        volatility_state: Optional[str],
        range_width: Optional[float],
        last_high: Optional[SwingPoint] = None,
        last_low: Optional[SwingPoint] = None,
        adx_value: Optional[float] = None,
        sweeps: Optional[List[LiquiditySweep]] = None,
    ) -> float:
        """
        محاسبه امتیاز کیفیت ساختار - نسخه بهینه شده برای اسکلپینگ چابک طلا
        """
        score = 0.0
        current_price = float(self.df['close'].iloc[-1])

        adx_threshold = self.GOLD_SETTINGS.get('ADX_THRESHOLD_WEAK', 15)

        score += 5.0

        if bos != "NONE":
            score += 45 * confidence
        elif choch != "NONE":
            score += 35 * confidence

        if last_high and current_price > last_high.price:
            penetration_bonus = 15.0 * (1.0 if confidence > 0.5 else 0.5)
            score += penetration_bonus
        elif last_low and current_price < last_low.price:
            penetration_bonus = 15.0 * (1.0 if confidence > 0.5 else 0.5)
            score += penetration_bonus

        if adx_value is not None:
            if adx_value > 25:
                score += 12.0
            elif adx_value > adx_threshold:
                score += 6.0
            else:
                score -= 10.0

        trend_score = 15 * trend_strength
        score += trend_score

        if volume_analysis:
            volume_zone = volume_analysis.get('volume_zone') or volume_analysis.get('zone', 'NORMAL')
            if volume_zone == "HIGH":
                score += 12
            elif volume_zone == "NORMAL":
                score += 6

        if volatility_state == "MODERATE_VOLATILITY":
            score += 8
        elif volatility_state == "HIGH_VOLATILITY":
            score -= 8
        elif volatility_state == "LOW_VOLATILITY":
            score -= 4

        if range_width and hasattr(self, 'atr') and self.atr > 0:
            atr_ratio = range_width / self.atr
            if atr_ratio < 1.0:
                score -= 8
            elif atr_ratio > 1.5:
                score += 8

        if last_high is not None:
            score += 2.5
        if last_low is not None:
            score += 2.5

        if sweeps:
            for sweep in sweeps:
                if sweep.type == 'BULLISH_SWEEP':
                    score += 8.0 * sweep.strength
                elif sweep.type == 'BEARISH_SWEEP':
                    score -= 8.0 * sweep.strength

        final_score = max(0.0, min(100.0, score))
        return round(final_score, 2)

    def _get_relevant_swings(self, major_swings: List[SwingPoint], lookback: int) -> List[SwingPoint]:
        """انتخاب سوینگ‌های مرتبط"""
        if len(major_swings) <= lookback:
            return major_swings

        recent_by_time = []
        last_time = self.df['time'].iloc[-1]

        for swing in reversed(major_swings):
            time_diff = (last_time - swing.time).total_seconds() / 3600
            if time_diff <= 24:
                recent_by_time.append(swing)
            if len(recent_by_time) >= lookback * 2:
                break

        if recent_by_time:
            recent_by_time.sort(key=lambda x: x.time, reverse=True)
            return recent_by_time[:lookback]

        return major_swings[-lookback:]

    def _determine_trend_with_confidence(
        self,
        swings: List[SwingPoint],
        current_price: float,
        volume_analysis: Optional[Dict] = None,
        volatility_state: Optional[str] = None,
        adx_value: Optional[float] = None,
    ) -> Tuple[MarketTrend, float, float]:
        """تشخیص روند با اطمینان بر اساس چندین فاکتور"""
        if len(swings) < 2:
            return MarketTrend.RANGING, 0.0, 0.0

        highs = [s for s in swings if s.side == 'HIGH']
        lows = [s for s in swings if s.side == 'LOW']

        if len(highs) < 2 or len(lows) < 2:
            return MarketTrend.RANGING, 0.0, 0.0

        higher_highs = sum(1 for i in range(1, len(highs)) if highs[i].price > highs[i - 1].price)
        higher_lows = sum(1 for i in range(1, len(lows)) if lows[i].price > lows[i - 1].price)
        lower_highs = sum(1 for i in range(1, len(highs)) if highs[i].price < highs[i - 1].price)
        lower_lows = sum(1 for i in range(1, len(lows)) if lows[i].price < lows[i - 1].price)

        if higher_highs > lower_highs and higher_lows > lower_lows:
            trend = MarketTrend.UPTREND
            strength = (higher_highs + higher_lows) / (len(highs) + len(lows) - 2)
        elif lower_highs > higher_highs and lower_lows > higher_lows:
            trend = MarketTrend.DOWNTREND
            strength = (lower_highs + lower_lows) / (len(highs) + len(lows) - 2)
        else:
            trend = MarketTrend.RANGING
            strength = 0.3

        if adx_value is not None:
            adx_strength = adx_value / 100.0
            strength = (strength * 0.6) + (adx_strength * 0.4)

        if volume_analysis:
            volume_factor = volume_analysis.get('rvol', 1.0)
            if volume_factor is None or pd.isna(volume_factor):
                volume_factor = 1.0
            volume_factor = min(float(volume_factor), 2.0) / 2.0
            strength = strength * (0.7 + 0.3 * volume_factor)

        confidence = strength * 0.7

        if volatility_state == "MODERATE_VOLATILITY":
            confidence *= 1.1
        elif volatility_state == "LOW_VOLATILITY":
            confidence *= 0.9

        self._log_debug(
            "[NDS][SMC][STRUCTURE] trend=%s strength=%.2f confidence=%.2f",
            trend.value,
            strength,
            confidence,
        )
        return trend, strength, min(1.0, confidence)

    def _detect_bos_choch(
        self,
        last_high: Optional[SwingPoint],
        last_low: Optional[SwingPoint],
        current_high: float,
        current_low: float,
        current_close: float,
        trend: MarketTrend,
        trend_strength: float,
        volume_analysis: Optional[Dict] = None,
        volatility_state: Optional[str] = None,
    ) -> Tuple[str, str, float]:
        """
        تشخیص BOS/CHOCH با تأیید چندمرحله‌ای
        """
        bos = "NONE"
        choch = "NONE"
        confidence = 0.0

        if not last_high or not last_low:
            self._log_debug("[NDS][SMC][BOS_CHOCH] insufficient swings")
            return bos, choch, confidence

        base_buffer = self._calculate_dynamic_buffer(
            atr=self.atr,
            trend_strength=trend_strength,
            volatility_state=volatility_state,
            volume_analysis=volume_analysis,
        )

        bos, bos_confidence = self._detect_bos_advanced(
            last_high=last_high,
            last_low=last_low,
            current_high=current_high,
            current_low=current_low,
            current_close=current_close,
            trend=trend,
            base_buffer=base_buffer,
            volume_analysis=volume_analysis,
        )

        choch, choch_confidence = self._detect_choch_advanced(
            last_high=last_high,
            last_low=last_low,
            current_high=current_high,
            current_low=current_low,
            current_close=current_close,
            trend=trend,
            base_buffer=base_buffer,
            bos_detected=(bos != "NONE"),
        )

        final_bos, final_choch, final_confidence = self._validate_with_price_action(
            bos=bos,
            choch=choch,
            bos_confidence=bos_confidence,
            choch_confidence=choch_confidence,
            current_high=current_high,
            current_low=current_low,
            current_close=current_close,
            last_high_price=last_high.price,
            last_low_price=last_low.price,
            df=self.df,
        )

        self._log_debug(
            "[NDS][SMC][BOS_CHOCH] result bos=%s choch=%s conf=%.2f",
            final_bos,
            final_choch,
            final_confidence,
        )
        return final_bos, final_choch, final_confidence

    def _calculate_dynamic_buffer(
        self,
        atr: float,
        trend_strength: float,
        volatility_state: Optional[str],
        volume_analysis: Optional[Dict],
    ) -> Dict[str, float]:
        """محاسبه بافر پویا بر اساس شرایط مختلف بازار"""
        buffers = {
            'bos': atr * 0.15,
            'choch': atr * 0.12,
            'aggressive': atr * 0.08,
            'conservative': atr * 0.2,
        }

        if trend_strength > 0.7:
            buffers['bos'] *= 0.8
            buffers['choch'] *= 0.7
        elif trend_strength < 0.3:
            buffers['bos'] *= 1.5
            buffers['choch'] *= 1.3

        if volatility_state == "HIGH_VOLATILITY":
            buffers['bos'] *= 1.2
            buffers['choch'] *= 1.1
        elif volatility_state == "LOW_VOLATILITY":
            buffers['bos'] *= 0.8
            buffers['choch'] *= 0.9

        if volume_analysis:
            volume_zone = volume_analysis.get('volume_zone') or volume_analysis.get('zone')
            if volume_zone == "HIGH":
                buffers['bos'] *= 0.9
                buffers['choch'] *= 0.85

        return buffers

    def _confirm_with_candle_pattern(
        self,
        current_high: float,
        current_low: float,
        current_close: float,
        last_high_price: float,
        last_low_price: float,
        trend: MarketTrend,
    ) -> bool:
        """تأیید شکست با الگوهای کندل استیک"""
        candle_size = abs(current_high - current_low)
        current_open = float(self.df['open'].iloc[-1])
        if pd.isna(current_open):
            return False
        body_size = abs(current_close - current_open)

        if trend == MarketTrend.UPTREND:
            if current_close > last_high_price and (current_close - last_high_price) > (candle_size * 0.3):
                if body_size > (candle_size * 0.4):
                    return True

        elif trend == MarketTrend.DOWNTREND:
            if current_close < last_low_price and (last_low_price - current_close) > (candle_size * 0.3):
                if body_size > (candle_size * 0.4):
                    return True

        return False

    def _check_reversal_patterns(
        self,
        current_high: float,
        current_low: float,
        current_close: float,
        pattern_type: str,
    ) -> bool:
        """بررسی الگوهای بازگشتی کندلی"""
        try:
            current_candle = self.df.iloc[-1]
            prev_candle = self.df.iloc[-2]

            current_open = current_candle['open']
            prev_open = prev_candle['open']
            prev_close = prev_candle['close']
            prev_high = prev_candle['high']
            prev_low = prev_candle['low']

            current_body = abs(current_close - current_open)
            prev_body = abs(prev_close - prev_open)
            current_range = current_high - current_low
            prev_range = prev_high - prev_low

            if pattern_type == "bullish":
                if current_low < prev_low and current_close > (current_open + (current_range * 0.6)):
                    return True

                if current_close > prev_open and current_open < prev_close and current_body > (prev_body * 1.5):
                    return True

            elif pattern_type == "bearish":
                if current_high > prev_high and current_close < (current_open - (current_range * 0.6)):
                    return True

                if current_close < prev_open and current_open > prev_close and current_body > (prev_body * 1.5):
                    return True

        except (IndexError, KeyError):
            pass

        return False

    def _calculate_bearish_pressure(self, recent_candles: pd.DataFrame) -> float:
        """محاسبه فشار فروش در کندل‌های اخیر"""
        if len(recent_candles) == 0:
            return 0.0

        bearish_count = 0
        total_candles = len(recent_candles)

        for _, candle in recent_candles.iterrows():
            if candle['close'] < candle['open']:
                bearish_count += 1

        return bearish_count / total_candles

    def _calculate_bullish_pressure(self, recent_candles: pd.DataFrame) -> float:
        """محاسبه فشار خرید در کندل‌های اخیر"""
        if len(recent_candles) == 0:
            return 0.0

        bullish_count = 0
        total_candles = len(recent_candles)

        for _, candle in recent_candles.iterrows():
            if candle['close'] > candle['open']:
                bullish_count += 1

        return bullish_count / total_candles

    def _detect_bos_advanced(
        self,
        last_high: SwingPoint,
        last_low: SwingPoint,
        current_high: float,
        current_low: float,
        current_close: float,
        trend: MarketTrend,
        base_buffer: Dict[str, float],
        volume_analysis: Optional[Dict],
    ) -> Tuple[str, float]:
        """تشخیص پیشرفته BOS با در نظر گرفتن تأییدیه‌های چندگانه"""
        bos = "NONE"
        confidence = 0.0

        last_high_price = last_high.price
        last_low_price = last_low.price

        price_break = False
        price_signal = ""

        self._log_debug(
            "[NDS][SMC][BOS_CHOCH] BOS check trend=%s last_high=%.2f last_low=%.2f close=%.2f buffer=%.4f",
            trend.value,
            last_high_price,
            last_low_price,
            current_close,
            base_buffer.get('bos', 0.0),
        )

        if trend == MarketTrend.UPTREND:
            if current_close > (last_high_price + base_buffer['bos']):
                price_break = True
                price_signal = "BULLISH_BOS"
        elif trend == MarketTrend.DOWNTREND:
            if current_close < (last_low_price - base_buffer['bos']):
                price_break = True
                price_signal = "BEARISH_BOS"

        volume_confirmation = False
        if volume_analysis:
            volume_ratio = volume_analysis.get('rvol', 1.0)
            volume_confirmation = volume_ratio > 1.2

        candle_confirmation = self._confirm_with_candle_pattern(
            current_high,
            current_low,
            current_close,
            last_high_price,
            last_low_price,
            trend,
        )
        self._log_debug(
            "[NDS][SMC][BOS_CHOCH] BOS confirmations volume=%s candle=%s",
            volume_confirmation,
            candle_confirmation,
        )

        if price_break:
            confidence = 0.4

            if volume_confirmation:
                confidence += 0.3

            if candle_confirmation:
                confidence += 0.3

            if confidence >= 0.6:
                bos = price_signal
                self._log_debug("[NDS][SMC][BOS_CHOCH] BOS confirmed=%s conf=%.1f%%", bos, confidence * 100)

        return bos, confidence

    def _detect_choch_advanced(
        self,
        last_high: SwingPoint,
        last_low: SwingPoint,
        current_high: float,
        current_low: float,
        current_close: float,
        trend: MarketTrend,
        base_buffer: Dict[str, float],
        bos_detected: bool,
    ) -> Tuple[str, float]:
        """تشخیص پیشرفته CHOCH - حساس به تغییر روند"""
        choch = "NONE"
        confidence = 0.0

        if bos_detected:
            return choch, confidence

        last_high_price = last_high.price
        last_low_price = last_low.price

        self._log_debug(
            "[NDS][SMC][BOS_CHOCH] CHOCH check trend=%s last_high=%.2f last_low=%.2f close=%.2f buffer=%.4f",
            trend.value,
            last_high_price,
            last_low_price,
            current_close,
            base_buffer.get('choch', 0.0),
        )

        if trend == MarketTrend.UPTREND:
            if current_close < (last_low_price - base_buffer['choch']):
                if self._check_reversal_patterns(current_high, current_low, current_close, "bearish"):
                    choch = "BEARISH_CHOCH"
                    confidence = 0.7

        elif trend == MarketTrend.DOWNTREND:
            if current_close > (last_high_price + base_buffer['choch']):
                if self._check_reversal_patterns(current_high, current_low, current_close, "bullish"):
                    choch = "BULLISH_CHOCH"
                    confidence = 0.7

        elif trend == MarketTrend.RANGING:
            range_buffer = base_buffer['choch'] * 1.5

            if current_close > (last_high_price + range_buffer):
                choch = "BULLISH_CHOCH"
                confidence = 0.6
            elif current_close < (last_low_price - range_buffer):
                choch = "BEARISH_CHOCH"
                confidence = 0.6

        return choch, confidence

    def _validate_with_price_action(
        self,
        bos: str,
        choch: str,
        bos_confidence: float,
        choch_confidence: float,
        current_high: float,
        current_low: float,
        current_close: float,
        last_high_price: float,
        last_low_price: float,
        df: pd.DataFrame,
    ) -> Tuple[str, str, float]:
        """اعتبارسنجی نهایی با پرایس اکشن چندکندلی"""
        final_bos = bos
        final_choch = choch
        final_confidence = max(bos_confidence, choch_confidence)

        if bos != "NONE" or choch != "NONE":
            recent_candles = df.iloc[-4:-1]

            if bos == "BULLISH_BOS":
                bearish_pressure = self._calculate_bearish_pressure(recent_candles)
                if bearish_pressure > 0.7:
                    self._log_debug("[NDS][SMC][BOS_CHOCH] Bullish BOS high bearish pressure")
                    final_confidence *= 0.7

            elif bos == "BEARISH_BOS":
                bullish_pressure = self._calculate_bullish_pressure(recent_candles)
                if bullish_pressure > 0.7:
                    self._log_debug("[NDS][SMC][BOS_CHOCH] Bearish BOS high bullish pressure")
                    final_confidence *= 0.7

        if final_confidence < 0.5:
            final_bos = "NONE"
            final_choch = "NONE"

        return final_bos, final_choch, final_confidence

    def analyze_premium_discount(self, structure: MarketStructure) -> Tuple[str, float]:
        """تحلیل مناطق Premium/Discount"""
        if not structure.last_high or not structure.last_low:
            return "NEUTRAL", 0.0

        if structure.trend == MarketTrend.RANGING:
            range_high = structure.last_high.price
            range_low = structure.last_low.price

            if range_high <= range_low:
                return "NEUTRAL", 0.0

            range_mid = (range_high + range_low) / 2
            current_price = structure.current_price

            discount_zone = range_low + (range_high - range_low) * 0.3
            premium_zone = range_low + (range_high - range_low) * 0.7

            if current_price < discount_zone:
                return "DISCOUNT", range_mid
            if current_price > premium_zone:
                return "PREMIUM", range_mid
            return "EQUILIBRIUM", range_mid

        range_high = structure.last_high.price
        range_low = structure.last_low.price
        range_mid = (range_high + range_low) / 2

        discount_zone = range_low + (range_high - range_low) * 0.33
        premium_zone = range_low + (range_high - range_low) * 0.66

        current_price = structure.current_price

        if current_price < discount_zone:
            return "DISCOUNT", range_mid
        if current_price > premium_zone:
            return "PREMIUM", range_mid
        return "EQUILIBRIUM", range_mid

    def analyze_range_position_gold(self, structure: MarketStructure) -> float:
        """تحلیل موقعیت قیمت در رنج مخصوص بازار طلا"""
        if not structure.range_width or structure.range_width < self.atr:
            return 0.0

        current_price = structure.current_price
        range_low = structure.last_low.price
        range_high = structure.last_high.price

        position = (current_price - range_low) / structure.range_width
        last_candle = self.df.iloc[-1]
        candle_range = last_candle['high'] - last_candle['low']

        score = 0.0

        if position < 0.3:
            lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']
            if lower_wick > candle_range * 0.4:
                score += 25
            elif lower_wick > candle_range * 0.25:
                score += 15
            else:
                score += 8

        elif position > 0.7:
            upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
            if upper_wick > candle_range * 0.4:
                score -= 25
            elif upper_wick > candle_range * 0.25:
                score -= 15
            else:
                score -= 8

        return score

    def get_market_trend(self, swings: List[SwingPoint]) -> MarketTrend:
        """
        نسخه ارتقا یافته برای تشخیص سریع‌تر تغییر روند در اسکلپینگ
        """
        if len(swings) < 4:
            return MarketTrend.RANGING

        last_price = self.df['close'].iloc[-1]
        high_swings = [s for s in swings if s.side == 'HIGH']
        low_swings = [s for s in swings if s.side == 'LOW']

        if not high_swings or not low_swings:
            return MarketTrend.RANGING

        last_high = high_swings[-1]
        last_low = low_swings[-1]
        prev_high = high_swings[-2] if len(high_swings) > 1 else last_high
        prev_low = low_swings[-2] if len(low_swings) > 1 else last_low

        if last_price > last_high.price:
            return MarketTrend.UPTREND

        if last_price < last_low.price:
            return MarketTrend.DOWNTREND

        is_hh = last_high.price > prev_high.price
        is_hl = last_low.price > prev_low.price
        is_lh = last_high.price < prev_high.price
        is_ll = last_low.price < prev_low.price

        if is_hh or (is_hl and last_price > last_low.price):
            return MarketTrend.UPTREND

        if is_ll or (is_lh and last_price < last_high.price):
            return MarketTrend.DOWNTREND

        return MarketTrend.RANGING
