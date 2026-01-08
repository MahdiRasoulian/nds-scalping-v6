"""
تحلیل ساختار بازار و الگوهای SMC
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
from datetime import datetime

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
        self._prepare_data()
    
    def _prepare_data(self):
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
            logger.warning(f"📊 Swing Detection: داده کافی نیست (داده: {len(df)} کندل، نیاز: {period*2+1})")
            return []
        
        # ۱. محاسبات اولیه و شناسایی فرکتال‌ها
        high_series = df['high']
        low_series = df['low']
        
        high_rolling_max = high_series.rolling(window=2*period+1, center=True).max()
        low_rolling_min = low_series.rolling(window=2*period+1, center=True).min()
        
        # شناسایی اندیس‌های معتبر
        valid_range = range(period, len(df) - period)
        high_indices = [i for i in high_series[high_series == high_rolling_max].index if i in valid_range]
        low_indices = [i for i in low_series[low_series == low_rolling_min].index if i in valid_range]
        
        # ✅ لاگ تعداد فرکتال‌های اولیه
        logger.info(f"📊 Swing Detection: فرکتال‌های اولیه - High: {len(high_indices)}, Low: {len(low_indices)}")
        
        if not high_indices and not low_indices:
            logger.warning("⚠️  Swing Detection: هیچ فرکتال اولیه‌ای یافت نشد!")
        
        # ۲. پارامترهای فیلتر (بهینه شده برای طلا)
        min_distance = self.atr * self.settings.get('MIN_ATR_DISTANCE_MULTIPLIER', 1.2)
        min_vol_mult = self.settings.get('MIN_VOLUME_MULTIPLIER', 0.6)
        
        # ۳. پردازش اولیه High Swings
        high_swings = []
        last_h_price = None
        for idx in high_indices:
            price = float(df['high'].iloc[idx])
            # فیلتر حجم: تایید قدرت در سقف
            avg_vol = df['volume'].iloc[max(0, idx-period):idx].mean() if 'volume' in df.columns else 1
            if (df['volume'].iloc[idx] > avg_vol * min_vol_mult) and \
               (last_h_price is None or abs(price - last_h_price) >= min_distance):
                high_swings.append(SwingPoint(
                    index=idx, 
                    price=price, 
                    time=df['time'].iloc[idx], 
                    type=SwingType.HIGH,
                    side='HIGH'
                ))
                last_h_price = price
        
        # ۴. پردازش اولیه Low Swings
        low_swings = []
        last_l_price = None
        for idx in low_indices:
            price = float(df['low'].iloc[idx])
            avg_vol = df['volume'].iloc[max(0, idx-period):idx].mean() if 'volume' in df.columns else 1
            if (df['volume'].iloc[idx] > avg_vol * min_vol_mult) and \
               (last_l_price is None or abs(price - last_l_price) >= min_distance):
                low_swings.append(SwingPoint(
                    index=idx, 
                    price=price, 
                    time=df['time'].iloc[idx], 
                    type=SwingType.LOW,
                    side='LOW'
                ))
                last_l_price = price
        
        # ✅ لاگ سوینگ‌های اولیه
        logger.info(f"📊 Swing Detection: سوینگ‌های اولیه - High: {len(high_swings)}, Low: {len(low_swings)}")
        
        # ۵. ترکیب، مرتب‌سازی و پاکسازی ساختاری
        all_swings = sorted(high_swings + low_swings, key=lambda x: x.index)
        
        # ✅ لاگ ترکیب شده
        logger.info(f"📊 Swing Detection: سوینگ‌های ترکیب شده: {len(all_swings)}")
        
        if not all_swings:
            logger.warning("⚠️  Swing Detection: هیچ سوینگ اولیه‌ای پس از فیلتر حجم/فاصله یافت نشد!")
            return []
        
        # پاکسازی ساختاری
        if all_swings:
            original_count = len(all_swings)
            cleaned = self._clean_consecutive_swings(all_swings)
            if cleaned:
                all_swings = cleaned
                logger.info(f"📊 Swing Cleaning: {original_count} → {len(all_swings)} سوینگ (حذف تکراری‌ها)")
            else:
                logger.warning("⚠️  Swing Cleaning: تمام سوینگ‌ها پس از پاکسازی تکراری‌ها حذف شدند!")
                return []
        
        # فیلتر حرکت‌های معنادار
        if all_swings:
            original_count = len(all_swings)
            meaningful = self._filter_meaningful_swings(all_swings)
            if meaningful:
                all_swings = meaningful
                logger.info(f"📊 Meaningful Filter: {original_count} → {len(all_swings)} سوینگ")
            else:
                logger.warning("⚠️  Meaningful Filter: تمام سوینگ‌ها پس از فیلتر حرکت معنادار حذف شدند!")
                return []
        
        # ۶. تعیین نهایی نوع سوینگ (HH, LH, HL, LL) - بعد از فیلترها
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
        
        # ✅ لاگ نهایی
        logger.info(f"✅ Swing Detection Final: {len(all_swings)} سوینگ شناسایی شد")
        
        # شمارش انواع سوینگ‌ها
        if all_swings:
            hh_count = sum(1 for s in all_swings if s.type == SwingType.HH)
            lh_count = sum(1 for s in all_swings if s.type == SwingType.LH)
            ll_count = sum(1 for s in all_swings if s.type == SwingType.LL)
            hl_count = sum(1 for s in all_swings if s.type == SwingType.HL)
            
            logger.info(f"📊 Swing Types: HH={hh_count}, LH={lh_count}, LL={ll_count}, HL={hl_count}")
            
            # نمایش 3 سوینگ آخر
            last_swings = all_swings[-3:] if len(all_swings) >= 3 else all_swings
            for i, swing in enumerate(last_swings):
                logger.info(f"📊 Swing {len(all_swings)-len(last_swings)+i+1}: {swing.side}@{swing.price:.2f} ({swing.type.value})")
        
        return all_swings
    
    def _clean_consecutive_swings(self, swings: List[SwingPoint]) -> List[SwingPoint]:
        """حذف سوینگ‌های تکراری در یک سمت برای به دست آوردن ساختار زیگزاگی تمیز"""
        if not swings:
            logger.debug("🔄 Swing Cleaning: لیست ورودی خالی است")
            return []
        
        original_count = len(swings)
        cleaned = []
        
        for s in swings:
            if not cleaned:
                cleaned.append(s)
                continue
            
            last = cleaned[-1]
            if last.side == s.side:
                # اگر دو سقف متوالی داریم، بالاترین را نگه دار
                if s.side == 'HIGH' and s.price > last.price:
                    cleaned[-1] = s
                    logger.debug(f"🔄 Swing Cleaning: جایگزینی سقف {last.price:.2f} → {s.price:.2f}")
                # اگر دو کف متوالی داریم، پایین‌ترین را نگه دار
                elif s.side == 'LOW' and s.price < last.price:
                    cleaned[-1] = s
                    logger.debug(f"🔄 Swing Cleaning: جایگزینی کف {last.price:.2f} → {s.price:.2f}")
            else:
                cleaned.append(s)
        
        removed_count = original_count - len(cleaned)
        if removed_count > 0:
            logger.info(f"🔄 Swing Cleaning: حذف {removed_count} سوینگ تکراری ({original_count} → {len(cleaned)})")
        
        return cleaned
    
    def _filter_meaningful_swings(self, swings: List[SwingPoint]) -> List[SwingPoint]:
        """حذف نوسانات فرسایشی که حرکت قیمتی موثری ندارند"""
        if len(swings) < 3:
            logger.debug(f"📏 Meaningful Filter: لیست کوتاه ({len(swings)} سوینگ) - پاس داده شد")
            return swings
        
        atr_threshold = self.atr * self.settings.get('MEANINGFUL_MOVE_MULT', 0.5)
        logger.debug(f"📏 Meaningful Filter: حداقل حرکت معنادار = {atr_threshold:.2f}")
        
        meaningful = []
        removed_indices = []
        
        for i, s in enumerate(swings):
            if i == 0 or i == len(swings) - 1:
                meaningful.append(s)
                continue
            
            # فاصله قیمتی از سوینگ قبلی
            move_size = abs(s.price - swings[i-1].price)
            if move_size >= atr_threshold:
                meaningful.append(s)
            elif i + 1 < len(swings):
                next_move_size = abs(swings[i+1].price - s.price)
                if next_move_size >= atr_threshold:
                    meaningful.append(s)
                else:
                    removed_indices.append(i)
            else:
                removed_indices.append(i)
        
        if removed_indices:
            logger.info(f"📏 Meaningful Filter: حذف {len(removed_indices)} سوینگ بی‌معنا ({len(swings)} → {len(meaningful)})")
            # نمایش سوینگ‌های حذف شده
            for idx in removed_indices[:3]:  # فقط 3 تا اول
                swing = swings[idx]
                prev_move = abs(swing.price - swings[idx-1].price) if idx > 0 else 0
                logger.debug(f"📏 حذف سوینگ #{idx}: {swing.side}@{swing.price:.2f} (حرکت: {prev_move:.2f} < {atr_threshold:.2f})")
        
        return meaningful
    
    def detect_fvgs(self) -> List[FVG]:
        """شناسایی FVGها با پارامترهای بهبودیافته"""
        df = self.df
        fvg_list = []
        
        if len(df) < 3:
            return fvg_list
        
        min_fvg_size = self.atr * self.settings.get('FVG_MIN_SIZE_MULTIPLIER', 0.1)
        
        for i in range(2, len(df)):
            # کندل میانی
            candle_2_high = df['high'].iloc[i-1]
            candle_2_low = df['low'].iloc[i-1]
            candle_2_close = df['close'].iloc[i-1]
            candle_2_open = df['open'].iloc[i-1]
            candle_2_body = abs(candle_2_close - candle_2_open)
            candle_2_range = candle_2_high - candle_2_low
            
            # شناسایی Bullish FVG
            candle_1_high = df['high'].iloc[i-2]
            candle_3_low = df['low'].iloc[i]
            
            if candle_3_low > candle_1_high:
                fvg_size = candle_3_low - candle_1_high
                
                # شرایط بهبودیافته برای FVG
                body_condition = candle_2_close > candle_2_open
                body_size_condition = candle_2_body > (candle_2_range * 0.3)
                fvg_size_condition = fvg_size >= min_fvg_size
                volume_condition = True
                
                if 'rvol' in df.columns:
                    volume_condition = df['rvol'].iloc[i-1] > 0.8
                
                if all([body_condition, body_size_condition, fvg_size_condition, volume_condition]):
                    # محاسبه قدرت FVG
                    strength = 1.0
                    if candle_2_body > candle_2_range * 0.7:
                        strength = 1.5
                    if 'rvol' in df.columns and df['rvol'].iloc[i-1] > 1.5:
                        strength *= 1.2
                    
                    fvg = FVG(
                        type=FVGType.BULLISH,
                        top=float(candle_3_low),
                        bottom=float(candle_1_high),
                        mid=float((candle_3_low + candle_1_high) / 2),
                        time=df['time'].iloc[i-1],
                        index=i-1,
                        size=float(fvg_size),
                        strength=strength
                    )
                    fvg_list.append(fvg)
            
            # شناسایی Bearish FVG
            candle_1_low = df['low'].iloc[i-2]
            candle_3_high = df['high'].iloc[i]
            
            if candle_1_low > candle_3_high:
                fvg_size = candle_1_low - candle_3_high
                
                # شرایط بهبودیافته برای FVG
                body_condition = candle_2_close < candle_2_open
                body_size_condition = candle_2_body > (candle_2_range * 0.3)
                fvg_size_condition = fvg_size >= min_fvg_size
                volume_condition = True
                
                if 'rvol' in df.columns:
                    volume_condition = df['rvol'].iloc[i-1] > 0.8
                
                if all([body_condition, body_size_condition, fvg_size_condition, volume_condition]):
                    # محاسبه قدرت FVG
                    strength = 1.0
                    if candle_2_body > candle_2_range * 0.7:
                        strength = 1.5
                    if 'rvol' in df.columns and df['rvol'].iloc[i-1] > 1.5:
                        strength *= 1.2
                    
                    fvg = FVG(
                        type=FVGType.BEARISH,
                        top=float(candle_1_low),
                        bottom=float(candle_3_high),
                        mid=float((candle_1_low + candle_3_high) / 2),
                        time=df['time'].iloc[i-1],
                        index=i-1,
                        size=float(fvg_size),
                        strength=strength
                    )
                    fvg_list.append(fvg)
        
        # بررسی پر شدن FVGها
        for fvg in fvg_list:
            if (len(df) - 1 - fvg.index) < 20:
                fvg.filled = False
                continue
            
            check_limit = min(fvg.index + 80, len(df))
            filled = False
            
            for j in range(fvg.index + 1, check_limit):
                if fvg.type == FVGType.BULLISH:
                    if df['low'].iloc[j] <= fvg.top:
                        filled = True
                        break
                elif fvg.type == FVGType.BEARISH:
                    if df['high'].iloc[j] >= fvg.bottom:
                        filled = True
                        break
            
            fvg.filled = filled
        
        unfilled_count = sum(1 for f in fvg_list if not f.filled)
        logger.info(f"Detected {len(fvg_list)} FVGs ({unfilled_count} unfilled)")
        
        return fvg_list
    
    def detect_order_blocks(self, lookback: int = 50) -> List[OrderBlock]:
            """
            شناسایی Order Block های معتبر به سبک SMC
            (کندل مخالف قبل از حرکت شارپ)
            """
            order_blocks = []
            df = self.df
            
            if len(df) < lookback + 5:
                return order_blocks
            
            # پارامترهای تشخیص
            atr = self.atr
            min_move_size = atr * 1.0  # حرکت بعد از OB باید حداقل 1 برابر ATR باشد
            
            # حلقه روی کندل‌ها (تا 3 کندل قبل از آخر، چون نیاز به تایید حرکت بعد داریم)
            for i in range(lookback, len(df) - 3):
                candle_a = df.iloc[i]     # کندل potential OB
                candle_b = df.iloc[i+1]   # کندل تایید 1 (حرکت انفجاری)
                candle_c = df.iloc[i+2]   # کندل تایید 2 (ادامه حرکت)
                
                # ---------------------------
                # 1. شناسایی BULLISH Order Block
                # (کندل نزولی که بعدش حرکت انفجاری صعودی رخ داده)
                # ---------------------------
                is_red_candle = candle_a['close'] < candle_a['open']
                
                # محاسبه قدرت حرکت صعودی بعد از کندل
                move_up = candle_b['close'] - candle_a['high']
                is_strong_move_up = (
                    candle_b['close'] > candle_a['high'] and  # بسته شدن بالای های OB
                    candle_b['close'] > candle_b['open'] and  # کندل بعدی سبز باشد
                    (move_up > min_move_size or (candle_b['close'] - candle_b['open']) > atr * 0.8) # حرکت قدرتمند
                )

                if is_red_candle and is_strong_move_up:
                    # محاسبه قدرت OB
                    strength = 1.0
                    
                    # اگر کندل بعدی (C) هم صعودی بود، اعتبار بیشتر می‌شود
                    if candle_c['close'] > candle_b['high']:
                        strength += 0.5
                    
                    # بررسی حجم (اگر موجود باشد)
                    if 'rvol' in df.columns and df['rvol'].iloc[i+1] > 1.2:
                        strength += 0.5

                    block = OrderBlock(
                        type='BULLISH_OB',
                        high=float(candle_a['high']), # ناحیه OB از High
                        low=float(candle_a['low']),   # تا Low کندل قرمز است
                        time=candle_a['time'],
                        index=i,
                        strength=strength
                    )
                    order_blocks.append(block)

                # ---------------------------
                # 2. شناسایی BEARISH Order Block
                # (کندل صعودی که بعدش حرکت انفجاری نزولی رخ داده)
                # ---------------------------
                is_green_candle = candle_a['close'] > candle_a['open']
                
                # محاسبه قدرت حرکت نزولی بعد از کندل
                move_down = candle_a['low'] - candle_b['close']
                is_strong_move_down = (
                    candle_b['close'] < candle_a['low'] and   # بسته شدن پایین لوی OB
                    candle_b['close'] < candle_b['open'] and  # کندل بعدی قرمز باشد
                    (move_down > min_move_size or (candle_b['open'] - candle_b['close']) > atr * 0.8) # حرکت قدرتمند
                )

                if is_green_candle and is_strong_move_down:
                    # محاسبه قدرت OB
                    strength = 1.0
                    
                    # تایید کندل دوم
                    if candle_c['close'] < candle_b['low']:
                        strength += 0.5
                        
                    # بررسی حجم
                    if 'rvol' in df.columns and df['rvol'].iloc[i+1] > 1.2:
                        strength += 0.5

                    block = OrderBlock(
                        type='BEARISH_OB',
                        high=float(candle_a['high']), # ناحیه OB از High
                        low=float(candle_a['low']),   # تا Low کندل سبز است
                        time=candle_a['time'],
                        index=i,
                        strength=strength
                    )
                    order_blocks.append(block)
            
            # فیلتر کردن OB های قدیمی و تست شده (Mitigated)
            # در نسخه ساده، فقط جدیدترین‌ها را برمی‌گردانیم
            logger.info(f"Detected {len(order_blocks)} raw order blocks")
            return order_blocks[-5:]  # فقط 5 تای آخر که نزدیک به قیمت هستند
    
    def detect_liquidity_sweeps(self, swings: List[SwingPoint], lookback_swings: int = 5) -> List[LiquiditySweep]:
            """
            شناسایی نفوذهای فیک (Liquidity Sweeps) با استانداردهای SMC
            """
            if not swings:
                return []
            
            sweeps = []
            # فقط 20 کندل آخر را بررسی می‌کنیم (سوئیپ باید تازه باشد)
            recent_data = self.df.tail(40) 
            recent_highs = [s for s in swings if s.side == 'HIGH'][-lookback_swings:]
            recent_lows = [s for s in swings if s.side == 'LOW'][-lookback_swings:]
            
            atr_value = self.atr
            min_penetration = atr_value * self.settings.get('MIN_SWEEP_PENETRATION_MULTIPLIER', 0.2)
            max_penetration = atr_value * 3.0  # 🔥 فیلتر جدید: نفوذ نباید بیش از حد عمیق باشد
            
            for idx, row in recent_data.iterrows():
                candle_range = row['high'] - row['low']
                
                # فیلتر کندل‌های بسیار کوچک (دوجی‌های بی‌ارزش)
                if candle_range < (atr_value * 0.5):
                    continue
                
                # ---------------------------
                # 1. بررسی سوئیپ نزولی (Bearish Sweep of Highs)
                # ---------------------------
                for swing in recent_highs:
                    # شرط زمانی: کندل باید بعد از سوینگ باشد
                    if row['time'] <= swing.time:
                        continue

                    # شرط اصلی: High بالاتر رفته اما Close پایین‌تر بسته شده (SFP)
                    if row['high'] > swing.price and row['close'] < swing.price:
                        
                        penetration = row['high'] - swing.price
                        
                        # فیلتر مقدار نفوذ (نه خیلی کم، نه خیلی زیاد)
                        if min_penetration <= penetration <= max_penetration:
                            
                            upper_wick = row['high'] - max(row['open'], row['close'])
                            body_size = abs(row['close'] - row['open'])
                            
                            # اعتبارسنجی قدرت سوئیپ
                            is_valid_shape = (
                                (upper_wick > body_size) or           # پین بار (شدوی بلند)
                                (row['close'] < row['open']) or       # کندل نزولی قوی
                                (upper_wick > candle_range * 0.4)     # شدو حداقل 40% کل کندل باشد
                            )
                            
                            # اگر حجم بالا باشد، شکل کندل اهمیت کمتری دارد
                            has_high_volume = 'rvol' in row and row['rvol'] > 1.5
                            
                            if is_valid_shape or has_high_volume:
                                strength = min(3.0, (penetration / atr_value) + (0.5 if has_high_volume else 0))
                                
                                sweep = LiquiditySweep(
                                    time=row['time'],
                                    type='BEARISH_SWEEP',
                                    level=swing.price,
                                    penetration=penetration,
                                    description=f"Bearish Sweep (RVOL: {row.get('rvol', 0):.1f}x)",
                                    strength=strength
                                )
                                sweeps.append(sweep)

                # ---------------------------
                # 2. بررسی سوئیپ صعودی (Bullish Sweep of Lows)
                # ---------------------------
                for swing in recent_lows:
                    if row['time'] <= swing.time:
                        continue

                    # شرط اصلی: Low پایین‌تر رفته اما Close بالاتر بسته شده
                    if row['low'] < swing.price and row['close'] > swing.price:
                        
                        penetration = swing.price - row['low']
                        
                        if min_penetration <= penetration <= max_penetration:
                            
                            lower_wick = min(row['open'], row['close']) - row['low']
                            body_size = abs(row['close'] - row['open'])
                            
                            is_valid_shape = (
                                (lower_wick > body_size) or           # پین بار
                                (row['close'] > row['open']) or       # کندل صعودی قوی
                                (lower_wick > candle_range * 0.4)
                            )
                            
                            has_high_volume = 'rvol' in row and row['rvol'] > 1.5
                            
                            if is_valid_shape or has_high_volume:
                                strength = min(3.0, (penetration / atr_value) + (0.5 if has_high_volume else 0))
                                
                                sweep = LiquiditySweep(
                                    time=row['time'],
                                    type='BULLISH_SWEEP',
                                    level=swing.price,
                                    penetration=penetration,
                                    description=f"Bullish Sweep (RVOL: {row.get('rvol', 0):.1f}x)",
                                    strength=strength
                                )
                                sweeps.append(sweep)

            # حذف تکراری‌ها و بازگرداندن جدیدترین‌ها
            unique_sweeps = []
            seen = set()
            # لیست را برعکس می‌کنیم تا از آخر به اول (جدیدترین‌ها) پردازش کنیم
            for sweep in reversed(sweeps):
                key = (sweep.time, sweep.type, round(sweep.level, 2))
                if key not in seen:
                    seen.add(key)
                    unique_sweeps.append(sweep)
            
            # برگرداندن به ترتیب زمانی
            unique_sweeps.reverse()
            
            logger.info(f"Detected {len(unique_sweeps)} fresh liquidity sweeps")
            return unique_sweeps
    
    def determine_market_structure(self, swings: List[SwingPoint], lookback_swings: int = 4, 
                                    volume_analysis: Optional[Dict] = None,
                                    volatility_state: Optional[str] = None,
                                    adx_value: Optional[float] = None) -> MarketStructure:
        """
        تعیین ساختار بازار با منطق NDS (Nodal Displacement Sequencing)
        تمرکز بر جابجایی نودها (Displacement) و تقارن فرکتالی
        """
        
        # ۱. شرایط اولیه - در NDS حتی با دیتای کم هم به دنبال نود هستیم
        if len(swings) < 3:
            current_price = float(self.df['close'].iloc[-1])
            return MarketStructure(
                trend=MarketTrend.RANGING,
                bos="NONE", choch="NONE",
                last_high=None, last_low=None,
                current_price=current_price,
                bos_choch_confidence=0.0,
                volume_analysis=volume_analysis,
                volatility_state=volatility_state,
                adx_value=adx_value,
                structure_score=10.0  # 🔴 اضافه شد: امتیاز پایه
            )
        
        # ۲. فیلتر سوینگ‌ها (نودها) با رویکرد NDS
        # در NDS فیلتر ATR 1.5 بسیار بزرگ است و نودهای فرکتالی را حذف می‌کند.
        # حساسیت را پویا می‌کنیم: در اسکلپینگ نودهای نزدیک‌تر اهمیت هندسی دارند.
        dynamic_multiplier = 0.75 if volatility_state == "HIGH" else 1.0 
        min_swing_distance = self.atr * dynamic_multiplier
        
        major_swings = []
        last_high_p, last_low_p = None, None
        
        # شناسایی نودهای معتبر برای توالی (Sequencing)
        for swing in swings:
            if swing.side == 'HIGH':
                if last_high_p is None or abs(swing.price - last_high_p) >= min_swing_distance:
                    major_swings.append(swing)
                    last_high_p = swing.price
            else:
                if last_low_p is None or abs(swing.price - last_low_p) >= min_swing_distance:
                    major_swings.append(swing)
                    last_low_p = swing.price
        
        # ۳. انتخاب نودهای مرتبط (Recent Nodes)
        recent_swings = self._get_relevant_swings(major_swings, lookback_swings)
        
        # ۴. شناسایی نودهای مرجع (Reference Nodes)
        last_high = next((s for s in reversed(recent_swings) if s.side == 'HIGH'), None)
        last_low = next((s for s in reversed(recent_swings) if s.side == 'LOW'), None)
        
        # ۵. داده‌های جاری قیمت برای تشخیص Displacement
        current_price = float(self.df['close'].iloc[-1])
        current_high = float(self.df['high'].iloc[-1])
        current_low = float(self.df['low'].iloc[-1])
        
        # ۶. تشخیص روند NDS (بر اساس جابجایی قیمت نسبت به آخرین نودها)
        # در NDS اگر قیمت از نود عبور کند، روند تغییر کرده است (حتی قبل از تشکیل سوینگ جدید)
        trend, trend_strength, trend_confidence = self._determine_trend_with_confidence(
            recent_swings, current_price, volume_analysis, volatility_state, adx_value
        )
        
        # اصلاح روند (NDS Override): اگر جابجایی (Displacement) رخ داده باشد
        nds_displacement = False
        if last_high and current_price > last_high.price:
            trend = MarketTrend.UPTREND
            nds_displacement = True
        elif last_low and current_price < last_low.price:
            trend = MarketTrend.DOWNTREND
            nds_displacement = True

        # ۷. تشخیص BOS/CHoCH (در NDS این‌ها جابجایی توالی نودها هستند)
        bos, choch, bos_choch_confidence = self._detect_bos_choch(
            last_high=last_high,
            last_low=last_low,
            current_high=current_high,
            current_low=current_low,
            current_close=current_price,
            trend=trend,
            trend_strength=trend_strength,
            volume_analysis=volume_analysis,
            volatility_state=volatility_state
        )
        
        # اگر NDS جابجایی را تایید کند اما BOS کلاسیک هنوز صادر نشده باشد
        if nds_displacement and bos == "NONE":
            bos = "BULLISH_DISPLACEMENT" if trend == MarketTrend.UPTREND else "BEARISH_DISPLACEMENT"
            bos_choch_confidence = max(bos_choch_confidence, 0.75)

        # ۸. محاسبه محدوده نوسان نودها (Nodal Range)
        range_width, range_mid = None, None
        if last_high and last_low:
            range_width = abs(last_high.price - last_low.price)
            range_mid = (last_high.price + last_low.price) / 2
            
            # در NDS رنج کوچک نشانه فشردگی برای جابجایی بزرگ است، پس حذفش نمی‌کنیم
            min_range = self.atr * 0.5 
            if range_width < min_range:
                range_width = None # هنوز رنج معتبری نداریم
        
        # ۹. محاسبه امتیاز ساختار (تطبیق داده شده با NDS)
        # 🔴 **اصلاح: ارسال پارامترهای اضافی**
        structure_score = self._calculate_structure_score(
            bos=bos,
            choch=choch,
            confidence=bos_choch_confidence,
            trend_strength=trend_strength,
            volume_analysis=volume_analysis,
            volatility_state=volatility_state,
            range_width=range_width,
            last_high=last_high,      # 🔴 اضافه شد
            last_low=last_low,        # 🔴 اضافه شد
            adx_value=adx_value       # 🔴 اضافه شد
        )
        
        # افزایش امتیاز در صورت وجود جابجایی صریح (NDS Core Rule)
        if nds_displacement:
            structure_score = min(100.0, structure_score + 20.0)
        
        # ۱۰. ایجاد خروجی نهایی
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
            volume_analysis=volume_analysis,
            volatility_state=volatility_state,
            adx_value=adx_value,
            structure_score=structure_score
        )
        
        logger.info(f"🏛️ NDS Structure: Trend={trend.value}, Signal={bos}, "
                    f"Conf={bos_choch_confidence:.1%}, Score={structure_score:.1f}")
        
        return structure

    def _calculate_structure_score(self, bos, choch, confidence, trend_strength,
                                        volume_analysis, volatility_state, range_width,
                                        last_high=None, last_low=None, adx_value=None, 
                                        sweeps=None) -> float: # 🔴 اضافه شدن sweeps به عنوان آرگومان اختیاری
                """
                محاسبه امتیاز کیفیت ساختار - نسخه بهینه شده برای اسکلپینگ چابک طلا
                بدون تغییر در نام متد یا حذف منطق‌های اصلی
                """
                score = 0.0
                current_price = float(self.df['close'].iloc[-1])
                
                # خواندن حد آستانه از تنظیمات (در صورت عدم وجود، پیش‌فرض 15)
                adx_threshold = self.GOLD_SETTINGS.get('ADX_THRESHOLD_WEAK', 15)
                
                print(f"🔍 DEBUG _calculate_structure_score:")
                print(f"   bos: {bos}, choch: {choch}")
                print(f"   confidence: {confidence}, trend_strength: {trend_strength}")
                print(f"   last_high: {last_high}, last_low: {last_low}")
                print(f"   adx_value: {adx_value} (Threshold: {adx_threshold})")
                print(f"   volatility_state: {volatility_state}")
                
                # ۱. امتیاز پایه برای داشتن ساختار (کاهش از 10 به 5 برای جلوگیری از تورم امتیاز)
                score += 5.0  
                print(f"   Base score: +5.0 = {score:.1f}")
                
                # ۲. امتیاز برای تاییدیه های ساختاری (BOS/CHoCH)
                if bos != "NONE":
                    if "DISPLACEMENT" in bos:
                        # استفاده از ضریب اطمینان برای واقعی‌تر کردن امتیاز
                        score += 30 * confidence  
                        print(f"   BOS DISPLACEMENT: +{30 * confidence:.1f} = {score:.1f}")
                    else:
                        score += 45 * confidence  
                        print(f"   BOS CLASSIC: +{45 * confidence:.1f} = {score:.1f}")
                elif choch != "NONE":
                    score += 35 * confidence
                    print(f"   CHoCH: +{35 * confidence:.1f} = {score:.1f}")
                
                # ۳. امتیاز برای نفوذ قیمت (اصلاح شده: فقط در صورت تایید جهت روند)
                # به جای 25 امتیاز ثابت، امتیاز را به قدرت نفوذ و اطمینان وابسته کردیم
                if last_high and current_price > last_high.price:
                    penetration_bonus = 20.0 * (1.0 if confidence > 0.5 else 0.5)
                    score += penetration_bonus
                    print(f"   Price above last high: +{penetration_bonus:.1f} = {score:.1f}")
                elif last_low and current_price < last_low.price:
                    penetration_bonus = 20.0 * (1.0 if confidence > 0.5 else 0.5)
                    score += penetration_bonus
                    print(f"   Price below last low: +{penetration_bonus:.1f} = {score:.1f}")
                
                # ۴. هم‌راستایی با قدرت روند (ADX Alignment) - متصل به متغیر داینامیک
                if adx_value:
                    if adx_value > 25:
                        score += 15.0  # پاداش بیشتر برای روند قوی
                        print(f"   ADX > 25: +15.0 = {score:.1f}")
                    elif adx_value > adx_threshold: # استفاده از متغیر جدید شما
                        score += 7.0   
                        print(f"   ADX > {adx_threshold}: +7.0 = {score:.1f}")
                    else:
                        score -= 10.0  # جریمه برای ADX زیر آستانه (رنج بودن بازار)
                        print(f"   ADX weak (< {adx_threshold}): -10.0 = {score:.1f}")
                
                # ۵. امتیاز برای قدرت روند (Trend Strength از سوینگ ها)
                trend_score = 15 * trend_strength
                score += trend_score
                print(f"   Trend strength ({trend_strength}): +{trend_score:.1f} = {score:.1f}")
                
                # ۶. فاکتور حجم
                if volume_analysis:
                    vol_zone = volume_analysis.get('zone', 'NORMAL')
                    if vol_zone == "HIGH":
                        score += 15
                        print(f"   Volume HIGH: +15.0 = {score:.1f}")
                    elif vol_zone == "NORMAL":
                        score += 7
                        print(f"   Volume NORMAL: +7.0 = {score:.1f}")
                
                # ۷. وضعیت نوسان (تطبیق با کلمات کلیدی استاندارد کد شما)
                if volatility_state in ["MEDIUM", "MODERATE_VOLATILITY"]:
                    score += 10
                    print(f"   Volatility MODERATE: +10.0 = {score:.1f}")
                elif volatility_state == "HIGH":
                    score -= 10   # افزایش جریمه برای نوسان مخرب در اسکلپینگ
                    print(f"   Volatility HIGH: -10.0 = {score:.1f}")
                elif volatility_state == "LOW":
                    score -= 5    # بازار خیلی آروم هم برای اسکلپر امتیاز منفی داره
                    print(f"   Volatility LOW: -5.0 = {score:.1f}")
                
                # ۸. اعتبار رنج (Range Validity)
                if range_width and hasattr(self, 'atr') and self.atr > 0:
                    atr_ratio = range_width / self.atr
                    if atr_ratio < 1.0:
                        score -= 10  
                        print(f"   Range small (ATR ratio {atr_ratio:.1f}): -10.0 = {score:.1f}")
                    elif atr_ratio > 1.5:
                        score += 10
                        print(f"   Range large (ATR ratio {atr_ratio:.1f}): +10.0 = {score:.1f}")

                # ۹. امتیاز برای داشتن سوینگ‌های معتبر (تعدیل شده)
                if last_high is not None:
                    score += 2.5
                    print(f"   Has last_high: +2.5 = {score:.1f}")
                if last_low is not None:
                    score += 2.5
                    print(f"   Has last_low: +2.5 = {score:.1f}")

                # ۱۰. 🟢 اضافه شدن بخش نقدینگی (Liquidity Sweeps)
                if sweeps:
                    for sweep in sweeps:
                        if sweep.type == 'BULLISH_SWEEP':
                            sweep_bonus = 15.0 * sweep.strength
                            score += sweep_bonus
                            print(f"   Bullish Sweep detected: +{sweep_bonus:.1f} = {score:.1f}")
                        elif sweep.type == 'BEARISH_SWEEP':
                            sweep_penalty = 15.0 * sweep.strength
                            score -= sweep_penalty
                            print(f"   Bearish Sweep detected: -{sweep_penalty:.1f} = {score:.1f}")
                
                # تضمین حداقل امتیاز
                if score < 0: score = 0.0
                
                # محدود کردن امتیاز بین 0 تا 100
                final_score = max(0.0, min(100.0, score))
                
                print(f"📊 FINAL Structure Score: {final_score:.1f}")
                
                return round(final_score, 2)
    
    def _get_relevant_swings(self, major_swings: List[SwingPoint], lookback: int) -> List[SwingPoint]:
        """انتخاب سوینگ‌های مرتبط"""
        if len(major_swings) <= lookback:
            return major_swings
        
        # روش 1: آخرین سوینگ‌ها با فاصله زمانی مناسب
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
    
    def _determine_trend_with_confidence(self, swings: List[SwingPoint], current_price: float,
                                    volume_analysis: Optional[Dict] = None,
                                    volatility_state: Optional[str] = None,
                                    adx_value: Optional[float] = None) -> Tuple[MarketTrend, float, float]:
        """تشخیص روند با اطمینان بر اساس چندین فاکتور"""
        
        # منطق پایه
        if len(swings) < 2:
            return MarketTrend.RANGING, 0.0, 0.0
        
        # ۱. تشخیص روند بر اساس سوینگ‌ها
        highs = [s for s in swings if s.side == 'HIGH']
        lows = [s for s in swings if s.side == 'LOW']
        
        if len(highs) < 2 or len(lows) < 2:
            return MarketTrend.RANGING, 0.0, 0.0
        
        # تحلیل HH/HL و LL/LH
        higher_highs = sum(1 for i in range(1, len(highs)) if highs[i].price > highs[i-1].price)
        higher_lows = sum(1 for i in range(1, len(lows)) if lows[i].price > lows[i-1].price)
        lower_highs = sum(1 for i in range(1, len(highs)) if highs[i].price < highs[i-1].price)
        lower_lows = sum(1 for i in range(1, len(lows)) if lows[i].price < lows[i-1].price)
        
        # ۲. تصمیم‌گیری روند
        if higher_highs > lower_highs and higher_lows > lower_lows:
            trend = MarketTrend.UPTREND
            strength = (higher_highs + higher_lows) / (len(highs) + len(lows) - 2)
        elif lower_highs > higher_highs and lower_lows > higher_lows:
            trend = MarketTrend.DOWNTREND
            strength = (lower_highs + lower_lows) / (len(highs) + len(lows) - 2)
        else:
            trend = MarketTrend.RANGING
            strength = 0.3  # قدرت پایه برای رنج
        
        # ۳. تنظیم قدرت بر اساس ADX
        if adx_value:
            adx_strength = adx_value / 100.0
            strength = (strength * 0.6) + (adx_strength * 0.4)
        
        # ۴. تنظیم بر اساس حجم
        if volume_analysis:
            volume_factor = min(volume_analysis.get('rvol', 1.0), 2.0) / 2.0
            strength = strength * (0.7 + 0.3 * volume_factor)
        
        # ۵. محاسبه اطمینان نهایی
        confidence = strength * 0.7  # پایه
        
        # افزایش اطمینان بر اساس نوسان مطلوب
        if volatility_state == "MEDIUM":
            confidence *= 1.2
        elif volatility_state == "LOW":
            confidence *= 0.9
        
        return trend, strength, min(1.0, confidence)
    
    def _detect_bos_choch(self, last_high: Optional[SwingPoint], last_low: Optional[SwingPoint],
                        current_high: float, current_low: float, current_close: float,
                        trend: MarketTrend, trend_strength: float, 
                        volume_analysis: Optional[Dict] = None,
                        volatility_state: Optional[str] = None) -> Tuple[str, str]:
        """
        🔥 نسخه حرفه‌ای تشخیص BOS/CHoCH - با هوش مصنوعی بازار
        پارامترهای جدید:
            current_close: قیمت بسته‌شدن (برای تأیید نهایی)
            volume_analysis: تحلیل حجم برای تأیید شکست
            volatility_state: وضعیت نوسان بازار
        """
        
        bos = "NONE"
        choch = "NONE"
        confidence = 0.0
        
        if not last_high or not last_low:
            logger.debug("⚠️ BOS/CHoCH: سوینگ‌های کافی برای تحلیل وجود ندارد")
            return bos, choch, confidence
        
        # ۱. محاسبات پیشرفته بافر بر اساس شرایط بازار
        base_buffer = self._calculate_dynamic_buffer(
            atr=self.atr,
            trend_strength=trend_strength,
            volatility_state=volatility_state,
            volume_analysis=volume_analysis
        )
        
        # ۲. تشخیص BOS (Break of Structure)
        bos, bos_confidence = self._detect_bos_advanced(
            last_high=last_high,
            last_low=last_low,
            current_high=current_high,
            current_low=current_low,
            current_close=current_close,
            trend=trend,
            base_buffer=base_buffer,
            volume_analysis=volume_analysis
        )
        
        # ۳. تشخیص CHoCH (Change of Character)
        choch, choch_confidence = self._detect_choch_advanced(
            last_high=last_high,
            last_low=last_low,
            current_high=current_high,
            current_low=current_low,
            current_close=current_close,
            trend=trend,
            base_buffer=base_buffer,
            bos_detected=(bos != "NONE")
        )
        
        # ۴. اعتبارسنجی نهایی با کندل استیک‌ها
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
            df=self.df  # برای تحلیل چندکندلی
        )
        
        return final_bos, final_choch, final_confidence


    def _calculate_dynamic_buffer(self, atr: float, trend_strength: float, 
                                volatility_state: str, volume_analysis: Dict) -> Dict:
        """
        محاسبه بافر پویا بر اساس شرایط مختلف بازار
        """
        buffers = {
            'bos': atr * 0.15,  # پیش‌فرض
            'choch': atr * 0.12,
            'aggressive': atr * 0.08,
            'conservative': atr * 0.2
        }
        
        # تنظیم بر اساس قدرت روند
        if trend_strength > 0.7:
            buffers['bos'] *= 0.8
            buffers['choch'] *= 0.7
        elif trend_strength < 0.3:
            buffers['bos'] *= 1.5
            buffers['choch'] *= 1.3
        
        # تنظیم بر اساس نوسان
        if volatility_state == "HIGH":
            buffers['bos'] *= 1.2
            buffers['choch'] *= 1.1
        elif volatility_state == "LOW":
            buffers['bos'] *= 0.8
            buffers['choch'] *= 0.9
        
        # تنظیم بر اساس حجم
        if volume_analysis and volume_analysis.get('zone') == "HIGH":
            buffers['bos'] *= 0.9
            buffers['choch'] *= 0.85
        
        return buffers



    def _confirm_with_candle_pattern(self, current_high: float, current_low: float, current_close: float,
                                last_high_price: float, last_low_price: float, trend: MarketTrend) -> bool:
        """
        تأیید شکست با الگوهای کندل استیک
        """
        # محاسبه اندازه کندل
        candle_size = abs(current_high - current_low)
        body_size = abs(current_close - ((current_high + current_low) / 2))
        
        # برای شکست صعودی
        if trend == MarketTrend.UPTREND:
            # کندل باید بسته‌شدن قوی بالای مقاومت داشته باشد
            if current_close > last_high_price and (current_close - last_high_price) > (candle_size * 0.3):
                # بدنه کندل باید حداقل ۴۰٪ اندازه کل کندل باشد
                if body_size > (candle_size * 0.4):
                    return True
        
        # برای شکست نزولی
        elif trend == MarketTrend.DOWNTREND:
            # کندل باید بسته‌شدن قوی زیر حمایت داشته باشد
            if current_close < last_low_price and (last_low_price - current_close) > (candle_size * 0.3):
                if body_size > (candle_size * 0.4):
                    return True
        
        return False

    def _check_reversal_patterns(self, current_high: float, current_low: float, current_close: float,
                            pattern_type: str) -> bool:
        """
        بررسی الگوهای بازگشتی کندلی
        """
        # دریافت کندل فعلی و قبلی
        try:
            current_candle = self.df.iloc[-1]
            prev_candle = self.df.iloc[-2]
            
            current_open = current_candle['open']
            prev_open = prev_candle['open']
            prev_close = prev_candle['close']
            prev_high = prev_candle['high']
            prev_low = prev_candle['low']
            
            # محاسبه اندازه‌ها
            current_body = abs(current_close - current_open)
            prev_body = abs(prev_close - prev_open)
            current_range = current_high - current_low
            prev_range = prev_high - prev_low
            
            if pattern_type == "bullish":
                # الگوهای بازگشت صعودی
                # پین‌بار صعودی
                if current_low < prev_low and current_close > (current_open + (current_range * 0.6)):
                    return True
                
                # اینگالف صعودی
                if current_close > prev_open and current_open < prev_close and current_body > (prev_body * 1.5):
                    return True
            
            elif pattern_type == "bearish":
                # الگوهای بازگشت نزولی
                # پین‌بار نزولی
                if current_high > prev_high and current_close < (current_open - (current_range * 0.6)):
                    return True
                
                # اینگالف نزولی
                if current_close < prev_open and current_open > prev_close and current_body > (prev_body * 1.5):
                    return True
        
        except (IndexError, KeyError):
            pass
        
        return False

    def _calculate_bearish_pressure(self, recent_candles) -> float:
        """محاسبه فشار فروش در کندل‌های اخیر"""
        if len(recent_candles) == 0:
            return 0.0
        
        bearish_count = 0
        total_candles = len(recent_candles)
        
        for _, candle in recent_candles.iterrows():
            if candle['close'] < candle['open']:  # کندل نزولی
                bearish_count += 1
        
        return bearish_count / total_candles

    def _calculate_bullish_pressure(self, recent_candles) -> float:
        """محاسبه فشار خرید در کندل‌های اخیر"""
        if len(recent_candles) == 0:
            return 0.0
        
        bullish_count = 0
        total_candles = len(recent_candles)
        
        for _, candle in recent_candles.iterrows():
            if candle['close'] > candle['open']:  # کندل صعودی
                bullish_count += 1
        
        return bullish_count / total_candles



    def _detect_bos_advanced(self, last_high, last_low, current_high, current_low,
                            current_close, trend, base_buffer, volume_analysis) -> Tuple[str, float]:
        """
        تشخیص پیشرفته BOS با در نظر گرفتن تأییدیه‌های چندگانه
        """
        bos = "NONE"
        confidence = 0.0
        
        last_high_price = last_high.price
        last_low_price = last_low.price
        
        # ۱. شکست قیمتی
        price_break = False
        price_signal = ""
        
        if trend == MarketTrend.UPTREND:
            if current_close > (last_high_price + base_buffer['bos']):
                price_break = True
                price_signal = "BULLISH_BOS"
        elif trend == MarketTrend.DOWNTREND:
            if current_close < (last_low_price - base_buffer['bos']):
                price_break = True
                price_signal = "BEARISH_BOS"
        
        # ۲. تأیید با حجم
        volume_confirmation = False
        if volume_analysis:
            volume_ratio = volume_analysis.get('rvol', 1.0)
            volume_confirmation = volume_ratio > 1.2  # حجم بالا تأیید می‌کند
        
        # ۳. تأیید با کندل بسته‌شدن
        candle_confirmation = self._confirm_with_candle_pattern(
            current_high, current_low, current_close,
            last_high_price, last_low_price, trend
        )
        
        # ۴. امتیازدهی نهایی
        if price_break:
            confidence = 0.4  # پایه
            
            if volume_confirmation:
                confidence += 0.3
            
            if candle_confirmation:
                confidence += 0.3
            
            if confidence >= 0.6:  # آستانه اعتماد
                bos = price_signal
                logger.info(f"✅ BOS تایید شده: {bos} با اطمینان {confidence:.1%}")
        
        return bos, confidence


    def _detect_choch_advanced(self, last_high, last_low, current_high, current_low,
                            current_close, trend, base_buffer, bos_detected) -> Tuple[str, float]:
        """
        تشخیص پیشرفته CHoCH - حساس به تغییر روند
        """
        choch = "NONE"
        confidence = 0.0
        
        # اگر BOS شناسایی شده، CHoCH کم‌اهمیت‌تر است
        if bos_detected:
            return choch, confidence
        
        last_high_price = last_high.price
        last_low_price = last_low.price
        
        # تشخیص CHoCH بستگی به روند دارد
        if trend == MarketTrend.UPTREND:
            # CHoCH نزولی: شکست حمایت در روند صعودی
            if current_close < (last_low_price - base_buffer['choch']):
                # تأیید با الگوهای بازگشتی
                if self._check_reversal_patterns(current_high, current_low, current_close, "bearish"):
                    choch = "BEARISH_CHOCH"
                    confidence = 0.7
        
        elif trend == MarketTrend.DOWNTREND:
            # CHoCH صعودی: شکست مقاومت در روند نزولی
            if current_close > (last_high_price + base_buffer['choch']):
                if self._check_reversal_patterns(current_high, current_low, current_close, "bullish"):
                    choch = "BULLISH_CHOCH"
                    confidence = 0.7
        
        elif trend == MarketTrend.RANGING:
            # در رنج، هر شکست قابل توجهی می‌تواند CHoCH باشد
            range_buffer = base_buffer['choch'] * 1.5
            
            if current_close > (last_high_price + range_buffer):
                choch = "BULLISH_CHOCH"
                confidence = 0.6
            elif current_close < (last_low_price - range_buffer):
                choch = "BEARISH_CHOCH"
                confidence = 0.6
        
        return choch, confidence


    def _validate_with_price_action(self, bos, choch, bos_confidence, choch_confidence,
                                current_high, current_low, current_close,
                                last_high_price, last_low_price, df) -> Tuple[str, str, float]:
        """
        اعتبارسنجی نهایی با پرایس اکشن چندکندلی
        """
        final_bos = bos
        final_choch = choch
        final_confidence = max(bos_confidence, choch_confidence)
        
        # اگر سیگنالی داریم، بررسی کندل‌های قبلی
        if bos != "NONE" or choch != "NONE":
            # بررسی ۳ کندل قبلی برای تأیید
            recent_candles = df.iloc[-4:-1]  # ۳ کندل قبل از آخرین
            
            if bos == "BULLISH_BOS":
                # تأیید صعودی: کندل‌های قبلی نباید نزولی قوی باشند
                bearish_pressure = self._calculate_bearish_pressure(recent_candles)
                if bearish_pressure > 0.7:  # فشار فروش بالا
                    logger.warning("⚠️ BOS صعودی با فشار فروش بالا - کاهش اطمینان")
                    final_confidence *= 0.7
            
            elif bos == "BEARISH_BOS":
                bullish_pressure = self._calculate_bullish_pressure(recent_candles)
                if bullish_pressure > 0.7:
                    logger.warning("⚠️ BOS نزولی با فشار خرید بالا - کاهش اطمینان")
                    final_confidence *= 0.7
        
        # فیلتر نهایی: حداقل اطمینان ۰.۵
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
            elif current_price > premium_zone:
                return "PREMIUM", range_mid
            else:
                return "EQUILIBRIUM", range_mid
        else:
            range_high = structure.last_high.price
            range_low = structure.last_low.price
            range_mid = (range_high + range_low) / 2
            
            discount_zone = range_low + (range_high - range_low) * 0.33
            premium_zone = range_low + (range_high - range_low) * 0.66
            
            current_price = structure.current_price
            
            if current_price < discount_zone:
                return "DISCOUNT", range_mid
            elif current_price > premium_zone:
                return "PREMIUM", range_mid
            else:
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
            
            # 1. استخراج آخرین وضعیت قیمت نسبت به سوینگ‌ها
            last_price = self.df['close'].iloc[-1]
            high_swings = [s for s in swings if s.side == 'HIGH']
            low_swings = [s for s in swings if s.side == 'LOW']
            
            if not high_swings or not low_swings:
                return MarketTrend.RANGING

            last_high = high_swings[-1]
            last_low = low_swings[-1]
            prev_high = high_swings[-2] if len(high_swings) > 1 else last_high
            prev_low = low_swings[-2] if len(low_swings) > 1 else last_low

            # 2. تشخیص سریع تغییر روند (Fast CHoCH Detection)
            # اگر قیمت جاری بالاتر از آخرین سقف نزولی باشد -> پتانسیل صعودی
            if last_price > last_high.price:
                return MarketTrend.UPTREND
            
            # اگر قیمت جاری پایین‌تر از آخرین کف صعودی باشد -> پتانسیل نزولی
            if last_price < last_low.price:
                return MarketTrend.DOWNTREND

            # 3. تحلیل کلاسیک ساختار (HH/HL یا LL/LH)
            is_hh = last_high.price > prev_high.price
            is_hl = last_low.price > prev_low.price
            is_lh = last_high.price < prev_high.price
            is_ll = last_low.price < prev_low.price

            # در اسکلپینگ، حتی یکی از این شرایط به همراه تایید قیمت کافیست
            if is_hh or (is_hl and last_price > last_low.price):
                return MarketTrend.UPTREND
                
            if is_ll or (is_lh and last_price < last_high.price):
                return MarketTrend.DOWNTREND

            return MarketTrend.RANGING        
