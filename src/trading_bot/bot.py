"""
ربات اصلی معاملات NDS برای طلا - نسخه اسکلپینگ
نسخه یکپارچه با risk_manager.py
"""

import sys
import time
import signal
import atexit
import logging
from datetime import datetime, timedelta
from pathlib import Path



# پیدا کردن مسیر اصلی پروژه (nds_bot)
# چون bot.py در src/trading_bot قرار دارد، سه پله به عقب برمی‌گردیم
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent # nds_bot
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# اضافه کردن پوشه src به مسیرها
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))



from typing import Dict, List, Any, Optional, Union

# تنظیم لاگر - باید در ابتدای فایل باشد
logger = logging.getLogger(__name__)

# ایمپورت‌های پروژه
from config.settings import config

from src.utils.telegram_notifier import TelegramNotifier



# ایمپورت مدیر ریسک اسکلپینگ
try:
    from src.trading_bot.risk_manager import create_scalping_risk_manager
    logger.info("✅ Scalping Risk Manager module imported successfully")
except ImportError as e:
    logger.critical(f"❌ Scalping Risk Manager module not found: {e}")
    print(f"\n❌ خطا: ماژول مدیریت ریسک اسکلپینگ یافت نشد")
    print(f"   لطفاً از وجود فایل‌های زیر اطمینان حاصل کنید:")
    print(f"   - src/trading_bot/risk_manager.py")
    sys.exit(1)

from src.trading_bot.state import BotState
from src.trading_bot.execution_reporting import generate_execution_report
from src.trading_bot.nds.models import LivePriceSnapshot
from src.trading_bot.realtime_price import RealTimePriceMonitor
from src.trading_bot.trade_tracker import TradeTracker
from src.trading_bot.user_controls import UserControls
from src.ui.cli import print_banner, print_help, update_config_interactive

# ایمپورت آنالایزر جدید به صورت ماژولار
try:
    from src.trading_bot.nds.analyzer import analyze_gold_market  # ✅ تابع اصلی
    from src.trading_bot.nds.analyzer import GoldNDSAnalyzer  # کلاس اصلی
    logger.info("✅ NDS analyzer module imported successfully")
except ImportError as e:
    logger.critical(f"❌ NDS analyzer module not found: {e}")
    print(f"\n❌ خطا: ماژول تحلیل NDS یافت نشد")
    print(f"   لطفاً از وجود فایل‌های زیر اطمینان حاصل کنید:")
    print(f"   - src/trading_bot/nds/analyzer.py")
    print(f"   - src/trading_bot/nds/models.py")
    print(f"   - src/trading_bot/nds/indicators.py")
    print(f"   - src/trading_bot/nds/smc.py")
    sys.exit(1)

# متغیر گلوبال برای سیگنال هندلر (برای دسترسی از بیرون کلاس)
bot_state_global = None





class NDSBot:
    """
    کلاس اصلی ربات NDS برای اسکلپینگ طلا - نسخه Real-Time
    شامل منطق ترید، مدیریت چرخه تحلیل و ارتباط با کاربر
    """

    def __init__(self, mt5_client_cls, risk_manager_cls=None, analyzer_cls=None, analyze_func=None):
        global bot_state_global
        self.bot_state = BotState()
        bot_state_global = self.bot_state  # اتصال به متغیر گلوبال
        
        # کلاس‌های پاس داده شده (Dependency Injection)
        self.MT5Client_cls = mt5_client_cls
        self.RiskManager_cls = risk_manager_cls  # اختیاری برای سازگاری با اسکلپینگ
        
        # استفاده از تابع تحلیل ماژولار (اگر analyze_func مشخص نشده)
        if analyze_func is None:
            self.analyze_market_func = analyze_gold_market  # ✅ استفاده از تابع ماژولار
        else:
            self.analyze_market_func = analyze_func
        
        self.mt5_client = None
        self.risk_manager = None
        self.config = config
        self.analyzer_config = None

        self.price_monitor = RealTimePriceMonitor(config=self.config, bot_state=self.bot_state, logger=logger)
        self.trade_tracker = TradeTracker()
        self.user_controls = UserControls(self, logger)

        self.notifier = TelegramNotifier()
    
    
    
    def initialize(self) -> bool:
            """🔥 مقداردهی اولیه ربات و اتصال به سرویس‌ها (نسخه Real-Time حرفه‌ای)"""
            
            logger.info("🔧 در حال راه‌اندازی ربات اسکلپینگ Real-Time...")
            print("\n🔧 در حال راه‌اندازی ربات اسکلپینگ Real-Time...")
            
            try:
                # 1. ایجاد یا بازیابی MT5 Client
                if self.mt5_client is None:
                    self.mt5_client = self.MT5Client_cls()
                
                # 🔥 دریافت اعتبارنامه‌ها و تنظیم داینامیک فواصل آپدیت
                credentials = self.config.get_mt5_credentials()
                tick_interval = self.config.get('trading_settings.TICK_UPDATE_INTERVAL')

                if credentials:
                    credentials['real_time_enabled'] = True
                    credentials['tick_update_interval'] = tick_interval
                    self.config.save_mt5_credentials(credentials)
                    logger.info(f"✅ تنظیمات Real-Time (Interval: {tick_interval}s) به کانفیگ MT5 اعمال شد")
                
                # 2. مدیریت ورود به حساب
                if not credentials or not all(k in credentials for k in ['login', 'password', 'server']):
                    logger.warning("❌ اطلاعات حساب MT5 ناقص است. درخواست ورودی از کاربر...")
                    # در اینجا می‌توان متد ورود دستی را صدا زد
                    return False

                self.mt5_client.login = int(credentials['login'])
                self.mt5_client.password = credentials['password']
                self.mt5_client.server = credentials['server']
                
                if not self.mt5_client.connect():
                    logger.error("❌ اتصال به MT5 ناموفق بود.")
                    return False
                
                # به‌روزرسانی موجودی حساب در سیستم
                account_info = self.mt5_client.get_account_info()
                if account_info:
                    current_equity = account_info['equity']
                    self.config.update_setting('ACCOUNT_BALANCE', current_equity)
                    logger.info(f"💰 حساب متصل شد | موجودی لحظه‌ای: ${current_equity:,.2f}")
                
                # 🔥 3. شروع مانیتورینگ قیمت لحظه‌ای
                self.price_monitor.set_mt5_client(self.mt5_client)
                self.price_monitor.start()
                
                # 4. آماده‌سازی هوشمند آنالایزر (تطبیق با نتایج بکتست موفق)
                logger.info("🧠 در حال هماهنگ‌سازی تنظیمات آنالایزر با استراتژی SMC...")
                self.analyzer_config = self.config.get_full_config_for_analyzer()
                
                # اطمینان از وجود ANALYZER_SETTINGS برای ماژول‌های داخلی
                if 'ANALYZER_SETTINGS' not in self.analyzer_config:
                    self.analyzer_config['ANALYZER_SETTINGS'] = self.config.get('technical_settings')

                # تزریق پارامترهای بهینه شده بکتست به صورت داینامیک
                tech_settings = self.analyzer_config.get('ANALYZER_SETTINGS', {})
                analyzer_settings = {
                    **tech_settings,
                    'ADX_THRESHOLD_WEAK': self.config.get('technical_settings.ADX_THRESHOLD_WEAK'),
                    'REAL_TIME_ENABLED': True,
                    'USE_CURRENT_PRICE_FOR_ANALYSIS': True
                }
                self.analyzer_config = {
                    **self.analyzer_config,
                    'ANALYZER_SETTINGS': analyzer_settings
                }
                
                # 5. ایجاد مدیر ریسک (Risk Manager)
                scalping_config = self.config.get_risk_manager_config()
                # تزریق تنظیمات فاصله کندلی و انحراف قیمت
                scalping_config.update({
                    'MIN_CANDLES_BETWEEN': self.config.get('trading_rules.MIN_CANDLES_BETWEEN'),
                    'MAX_PRICE_DEVIATION': self.config.get('risk_settings.MAX_PRICE_DEVIATION_PIPS'),
                    'real_time_enabled': True
                })
                
                self.risk_manager = create_scalping_risk_manager(overrides=scalping_config)
                
                logger.info("✅ ربات با موفقیت عملیاتی شد.")
                
                # 🔥 نمایش گزارش وضعیت واقعی (نه مقادیر فیکس!)
                self._log_real_time_status()


                logger.info("🔄 بازیابی پوزیشن‌های باز از MT5...")
                self._monitor_open_trades()
                
                return True
                
            except Exception as e:
                logger.critical(f"❌ خطای بحرانی در Initialize: {e}", exc_info=True)
                return False

    def _log_real_time_status(self):
        """🔥 گزارش وضعیت واقعی و داینامیک سیستم (بدون مقادیر Fixed)"""
        try:
            symbol = self.config.get('trading_settings.SYMBOL')
            current_price = self.price_monitor.get_current_price(symbol)
            
            # استخراج مقادیر واقعی از کانتستنت‌ها و وضعیت جاری
            conn_status = "✅ Connected" if self.mt5_client and self.mt5_client.connected else "❌ Disconnected"
            monitor_status = "✅ Active" if hasattr(self.mt5_client, 'real_time_monitor') and self.mt5_client.real_time_monitor else "⚠️ Inactive"
            
            # محاسبه انحراف قیمت واقعی از تنظیمات
            max_dev = self.config.get('risk_settings.MAX_PRICE_DEVIATION_PIPS')
            min_candles = self.config.get('trading_rules.MIN_CANDLES_BETWEEN')
            
            status_report = f"""
            🎯 گزارش وضعیت لحظه‌ای سیستم (Real-Time)
            ==========================================
            📊 وضعیت اتصال: {conn_status}
            🎯 مانیتور قیمت: {monitor_status}
            💰 اکوئیتی جاری: ${self.config.get('ACCOUNT_BALANCE'):,.2f}
            
            📈 وضعیت بازار لحظه‌ای:
            نماد: {symbol}
            Bid: {current_price.get('bid', 0.0):.2f} | Ask: {current_price.get('ask', 0.0):.2f}
            اسپرد: {current_price.get('spread', 0.0):.2f}
            منبع قیمت: {current_price.get('source', 'Unknown')}
            
            ⚙️ پارامترهای فعال معاملاتی:
            فاصله استراحت: {min_candles} کندل
            حداکثر انحراف مجاز: {max_dev} Pips
            آپدیت قیمت: هر {self.config.get('trading_settings.TICK_UPDATE_INTERVAL')} ثانیه
            ==========================================
            """
            
            logger.info(status_report)
            print(status_report)
            
        except Exception as e:
            logger.error(f"❌ خطا در تولید گزارش وضعیت: {e}")

    def run_analysis_cycle(self, cycle_number: int):
            """اجرای یک سیکل کامل تحلیل بازار اسکلپینگ با فیلتر فاصله کندلی"""
            # ۱. استخراج مستقیم تنظیمات از bot_config.json
            SYMBOL = self.config.get('trading_settings.SYMBOL')
            TIMEFRAME = self.config.get('trading_settings.TIMEFRAME')
            BARS_TO_FETCH = self.config.get('trading_settings.BARS_TO_FETCH')
            ENABLE_AUTO_TRADING = self.config.get('trading_settings.ENABLE_AUTO_TRADING')
            ENABLE_DRY_RUN = self.config.get('trading_settings.ENABLE_DRY_RUN')
            
            # استفاده از متغیر جدید بر پایه کندل
            MIN_CANDLES_BETWEEN = self.config.get('trading_rules.MIN_CANDLES_BETWEEN')
            MAX_POS = self.config.get('trading_rules.MAX_POSITIONS')
            WAIT_CLOSE = self.config.get('trading_rules.WAIT_FOR_CLOSE_BEFORE_NEW_TRADE')
            
            ENTRY_FACTOR = self.config.get('technical_settings.ENTRY_FACTOR')
            MIN_CONFIDENCE = self.config.get('technical_settings.SCALPING_MIN_CONFIDENCE')
            
            RISK_AMOUNT_USD = self.config.get('risk_settings.RISK_AMOUNT_USD')
            ACCOUNT_BALANCE = self.config.get('ACCOUNT_BALANCE')

            logger.info(f"⚙️ تنظیمات نهایی بارگذاری شد: Timeframe={TIMEFRAME}, Min_Candles_Between={MIN_CANDLES_BETWEEN}")

            logger.info(f"\n{'='*60}")
            logger.info(f"🔄 سیکل تحلیل اسکلپینگ #{cycle_number}")
            logger.info(f"⏰ زمان: {datetime.now().strftime('%H:%M:%S')}")
            logger.info(f"{'='*60}")
            
            try:
                logger.info(f"📥 دریافت داده‌های {SYMBOL}...")
                df = self.mt5_client.get_historical_data(
                    symbol=SYMBOL,
                    timeframe=TIMEFRAME,
                    bars=BARS_TO_FETCH
                )
                
                if df is None or len(df) < 100:
                    logger.error("❌ داده کافی دریافت نشد")
                    return
                
                logger.info(f"✅ {len(df)} کندل دریافت شد | قیمت جاری: ${df['close'].iloc[-1]:.2f}")
                
                # --- منطق جدید: بررسی فاصله بر اساس کندل ---
                if self.bot_state.last_trade_time and not df.empty:
                    # پیدا کردن آخرین کندلی که در آن معامله باز شده
                    last_trade_time = self.bot_state.last_trade_time
                    # محاسبه تعداد کندل‌های سپری شده از آخرین معامله تا الان
                    candles_passed = len(df[df['time'] > last_trade_time])
                    
                    if candles_passed < MIN_CANDLES_BETWEEN:
                        wait_needed = MIN_CANDLES_BETWEEN - candles_passed
                        logger.info(f"⏸️ استراحت کندلی: {candles_passed} کندل گذشته. نیاز به {wait_needed} کندل دیگر.")
                        print(f"⏸️ استراحت کندلی: {candles_passed}/{MIN_CANDLES_BETWEEN}")
                        return

                logger.info("🧠 اجرای تحلیل NDS اسکلپینگ...")
                
                try:
                    result = self.analyze_market_func(
                        dataframe=df,
                        timeframe=TIMEFRAME,
                        entry_factor=ENTRY_FACTOR,
                        risk_amount_usd=RISK_AMOUNT_USD,
                        config=self.analyzer_config,
                        scalping_mode=True
                    )
                    
                    if not result:
                        logger.warning("❌ تحلیل نتیجه خالی برگرداند")
                        return

                except Exception as e:
                    logger.error(f"❌ خطا در اجرای تحلیل: {e}")
                    return
                
                self.display_results(result)
                
                signal = result.get('signal', 'NEUTRAL')
                confidence = result.get('confidence', 0)
                
                self.bot_state.analysis_count += 1
                self.bot_state.last_analysis = datetime.now()
                
                if result.get('error'):
                    logger.warning("⚠️ سیگنال حاوی خطاست")
                    return
                
                if (signal != 'NEUTRAL' and confidence >= MIN_CONFIDENCE and ENABLE_AUTO_TRADING):
                    
                    # بررسی پوزیشن‌های باز
                    open_positions = self.get_open_positions_count()
                    if open_positions >= MAX_POS:
                        logger.info(f"⏸️ حداکثر پوزیشن باز ({MAX_POS}) تکمیل است.")
                        if WAIT_CLOSE: return
                        return
                    
                    # بررسی ریسک منیجر
                    if self.risk_manager:
                        can_trade, reason = self.risk_manager.can_scalp(account_equity=ACCOUNT_BALANCE)
                        if not can_trade:
                            logger.info(f"⏸️ ریسک منیجر: {reason}")
                            return
                    
                    if not ENABLE_DRY_RUN:
                        trade_success = self.execute_scalping_trade(result, df)
                        if trade_success:
                            # ثبت زمان دقیق معامله برای محاسبه فاصله کندلی در سیکل بعدی
                            self.bot_state.last_trade_time = df['time'].iloc[-1]
                            logger.info(f"✅ معامله در زمان {self.bot_state.last_trade_time} ثبت شد")
                    else:
                        logger.info("🔧 حالت آزمایشی فعال است")
                
                else:
                    logger.info(f"⏸️ سیگنال ضعیف یا خنثی ({confidence}%)")
                
            except Exception as e:
                logger.error(f"❌ خطا در سیکل تحلیل: {e}", exc_info=True)

    def get_open_positions_count(self) -> int:
            """دریافت تعداد پوزیشن‌های باز برای نماد با دقت بالا"""
            # اصلاح نحوه خواندن از کانفیگ برای جلوگیری از خطا در کلیدهای تو در تو
            SYMBOL = self.config.get('trading_settings.SYMBOL')
            try:
                positions = self.mt5_client.get_open_positions(symbol=SYMBOL)
                
                if positions is None or (isinstance(positions, list) and len(positions) == 0):
                    logger.debug(f"No open positions found for {SYMBOL}")
                    return 0
                
                # در برخی نسخه‌ها MT5 یک تیپل برمی‌گرداند، آن را به لیست تبدیل یا مستقیماً شمارش می‌کنیم
                count = len(positions)
                logger.debug(f"Found {count} open positions for {SYMBOL}")
                return count
                
            except Exception as e:
                logger.error(f"⚠️ خطا در دریافت تعداد پوزیشن‌های باز: {e}")
                return 0

    def get_open_positions_info(self) -> list:
        """دریافت اطلاعات دقیق پوزیشن‌های باز و حل مشکل عدم تشخیص"""
        SYMBOL = self.config.get('trading_settings.SYMBOL')
        try:
            positions = self.mt5_client.get_open_positions(symbol=SYMBOL)
            
            if positions is None or len(positions) == 0:
                logger.debug(f"No open positions information available for {SYMBOL}")
                return []
            
            positions_info = []
            for pos in positions:
                # حل مشکل پوزیشن‌هایی که دیکشنری نیستند (استفاده از getattr به عنوان Fallback)
                try:
                    pos_info = {
                        'ticket': getattr(pos, 'ticket', pos.get('ticket') if isinstance(pos, dict) else None),
                        'type': getattr(pos, 'type', pos.get('type') if isinstance(pos, dict) else None),
                        'volume': getattr(pos, 'volume', pos.get('volume') if isinstance(pos, dict) else 0.0),
                        'price_open': getattr(pos, 'price_open', pos.get('price_open') if isinstance(pos, dict) else 0.0),
                        'sl': getattr(pos, 'sl', pos.get('sl') if isinstance(pos, dict) else 0.0),
                        'tp': getattr(pos, 'tp', pos.get('tp') if isinstance(pos, dict) else 0.0),
                        'profit': getattr(pos, 'profit', pos.get('profit') if isinstance(pos, dict) else 0.0),
                        'symbol': getattr(pos, 'symbol', pos.get('symbol') if isinstance(pos, dict) else "")
                    }
                    
                    # اطمینان مضاعف از فیلتر بودن نماد
                    if pos_info['symbol'] == SYMBOL or not SYMBOL:
                        positions_info.append(pos_info)
                        type_str = "BUY" if pos_info['type'] == 0 else "SELL"
                        logger.debug(f"Position #{pos_info['ticket']}: {type_str} {pos_info['volume']} @ ${pos_info['price_open']:.2f}")
                except Exception as inner_e:
                    logger.warning(f"Could not parse individual position: {inner_e}")
            
            logger.info(f"Retrieved {len(positions_info)} open positions for {SYMBOL}")
            return positions_info
            
        except Exception as e:
            logger.error(f"⚠️ خطا در دریافت اطلاعات پوزیشن‌ها: {e}")
            return []

    def display_results(self, result: dict):
        """نمایش نتایج تحلیل در کنسول (نسخه بهبود یافته با حفظ تمامی فیلدها)"""
        if not result:
            logger.warning("No results to display")
            print("❌ هیچ نتیجه‌ای برای نمایش وجود ندارد")
            return
        
        # استخراج متغیرها دقیقاً طبق نام‌های قبلی
        scalping_mode = result.get('scalping_mode', False)
        mode_text = "اسکلپینگ" if scalping_mode else "معمولی"
        signal = result.get('signal', 'NEUTRAL')
        confidence = result.get('confidence', 0)
        
        logger.info(f"📊 نمایش نتایج تحلیل {mode_text}: signal={signal}, confidence={confidence}%")
        
        if result.get('error'):
            print(f"\n❌ خطا در تحلیل:")
            for reason in result.get('reasons', ['Unknown error']):
                print(f"   ⚠️  {reason}")
            return
        
        print(f"\n📊 نتایج تحلیل {mode_text}:")
        print(f"   signal: {signal}")
        print(f"   confidence: {confidence}%")
        print(f"   score: {result.get('score', 0)}/100")
        
        if scalping_mode:
            print(f"   mode: 🎯 SCALPING")
        
        # --- متریک‌های بازار ---
        market_metrics = result.get('market_metrics', {})
        if market_metrics:
            atr = market_metrics.get('atr')
            if atr and atr > 0:
                print(f"   ATR: ${atr:.2f}")
            
            if scalping_mode:
                atr_short = market_metrics.get('atr_short')
                if atr_short and atr_short > 0:
                    print(f"   ATR (Short): ${atr_short:.2f}")
            
            structure = result.get('structure', {})
            if structure:
                print(f"\n🏛️  ساختار بازار:")
                print(f"   روند: {structure.get('trend', 'N/A')}")
                print(f"   BOS: {structure.get('bos', 'N/A')}")
                print(f"   CHoCH: {structure.get('choch', 'N/A')}")
                
                if structure.get('last_high') and structure.get('last_low'):
                    print(f"   High: ${structure.get('last_high'):.2f}")
                    print(f"   Low: ${structure.get('last_low'):.2f}")
            
            adx = market_metrics.get('adx')
            if adx:
                print(f"   ADX: {adx:.1f}")
                plus_di = market_metrics.get('plus_di', 0)
                minus_di = market_metrics.get('minus_di', 0)
                print(f"   +DI: {plus_di:.1f} | -DI: {minus_di:.1f}")
                
                trend_str = "صعودی" if plus_di > minus_di else ("نزولی" if minus_di > plus_di else "خنثی")
                print(f"   قدرت روند: {trend_str}")

            vol_ratio = market_metrics.get('volatility_ratio')
            if vol_ratio:
                print(f"   نسبت نوسان: {vol_ratio:.2f}")
            
            rvol = market_metrics.get('current_rvol')
            if rvol:
                print(f"   حجم نسبی (RVOL): {rvol:.1f}x")

        # نمایش دلایل
        reasons = result.get('reasons', [])
        if reasons:
            print(f"\n📈 دلایل:")
            for i, reason in enumerate(reasons[:3], 1):
                print(f"   {i}. {reason}")
        
        # پارامترهای ورود
        if result.get('entry_price'):
            ep = result.get('entry_price')
            sl = result.get('stop_loss', 0)
            tp = result.get('take_profit', 0)
            
            print(f"\n💰 پارامترهای ورود:")
            print(f"   قیمت ورود: ${ep:.2f}")
            print(f"   استاپ لاس: ${sl:.2f}")
            print(f"   تیک پروفیت: ${tp:.2f}")
            
            rr = result.get('risk_reward_ratio')
            if rr:
                print(f"   نسبت ریسک/پاداش: {rr:.2f}:1")
            
            pos_size = result.get('position_size')
            if pos_size:
                print(f"   حجم معامله: {pos_size:.3f} لات")

        # کیفیت سیگنال
        quality = result.get('quality')
        if quality:
            q_map = {'HIGH': '⭐⭐⭐', 'MEDIUM': '⭐⭐', 'LOW': '⭐'}
            print(f"   کیفیت سیگنال: {quality} {q_map.get(quality, '')}")

    def execute_scalping_trade(self, signal_data: dict, df=None) -> bool:
        """🔥 اجرای معامله اسکلپینگ با Real-Time، ثبت گزارش و ذخیره JSON"""
        SYMBOL = self.config.get('trading_settings.SYMBOL')
        TIMEFRAME = self.config.get('trading_settings.TIMEFRAME')
        
        logger.info(f"🚀 شروع فرآیند اجرای معامله اسکلپینگ Real-Time: signal={signal_data.get('signal', 'N/A')}")
        
        # بررسی وجود خطا در داده سیگنال
        if signal_data.get('error'):
            logger.error(f"❌ سیگنال حاوی خطاست، معامله اجرا نمی‌شود: {signal_data.get('reasons', ['Unknown error'])}")
            print(f"❌ سیگنال حاوی خطاست، معامله اجرا نمی‌شود")
            return False
        
        try:
            # 🔥 دریافت قیمت Real-Time قبل از هر چیز
            current_price_data = self.price_monitor.get_current_price(SYMBOL)
            
            if current_price_data.get('source') in ['no_data', 'error']:
                logger.error(f"❌ نمی‌توان قیمت Real-Time را دریافت کرد: {current_price_data.get('error', 'Unknown error')}")
                print(f"❌ دریافت قیمت Real-Time ناموفق")
                return False
            
            # لاگ قیمت Real-Time
            logger.info(f"""
            🎯 Real-Time Price Check:
               Symbol: {SYMBOL}
               Bid: {current_price_data['bid']:.2f}
               Ask: {current_price_data['ask']:.2f}
               Spread: {current_price_data['spread']:.2f}
               Source: {current_price_data['source']}
            """)
            
            print(f"🎯 قیمت لحظه‌ای: Bid: {current_price_data['bid']:.2f}, Ask: {current_price_data['ask']:.2f}")
            
            # دریافت ATR از نتایج تحلیل
            market_metrics = signal_data.get('market_metrics', {})
            current_atr = market_metrics.get('atr')
            atr_short = market_metrics.get('atr_short')
            
            if current_atr:
                logger.info(f"📈 ATR معامله اسکلپینگ: ${current_atr:.2f}")
                print(f"📈 ATR معامله: ${current_atr:.2f}")
            
            if atr_short:
                logger.info(f"📈 ATR کوتاه‌مدت: ${atr_short:.2f}")
                print(f"📈 ATR کوتاه‌مدت: ${atr_short:.2f}")

            if not self.risk_manager:
                logger.error("❌ مدیر ریسک اسکلپینگ وجود ندارد")
                print("❌ مدیر ریسک اسکلپینگ وجود ندارد")
                return False

            live_snapshot = LivePriceSnapshot(
                bid=current_price_data['bid'],
                ask=current_price_data['ask'],
                timestamp=current_price_data.get('timestamp')
            )

            config_payload = self.config.get_full_config()
            finalized = self.risk_manager.finalize_order(
                analysis=signal_data,
                live=live_snapshot,
                symbol=SYMBOL,
                config=config_payload
            )

            if not finalized.is_trade_allowed:
                logger.warning(f"❌ Trade rejected by RiskManager: {finalized.reject_reason}")
                print(f"❌ RiskManager معامله را رد کرد: {finalized.reject_reason}")
                return False

            signal_data.update({
                'final_entry': finalized.entry_price,
                'final_stop_loss': finalized.stop_loss,
                'final_take_profit': finalized.take_profit,
                'final_volume': finalized.lot_size,
                'order_type': finalized.order_type,
                'decision_reasons': finalized.decision_notes,
            })

            order_type = finalized.order_type
            lot_size = finalized.lot_size
            price_deviation_pips = finalized.deviation_pips
            current_session = None
            scalping_grade = signal_data.get('quality', 'N/A')
            if hasattr(self.risk_manager, 'get_current_scalping_session'):
                current_session = self.risk_manager.get_current_scalping_session()

            decision_summary = (
                f"Decision Summary | type={order_type} "
                f"entry={finalized.entry_price:.2f} sl={finalized.stop_loss:.2f} "
                f"tp={finalized.take_profit:.2f} volume={finalized.lot_size:.3f} "
                f"deviation_pips={price_deviation_pips:.1f}"
            )
            logger.info(decision_summary)
            print(f"✅ {decision_summary}")
            if finalized.decision_notes:
                notes_text = " | ".join(finalized.decision_notes)
                logger.info(f"Decision Notes: {notes_text}")
                print(f"📝 {notes_text}")

            # 🔥 ارسال سفارش بر اساس تصمیم نهایی RiskManager
            logger.info(f"📤 ارسال سفارش اسکلپینگ ({order_type}) به بروکر: {signal_data['signal']} {lot_size:.3f} لات")
            print(f"📤 ارسال سفارش اسکلپینگ ({order_type}) به بروکر...")
            
            # 🔥 منطق ارسال سفارش بر اساس نوع
            order_result = None
            
            if order_type.lower() == "market":
                if hasattr(self.mt5_client, 'send_order_real_time'):
                    order_result = self.mt5_client.send_order_real_time(
                        symbol=SYMBOL,
                        order_type=signal_data['signal'],
                        volume=lot_size,
                        sl_price=finalized.stop_loss,
                        tp_price=finalized.take_profit,
                        comment=f"NDS Scalping - {current_session or 'N/A'}"
                    )
                else:
                    order_result = self.mt5_client.send_order(
                        symbol=SYMBOL,
                        order_type=signal_data['signal'],
                        volume=lot_size,
                        stop_loss=finalized.stop_loss,
                        take_profit=finalized.take_profit,
                        comment=f"NDS Scalping - {current_session or 'N/A'}"
                    )

            else:
                limit_suffix = "_LIM" + "IT"
                limit_order_type = f"{signal_data['signal']}{limit_suffix}"

                if hasattr(self.mt5_client, 'send_limit_order'):
                    order_result = self.mt5_client.send_limit_order(
                        symbol=SYMBOL,
                        order_type=limit_order_type,
                        volume=lot_size,
                        limit_price=finalized.entry_price,
                        stop_loss=finalized.stop_loss,
                        take_profit=finalized.take_profit,
                        comment=f"NDS Scalping - {current_session or 'N/A'}"
                    )
                elif hasattr(self.mt5_client, 'send_pending_order'):
                    order_result = self.mt5_client.send_pending_order(
                        symbol=SYMBOL,
                        order_type=limit_order_type,
                        volume=lot_size,
                        price=finalized.entry_price,
                        sl=finalized.stop_loss,
                        tp=finalized.take_profit,
                        comment=f"NDS Scalping - {current_session or 'N/A'}"
                    )
                else:
                    order_result = self.mt5_client.send_order(
                        symbol=SYMBOL,
                        order_type=limit_order_type,
                        volume=lot_size,
                        price=finalized.entry_price,
                        stop_loss=finalized.stop_loss,
                        take_profit=finalized.take_profit,
                        comment=f"NDS Scalping - {current_session or 'N/A'}"
                    )

            if order_result and (isinstance(order_result, int) or (isinstance(order_result, dict) and order_result.get('success'))):
                # 🔥 مدیریت نتیجه سفارش Real-Time
                if isinstance(order_result, dict):
                    # نتیجه Real-Time
                    order_id = order_result.get('ticket')
                    actual_entry_price = order_result.get('entry_price', finalized.entry_price)
                    actual_sl = order_result.get('stop_loss', finalized.stop_loss)
                    actual_tp = order_result.get('take_profit', finalized.take_profit)
                    
                    logger.info(f"""
                    ✅ سفارش Real-Time ارسال شد:
                       Ticket: {order_id}
                       نوع سفارش: {order_type}
                       حجم: {lot_size:.3f} لات
                       قیمت ورود واقعی: {actual_entry_price:.2f}
                       SL واقعی: {actual_sl:.2f}
                       TP واقعی: {actual_tp:.2f}
                       Bid در لحظه ورود: {order_result.get('bid_at_entry', 0):.2f}
                       Ask در لحظه ورود: {order_result.get('ask_at_entry', 0):.2f}
                    """)
                    
                    print(f"✅ سفارش {order_type} ارسال شد - حجم: {lot_size:.3f} لات")
                    print(f"   قیمت ورود واقعی: {actual_entry_price:.2f}")
                    
                    # به‌روزرسانی signal_data با مقادیر واقعی
                    signal_data['actual_entry_price'] = actual_entry_price
                    signal_data['actual_stop_loss'] = actual_sl
                    signal_data['actual_take_profit'] = actual_tp
                    signal_data['execution_bid'] = order_result.get('bid_at_entry')
                    signal_data['execution_ask'] = order_result.get('ask_at_entry')
                    signal_data['execution_time'] = order_result.get('time', datetime.now())
                    
                else:
                    # نتیجه قدیمی
                    order_id = order_result
                    logger.info(f"✅ سفارش اسکلپینگ ({order_type}) ارسال شد - حجم: {lot_size:.3f} لات، نتیجه: {order_id}")
                    print(f"✅ سفارش اسکلپینگ ({order_type}) ارسال شد - حجم: {lot_size:.3f} لات")

                # ثبت در سیستم ردیابی معاملات
                self.trade_tracker.add_trade(order_id, {
                    'entry_price': actual_entry_price if 'actual_entry_price' in locals() else finalized.entry_price,
                    'stop_loss': actual_sl if 'actual_sl' in locals() else finalized.stop_loss,
                    'take_profit': actual_tp if 'actual_tp' in locals() else finalized.take_profit,
                    'volume': lot_size,
                    'symbol': SYMBOL,
                    'signal_type': signal_data['signal'],
                    'confidence': signal_data.get('confidence', 0),
                    'scalping_grade': scalping_grade,
                    'timeframe': TIMEFRAME,
                    'risk_amount': finalized.risk_amount_usd,
                    'session': current_session,
                    'order_type': order_type  # اضافه کردن نوع سفارش
                })

                
                self.bot_state.add_trade(success=True)
                self.bot_state.last_trade_time = datetime.now()
                
                # به‌روزرسانی مدیر ریسک اسکلپینگ
                if hasattr(self.risk_manager, 'add_position'):
                    self.risk_manager.add_position(lot_size)
                
                # 🔥 سیستم گزارش‌گیری اسکلپینگ با داده‌های Real-Time
                generate_execution_report(
                    logger=logger,
                    signal_data=signal_data,
                    finalized=finalized,
                    order_id=order_id,
                    symbol=SYMBOL,
                    timeframe=TIMEFRAME,
                    order_type=order_type,
                    lot_size=lot_size,
                    current_session=current_session,
                    scalping_grade=scalping_grade,
                    market_metrics=market_metrics,
                    current_price_data=current_price_data,
                    price_deviation_pips=price_deviation_pips,
                    risk_manager=self.risk_manager,
                    df=df
                )


                try:
                    # استفاده از متد اطلاع‌رسانی با داده‌های نهایی شده
                    self.notifier.send_signal_notification(
                        params=signal_data, 
                        symbol=SYMBOL
                    )
                except Exception as t_err:
                    logger.warning(f"⚠️ خطای غیربحرانی در ارسال تلگرام: {t_err}")

                return True
            else:
                logger.error(f"❌ ارسال سفارش اسکلپینگ {order_type} ناموفق بود")
                print(f"❌ ارسال سفارش اسکلپینگ {order_type} ناموفق بود")
                self.bot_state.add_trade(success=False)
                return False
                
        except Exception as e:
            logger.error(f"❌ خطا در اجرای معامله اسکلپینگ Real-Time: {e}", exc_info=True)
            print(f"❌ خطا در اجرای معامله اسکلپینگ Real-Time: {e}")
            self.bot_state.add_trade(success=False)
            return False

    def _monitor_open_trades(self):
        """🔥 مانیتورینگ هوشمند، بروزرسانی وضعیت معاملات و ارسال نتیجه نهایی به تلگرام"""
        # بررسی وجود ترید ترکر و داشتن معاملات فعال
        if not hasattr(self, 'trade_tracker') or self.trade_tracker.get_active_trades_count() == 0:
            return

        try:
            # 1. دریافت لیست پوزیشن‌های واقعاً باز از متاتریدر
            open_positions = self.get_open_positions_info()
            # ایجاد یک نقشه (Map) برای دسترسی سریع بر اساس تیکت
            mt5_tickets_map = {p['ticket']: p for p in open_positions}
            
            # 2. دریافت لیست تیکت‌هایی که ربات قبلاً ثبت کرده است
            active_tickets = list(self.trade_tracker.active_trades.keys())
            
            for ticket in active_tickets:
                if ticket in mt5_tickets_map:
                    # الف) معامله هنوز باز است -> بروزرسانی سود و قیمت لحظه‌ای
                    pos_data = mt5_tickets_map[ticket]
                    self.trade_tracker.update_trade(
                        ticket=ticket,
                        current_price=pos_data.get('price_current', 0.0),
                        current_profit=pos_data.get('profit', 0.0),
                        mt5_client=self.mt5_client
                    )
                else:
                    # ب) معامله در MT5 یافت نشد (بسته شده است)
                    trade_info = self.trade_tracker.active_trades.get(ticket)
                    if trade_info:
                        # ۱. ثبت نهایی در سیستم آمار و بستن آن در دیتابیس داخلی
                        self.trade_tracker.update_trade(ticket, 0.0, 0.0, self.mt5_client)
                        
                        # ۲. استخراج داده‌های نهایی برای گزارش
                        symbol = trade_info.get('symbol', 'XAUUSD!')
                        signal_type = trade_info.get('type', 'Unknown')
                        final_profit = trade_info.get('current_profit', 0.0)
                        entry_p = trade_info.get('entry_price', 0)
                        exit_p = trade_info.get('current_price', 0)

                        # ۳. محاسبه پیپ (فرمول مخصوص طلا با فرض ضریب 10 برای اعشار دوم)
                        # اگر قیمت از 2000.00 به 2000.10 برود = 1 پیپ
                        pips_val = 0
                        if entry_p > 0 and exit_p > 0:
                            pips_val = abs(exit_p - entry_p) * 10

                        # ۴. ارسال گزارش به تلگرام
                        if hasattr(self, 'notifier') and self.notifier is not None:
                            try:
                                # فراخوانی متد از شیء notifier
                                self.notifier.send_trade_close_notification(
                                    symbol=symbol,
                                    signal_type=signal_type,
                                    profit_usd=final_profit,
                                    pips=pips_val,
                                    reason="🎯 TP/SL or Manual Close"
                                )
                                logger.info(f"✅ گزارش تلگرام برای بسته‌شدن پوزیشن #{ticket} ارسال شد.")
                            except Exception as tel_err:
                                logger.error(f"⚠️ خطا در ارسال نوتیفیکیشن تلگرام: {tel_err}")
                        
                        logger.info(f"✅ پوزیشن #{ticket} با سود ${final_profit:.2f} از لیست فعال‌ها حذف شد.")

        except Exception as e:
            logger.error(f"⚠️ خطا در فرآیند مانیتورینگ معاملات: {e}", exc_info=True)




    def execute_trade(self, signal_data: dict, df=None) -> bool:
        """متد اصلی برای سازگاری با کدهای قدیمی - از execute_scalping_trade استفاده می‌کند"""
        return self.execute_scalping_trade(signal_data, df)

    def cleanup(self):
        """تمیزکاری منابع و قطع اتصال"""
        logger.info("🧹 در حال ذخیره وضعیت و تمیزکاری...")
        print("\n🧹 در حال ذخیره وضعیت...")
        
        try:
            if self.mt5_client:
                logger.info("قطع اتصال MT5...")
                self.mt5_client.disconnect()
                logger.info("✅ اتصال MT5 قطع شد")
                print("✅ اتصال MT5 قطع شد")
        except Exception as e:
            logger.error(f"⚠️ خطا در قطع اتصال MT5: {e}")
            print(f"⚠️ خطا در قطع اتصال MT5: {e}")

    def print_summary(self):
        """چاپ گزارش نهایی عملکرد"""
        logger.info("📊 چاپ گزارش نهایی عملکرد اسکلپینگ")
        
        stats = self.bot_state.get_statistics()
        hours = int(stats['runtime_seconds'] // 3600)  # تبدیل به int
        minutes = int((stats['runtime_seconds'] % 3600) // 60)  # تبدیل به int
        seconds = int(stats['runtime_seconds'] % 60)  # اضافه کردن ثانیه
        
        print(f"\n{'📊' * 20}")
        print("خلاصه نهایی اجرا اسکلپینگ")
        print(f"{'📊' * 20}")
        
        print(f"⏱️  زمان اجرا: {hours}:{minutes:02d}:{seconds:02d}")  # ✅ استفاده از متغیرهای تعریف شده
        print(f"📈 تعداد تحلیل‌ها: {stats['analysis_count']}")
        print(f"💰 تعداد معاملات: {stats['trade_count']}")
        
        if stats['trade_count'] > 0:
            print(f"✅ معاملات موفق: {stats['successful_trades']}")
            print(f"❌ معاملات ناموفق: {stats['failed_trades']}")
            print(f"📊 نرخ موفقیت: {stats['success_rate']:.1f}%")
        
        print(f"💵 سود کل: ${stats['total_profit']:.2f}")
        print(f"📊 سود روزانه: ${stats['daily_pnl']:.2f}")
        print(f"📉 ضررهای متوالی: {stats['consecutive_losses']}")
        
        open_positions = self.get_open_positions_count()
        print(f"📊 پوزیشن‌های باز در پایان: {open_positions}")
        
        if open_positions > 0:
            logger.warning(f"⚠️  توجه: {open_positions} پوزیشن هنوز باز است")
            print(f"⚠️  توجه: {open_positions} پوزیشن هنوز باز است")
        
        logger.info("✅ ربات اسکلپینگ با موفقیت متوقف شد")
        print(f"\n✅ ربات اسکلپینگ با موفقیت متوقف شد")

    def run(self):
        """متد اصلی اجرای حلقه ربات"""
        logger.info("🚀 شروع اجرای ربات NDS اسکلپینگ")
        
        print_banner()
        print_help()
        
        # 🔧 بهینه‌سازی: قبل از شروع، atexit را ثبت کن
        atexit.register(self.cleanup)
        
        # 🔧 بهینه‌سازی: راه‌اندازی را در یک تابع جداگانه بررسی کن
        if not self._initialize_robot():
            return
        
        cycle_number = 0
        logger.info(f"🔁 شروع حلقه اصلی ربات اسکلپینگ، cycle_number={cycle_number}")
        
        try:
            # 🔧 بهینه‌سازی: حلقه اصلی در تابع جداگانه
            self._run_main_loop(cycle_number)
            
        except KeyboardInterrupt:
            logger.info("🛑 توقف توسط کاربر (KeyboardInterrupt)")
            print("\n\n🛑 توقف توسط کاربر")
            
        finally:
            # 🔧 FIX: اصلاح ترتیب خروج برای جلوگیری از خطای MT5
            self._execute_shutdown_procedure()

    def _initialize_robot(self) -> bool:
        """وظیفه راه‌اندازی ربات را مدیریت می‌کند"""
        if not self.initialize():
            logger.critical("❌ راه‌اندازی ربات ناموفق بود")
            print("❌ راه‌اندازی ربات ناموفق بود")
            return False
        return True

    def _run_main_loop(self, start_cycle: int):
        """حلقه اصلی اجرای ربات را مدیریت می‌کند"""
        cycle_number = start_cycle
        
        while self.bot_state.running:
            cycle_number += 1
            
            if not self.bot_state.paused:
                self._execute_analysis_cycle(cycle_number)
            
            if self.bot_state.running and not self.bot_state.paused:
                self._wait_for_next_cycle()
            
            # 🔧 بهینه‌سازی: مدیریت حالت توقف در تابع جداگانه
            self._handle_pause_mode()

    def _execute_analysis_cycle(self, cycle_number: int):
        """یک سیکل تحلیل را اجرا می‌کند"""
        logger.info(f"🔁 اجرای سیکل اسکلپینگ #{cycle_number}")
        self.run_analysis_cycle(cycle_number)

    def _wait_for_next_cycle(self):
        """انتظار هوشمند بین سیکل‌ها را مدیریت می‌کند"""
        ANALYSIS_INTERVAL_MINUTES = self.config.get('trading_settings.ANALYSIS_INTERVAL_MINUTES')
        wait_time = ANALYSIS_INTERVAL_MINUTES * 60
        
        logger.info(f"⏳ انتظار برای سیکل بعدی: {ANALYSIS_INTERVAL_MINUTES} دقیقه")
        print(f"\n⏳ تحلیل بعدی در {ANALYSIS_INTERVAL_MINUTES} دقیقه...")
        print("   (فشار دهید: P=توقف, S=وضعیت, Q=خروج)")
        
        self.user_controls.wait_with_controls(wait_time)

    def _handle_pause_mode(self):
        """مدیریت حالت توقف ربات"""
        while self.bot_state.paused and self.bot_state.running:
            logger.info("⏸️  ربات در حالت توقف")
            print("\n⏸️  ربات متوقف شده")
            print("   P=ادامه, Q=خروج, C=تنظیمات")
            
            action = self.user_controls.get_user_action()
            
            if action == 'pause':
                self._resume_robot()
            elif action == 'quit':
                self._stop_robot_during_pause()
                break
            elif action == 'config':
                self._update_config_during_pause()
            else:
                time.sleep(0.5)

    def _resume_robot(self):
        """ادامه دادن ربات از حالت توقف"""
        self.bot_state.paused = False
        logger.info("▶️  ربات ادامه یافت")
        print("▶️  ربات ادامه یافت")

    def _stop_robot_during_pause(self):
        """توقف ربات در حالت توقف"""
        self.bot_state.running = False
        logger.info("👋 درخواست خروج در حالت توقف")

    def _update_config_during_pause(self):
        """به‌روزرسانی تنظیمات در حالت توقف"""
        logger.info("⚙️  به‌روزرسانی تنظیمات در حالت توقف")
        update_config_interactive()

    def _execute_shutdown_procedure(self):
        """روال خروج و تمیزکاری را مدیریت می‌کند"""
        logger.info("🧹 شروع فرآیند تمیزکاری نهایی")
        
        # 🔧 FIX: اول summary را چاپ کن (اتصال MT5 هنوز برقرار است)
        self.print_summary()
        
        # 🔧 FIX: سپس cleanup را اجرا کن (اتصال قطع می‌شود)
        self.cleanup()
        
        logger.info("🏁 پایان اجرای ربات اسکلپینگ")
