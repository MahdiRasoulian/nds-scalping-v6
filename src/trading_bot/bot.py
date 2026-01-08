"""
ربات اصلی معاملات NDS برای طلا - نسخه اسکلپینگ
نسخه یکپارچه با risk_manager.py
نسخه بهبود یافته با:
- سازگاری کامل با mt5_client.py (Real-Time + positions/pending)
- رفع مشکل عدم تشخیص بسته شدن پوزیشن (مانیتورینگ پیوسته + تشخیص pending vs position)
- یکپارچه‌سازی قرارداد خروجی Analyzer (AnalysisResult/dataclass -> dict)
- بهبود گزارش‌گیری lifecycle (OPEN/UPDATE/CLOSE) + تلگرام
- اصلاح ناسازگاری NONE/NEUTRAL و جلوگیری از ترید روی سیگنال خنثی
"""

import sys
import time
import atexit
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List

# پیدا کردن مسیر اصلی پروژه (nds_bot)
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent  # nds_bot
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# اضافه کردن پوشه src به مسیرها
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

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
from src.trading_bot.contracts import ExecutionEvent, PositionContract, compute_pips
from src.trading_bot.nds.models import LivePriceSnapshot
from src.trading_bot.realtime_price import RealTimePriceMonitor
from src.trading_bot.trade_tracker import TradeTracker
from src.trading_bot.user_controls import UserControls
from src.ui.cli import print_banner, print_help, update_config_interactive

# ایمپورت آنالایزر جدید به صورت ماژولار
try:
    from src.trading_bot.nds.analyzer import analyze_gold_market
    from src.trading_bot.nds.analyzer import GoldNDSAnalyzer
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
        bot_state_global = self.bot_state

        # DI
        self.MT5Client_cls = mt5_client_cls
        self.RiskManager_cls = risk_manager_cls

        self.analyze_market_func = analyze_func or analyze_gold_market

        self.mt5_client = None
        self.risk_manager = None
        self.config = config
        self.analyzer_config = None

        self.price_monitor = RealTimePriceMonitor(config=self.config, bot_state=self.bot_state, logger=logger)
        self.trade_tracker = TradeTracker()
        self.user_controls = UserControls(self, logger)

        self.notifier = TelegramNotifier()

        # مانیتورینگ معامله
        self._last_trade_monitor_ts = 0.0
        self._trade_monitor_interval_sec = 2.0  # هر 2 ثانیه بررسی تریدها (قابل تغییر)

    # ----------------------------
    # Helpers
    # ----------------------------
    def _result_to_dict(self, result: Any) -> Dict[str, Any]:
        """
        سازگارکننده خروجی آنالایزر:
        - اگر dict باشد همان را می‌دهد
        - اگر AnalysisResult/dataclass باشد به dict تبدیل می‌کند
        - keyهای context را برای display_results و trade حفظ می‌کند
        """
        if result is None:
            return {}

        if isinstance(result, dict):
            return result

        # dataclass / pydantic-like
        if hasattr(result, "__dict__"):
            d = dict(result.__dict__)
            ctx = d.get("context")
            if isinstance(ctx, dict):
                # merge برخی کلیدهای مورد انتظار bot.py
                for k in (
                    "market_metrics",
                    "structure",
                    "analysis_data",
                    "session_analysis",
                    "scalping_mode",
                    "reasons",
                    "entry_price",
                    "stop_loss",
                    "take_profit",
                    "position_size",
                    "risk_reward_ratio",
                    "quality",
                    "score",
                ):
                    if k not in d and k in ctx:
                        d[k] = ctx[k]
            return d

        return {}

    def _normalize_signal(self, signal_value: str) -> str:
        """
        استانداردسازی سیگنال:
        Analyzer: BUY/SELL/NONE
        برخی نسخه‌ها: NEUTRAL
        """
        sig = (signal_value or "NONE").upper()
        if sig == "NEUTRAL":
            sig = "NONE"
        if sig not in ("BUY", "SELL", "NONE"):
            # هر چیزی غیر از BUY/SELL را خنثی در نظر بگیر
            sig = "NONE"
        return sig

    def _maybe_monitor_trades(self, force: bool = False):
        """مانیتورینگ معاملات با throttle برای جلوگیری از فشار"""
        now = time.time()
        if force or (now - self._last_trade_monitor_ts) >= self._trade_monitor_interval_sec:
            self._last_trade_monitor_ts = now
            self._monitor_open_trades()

    # ----------------------------
    # Initialize
    # ----------------------------
    def initialize(self) -> bool:
        """🔥 مقداردهی اولیه ربات و اتصال به سرویس‌ها (نسخه Real-Time حرفه‌ای)"""
        logger.info("🔧 در حال راه‌اندازی ربات اسکلپینگ Real-Time...")
        print("\n🔧 در حال راه‌اندازی ربات اسکلپینگ Real-Time...")

        try:
            # 1) ایجاد MT5 Client
            if self.mt5_client is None:
                self.mt5_client = self.MT5Client_cls()

            # 2) اعمال credential های real-time در config متمرکز (در صورت وجود)
            credentials = self.config.get_mt5_credentials()
            tick_interval = self.config.get("trading_settings.TICK_UPDATE_INTERVAL")

            if credentials:
                credentials["real_time_enabled"] = True
                credentials["tick_update_interval"] = tick_interval
                self.config.save_mt5_credentials(credentials)
                logger.info(f"✅ تنظیمات Real-Time (Interval: {tick_interval}s) به کانفیگ MT5 اعمال شد")

            # 3) مدیریت ورود/اتصال
            if not credentials or not all(k in credentials for k in ["login", "password", "server"]):
                logger.warning("❌ اطلاعات حساب MT5 ناقص است.")
                print("❌ اطلاعات حساب MT5 ناقص است. لطفاً در config/bot_config.json تکمیل کنید.")
                return False

            # این فیلدها در MT5Client شما داخل ConnectionConfig استفاده می‌شود،
            # اما نگه می‌داریم چون شاید در کلاس شما استفاده می‌شود.
            self.mt5_client.login = int(credentials["login"])
            self.mt5_client.password = credentials["password"]
            self.mt5_client.server = credentials["server"]

            if not self.mt5_client.connect():
                logger.error("❌ اتصال به MT5 ناموفق بود.")
                return False

            # 4) آپدیت موجودی
            account_info = self.mt5_client.get_account_info()
            if account_info:
                current_equity = account_info.get("equity") or account_info.get("balance") or 0.0
                self.config.update_setting("ACCOUNT_BALANCE", current_equity)
                logger.info(f"💰 حساب متصل شد | موجودی لحظه‌ای: ${current_equity:,.2f}")

            # 5) شروع مانیتورینگ قیمت (سیستم داخلی پروژه)
            self.price_monitor.set_mt5_client(self.mt5_client)
            self.price_monitor.start()

            # 6) آماده‌سازی آنالایزر
            logger.info("🧠 در حال هماهنگ‌سازی تنظیمات آنالایزر با استراتژی SMC...")
            self.analyzer_config = self.config.get_full_config_for_analyzer()

            if "ANALYZER_SETTINGS" not in self.analyzer_config:
                self.analyzer_config["ANALYZER_SETTINGS"] = self.config.get("technical_settings")

            tech_settings = self.analyzer_config.get("ANALYZER_SETTINGS", {})
            analyzer_settings = {
                **tech_settings,
                "ADX_THRESHOLD_WEAK": self.config.get("technical_settings.ADX_THRESHOLD_WEAK"),
                "REAL_TIME_ENABLED": True,
                "USE_CURRENT_PRICE_FOR_ANALYSIS": True,
            }
            self.analyzer_config = {**self.analyzer_config, "ANALYZER_SETTINGS": analyzer_settings}

            # 7) ایجاد Risk Manager
            scalping_config = {
                "risk_manager_config": self.config.get_risk_manager_config(),
                "trading_rules": {
                    "MIN_CANDLES_BETWEEN": self.config.get("trading_rules.MIN_CANDLES_BETWEEN"),
                },
                "risk_settings": {
                    "MAX_PRICE_DEVIATION_PIPS": self.config.get("risk_settings.MAX_PRICE_DEVIATION_PIPS"),
                },
            }
            self.risk_manager = create_scalping_risk_manager(overrides=scalping_config)

            logger.info("✅ ربات با موفقیت عملیاتی شد.")
            self._log_real_time_status()

            # بازیابی/مانیتور اولیه (فقط برای همگام‌سازی)
            logger.info("🔄 همگام‌سازی اولیه وضعیت معاملات با MT5...")
            self._maybe_monitor_trades(force=True)

            return True

        except Exception as e:
            logger.critical(f"❌ خطای بحرانی در Initialize: {e}", exc_info=True)
            return False

    def _log_real_time_status(self):
        """🔥 گزارش وضعیت واقعی و داینامیک سیستم"""
        try:
            symbol = self.config.get("trading_settings.SYMBOL")
            current_price = self.price_monitor.get_current_price(symbol)

            conn_status = "✅ Connected" if self.mt5_client and getattr(self.mt5_client, "connected", False) else "❌ Disconnected"
            monitor_status = "✅ Active" if getattr(self.mt5_client, "real_time_monitor", None) else "⚠️ Inactive"

            max_dev = self.config.get("risk_settings.MAX_PRICE_DEVIATION_PIPS")
            min_candles = self.config.get("trading_rules.MIN_CANDLES_BETWEEN")

            status_report = f"""
🎯 گزارش وضعیت لحظه‌ای سیستم (Real-Time)
==========================================
📊 وضعیت اتصال: {conn_status}
🎯 مانیتور قیمت MT5: {monitor_status}
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
            logger.error(f"❌ خطا در تولید گزارش وضعیت: {e}", exc_info=True)

    # ----------------------------
    # Main Cycle
    # ----------------------------
    def run_analysis_cycle(self, cycle_number: int):
        """اجرای یک سیکل کامل تحلیل بازار اسکلپینگ با فیلتر فاصله کندلی + مانیتورینگ ترید"""
        SYMBOL = self.config.get("trading_settings.SYMBOL")
        TIMEFRAME = self.config.get("trading_settings.TIMEFRAME")
        BARS_TO_FETCH = self.config.get("trading_settings.BARS_TO_FETCH")
        ENABLE_AUTO_TRADING = self.config.get("trading_settings.ENABLE_AUTO_TRADING")
        ENABLE_DRY_RUN = self.config.get("trading_settings.ENABLE_DRY_RUN")

        MIN_CANDLES_BETWEEN = self.config.get("trading_rules.MIN_CANDLES_BETWEEN")
        MAX_POS = self.config.get("trading_rules.MAX_POSITIONS")
        WAIT_CLOSE = self.config.get("trading_rules.WAIT_FOR_CLOSE_BEFORE_NEW_TRADE")

        ENTRY_FACTOR = self.config.get("technical_settings.ENTRY_FACTOR")
        MIN_CONFIDENCE = self.config.get("technical_settings.SCALPING_MIN_CONFIDENCE")

        ACCOUNT_BALANCE = self.config.get("ACCOUNT_BALANCE")

        logger.info(f"⚙️ تنظیمات نهایی بارگذاری شد: Timeframe={TIMEFRAME}, Min_Candles_Between={MIN_CANDLES_BETWEEN}")

        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 سیکل تحلیل اسکلپینگ #{cycle_number}")
        logger.info(f"⏰ زمان: {datetime.now().strftime('%H:%M:%S')}")
        logger.info(f"{'='*60}")

        try:
            # 0) مانیتورینگ تریدها (برای تشخیص بسته شدن/آپدیت سود)
            self._maybe_monitor_trades(force=True)

            logger.info(f"📥 دریافت داده‌های {SYMBOL}...")
            df = self.mt5_client.get_historical_data(symbol=SYMBOL, timeframe=TIMEFRAME, bars=BARS_TO_FETCH)

            if df is None or len(df) < 100:
                logger.error("❌ داده کافی دریافت نشد")
                return

            logger.info(f"✅ {len(df)} کندل دریافت شد | قیمت جاری: ${df['close'].iloc[-1]:.2f}")

            # --- استراحت کندلی (استاندارد: زمان کندل) ---
            if self.bot_state.last_trade_candle_time and not df.empty:
                last_trade_time = self.bot_state.last_trade_candle_time
                candles_passed = len(df[df["time"] > last_trade_time])
                if candles_passed < MIN_CANDLES_BETWEEN:
                    wait_needed = MIN_CANDLES_BETWEEN - candles_passed
                    logger.info(f"⏸️ استراحت کندلی: {candles_passed} کندل گذشته. نیاز به {wait_needed} کندل دیگر.")
                    print(f"⏸️ استراحت کندلی: {candles_passed}/{MIN_CANDLES_BETWEEN}")
                    # حتی در حالت استراحت هم مانیتور را نگه دار
                    self._maybe_monitor_trades()
                    return

            logger.info("🧠 اجرای تحلیل NDS اسکلپینگ...")

            try:
                # 🔥 FIX: risk_amount_usd از امضای analyze_gold_market حذف شد
                raw_result = self.analyze_market_func(
                    dataframe=df,
                    timeframe=TIMEFRAME,
                    entry_factor=ENTRY_FACTOR,
                    config=self.analyzer_config,
                    scalping_mode=True,
                )
                result = self._result_to_dict(raw_result)
                if not result:
                    logger.warning("❌ تحلیل نتیجه خالی برگرداند")
                    return
            except Exception as e:
                logger.error(f"❌ خطا در اجرای تحلیل: {e}", exc_info=True)
                return

            # نرمال‌سازی سیگنال
            result["signal"] = self._normalize_signal(result.get("signal", "NONE"))

            self.display_results(result)

            signal_value = result.get("signal", "NONE")
            confidence = float(result.get("confidence", 0) or 0)

            self.bot_state.analysis_count += 1
            self.bot_state.last_analysis = datetime.now()

            if result.get("error"):
                logger.warning("⚠️ سیگنال حاوی خطاست")
                return

            # فقط BUY/SELL اجازه ترید دارند
            if (signal_value in ("BUY", "SELL")) and (confidence >= MIN_CONFIDENCE) and ENABLE_AUTO_TRADING:
                # محدودیت تعداد پوزیشن
                open_positions = self.get_open_positions_count()
                if open_positions >= MAX_POS:
                    logger.info(f"⏸️ حداکثر پوزیشن باز ({MAX_POS}) تکمیل است.")
                    if WAIT_CLOSE:
                        return
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
                        # ✅ زمان معامله بر اساس زمان کندل (برای محاسبه candles_passed)
                        self.bot_state.last_trade_candle_time = df["time"].iloc[-1]
                        self.bot_state.last_trade_wall_time = datetime.now()
                        self.bot_state.last_trade_time = self.bot_state.last_trade_wall_time
                        logger.info(f"✅ معامله در زمان کندل {self.bot_state.last_trade_candle_time} ثبت شد")
                        # مانیتورینگ فوری بعد از ارسال سفارش
                        self._maybe_monitor_trades(force=True)
                else:
                    logger.info("🔧 حالت آزمایشی فعال است (Dry Run)")
            else:
                logger.info(f"⏸️ سیگنال خنثی/ضعیف | signal={signal_value} confidence={confidence}%")

            # در پایان هر سیکل، مجدداً مانیتور کنیم تا closeها از دست نرود
            self._maybe_monitor_trades(force=True)

        except Exception as e:
            logger.error(f"❌ خطا در سیکل تحلیل: {e}", exc_info=True)

    # ----------------------------
    # Positions/Pending (MT5)
    # ----------------------------
    def get_open_positions_count(self) -> int:
        """دریافت تعداد پوزیشن‌های باز برای نماد با سازگاری با MT5Client"""
        SYMBOL = self.config.get("trading_settings.SYMBOL")
        try:
            positions = self.mt5_client.get_open_positions(symbol=SYMBOL)
            if not positions:
                logger.debug(f"No open positions found for {SYMBOL}")
                return 0
            count = len(positions)
            logger.debug(f"Found {count} open positions for {SYMBOL}")
            return count
        except Exception as e:
            logger.error(f"⚠️ خطا در دریافت تعداد پوزیشن‌های باز: {e}", exc_info=True)
            return 0

    def get_open_positions_info(self) -> List[PositionContract]:
        """
        دریافت اطلاعات دقیق پوزیشن‌های باز
        سازگار با mt5_client.get_open_positions که لیست dict برمی‌گرداند
        """
        SYMBOL = self.config.get("trading_settings.SYMBOL")
        try:
            positions: List[PositionContract] = self.mt5_client.get_open_positions(symbol=SYMBOL)
            if not positions:
                logger.debug(f"No open positions information available for {SYMBOL}")
                return []

            for pos in positions:
                logger.debug(
                    "Position #%s: %s %.3f @ $%.2f | cur=$%.2f | pnl=$%.2f",
                    pos["position_ticket"],
                    pos["side"],
                    pos["volume"],
                    pos["entry_price"],
                    pos["current_price"],
                    pos["profit"],
                )

            logger.info(f"Retrieved {len(positions)} open positions for {SYMBOL}")
            return positions

        except Exception as e:
            logger.error(f"⚠️ خطا در دریافت اطلاعات پوزیشن‌ها: {e}", exc_info=True)
            return []

    def get_pending_orders_info(self) -> List[Dict[str, Any]]:
        """دریافت سفارش‌های pending برای جلوگیری از false-close در tracker"""
        SYMBOL = self.config.get("trading_settings.SYMBOL")
        try:
            if hasattr(self.mt5_client, "get_pending_orders"):
                orders = self.mt5_client.get_pending_orders(symbol=SYMBOL)
                return orders or []
            return []
        except Exception as e:
            logger.error(f"⚠️ خطا در دریافت pending orders: {e}", exc_info=True)
            return []

    # ----------------------------
    # Display
    # ----------------------------
    def display_results(self, result: dict):
        """نمایش نتایج تحلیل در کنسول (نسخه بهبود یافته با حفظ تمامی فیلدها)"""
        if not result:
            logger.warning("No results to display")
            print("❌ هیچ نتیجه‌ای برای نمایش وجود ندارد")
            return

        scalping_mode = bool(result.get("scalping_mode", False))
        mode_text = "اسکلپینگ" if scalping_mode else "معمولی"
        signal_value = result.get("signal", "NONE")
        confidence = result.get("confidence", 0)

        logger.info(f"📊 نمایش نتایج تحلیل {mode_text}: signal={signal_value}, confidence={confidence}%")

        if result.get("error"):
            print(f"\n❌ خطا در تحلیل:")
            for reason in result.get("reasons", ["Unknown error"]):
                print(f"   ⚠️  {reason}")
            return

        print(f"\n📊 نتایج تحلیل {mode_text}:")
        print(f"   signal: {signal_value}")
        print(f"   confidence: {confidence}%")
        print(f"   score: {result.get('score', 0)}/100")

        if scalping_mode:
            print(f"   mode: 🎯 SCALPING")

        market_metrics = result.get("market_metrics", {}) or {}
        if market_metrics:
            atr = market_metrics.get("atr")
            if atr and atr > 0:
                print(f"   ATR: ${atr:.2f}")

            if scalping_mode:
                atr_short = market_metrics.get("atr_short")
                if atr_short and atr_short > 0:
                    print(f"   ATR (Short): ${atr_short:.2f}")

            structure = result.get("structure", {}) or {}
            if structure:
                print(f"\n🏛️  ساختار بازار:")
                print(f"   روند: {structure.get('trend', 'N/A')}")
                print(f"   BOS: {structure.get('bos', 'N/A')}")
                print(f"   CHoCH: {structure.get('choch', 'N/A')}")

                if structure.get("last_high") and structure.get("last_low"):
                    print(f"   High: ${structure.get('last_high'):.2f}")
                    print(f"   Low: ${structure.get('last_low'):.2f}")

            adx = market_metrics.get("adx")
            if adx is not None:
                try:
                    adx_val = float(adx)
                    print(f"   ADX: {adx_val:.1f}")
                except Exception:
                    pass

                plus_di = market_metrics.get("plus_di", 0)
                minus_di = market_metrics.get("minus_di", 0)
                try:
                    print(f"   +DI: {float(plus_di):.1f} | -DI: {float(minus_di):.1f}")
                    trend_str = "صعودی" if plus_di > minus_di else ("نزولی" if minus_di > plus_di else "خنثی")
                    print(f"   قدرت روند: {trend_str}")
                except Exception:
                    pass

            vol_ratio = market_metrics.get("volatility_ratio")
            if vol_ratio:
                print(f"   نسبت نوسان: {vol_ratio:.2f}")

            rvol = market_metrics.get("current_rvol")
            if rvol:
                print(f"   حجم نسبی (RVOL): {rvol:.1f}x")

        reasons = result.get("reasons", []) or []
        if reasons:
            print(f"\n📈 دلایل:")
            for i, reason in enumerate(reasons[:3], 1):
                print(f"   {i}. {reason}")

        # پارامترهای ورود
        if result.get("entry_price"):
            ep = float(result.get("entry_price") or 0)
            sl = float(result.get("stop_loss") or 0)
            tp = float(result.get("take_profit") or 0)

            print(f"\n💰 پارامترهای ورود:")
            print(f"   قیمت ورود: ${ep:.2f}")
            print(f"   استاپ لاس: ${sl:.2f}")
            print(f"   تیک پروفیت: ${tp:.2f}")

            rr = result.get("risk_reward_ratio")
            if rr:
                try:
                    print(f"   نسبت ریسک/پاداش: {float(rr):.2f}:1")
                except Exception:
                    pass

            pos_size = result.get("position_size")
            if pos_size:
                try:
                    print(f"   حجم معامله: {float(pos_size):.3f} لات")
                except Exception:
                    pass

        quality = result.get("quality")
        if quality:
            q_map = {"HIGH": "⭐⭐⭐", "MEDIUM": "⭐⭐", "LOW": "⭐"}
            print(f"   کیفیت سیگنال: {quality} {q_map.get(quality, '')}")

    # ----------------------------
    # Trade Execution
    # ----------------------------
    def execute_scalping_trade(self, signal_data: dict, df=None) -> bool:
        """🔥 اجرای معامله اسکلپینگ با Real-Time، ثبت گزارش و ذخیره JSON"""
        SYMBOL = self.config.get("trading_settings.SYMBOL")
        TIMEFRAME = self.config.get("trading_settings.TIMEFRAME")

        # ایمنی: سیگنال باید BUY/SELL باشد
        signal_data["signal"] = self._normalize_signal(signal_data.get("signal", "NONE"))
        if signal_data["signal"] not in ("BUY", "SELL"):
            logger.info(f"⏸️ execute_scalping_trade skipped | signal={signal_data.get('signal')}")
            return False

        logger.info(f"🚀 شروع فرآیند اجرای معامله اسکلپینگ Real-Time: signal={signal_data.get('signal', 'N/A')}")

        if signal_data.get("error"):
            logger.error(f"❌ سیگنال حاوی خطاست، معامله اجرا نمی‌شود: {signal_data.get('reasons', ['Unknown error'])}")
            print("❌ سیگنال حاوی خطاست، معامله اجرا نمی‌شود")
            return False

        try:
            # قیمت Real-Time از PriceMonitor داخلی
            current_price_data = self.price_monitor.get_current_price(SYMBOL)
            if current_price_data.get("source") in ["no_data", "error"]:
                logger.error(f"❌ نمی‌توان قیمت Real-Time را دریافت کرد: {current_price_data.get('error', 'Unknown error')}")
                print("❌ دریافت قیمت Real-Time ناموفق")
                return False

            logger.info(
                "🎯 Real-Time Price Check: Symbol=%s Bid=%.2f Ask=%.2f Spread=%.2f Source=%s",
                SYMBOL,
                float(current_price_data.get("bid", 0.0) or 0.0),
                float(current_price_data.get("ask", 0.0) or 0.0),
                float(current_price_data.get("spread", 0.0) or 0.0),
                current_price_data.get("source", "Unknown"),
            )
            print(f"🎯 قیمت لحظه‌ای: Bid: {current_price_data['bid']:.2f}, Ask: {current_price_data['ask']:.2f}")

            market_metrics = signal_data.get("market_metrics", {}) or {}
            current_atr = market_metrics.get("atr")
            atr_short = market_metrics.get("atr_short")

            if current_atr:
                logger.info(f"📈 ATR معامله اسکلپینگ: ${float(current_atr):.2f}")
                print(f"📈 ATR معامله: ${float(current_atr):.2f}")

            if atr_short:
                logger.info(f"📈 ATR کوتاه‌مدت: ${float(atr_short):.2f}")
                print(f"📈 ATR کوتاه‌مدت: ${float(atr_short):.2f}")

            if not self.risk_manager:
                logger.error("❌ مدیر ریسک اسکلپینگ وجود ندارد")
                print("❌ مدیر ریسک اسکلپینگ وجود ندارد")
                return False

            live_snapshot = LivePriceSnapshot(
                bid=current_price_data["bid"],
                ask=current_price_data["ask"],
                timestamp=current_price_data.get("timestamp"),
            )

            config_payload = self.config.get_full_config()
            finalized = self.risk_manager.finalize_order(
                analysis=signal_data,
                live=live_snapshot,
                symbol=SYMBOL,
                config=config_payload,
            )

            if not finalized.is_trade_allowed:
                logger.warning(f"❌ Trade rejected by RiskManager: {finalized.reject_reason}")
                print(f"❌ RiskManager معامله را رد کرد: {finalized.reject_reason}")
                return False

            signal_data.update(
                {
                    "final_entry": finalized.entry_price,
                    "final_stop_loss": finalized.stop_loss,
                    "final_take_profit": finalized.take_profit,
                    "final_volume": finalized.lot_size,
                    "order_type": finalized.order_type,
                    "decision_reasons": finalized.decision_notes,
                }
            )

            order_type = finalized.order_type
            lot_size = finalized.lot_size
            price_deviation_pips = finalized.deviation_pips
            current_session = None
            scalping_grade = signal_data.get("quality", "N/A")
            if hasattr(self.risk_manager, "get_current_scalping_session"):
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

            logger.info(f"📤 ارسال سفارش اسکلپینگ ({order_type}) به بروکر: {signal_data['signal']} {lot_size:.3f} لات")
            print(f"📤 ارسال سفارش اسکلپینگ ({order_type}) به بروکر...")

            order_result = None

            # MT5Client شما send_order_real_time دارد و dict برمی‌گرداند.
            # Pending هم send_limit_order / send_pending_order را دارد.
            if str(order_type).lower() == "market":
                if hasattr(self.mt5_client, "send_order_real_time"):
                    order_result = self.mt5_client.send_order_real_time(
                        symbol=SYMBOL,
                        order_type=signal_data["signal"],
                        volume=lot_size,
                        sl_price=finalized.stop_loss,
                        tp_price=finalized.take_profit,
                        comment=f"NDS Scalping - {current_session or 'N/A'}",
                    )
                else:
                    order_result = self.mt5_client.send_order(
                        symbol=SYMBOL,
                        order_type=signal_data["signal"],
                        volume=lot_size,
                        stop_loss=finalized.stop_loss,
                        take_profit=finalized.take_profit,
                        comment=f"NDS Scalping - {current_session or 'N/A'}",
                    )
            else:
                # Limit/Pending
                limit_order_type = f"{signal_data['signal']}_LIMIT"  # BUY_LIMIT / SELL_LIMIT

                if hasattr(self.mt5_client, "send_limit_order"):
                    order_result = self.mt5_client.send_limit_order(
                        symbol=SYMBOL,
                        order_type=limit_order_type,
                        volume=lot_size,
                        limit_price=finalized.entry_price,
                        stop_loss=finalized.stop_loss,
                        take_profit=finalized.take_profit,
                        comment=f"NDS Scalping - {current_session or 'N/A'}",
                    )
                elif hasattr(self.mt5_client, "send_pending_order"):
                    order_result = self.mt5_client.send_pending_order(
                        symbol=SYMBOL,
                        order_type=limit_order_type,
                        volume=lot_size,
                        pending_price=finalized.entry_price,
                        stop_loss=finalized.stop_loss,
                        take_profit=finalized.take_profit,
                        comment=f"NDS Scalping - {current_session or 'N/A'}",
                    )
                else:
                    order_result = self.mt5_client.send_order(
                        symbol=SYMBOL,
                        order_type=limit_order_type,
                        volume=lot_size,
                        stop_loss=finalized.stop_loss,
                        take_profit=finalized.take_profit,
                        comment=f"NDS Scalping - {current_session or 'N/A'}",
                        order_action="LIMIT",
                    )

            # ارزیابی نتیجه
            success = False
            order_id = None
            position_ticket = None
            actual_entry_price = finalized.entry_price
            actual_sl = finalized.stop_loss
            actual_tp = finalized.take_profit

            if isinstance(order_result, dict):
                success = bool(order_result.get("success"))
                order_id = order_result.get("order_ticket") or order_result.get("ticket")
                position_ticket = order_result.get("position_ticket")
                # در send_order_real_time مقادیر entry/sl/tp برمی‌گردند
                actual_entry_price = float(order_result.get("entry_price", actual_entry_price) or actual_entry_price)
                actual_sl = float(order_result.get("stop_loss", actual_sl) or actual_sl)
                actual_tp = float(order_result.get("take_profit", actual_tp) or actual_tp)
                signal_data["execution_time"] = order_result.get("time", datetime.now())
            elif isinstance(order_result, int):
                success = True
                order_id = order_result

            if success and order_id:
                signal_data["order_ticket"] = order_id
                signal_data["position_ticket"] = position_ticket
                logger.info(
                    "✅ [TRADE][OPEN] ticket=%s position=%s symbol=%s side=%s entry=%.2f sl=%.2f tp=%.2f vol=%.3f order_type=%s",
                    order_id,
                    position_ticket,
                    SYMBOL,
                    signal_data["signal"],
                    actual_entry_price,
                    actual_sl,
                    actual_tp,
                    lot_size,
                    order_type,
                )
                print(f"✅ سفارش {order_type} ارسال شد - ticket={order_id} | حجم: {lot_size:.3f} لات")

                open_event: ExecutionEvent = {
                    "event_type": "OPEN",
                    "event_time": datetime.now(),
                    "symbol": SYMBOL,
                    "order_ticket": order_id,
                    "position_ticket": position_ticket,
                    "side": signal_data["signal"],
                    "volume": lot_size,
                    "entry_price": actual_entry_price,
                    "exit_price": None,
                    "sl": actual_sl,
                    "tp": actual_tp,
                    "profit": None,
                    "pips": None,
                    "reason": None,
                    "metadata": {
                        "confidence": signal_data.get("confidence", 0),
                        "scalping_grade": scalping_grade,
                        "timeframe": TIMEFRAME,
                        "risk_amount": getattr(finalized, "risk_amount_usd", None),
                        "session": current_session,
                        "order_type": order_type,
                        "magic": getattr(finalized, "magic", None),
                        "comment": order_result.get("comment") if isinstance(order_result, dict) else None,
                        "price_deviation_pips": price_deviation_pips,
                        "market_metrics": market_metrics,
                        "decision_notes": finalized.decision_notes,
                        "analysis_snapshot": signal_data,
                        "rr_ratio": getattr(finalized, "rr_ratio", None),
                    },
                }
                self.trade_tracker.add_trade_open(open_event)

                self.bot_state.add_trade(success=True)

                # برای candle-based cooldown، اینجا datetime.now نگذار (در run_analysis_cycle set می‌شود)
                # اگر df نبود، حداقل local زمان را بگذار
                if df is None or df.empty:
                    self.bot_state.last_trade_wall_time = datetime.now()
                    self.bot_state.last_trade_time = self.bot_state.last_trade_wall_time

                # آپدیت ریسک منیجر
                if hasattr(self.risk_manager, "add_position"):
                    self.risk_manager.add_position(lot_size)

                # گزارش اجرا
                generate_execution_report(
                    logger=logger,
                    event=open_event,
                    df=df,
                )

                # تلگرام
                try:
                    self.notifier.send_signal_notification(params=signal_data, symbol=SYMBOL)
                except Exception as t_err:
                    logger.warning(f"⚠️ خطای غیربحرانی در ارسال تلگرام: {t_err}", exc_info=True)

                # مانیتورینگ فوری بعد از باز شدن
                self._maybe_monitor_trades(force=True)
                return True

            logger.error(f"❌ ارسال سفارش اسکلپینگ {order_type} ناموفق بود | result={order_result}")
            print(f"❌ ارسال سفارش اسکلپینگ {order_type} ناموفق بود")
            self.bot_state.add_trade(success=False)
            return False

        except Exception as e:
            logger.error(f"❌ خطا در اجرای معامله اسکلپینگ Real-Time: {e}", exc_info=True)
            print(f"❌ خطا در اجرای معامله اسکلپینگ Real-Time: {e}")
            self.bot_state.add_trade(success=False)
            return False

    def execute_trade(self, signal_data: dict, df=None) -> bool:
        """سازگاری با کدهای قدیمی"""
        return self.execute_scalping_trade(signal_data, df)

    # ----------------------------
    # Trade Monitoring (Open/Close)
    # ----------------------------
    def _monitor_open_trades(self):
        """
        🔥 مانیتورینگ هوشمند:
        - بروزرسانی سود/قیمت برای پوزیشن‌های باز
        - جلوگیری از false-close با بررسی pending orders
        - تشخیص بسته‌شدن پوزیشن و ارسال نتیجه به تلگرام
        """
        if not hasattr(self, "trade_tracker"):
            return

        try:
            SYMBOL = self.config.get("trading_settings.SYMBOL")
            open_positions = self.get_open_positions_info()
            added_count, updated_count, closed_candidates = self.trade_tracker.reconcile_with_open_positions(open_positions)

            if added_count or updated_count:
                logger.debug("🔄 Trade reconciliation: added=%s updated=%s", added_count, updated_count)

            for record in closed_candidates:
                identity = record.get("trade_identity", {})
                position_ticket = identity.get("position_ticket")
                if not position_ticket:
                    continue

                history = self.mt5_client.get_position_history(position_ticket)
                if not history or not history.get("close_time"):
                    self.trade_tracker.mark_trade_unknown(position_ticket, "history_not_found")
                    logger.debug("⏳ Close not confirmed for position %s. Will retry.", position_ticket)
                    continue

                symbol = identity.get("symbol") or SYMBOL
                side = record.get("open_event", {}).get("side")
                entry_price = record.get("open_event", {}).get("entry_price")
                exit_price = history.get("exit_price") or record.get("last_update_event", {}).get("metadata", {}).get("current_price")
                profit = history.get("total_profit")
                close_time = history.get("close_time")
                reason = history.get("reason")

                pips_val = compute_pips(symbol, entry_price or 0.0, exit_price or 0.0)

                close_event: ExecutionEvent = {
                    "event_type": "CLOSE",
                    "event_time": close_time or datetime.now(),
                    "symbol": symbol,
                    "order_ticket": identity.get("order_ticket"),
                    "position_ticket": position_ticket,
                    "side": side,
                    "volume": record.get("open_event", {}).get("volume"),
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "sl": record.get("open_event", {}).get("sl"),
                    "tp": record.get("open_event", {}).get("tp"),
                    "profit": profit,
                    "pips": pips_val,
                    "reason": reason,
                    "metadata": {"history": history},
                }

                self.trade_tracker.close_trade_event(close_event)
                generate_execution_report(logger=logger, event=close_event)

                logger.info(
                    "[TRADE][CLOSE] position=%s profit=%.2f pips=%.1f reason=%s",
                    position_ticket,
                    float(profit or 0.0),
                    float(pips_val or 0.0),
                    reason,
                )

                if hasattr(self, "notifier") and self.notifier is not None:
                    try:
                        self.notifier.send_trade_close_notification(
                            symbol=symbol,
                            signal_type=side or "Unknown",
                            profit_usd=float(profit or 0.0),
                            pips=float(pips_val or 0.0),
                            reason=reason or "Manual/Other",
                        )
                        logger.info(f"✅ گزارش تلگرام برای بسته‌شدن پوزیشن #{position_ticket} ارسال شد.")
                    except Exception as tel_err:
                        logger.error(f"⚠️ خطا در ارسال نوتیفیکیشن تلگرام: {tel_err}", exc_info=True)

        except Exception as e:
            logger.error(f"⚠️ خطا در فرآیند مانیتورینگ معاملات: {e}", exc_info=True)

    # ----------------------------
    # Cleanup/Summary
    # ----------------------------
    def cleanup(self):
        """تمیزکاری منابع و قطع اتصال"""
        logger.info("🧹 در حال ذخیره وضعیت و تمیزکاری...")
        print("\n🧹 در حال ذخیره وضعیت...")

        try:
            # یک بار آخر مانیتورینگ تا closeها ثبت شوند
            self._maybe_monitor_trades(force=True)
        except Exception:
            pass

        try:
            if self.mt5_client:
                logger.info("قطع اتصال MT5...")
                self.mt5_client.disconnect()
                logger.info("✅ اتصال MT5 قطع شد")
                print("✅ اتصال MT5 قطع شد")
        except Exception as e:
            logger.error(f"⚠️ خطا در قطع اتصال MT5: {e}", exc_info=True)
            print(f"⚠️ خطا در قطع اتصال MT5: {e}")

    def print_summary(self):
        """چاپ گزارش نهایی عملکرد"""
        logger.info("📊 چاپ گزارش نهایی عملکرد اسکلپینگ")

        stats = self.bot_state.get_statistics()
        hours = int(stats["runtime_seconds"] // 3600)
        minutes = int((stats["runtime_seconds"] % 3600) // 60)
        seconds = int(stats["runtime_seconds"] % 60)

        print(f"\n{'📊' * 20}")
        print("خلاصه نهایی اجرا اسکلپینگ")
        print(f"{'📊' * 20}")

        print(f"⏱️  زمان اجرا: {hours}:{minutes:02d}:{seconds:02d}")
        print(f"📈 تعداد تحلیل‌ها: {stats['analysis_count']}")
        print(f"💰 تعداد معاملات: {stats['trade_count']}")

        if stats["trade_count"] > 0:
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
        print("\n✅ ربات اسکلپینگ با موفقیت متوقف شد")

    # ----------------------------
    # Main Loop
    # ----------------------------
    def run(self):
        """متد اصلی اجرای حلقه ربات"""
        logger.info("🚀 شروع اجرای ربات NDS اسکلپینگ")

        print_banner()
        print_help()

        atexit.register(self.cleanup)

        if not self._initialize_robot():
            return

        cycle_number = 0
        logger.info(f"🔁 شروع حلقه اصلی ربات اسکلپینگ، cycle_number={cycle_number}")

        try:
            self._run_main_loop(cycle_number)
        except KeyboardInterrupt:
            logger.info("🛑 توقف توسط کاربر (KeyboardInterrupt)")
            print("\n\n🛑 توقف توسط کاربر")
        finally:
            self._execute_shutdown_procedure()

    def _initialize_robot(self) -> bool:
        if not self.initialize():
            logger.critical("❌ راه‌اندازی ربات ناموفق بود")
            print("❌ راه‌اندازی ربات ناموفق بود")
            return False
        return True

    def _run_main_loop(self, start_cycle: int):
        cycle_number = start_cycle

        while self.bot_state.running:
            cycle_number += 1

            if not self.bot_state.paused:
                self._execute_analysis_cycle(cycle_number)

            if self.bot_state.running and not self.bot_state.paused:
                self._wait_for_next_cycle()

            self._handle_pause_mode()

    def _execute_analysis_cycle(self, cycle_number: int):
        logger.info(f"🔁 اجرای سیکل اسکلپینگ #{cycle_number}")
        self.run_analysis_cycle(cycle_number)

    def _wait_for_next_cycle(self):
        ANALYSIS_INTERVAL_MINUTES = self.config.get("trading_settings.ANALYSIS_INTERVAL_MINUTES")
        wait_time = ANALYSIS_INTERVAL_MINUTES * 60

        logger.info(f"⏳ انتظار برای سیکل بعدی: {ANALYSIS_INTERVAL_MINUTES} دقیقه")
        print(f"\n⏳ تحلیل بعدی در {ANALYSIS_INTERVAL_MINUTES} دقیقه...")
        print("   (فشار دهید: P=توقف, S=وضعیت, Q=خروج)")

        # در زمان انتظار، user_controls خودش loop دارد؛ بعد از پایان، مانیتور کنیم تا closeها از دست نرود
        self.user_controls.wait_with_controls(wait_time)
        self._maybe_monitor_trades(force=True)

    def _handle_pause_mode(self):
        while self.bot_state.paused and self.bot_state.running:
            logger.info("⏸️  ربات در حالت توقف")
            print("\n⏸️  ربات متوقف شده")
            print("   P=ادامه, Q=خروج, C=تنظیمات")

            action = self.user_controls.get_user_action()

            if action == "pause":
                self._resume_robot()
            elif action == "quit":
                self._stop_robot_during_pause()
                break
            elif action == "config":
                self._update_config_during_pause()
            else:
                # حتی در pause هم گهگاهی مانیتور معاملات را انجام بده
                self._maybe_monitor_trades()
                time.sleep(0.5)

    def _resume_robot(self):
        self.bot_state.paused = False
        logger.info("▶️  ربات ادامه یافت")
        print("▶️  ربات ادامه یافت")

    def _stop_robot_during_pause(self):
        self.bot_state.running = False
        logger.info("👋 درخواست خروج در حالت توقف")

    def _update_config_during_pause(self):
        logger.info("⚙️  به‌روزرسانی تنظیمات در حالت توقف")
        update_config_interactive()

    def _execute_shutdown_procedure(self):
        logger.info("🧹 شروع فرآیند تمیزکاری نهایی")

        # ابتدا summary (هنوز اتصال برقرار است)
        try:
            self.print_summary()
        except Exception as e:
            logger.error(f"⚠️ خطا در چاپ summary: {e}", exc_info=True)

        # سپس cleanup
        self.cleanup()

        logger.info("🏁 پایان اجرای ربات اسکلپینگ")
