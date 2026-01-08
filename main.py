"""
NDS Trading Bot Pro - Main Entry Point
نسخه ماژولار - منطبق با ساختار src.trading_bot.*
یکپارچه با ConfigManager موجود در config/settings.py
بهبود یافته برای جلوگیری از دو منبع RiskManager و خروج ایمن
"""

import sys
import os
import signal
import logging
from pathlib import Path
from typing import Any, Dict, Optional


# ------------------------------------------------------------
# 1) Standard Path Setup (Safe / Non-duplicative)
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ------------------------------------------------------------
# 2) Imports (Project)
# ------------------------------------------------------------
try:
    from src.utils.logger import setup_windows_encoding, setup_logging
    from config.settings import config as config_manager
    from src.trading_bot.bot import NDSBot
    from src.trading_bot.mt5_client import MT5Client
    from src.trading_bot.nds.analyzer import analyze_gold_market
except ImportError as e:
    print(f"❌ خطای ساختار پروژه: {e}")
    print("نکته: مطمئن شوید فایل‌ها در مسیرهای صحیح قرار دارند.")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# ------------------------------------------------------------
# 3) Logging Setup
# ------------------------------------------------------------
setup_windows_encoding()
setup_logging()
logger = logging.getLogger(__name__)


def _safe_get_full_config() -> Dict[str, Any]:
    """
    تلاش برای دریافت کانفیگ کامل بدون وابستگی شکننده به private field.
    """
    # اگر متد رسمی دارید، اولویت با آن است:
    for method_name in ("get_full_config", "get_config", "to_dict"):
        method = getattr(config_manager, method_name, None)
        if callable(method):
            try:
                cfg = method()
                if isinstance(cfg, dict):
                    return cfg
            except Exception:
                pass

    # fallback: دسترسی به _config با محافظ
    cfg = getattr(config_manager, "_config", None)
    return cfg if isinstance(cfg, dict) else {}


def _print_active_settings(full_config: Dict[str, Any]) -> None:
    """
    نمایش تنظیمات فعال (فقط برای اطمینان اپراتور).
    """
    trading_settings = full_config.get("trading_settings", {}) or {}
    risk_settings = full_config.get("risk_settings", {}) or {}

    print("\n⚙️  تنظیمات نهایی (Active):")
    print(f"  • نماد: {trading_settings.get('SYMBOL', 'XAUUSD!')}")
    print(f"  • تایم‌فریم: {trading_settings.get('TIMEFRAME', 'M5')}")
    print(f"  • بازه تحلیل: {trading_settings.get('ANALYSIS_INTERVAL_MINUTES', 5)} دقیقه")
    print(f"  • BARS: {trading_settings.get('BARS_TO_FETCH', 'N/A')}")
    print(f"  • AutoTrading: {trading_settings.get('ENABLE_AUTO_TRADING', False)}")
    print(f"  • DryRun: {trading_settings.get('ENABLE_DRY_RUN', False)}")

    # ریسک
    print("\n🛡️  تنظیمات ریسک (Config):")
    # ممکن است پروژه شما هم RISK_PERCENT داشته باشد هم RISK_AMOUNT_USD؛ هر دو را نمایش می‌دهیم
    if "RISK_AMOUNT_USD" in risk_settings:
        print(f"  • ریسک ثابت دلاری: ${risk_settings.get('RISK_AMOUNT_USD', 0.0)}")
    if "RISK_PERCENT" in risk_settings:
        print(f"  • ریسک درصدی: {risk_settings.get('RISK_PERCENT', 0.0)}%")

    # حداقل اطمینان (ممکن است در technical_settings باشد؛ اینجا فقط اگر در risk_settings بود)
    if "MIN_CONFIDENCE" in risk_settings:
        print(f"  • حداقل اعتماد (risk_settings): {risk_settings.get('MIN_CONFIDENCE', 0)}%")


def main() -> None:
    """
    تابع اصلی اجرای برنامه
    """

    # پاکسازی کنسول (اختیاری)
    try:
        os.system("cls" if os.name == "nt" else "clear")
    except Exception:
        pass

    print("🚀 NDS Gold Scalping Bot - در حال اجرا ...")

    try:
        # 1) Load config safely
        print("⏳ در حال بارگذاری تنظیمات از config/bot_config.json ...")
        full_config = _safe_get_full_config()

        if not full_config:
            print("⚠️  هشدار: کانفیگ کامل بارگذاری نشد. بررسی config/settings.py و bot_config.json ضروری است.")
            logger.warning("Full config is empty or not loaded.")

        # 2) Minimal sanity checks (credentials existence)
        try:
            creds = config_manager.get_mt5_credentials()
        except Exception:
            creds = None

        if not creds or not all(k in creds for k in ("login", "password", "server")):
            print("⚠️  اطلاعات MT5 کامل نیست. لطفاً mt5_credentials را در config تنظیم کنید.")
            logger.warning("MT5 credentials incomplete.")

        # 3) Print active settings snapshot
        _print_active_settings(full_config)

        # 4) Create bot (Single source of truth: bot will create its own RiskManager + monitors)
        print("\n📦 ماژول‌های فعال:")
        print(f"  • MT5 Client: {MT5Client.__name__}")
        print("  • Risk Manager: managed inside NDSBot.initialize()")
        print("  • Analyzer: analyze_gold_market (NDS/SMC Modular)")

        bot = NDSBot(
            mt5_client_cls=MT5Client,
            risk_manager_cls=None,   # مدیریت داخلی توسط initialize
            analyzer_cls=None,
            analyze_func=analyze_gold_market
        )

        # 5) Signal handling (Safe stop: do NOT sys.exit immediately)
        def signal_handler(sig, frame):
            print("\n🛑 درخواست توقف دریافت شد. ربات به صورت ایمن متوقف می‌شود...")
            logger.info("SIGINT received. Requesting safe shutdown...")

            try:
                if hasattr(bot, "bot_state") and bot.bot_state:
                    bot.bot_state.running = False
            except Exception:
                pass

        signal.signal(signal.SIGINT, signal_handler)

        # 6) Run bot
        print("\n🎯 شروع چرخه معاملاتی اسکلپینگ طلا")
        bot.run()

        # 7) After run finishes (normal shutdown)
        print("\n✅ اجرای ربات پایان یافت.")
        logger.info("Bot run finished normally.")

    except KeyboardInterrupt:
        # در حالت عادی signal handler این را مدیریت می‌کند؛ این فقط fallback است
        print("\n🛑 توقف توسط کاربر (KeyboardInterrupt)")
        logger.info("KeyboardInterrupt in main().")

    except Exception as e:
        print(f"\n💥 خطای بحرانی در لایه Main: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.critical("Critical failure in main", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
