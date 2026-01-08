"""
NDS Trading Bot Pro - Main Entry Point
نسخه ماژولار v5.0 - کاملاً منطبق با ساختار پکیج src.trading_bot.nds
یکپارچه شده با ConfigManager موجود در config/settings.py
نسخه به‌روز شده برای یکپارچگی با bot_config.json
"""

import sys
import os
import signal
import logging
from pathlib import Path

# ۱. تنظیم مسیرهای پروژه (Standard Path Setup)
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# ۲. ایمپورت ماژول‌های اصلی پروژه با ساختار جدید
try:
    from src.utils.logger import setup_windows_encoding, setup_logging
    # اصلاح مسیر ایمپورت بر اساس فایل شما:
    from config.settings import config as config_manager  # ✅ استفاده از اینستنس جهانی شما
    from src.trading_bot.bot import NDSBot
    from src.trading_bot.mt5_client import MT5Client
    from src.trading_bot.risk_manager import create_scalping_risk_manager
    from src.trading_bot.nds.analyzer import analyze_gold_market 
except ImportError as e:
    print(f"❌ خطای ساختار پروژه: {e}")
    print("نکته: مطمئن شوید فایل‌ها در مسیرهای صحیح قرار دارند.")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# تنظیمات اولیه
setup_windows_encoding()
setup_logging()
logger = logging.getLogger(__name__)

def main():
    """تابع اصلی اجرای برنامه"""
    
    # پاکسازی کنسول
    os.system('cls' if os.name == 'nt' else 'clear')
    print("🚀 NDS Gold Scalping Bot v5.0 (Modular Edition) در حال اجراست...")
    
    try:
        # ۳. بارگذاری و ادغام تنظیمات از config متمرکز
        print("⏳ در حال بارگذاری تنظیمات از config/bot_config.json ...")
        
        # دریافت تنظیمات کامل از config متمرکز
        full_config = config_manager._config
        
        # الف) آماده‌سازی تنظیمات برای تحلیل‌گر
        analyzer_config = config_manager.get_full_config_for_analyzer()

        # 🔧 اصلاح: اگر ANALYZER_SETTINGS خالی است، مستقیماً از config بگیر
        if not analyzer_config.get('ANALYZER_SETTINGS'):
            print("⚠️  ANALYZER_SETTINGS خالی است. پر کردن از config اصلی...")
            analyzer_config['ANALYZER_SETTINGS'] = config_manager.get_technical_settings()
            
        # همچنین TRADING_SESSIONS را اضافه کن
        if 'TRADING_SESSIONS' not in analyzer_config:
            sessions_config = config_manager.get_sessions_config()
            analyzer_config['TRADING_SESSIONS'] = sessions_config.get('TRADING_SESSIONS', {})

        print(f"✅ تنظیمات آنالایزر آماده شد: {len(analyzer_config.get('ANALYZER_SETTINGS', {}))} تنظیم تکنیکال")


        
        # ب) آماده‌سازی تنظیمات برای مدیر ریسک
        risk_manager_config = config_manager.get_risk_manager_config()
        
        # اضافه کردن تنظیمات مورد نیاز از config اصلی
        if risk_manager_config:
            # تنظیمات ضروری برای سازگاری
            scalping_config = {
                'risk_settings': config_manager.get('risk_settings', {}),
                'technical_settings': config_manager.get('technical_settings', {}),
                'sessions_config': config_manager.get('sessions_config', {}),
                'trading_rules': config_manager.get('trading_rules', {}),
                'risk_manager_config': risk_manager_config,
            }
            
            print("✅ تنظیمات مدیر ریسک اسکلپینگ از config متمرکز بارگیری شد.")
        else:
            print("⚠️ بخش risk_manager_config در config متمرکز یافت نشد. استفاده از تنظیمات پیش‌فرض.")
            scalping_config = {
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

        # ۴. ایجاد مدیر ریسک اسکلپینگ با تنظیمات یکپارچه
        try:
            risk_manager = create_scalping_risk_manager(overrides=scalping_config)
            print("✅ مدیر ریسک اسکلپینگ با موفقیت ایجاد شد")
        except Exception as e:
            logger.error(f"❌ خطا در ایجاد مدیر ریسک اسکلپینگ: {e}", exc_info=True)
            print(f"❌ خطا در ایجاد مدیر ریسک اسکلپینگ: {e}")
            
            # ایجاد مدیر ریسک با تنظیمات ساده‌تر
            risk_manager = create_scalping_risk_manager(overrides={})
            print("⚠️ مدیر ریسک با تنظیمات حداقلی ایجاد شد")
        
        # ۵. مقداردهی اولیه ربات (Dependency Injection)
        try:
            bot = NDSBot(
                mt5_client_cls=MT5Client,
                analyzer_cls=None,
                risk_manager_cls=None,  # از متد initialize ربات استفاده می‌شود
                analyze_func=analyze_gold_market
            )
            
            # تنظیم config تحلیل‌گر برای استفاده در ربات
            bot.analyzer_config = analyzer_config
            
            print("✅ ربات NDS با موفقیت مقداردهی اولیه شد")
        except Exception as e:
            logger.error(f"❌ خطا در ایجاد ربات: {e}", exc_info=True)
            print(f"❌ خطا در ایجاد ربات: {e}")
            raise
        
        print("✅ تمام ماژول‌ها (MT5, Scalping Risk Manager, SMC Analyzer) با موفقیت بارگذاری شدند.")
        
        # نمایش اطلاعات نسخه‌ها و تنظیمات
        print("\n📦 ماژول‌های فعال:")
        print(f"  • MT5 Client: {MT5Client.__name__}")
        print(f"  • Scalping Risk Manager: v{risk_manager.__class__.__name__}")
        print(f"  • SMC Analyzer: Gold Scalping v5.0")
        
        # نمایش تنظیمات فعال (جهت اطمینان کاربر)
        print("\n⚙️  تنظیمات نهایی (Active):")
        print(f"  • حداکثر ریسک: {scalping_config.get('MAX_RISK_PERCENT', 0.5)}%")
        print(f"  • ضریب ATR استاپ: {scalping_config.get('ATR_SL_MULTIPLIER', 1.0)}x")
        print(f"  • حداکثر معاملات روزانه: {scalping_config.get('MAX_DAILY_TRADES', 20)}")
        
        # نمایش تنظیمات معاملاتی
        trading_settings = full_config.get('trading_settings', {})
        if trading_settings:
            print(f"  • نماد: {trading_settings.get('SYMBOL', 'XAUUSD!')}")
            print(f"  • تایم‌فریم: {trading_settings.get('TIMEFRAME', 'M15')}")
            print(f"  • بازه تحلیل: {trading_settings.get('ANALYSIS_INTERVAL_MINUTES', 5)} دقیقه")
        
        # نمایش تنظیمات ریسک
        risk_settings = full_config.get('risk_settings', {})
        if risk_settings:
            print(f"  • ریسک هر معامله: {risk_settings.get('RISK_PERCENT', 2.0)}%")
            print(f"  • حداقل اعتماد: {risk_settings.get('MIN_CONFIDENCE', 65)}%")
        
        # ۶. مدیریت سیگنال خروج (Ctrl+C)
        def signal_handler(sig, frame):
            print(f"\n{'🛑' * 15} توقف ایمن ربات اسکلپینگ... {'🛑' * 15}")
            
            if hasattr(bot, 'bot_state'):
                bot.bot_state.running = False
            
            try:
                # چک کردن وجود متد در کلاس RiskManager
                if hasattr(risk_manager, 'get_scalping_summary'):
                    summary = risk_manager.get_scalping_summary()
                    print("\n📊 خلاصه وضعیت نهایی اسکلپینگ:")
                    print(f"  • سود/زیان امروز: ${summary.get('daily_profit_loss', 0.0):.2f}")
                    print(f"  • کل معاملات امروز: {summary.get('trades_today', 0)}")
                    print(f"  • سشن در لحظه خروج: {summary.get('current_session', 'N/A')}")
                    
                    # نمایش آمار اسکلپینگ
                    stats = summary.get('scalping_stats', {})
                    if stats.get('total_scalps', 0) > 0:
                        total_scalps = stats.get('total_scalps', 0)
                        winning_scalps = stats.get('winning_scalps', 0)
                        win_rate = (winning_scalps / total_scalps * 100) if total_scalps > 0 else 0
                        print(f"  • آمار اسکلپینگ: {total_scalps} معامله")
                        print(f"  • وین ریت: {win_rate:.1f}%")
                
                # نمایش وضعیت ربات
                if hasattr(bot, 'bot_state'):
                    bot_stats = bot.bot_state.get_statistics()
                    print(f"\n📊 آمار کلی ربات:")
                    print(f"  • کل تحلیل‌ها: {bot_stats.get('analysis_count', 0)}")
                    print(f"  • کل معاملات: {bot_stats.get('trade_count', 0)}")
                    print(f"  • نرخ موفقیت: {bot_stats.get('success_rate', 0):.1f}%")
                    
            except Exception as e:
                print(f"⚠️  خطا در نمایش خلاصه: {e}")
            
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # ۷. شروع چرخه فعالیت ربات
        print(f"\n{'🎯' * 5} شروع چرخه معاملاتی اسکلپینگ طلا {'🎯' * 5}")
        
        # اطلاعات حساب از config
        account_balance = full_config.get('ACCOUNT_BALANCE', 893.93)
        print(f"💰 موجودی حساب از config: ${account_balance:.2f}")
        
        # اطلاعات زمان‌بندی
        analysis_interval = trading_settings.get('ANALYSIS_INTERVAL_MINUTES', 5)
        print(f"⏰ بازه تحلیل: هر {analysis_interval} دقیقه")
        
        # شروع اجرای ربات
        bot.run()
        
    except KeyboardInterrupt:
        print(f"\n{'🛑' * 10} توقف توسط کاربر {'🛑' * 10}")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n💥 خطای بحرانی در لایه Main: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.critical("Critical failure in main loop", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
