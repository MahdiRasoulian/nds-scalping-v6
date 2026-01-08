import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import pytz

def extract_mt5_data(symbol: str, timeframe, start_date: datetime, end_date: datetime = None):
    """
    اتصال به MT5 و استخراج داده کندلی - نسخه بهبودیافته
    """
    # ۱. اتصال به MT5 با تنظیمات بهتر
    if not mt5.initialize(
        path="C:/Users/uep.ops.supv/AppData/Roaming/MetaTrader 5/terminal64.exe",  # مسیر MT5
        login=600108041,          # شماره حساب
        password="3Bl!8705",    # رمز
        server="Opogroup-Server1",  # سرور
        timeout=60000           # تایم‌اوت 60 ثانیه
    ):
        print(f"❌ MT5 initialization failed: {mt5.last_error()}")
        return None
    
    print(f"✅ Connected to MT5 - Version: {mt5.version()}")

    # ۲. بررسی نماد
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"❌ Symbol {symbol} not found")
        mt5.shutdown()
        return None
    
    # ۳. فعال‌سازی نماد اگر غیرفعال است
    if not symbol_info.visible:
        print(f"⚠️ Symbol {symbol} is not visible, trying to activate...")
        if not mt5.symbol_select(symbol, True):
            print(f"❌ Failed to activate {symbol}")
            mt5.shutdown()
            return None
    
    # ۴. تنظیم تاریخ‌ها (end_date پیش‌فرض حال حاضر)
    if end_date is None:
        end_date = datetime.now()
    
    timezone = pytz.timezone("Etc/UTC")
    date_from = timezone.localize(start_date)
    date_to = timezone.localize(end_date)
    
    print(f"📅 Requesting data: {symbol} | {timeframe} | {date_from} to {date_to}")
    
    # ۵. استخراج داده با کنترل خطا
    try:
        rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)
    except Exception as e:
        print(f"❌ Error fetching rates: {e}")
        mt5.shutdown()
        return None
    
    if rates is None or len(rates) == 0:
        print(f"⚠️ No data retrieved for {symbol}. Check date range.")
        
        # تست با بازه کوچک‌تر برای دیباگ
        test_end = date_from + timedelta(days=7)
        test_rates = mt5.copy_rates_range(symbol, timeframe, date_from, test_end)
        
        if test_rates is not None and len(test_rates) > 0:
            print(f"⚠️ But found {len(test_rates)} bars for 7-day test period")
            rates = test_rates
        else:
            mt5.shutdown()
            return None
    
    print(f"✅ Retrieved {len(rates)} bars")
    
    # ۶. تبدیل به DataFrame با ستون‌های کامل
    df = pd.DataFrame(rates)
    
    # تبدیل زمان
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # نامگذاری ستون‌ها (مطابق نیاز NDS Analyzer)
    column_mapping = {
        'open': 'open',
        'high': 'high', 
        'low': 'low',
        'close': 'close',
        'tick_volume': 'volume',
        'real_volume': 'real_volume',  # اگر موجود باشد
        'spread': 'spread'
    }
    
    # فقط ستون‌های موجود را نگه دار
    available_cols = [col for col in column_mapping.keys() if col in df.columns]
    df = df[['time'] + available_cols]
    
    # تغییر نام ستون‌ها
    df.columns = ['time'] + [column_mapping[col] for col in available_cols]
    
    # ۷. اعتبارسنجی داده‌ها
    # حذف سطرهای با قیمت نامعتبر
    df = df[
        (df['open'] > 0) & 
        (df['high'] > 0) & 
        (df['low'] > 0) & 
        (df['close'] > 0) &
        (df['high'] >= df['low']) &
        (df['high'] >= df['open']) & 
        (df['high'] >= df['close']) &
        (df['low'] <= df['open']) & 
        (df['low'] <= df['close'])
    ].copy()
    
    # مرتب‌سازی زمانی
    df.sort_values('time', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # ۸. اطلاعات آماری
    print(f"📊 Data Statistics:")
    print(f"   Period: {df['time'].min()} to {df['time'].max()}")
    print(f"   Total bars: {len(df)}")
    print(f"   Avg. spread: {df.get('spread', pd.Series([0])).mean():.1f}")
    print(f"   Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    
    # ۹. قطع اتصال
    mt5.shutdown()
    
    return df

def save_data_with_metadata(df: pd.DataFrame, symbol: str, timeframe: int, 
                           start_date: datetime, end_date: datetime):
    """ذخیره داده با متادیتای کامل"""
    
    # نام فایل با فرمت استاندارد
    tf_name = {
        mt5.TIMEFRAME_M1: "M1",
        mt5.TIMEFRAME_M5: "M5", 
        mt5.TIMEFRAME_M15: "M15"
    }.get(timeframe, f"TF{timeframe}")
    
    file_name = f"{symbol.replace('!', '')}_{tf_name}_{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}.csv"
    
    # ذخیره داده
    df.to_csv(file_name, index=False)
    
    # ایجاد فایل متادیتا
    metadata = {
        'symbol': symbol,
        'timeframe': tf_name,
        'timeframe_mt5': timeframe,
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'total_bars': len(df),
        'price_range': f"{df['low'].min():.2f} - {df['high'].max():.2f}",
        'created_at': datetime.now().isoformat(),
        'columns': list(df.columns)
    }
    
    import json
    with open(file_name.replace('.csv', '_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Data saved to: {file_name}")
    print(f"✅ Metadata saved to: {file_name.replace('.csv', '_metadata.json')}")
    
    return file_name

if __name__ == '__main__':
    # --- تنظیمات استخراج ---
    SYMBOL = "XAUUSD!"
    
    # تست با چند تایم‌فریم مختلف
    TIMEFRAMES = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
    }
    
    # محدوده زمانی (آخرین 90 روز)
    END_DATE = datetime.now()
    START_DATE = END_DATE - timedelta(days=2)
    
    print(f"📊 Starting data extraction for {SYMBOL}")
    print(f"   Period: {START_DATE} to {END_DATE}")
    
    for tf_name, tf_value in TIMEFRAMES.items():
        print(f"\n{'='*50}")
        print(f"Extracting {tf_name} data...")
        
        data = extract_mt5_data(SYMBOL, tf_value, START_DATE, END_DATE)
        
        if data is not None and len(data) > 0:
            file_path = save_data_with_metadata(data, SYMBOL, tf_value, START_DATE, END_DATE)
            
            # برای بکتست: M5 را پیشنهاد می‌دهیم
            if tf_name == 'M5':
                print(f"\n💡 For backtesting, use this file: {file_path}")
                print("   Command: python scripts/run_backtest.py --data \"" + file_path + "\"")
    
    print(f"\n{'='*50}")
    print("✅ All extractions completed!")