# src/backtester/main.py

import sys
import os
import pandas as pd
import numpy as np
import logging

# افزودن مسیر ریشه برای ایمپورت‌ها
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.backtester.backtester import NDSBacktester
# ایمپورت کردن هر دو تابع گزارش‌دهی از ریپورتر جدید
from src.backtester.reporter import plot_best_run, generate_detailed_report

def run_backtest_system():
    """سیستم اجرای بکتست، بهینه‌سازی و تولید گزارشات تفصیلی"""
    
    # 1. تنظیم فایل دیتا
    csv_file = "XAUUSD_M1_20251202-20260101.csv"
    
    if not os.path.exists(csv_file):
        print(f"❌ Data file not found: {csv_file}")
        return

    # 2. تعریف پارامترهای بهینه‌سازی (مطابق با متغیرهای تعریف شده در backtester.py)
    param_grid = {
        'TIMEFRAME': ['M5'],
        'ENTRY_FACTOR': [0.25],
        'SCALPING_MIN_CONFIDENCE': [44],
        'ATR_SL_MULTIPLIER': [2.0],
        'MIN_RVOL_SCALPING': [0.75],
        
        # پارامترهای جدید اسکلپینگ
        'SCALPING_MAX_BARS_BACK': [600],
        'SCALPING_MAX_DISTANCE_ATR': [2.5],
        "ADX_THRESHOLD_WEAK": [28],

        'MIN_CANDLES_BETWEEN': [5, 8],
        
        # مدیریت ریسک
        'MIN_RISK_REWARD': [0.66, 0.8],
        'MIN_SESSION_WEIGHT': [0.5]
    }

    print("\n🚀 Initializing NDS Backtester...")
    backtester = NDSBacktester(
        csv_path=csv_file,
        initial_balance=500.0,
        start_date="2025-12-08",
        end_date="2025-12-19" 
    )

    # 3. اجرای بهینه‌سازی
    print(f"\n🔄 Running Optimization on {len(param_grid)} parameters...")
    # نکته: برای تست سریع n_samples را پایین نگه دارید، برای نتیجه دقیق mode را 'grid' کنید
    results = backtester.run_optimization(param_grid, mode='random', n_samples=20)
    
    # 4. پردازش و نمایش نتایج
    if not results.empty and 'Total Trades' in results.columns:
        print("\n🏆 OPTIMIZATION COMPLETE")
        
        # جدا کردن مواردی که حداقل یک معامله داشته‌اند
        trades_only = results[results['Total Trades'] > 0].copy()
        
        if not trades_only.empty:
            # ذخیره کل نتایج معتبر در CSV
            trades_only.to_csv("optimization_results_valid.csv", index=False)
            print(f"💾 Saved all valid combinations to 'optimization_results_valid.csv'")

            # انتخاب بهترین اجرا (اولین سطر چون در بکتستر سورت شده است)
            best_run = trades_only.iloc[0]
            
            print("\n🌟 BEST CONFIGURATION DETAILS:")
            print(f"   Timeframe: {best_run['TIMEFRAME']}")
            print(f"   Net Profit: ${best_run['Net Profit ($)']:.2f}")
            print(f"   Win Rate: {best_run['Win Rate (%)']:.1f}%")
            print(f"   Total Trades: {best_run['Total Trades']}")
            print(f"   Max Drawdown: {best_run['Max Drawdown (%)']:.2f}%")
            
            # --- بخش جدید: تولید گزارشات تفصیلی ---
            print("\n📊 Generating comprehensive reports...")
            
            # 1. رسم نمودار کلی مقایسه‌ای (نسخه قدیمی بهبود یافته)
            plot_best_run(best_run.to_dict())
            
            # 2. تولید گزارش اختصاصی برای 3 اجرای برتر (شامل لیست تریدها و گراف اکوئیتی)
            top_n = min(3, len(trades_only))
            for i in range(top_n):
                run_data = trades_only.iloc[i].to_dict()
                run_name = f"TopRun_{i+1}_{run_data['TIMEFRAME']}"
                generate_detailed_report(run_data, run_name=run_name)
            
            print(f"\n✨ All reports and charts are ready in 'backtest_reports' folder.")
            
        else:
            print("⚠️ No configuration produced any trades. Try lowering SCALPING_MIN_CONFIDENCE.")
    else:
        print("❌ No results generated. Check your data or parameters.")

if __name__ == "__main__":
    try:
        run_backtest_system()
    except KeyboardInterrupt:
        print("\n🛑 Backtest stopped by user.")
    except Exception as e:
        print(f"❌ Critical Error: {e}")