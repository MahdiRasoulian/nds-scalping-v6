# src/backtester/reporter.py

import matplotlib.pyplot as plt
import pandas as pd
import logging
import os

# تنظیم لاگ
logger = logging.getLogger("NDS_Reporter")

def plot_best_run(params):
    """رسم نمودار خلاصه برای بهترین اجرا (نسخه اصلی بهبود یافته)"""
    try:
        plt.figure(figsize=(14, 10))
        
        # 1. نمایش اطلاعات متنی و پارامترها
        plt.subplot(2, 2, 1)
        plt.title(f"Best Configuration Summary", fontsize=16, fontweight='bold')
        
        info_text = (
            f"--- Parameters ---\n"
            f"Timeframe: {params.get('TIMEFRAME')}\n"
            f"Entry Factor: {params.get('ENTRY_FACTOR')}\n"
            f"Min Confidence: {params.get('SCALPING_MIN_CONFIDENCE')}%\n"
            f"SL Multiplier: {params.get('ATR_SL_MULTIPLIER')}\n"
            f"RVOL Min: {params.get('MIN_RVOL_SCALPING')}\n"
            f"\n--- Performance ---\n"
            f"Net Profit: ${params.get('Net Profit ($)', 0):.2f}\n"
            f"Total Trades: {params.get('Total Trades', 0)}\n"
            f"Win Rate: {params.get('Win Rate (%)', 0):.1f}%\n"
            f"Max Drawdown: {params.get('Max Drawdown (%)', 0):.2f}%"
        )
        
        plt.text(0.1, 0.5, info_text, transform=plt.gca().transAxes,
                 fontsize=11, verticalalignment='center', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        plt.axis('off')
        
        # 2. نمودار میله‌ای سود و تعداد معاملات
        plt.subplot(2, 2, 2)
        metrics = ['Net Profit ($)', 'Total Trades']
        values = [params.get(m, 0) for m in metrics]
        bars = plt.bar(metrics, values, color=['#27ae60', '#2980b9'])
        plt.title('Key Performance Metrics')
        plt.grid(True, axis='y', alpha=0.3)
        # افزودن عدد روی هر میله
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.1f}", va='bottom', ha='center')
        
        # 3. نمودار دایره‌ای توزیع برد و باخت
        plt.subplot(2, 2, 3)
        if params.get('Total Trades', 0) > 0:
            win_rate = params.get('Win Rate (%)', 0)
            loss_rate = 100 - win_rate
            plt.pie([win_rate, loss_rate], labels=['Wins', 'Losses'], 
                   colors=['#4CAF50', '#F44336'], autopct='%1.1f%%', startangle=90, explode=(0.05, 0))
            plt.title('Win/Loss Distribution')
        else:
            plt.text(0.5, 0.5, 'No Trades Recorded', ha='center', va='center')

        # 4. نمودار کوچک اکوئیتی (اگر داده‌ها موجود باشند)
        if 'equity_curve' in params:
            plt.subplot(2, 2, 4)
            plt.plot(params['equity_curve'], color='#8e44ad', linewidth=2)
            plt.title('Equity Growth Trend')
            plt.xlabel('Trade Sequence')
            plt.ylabel('Balance ($)')
            plt.grid(True, alpha=0.2)

        plt.tight_layout()
        filename = "nds_best_run_analysis.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"🖼️ Analysis chart saved to '{filename}'")
        
    except Exception as e:
        logger.error(f"Error in plot_best_run: {e}")

def generate_detailed_report(run_data, run_name="test_run"):
    """
    تولید گزارش کامل شامل:
    1. فایل CSV معاملات
    2. نمودار بزرگ روند اکوئیتی (Equity Curve)
    """
    try:
        # ایجاد پوشه خروجی
        output_folder = "backtest_reports"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        # 1. ذخیره لیست معاملات در CSV
        trades_list = run_data.get('trades_list', [])
        if trades_list:
            df_trades = pd.DataFrame(trades_list)
            csv_path = os.path.join(output_folder, f"{run_name}_trades.csv")
            df_trades.to_csv(csv_path, index=False)
            logger.info(f"✅ Trade details saved to: {csv_path}")
            
        # 2. رسم نمودار اختصاصی Equity Curve
        equity_curve = run_data.get('equity_curve', [])
        if equity_curve:
            plt.figure(figsize=(12, 6))
            plt.plot(equity_curve, color='#2ecc71', linewidth=2.5, label='Account Equity')
            plt.fill_between(range(len(equity_curve)), equity_curve, min(equity_curve), alpha=0.15, color='#2ecc71')
            
            plt.title(f"Equity Growth Curve: {run_name}", fontsize=14)
            plt.xlabel("Trade Count")
            plt.ylabel("Balance ($)")
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            
            chart_path = os.path.join(output_folder, f"{run_name}_equity.png")
            plt.savefig(chart_path, dpi=200, bbox_inches='tight')
            plt.close()
            logger.info(f"📈 Equity chart saved to: {chart_path}")
            
    except Exception as e:
        logger.error(f"Error generating detailed report: {e}")