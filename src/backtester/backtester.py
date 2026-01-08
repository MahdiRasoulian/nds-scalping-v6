import pandas as pd
import numpy as np
import itertools
import logging
from tqdm import tqdm
import traceback
from datetime import datetime
import random
import os
from typing import Dict, List, Any

# تنظیم لاگ جامع
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest_debug.log', mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("NDS_Backtester")

class NDSBacktester:
    """کلاس بکتستر پیشرفته برای استراتژی NDS"""
    
    def __init__(self, csv_path: str, initial_balance: float = 1000.0, 
                 start_date: str = None, end_date: str = None):
        """مقداردهی اولیه بکتستر"""
        self.csv_path = csv_path
        self.initial_balance = initial_balance
        self.start_date = start_date
        self.end_date = end_date
        self.raw_data = self._load_data()
        
        if self.raw_data.empty:
            logger.error("❌ Failed to load data. Backtester cannot continue.")
            raise ValueError("No data loaded.")

    def _load_data(self) -> pd.DataFrame:
        """بارگذاری و آماده‌سازی داده‌ها"""
        try:
            logger.info(f"📂 Loading data from: {self.csv_path}")
            df = pd.read_csv(self.csv_path)
            
            df.columns = [col.strip().lower() for col in df.columns]
            
            time_column = None
            for col in ['time', 'date', 'datetime', 'timestamp']:
                if col in df.columns:
                    time_column = col
                    break
            
            if not time_column:
                for col in df.columns:
                    try:
                        pd.to_datetime(df[col].iloc[0])
                        time_column = col
                        break
                    except:
                        continue
            
            if not time_column:
                logger.error("❌ Time column not found!")
                return pd.DataFrame()

            df[time_column] = pd.to_datetime(df[time_column])
            df.set_index(time_column, inplace=True)
            df.sort_index(inplace=True)
            
            rename_map = {}
            for req in ['open', 'high', 'low', 'close', 'volume']:
                if req not in df.columns:
                    for col in df.columns:
                        if req in col:
                            rename_map[col] = req
                            break
            
            if rename_map:
                df.rename(columns=rename_map, inplace=True)

            if self.start_date:
                df = df[df.index >= pd.to_datetime(self.start_date)]
            if self.end_date:
                df = df[df.index <= pd.to_datetime(self.end_date)]
                
            # بهبود: مدیریت مقادیر گم شده برای جلوگیری از خطا در محاسبات اندیکاتور
            df = df.ffill().bfill()
            
            required = ['open', 'high', 'low', 'close']
            missing = [c for c in required if c not in df.columns]
            if missing:
                logger.error(f"❌ Missing columns: {missing}")
                return pd.DataFrame()
            
            logger.info(f"✅ Loaded {len(df)} rows from {df.index[0]} to {df.index[-1]}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return pd.DataFrame()

    def resample_data(self, timeframe: str) -> pd.DataFrame:
        """تبدیل تایم‌فریم"""
        rule_map = {'M1': '1min', 'M5': '5min', 'M15': '15min', 'H1': '1h'}
        rule = rule_map.get(timeframe, '1min')
        
        try:
            resampled = self.raw_data.resample(rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            return resampled
        except Exception as e:
            logger.error(f"❌ Error resampling: {e}")
            return pd.DataFrame()

    def run_optimization(self, param_grid: Dict[str, List[Any]], mode: str = 'grid', n_samples: int = 100) -> pd.DataFrame:
        """اجرای بهینه‌سازی و بازگرداندن نتایج تفصیلی"""
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        all_combinations = list(itertools.product(*values))
        
        if mode == 'random' and n_samples < len(all_combinations):
            logger.info(f"🎲 Random sampling: {n_samples} out of {len(all_combinations)}")
            combinations_to_test = random.sample(all_combinations, n_samples)
        else:
            logger.info(f"📊 Full grid search: {len(all_combinations)} combinations")
            combinations_to_test = all_combinations

        results = []
        
        for combo in tqdm(combinations_to_test, desc="Optimizing"):
            params = dict(zip(keys, combo))
            try:
                stats = self._run_single_backtest(params)
                result_row = {**params, **stats}
                results.append(result_row)
            except Exception as e:
                logger.error(f"Error in combo {params}: {e}")
                logger.error(traceback.format_exc())
                continue
                
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            return self._sort_results(results_df)
        return results_df

    def _run_single_backtest(self, params: Dict) -> Dict:
        """اجرای یک دور بکتست با اعمال محدودیت فاصله بین معاملات"""
        timeframe = params.get('TIMEFRAME', 'M15')
        df = self.resample_data(timeframe)
        
        if df.empty:
            return self._get_empty_metrics()

        config = self._create_config(params)
        min_candles_between = params.get('MIN_CANDLES_BETWEEN', 4) 
        max_positions = params.get('MAX_POSITIONS', 4)

        trades = []
        balance = self.initial_balance
        equity_curve = [balance]
        
        last_exit_idx = -min_candles_between 
        active_trades = [] 
        window_size = params.get('WINDOW_SIZE', 100)

        try:
            from src.trading_bot.nds.analyzer import GoldNDSAnalyzer
        except ImportError:
            logger.error("❌ Could not import GoldNDSAnalyzer")
            return self._get_empty_metrics()

        for i in range(window_size, len(df)):
            # ۱. مدیریت پوزیشن‌های باز و خروج‌ها (بر اساس کندل فعلی i)
            still_active = []
            for t in active_trades:
                # چک کردن خروج با دیتای کندل فعلی
                exit_check = self._check_individual_trade_exit(df.iloc[i], t, params.get('RISK_AMOUNT_USD', 50.0))
                if exit_check:
                    balance += exit_check['pnl']
                    last_exit_idx = i
                    t.update({
                        'exit_time': df.index[i],
                        'exit_price': exit_check['exit_price'],
                        'pnl': exit_check['pnl'],
                        'duration': i - t['start_idx'],
                        'result': 'WIN' if exit_check['pnl'] > 0 else 'LOSS'
                    })
                    trades.append(t)
                else:
                    still_active.append(t)
            active_trades = still_active

            # ۲. تحلیل برای سیگنال جدید (با رعایت عدم نگاه به آینده i-1)
            can_open = len(active_trades) < max_positions and i >= last_exit_idx + min_candles_between
            
            if can_open:
                current_data = df.iloc[:i].copy() # استفاده از i برای حذف Look-ahead bias
                current_data['time'] = current_data.index
                
                analyzer = GoldNDSAnalyzer(current_data, config=config)
                signal_result = analyzer.generate_trading_signal(
                    timeframe=timeframe,
                    entry_factor=params.get('ENTRY_FACTOR', 0.2),
                    scalping_mode=True
                )
                
                if signal_result and signal_result.get('signal') in ['BUY', 'SELL']:
                    # ورود در قیمت شروع کندل فعلی یا قیمت اعلامی آنالایزر
                    entry_price = signal_result.get('entry_price', df.iloc[i]['open'])
                    sl = signal_result.get('stop_loss')
                    tp = signal_result.get('take_profit')
                    
                    if entry_price and sl and tp:
                        active_trades.append({
                            'start_idx': i,
                            'entry_time': df.index[i],
                            'type': signal_result['signal'],
                            'entry_price': entry_price,
                            'sl': sl,
                            'tp': tp
                        })

            equity_curve.append(balance)

        return self._calculate_metrics(trades, equity_curve)

    def _check_individual_trade_exit(self, current_candle, trade, risk_amount):
        """متد کمکی برای بررسی خروج یک معامله در کندل جاری (جلوگیری از Look-ahead)"""
        high, low = current_candle['high'], current_candle['low']
        
        if trade['type'] == 'BUY':
            if low <= trade['sl']: # اولویت با ضرر در صورت لمس هر دو در یک کندل
                return {'pnl': -risk_amount, 'exit_price': trade['sl']}
            if high >= trade['tp']:
                rr = abs(trade['tp'] - trade['entry_price']) / abs(trade['entry_price'] - trade['sl']) if abs(trade['entry_price'] - trade['sl']) != 0 else 1
                return {'pnl': risk_amount * rr, 'exit_price': trade['tp']}
        
        elif trade['type'] == 'SELL':
            if high >= trade['sl']:
                return {'pnl': -risk_amount, 'exit_price': trade['sl']}
            if low <= trade['tp']:
                rr = abs(trade['entry_price'] - trade['tp']) / abs(trade['sl'] - trade['entry_price']) if abs(trade['sl'] - trade['entry_price']) != 0 else 1
                return {'pnl': risk_amount * rr, 'exit_price': trade['tp']}
        return None

    def _simulate_trade_outcome(self, df, start_idx, signal_type, entry, sl, tp, spread):
        """شبیه‌سازی نتیجه معامله (حفظ شده برای سازگاری با متدهای دیگر)"""
        look_forward = min(len(df), start_idx + 300)
        risk_amount = 50.0

        for j in range(start_idx, look_forward):
            res = self._check_individual_trade_exit(df.iloc[j], 
                                                 {'type': signal_type, 'entry_price': entry, 'sl': sl, 'tp': tp}, 
                                                 risk_amount)
            if res:
                res['duration'] = j - start_idx
                return res
        
        # در صورت انقضای زمان
        final_price = df.iloc[look_forward-1]['close']
        pnl = (final_price - entry) if signal_type == 'BUY' else (entry - final_price)
        return {'pnl': pnl, 'duration': look_forward - start_idx, 'exit_price': final_price, 'reason': 'timeout'}

    def _create_config(self, params: Dict) -> Dict:
        """ایجاد ساختار کانفیگ تودرتو منطبق با هسته بات"""
        return {
            "trading_settings": {
                "TIMEFRAME": params.get('TIMEFRAME', 'M15'),
            },
            "technical_settings": {
                "ENTRY_FACTOR": params.get('ENTRY_FACTOR', 0.2),
                "SCALPING_MIN_CONFIDENCE": params.get('SCALPING_MIN_CONFIDENCE', 35),
                "ATR_SL_MULTIPLIER": params.get('ATR_SL_MULTIPLIER', 2.0),
                "MIN_RVOL_SCALPING": params.get('MIN_RVOL_SCALPING', 0.8),
                "ATR_WINDOW": 14,
                "SWING_PERIOD": 5,
                "SCALPING_MAX_BARS_BACK": params.get('SCALPING_MAX_BARS_BACK', 500),
                "SCALPING_MAX_DISTANCE_ATR": params.get('SCALPING_MAX_DISTANCE_ATR', 2.5),
                "ADX_THRESHOLD_WEAK": params.get('ADX_THRESHOLD_WEAK', 20),
                "MIN_STRUCTURE_SCORE": 20.0
            },
            "risk_settings": {
                "MIN_CONFIDENCE": params.get('SCALPING_MIN_CONFIDENCE', 35.0),
                "RISK_AMOUNT_USD": 50.0,
                "MIN_RISK_REWARD": params.get('MIN_RISK_REWARD', 1.0),
                "MAX_PRICE_DEVIATION_PIPS": 55.0
            },
            "sessions_config": {
                "MIN_SESSION_WEIGHT": params.get('MIN_SESSION_WEIGHT', 0.3)
            }
        }

    def _calculate_metrics(self, trades, equity_curve):
        """محاسبه جامع شاخص‌های عملکرد"""
        if not trades:
            metrics = self._get_empty_metrics()
            metrics['equity_curve'] = equity_curve
            return metrics
            
        pnls = [t['pnl'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        net_profit = sum(pnls)
        win_rate = (len(wins) / len(trades)) * 100
        
        eq_series = pd.Series(equity_curve)
        rolling_max = eq_series.cummax()
        drawdowns = (eq_series - rolling_max) / rolling_max * 100
        max_dd = abs(drawdowns.min()) if not drawdowns.empty else 0

        returns = eq_series.pct_change().dropna()
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 and np.std(returns) != 0 else 0

        return {
            'Total Trades': len(trades),
            'Win Rate (%)': win_rate,
            'Net Profit ($)': net_profit,
            'Max Drawdown (%)': max_dd,
            'Profit Factor': sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 999,
            'Average Win ($)': np.mean(wins) if wins else 0,
            'Average Loss ($)': np.mean(losses) if losses else 0,
            'Sharpe Ratio': sharpe,
            'trades_list': trades,      
            'equity_curve': equity_curve 
        }

    def _get_empty_metrics(self):
        return {
            'Total Trades': 0, 
            'Win Rate (%)': 0, 
            'Net Profit ($)': 0, 
            'Max Drawdown (%)': 0,
            'Profit Factor': 0,
            'Sharpe Ratio': 0,
            'trades_list': [],
            'equity_curve': [self.initial_balance]
        }

    def _sort_results(self, df):
        if 'Net Profit ($)' in df.columns:
            return df.sort_values('Net Profit ($)', ascending=False)
        return df