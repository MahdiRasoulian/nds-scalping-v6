"""Trade tracking utilities for NDS bot."""

from datetime import datetime


class TradeTracker:
    """ردیاب کامل معاملات از باز شدن تا بسته شدن"""

    def __init__(self):
        self.active_trades = {}  # {ticket: {open_time, entry_price, volume, ...}}
        self.closed_trades = []  # لیست معاملات بسته شده
        self.max_daily_profit = 0.0
        self.daily_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_profit': 0.0
        }

    def add_trade(self, ticket: int, entry_data: dict):
        """ثبت معامله جدید"""
        self.active_trades[ticket] = {
            **entry_data,
            'status': 'OPEN',
            'open_time': datetime.now(),
            'max_profit': 0.0,
            'max_loss': 0.0,
            'current_profit': 0.0,
            'current_price': entry_data.get('entry_price', 0.0),
            'last_update': datetime.now()
        }
        self.daily_stats['total_trades'] += 1

    def update_trade(self, ticket: int, current_price: float, current_profit: float, mt5_client=None):
        """بروزرسانی وضعیت معامله باز"""
        if ticket in self.active_trades:
            trade = self.active_trades[ticket]

            # اگر قیمت یا سود صفر پاس داده شده باشد (نشانه بسته شدن از طرف مانیتورینگ)
            # و در عین حال در MT5 هم پوزیشن نباشد، عملیات بستن را انجام بده
            if current_price == 0.0 and current_profit == 0.0:
                # استفاده از آخرین سود ثبت شده برای گزارش نهایی
                final_p = trade.get('current_profit', 0.0)
                self._close_trade_automatically(ticket, final_p)
                return

            # بروزرسانی مقادیر لحظه‌ای
            trade['current_price'] = current_price
            trade['current_profit'] = current_profit
            trade['last_update'] = datetime.now()

            # ثبت حداکثر سود/ضرر
            trade['max_profit'] = max(trade['max_profit'], current_profit)
            trade['max_loss'] = min(trade['max_loss'], current_profit)

            # --- بخش بهبود یافته ---
            # به جای درخواست مجدد از MT5، از منطق مانیتورینگ bot.py تبعیت می‌کنیم.
            # اگر لازم باشد از داخل اینجا هم چک شود، فقط به شرطی که لیست پوزیشن‌ها از قبل گرفته نشده باشد.
            # اما طبق معماری جدید ما، نیازی به کدهای زیر نیست و کامنت می‌شوند تا سرعت بالا برود:
            """
            try:
                positions = mt5_client.get_open_positions()
                if positions:
                    open_tickets = [p.get('ticket') for p in positions if p]
                    if ticket not in open_tickets:
                        self._close_trade_automatically(ticket, current_profit)
            except:
                pass
            """

    def _close_trade_automatically(self, ticket: int, final_profit: float):
        """بستن خودکار معامله وقتی از MT5 حذف شده"""
        if ticket in self.active_trades:
            trade = self.active_trades[ticket]
            trade.update({
                'status': 'CLOSED',
                'close_time': datetime.now(),
                'close_profit': final_profit,
                'final_profit': final_profit,
                'close_reason': 'auto_detected'
            })

            self.closed_trades.append(trade)

            # آمار روزانه
            self.daily_stats['total_profit'] += final_profit
            if final_profit > 0:
                self.daily_stats['winning_trades'] += 1

            # حداکثر سود روزانه
            if final_profit > self.max_daily_profit:
                self.max_daily_profit = final_profit

            del self.active_trades[ticket]
            return True
        return False

    def get_active_trades_count(self) -> int:
        """تعداد معاملات فعال"""
        return len(self.active_trades)

    def get_daily_stats(self) -> dict:
        """آمار روزانه"""
        win_rate = 0
        if self.daily_stats['total_trades'] > 0:
            win_rate = (self.daily_stats['winning_trades'] / self.daily_stats['total_trades']) * 100

        return {
            **self.daily_stats,
            'win_rate': win_rate,
            'max_daily_profit': self.max_daily_profit,
            'active_trades': self.get_active_trades_count(),
            'closed_trades': len(self.closed_trades)
        }
