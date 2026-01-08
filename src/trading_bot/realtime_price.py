"""Real-time price monitoring utilities."""

import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional


class RealTimePriceMonitor:
    """مانیتورینگ و کش قیمت‌های Real-Time"""

    def __init__(self, config, bot_state, logger):
        self.config = config
        self.bot_state = bot_state
        self.logger = logger
        self.mt5_client = None
        self.real_time_prices: Dict[str, Dict[str, Any]] = {}
        self.last_tick_time: Dict[str, datetime] = {}
        self.price_monitor_thread: Optional[threading.Thread] = None
        self._last_price_log = datetime.min

    def set_mt5_client(self, mt5_client) -> None:
        self.mt5_client = mt5_client

    def start(self) -> None:
        """🔥 شروع مانیتورینگ Real-Time قیمت‌ها"""
        if not self.mt5_client or not self.mt5_client.connected:
            self.logger.warning("⚠️ Cannot start Real-Time monitor: MT5 not connected")
            return

        try:
            if hasattr(self.mt5_client, 'real_time_monitor'):
                if self.mt5_client.real_time_monitor:
                    self.logger.info("✅ Real-Time monitor already active")
                    return

                self.mt5_client.real_time_monitor.start()
                self.logger.info("🎯 Real-Time Price Monitor Started")
            else:
                self._start_legacy_price_monitor()

        except Exception as e:
            self.logger.error(f"❌ Error starting Real-Time monitor: {e}")

    def _start_legacy_price_monitor(self) -> None:
        """🔥 مانیتورینگ Real-Time برای نسخه‌های قدیمی MT5 Client"""
        def monitor_loop():
            self.logger.info("🔄 Legacy Real-Time Monitor started")
            while getattr(self.bot_state, 'is_running', self.bot_state.running) and self.mt5_client and self.mt5_client.connected:
                try:
                    symbol = self.config.get('trading_settings.SYMBOL')
                    tick = self.mt5_client.get_current_tick(symbol)

                    if tick:
                        self.real_time_prices[symbol] = {
                            'bid': tick['bid'],
                            'ask': tick['ask'],
                            'last': tick['last'],
                            'time': tick['time'],
                            'spread': tick['spread']
                        }
                        self.last_tick_time[symbol] = datetime.now()

                        current_time = datetime.now()
                        if (current_time - self._last_price_log).seconds >= 30:
                            self.logger.debug(
                                f"📊 Real-Time Price: {symbol} - Bid: {tick['bid']:.2f}, "
                                f"Ask: {tick['ask']:.2f}, Spread: {tick['spread']:.2f}"
                            )
                            self._last_price_log = current_time

                    time.sleep(1)

                except Exception as e:
                    self.logger.error(f"Real-Time monitor error: {e}")
                    time.sleep(5)

            self.logger.info("⏹️ Legacy Real-Time Monitor stopped")

        self.price_monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.price_monitor_thread.start()

    def get_current_price(self, symbol: str) -> Dict[str, Any]:
        """🔥 دریافت قیمت لحظه‌ای از کش یا دریافت مستقیم"""
        try:
            if symbol in self.real_time_prices:
                price_data = self.real_time_prices[symbol]
                if self.last_tick_time.get(symbol):
                    age = (datetime.now() - self.last_tick_time[symbol]).total_seconds()
                    if age < 3:
                        return {
                            **price_data,
                            'source': 'real_time_cache',
                            'age_seconds': age
                        }

            if self.mt5_client and self.mt5_client.connected:
                tick = self.mt5_client.get_current_tick(symbol)
                if tick:
                    return {
                        'bid': tick.get('bid', 0),
                        'ask': tick.get('ask', 0),
                        'last': tick.get('last', 0),
                        'time': tick.get('time', datetime.now()),
                        'spread': tick.get('spread', 0),
                        'source': 'direct_fetch'
                    }

            return {
                'bid': 0,
                'ask': 0,
                'last': 0,
                'time': datetime.now(),
                'spread': 0,
                'source': 'no_data',
                'error': 'No price data available'
            }

        except Exception as e:
            self.logger.error(f"Error getting current price: {e}")
            return {
                'bid': 0,
                'ask': 0,
                'last': 0,
                'time': datetime.now(),
                'spread': 0,
                'source': 'error',
                'error': str(e)
            }
