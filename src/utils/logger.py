# src/utils/logger.py

import sys
import os
import logging
import re
from config.settings import config

# نگاشت سطوح لاگ
LOG_LEVEL_MAP = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

def setup_windows_encoding():
    """تنظیمات encoding برای ویندوز (قبل از هر چیز)"""
    if os.name == 'nt':
        # تغییر کدپیج کنسول به UTF-8
        os.system('chcp 65001 > nul')
        
        # پیکربندی stdout و stderr برای UTF-8
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except (AttributeError, Exception):
            # جایگزین برای پایتون قدیمی
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def setup_logging():
    """تنظیمات یکپارچه logging برای کل پروژه"""
    
    # دریافت سطح لاگ از کانفیگ
    log_level = config.get('LOG_LEVEL', "INFO")
    
    # حذف همه handlers موجود
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # تنظیم formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handler با UTF-8
    file_handler = logging.FileHandler(
        config.get('DEBUG_LOG_FILE', 'debug.log'), 
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    
    # Stream handler با UTF-8
    class SafeStreamHandler(logging.StreamHandler):
        """Handler ایمن برای ویندوز با UTF-8"""
        def __init__(self, stream=None):
            super().__init__(stream)
        
        def emit(self, record):
            try:
                msg = self.format(record)
                stream = self.stream
                stream.write(msg + self.terminator)
                self.flush()
            except UnicodeEncodeError:
                # اگر خطای encoding داشت، ایموجی‌ها را حذف کن
                try:
                    msg = self.format(record)
                    msg_clean = re.sub(r'[^\x00-\x7F]+', '', msg)
                    stream = self.stream
                    stream.write(msg_clean + self.terminator)
                    self.flush()
                except:
                    self.handleError(record)
            except:
                self.handleError(record)
    
    stream_handler = SafeStreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    
    # تنظیم level و اضافه کردن handlers
    logging.root.setLevel(LOG_LEVEL_MAP.get(log_level, logging.INFO))
    logging.root.addHandler(file_handler)
    logging.root.addHandler(stream_handler)