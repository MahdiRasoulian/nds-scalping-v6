# config/settings.py
import json
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """مدیریت متمرکز تنظیمات از bot_config.json (منبع واحد حقیقت)"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """بارگذاری config از config/bot_config.json به عنوان منبع واحد"""
        config_path = Path(__file__).resolve().parent / "bot_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"bot_config.json not found at {config_path}")

        with config_path.open('r', encoding='utf-8') as f:
            self._config = json.load(f)

        self._validate_required_keys()
        logger.info("✅ Config loaded from config/bot_config.json")

    def _validate_required_keys(self) -> None:
        required_paths = [
            "bot_name",
            "version",
            "ACCOUNT_BALANCE",
            "trading_rules",
            "risk_settings",
            "trading_settings",
            "technical_settings",
            "sessions_config",
            "risk_manager_config",
        ]

        for key in required_paths:
            self._require_path(key)

    def _require_path(self, key: str) -> Any:
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                raise KeyError(f"Missing required config key: {key}")
        return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """دریافت مقدار از config با پشتیبانی از مسیر نقطه‌ای (Dot Notation)"""
        value = self._require_path(key)
        return value if value is not None else default
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        return self.get(key, default)
    
    def get_trading_rules(self) -> Dict:
        return self._config.get('trading_rules', {})
    
    def get_risk_settings(self) -> Dict:
        return self._config.get('risk_settings', {})
    
    def get_trading_settings(self) -> Dict:
        return self._config.get('trading_settings', {})
    
    def get_technical_settings(self) -> Dict:
        return self._config.get('technical_settings', {})
    
    def get_sessions_config(self) -> Dict:
        return self._config.get('sessions_config', {})
    
    def get_risk_manager_config(self) -> Dict:
        return self._config.get('risk_manager_config', {})
    
    def get_full_config_for_analyzer(self) -> Dict:
        """دریافت کامل تنظیمات برای آنالایزر بدون مقادیر هاردکد شده"""
        technical = self.get_technical_settings()
        sessions = self.get_sessions_config()

        return {
            'ANALYZER_SETTINGS': technical.copy(),
            'sessions_config': sessions.copy(),
            'TRADING_SESSIONS': sessions.get('TRADING_SESSIONS', {})
        }

    def get_full_config(self) -> Dict:
        """دریافت کل تنظیمات به صورت دیکشنری"""
        return self._config.copy()
    
    def get_mt5_credentials(self) -> Dict:
        creds_path = "config/mt5_credentials.json"
        if Path(creds_path).exists():
            try:
                with open(creds_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except: pass
        return {}

    def save_mt5_credentials(self, credentials: Dict):
        creds_path = "config/mt5_credentials.json"
        try:
            Path("config").mkdir(exist_ok=True)
            with open(creds_path, 'w', encoding='utf-8') as f:
                json.dump(credentials, f, indent=2, ensure_ascii=False)
            logger.info("✅ MT5 credentials saved")
        except Exception as e:
            logger.error(f"❌ Error saving MT5 credentials: {e}")
    
    def update_setting(self, key: str, value: Any):
        keys = key.split('.')
        config_ref = self._config
        for k in keys[:-1]:
            if k not in config_ref: config_ref[k] = {}
            config_ref = config_ref[k]
        config_ref[keys[-1]] = value
        logger.info(f"Config updated: {key} = {value}")
    
config = ConfigManager()
