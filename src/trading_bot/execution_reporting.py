"""Execution reporting utilities for finalized orders."""

import json
from datetime import datetime
from pathlib import Path


def generate_execution_report(
    logger,
    signal_data: dict,
    finalized,
    order_id,
    symbol: str,
    timeframe: str,
    order_type: str,
    lot_size: float,
    current_session: str,
    scalping_grade: str,
    market_metrics: dict,
    current_price_data: dict,
    price_deviation_pips: float,
    risk_manager=None,
    df=None
) -> None:
    """Generate JSON execution report and optional full report."""
    try:
        execution_entry_price = signal_data.get('actual_entry_price', finalized.entry_price)
        execution_stop_loss = signal_data.get('actual_stop_loss', finalized.stop_loss)
        execution_take_profit = signal_data.get('actual_take_profit', finalized.take_profit)
        planned_stop_loss = signal_data.get('stop_loss', finalized.stop_loss)
        planned_take_profit = signal_data.get('take_profit', finalized.take_profit)
        planned_entry = signal_data.get('entry_price', finalized.entry_price)

        session_multiplier = 1.0
        if risk_manager and hasattr(risk_manager, 'get_scalping_multiplier'):
            session_multiplier = risk_manager.get_scalping_multiplier(current_session or 'N/A')

        execution_report = {
            'order_id': order_id,
            'symbol': symbol,
            'signal': signal_data['signal'],
            'order_type': order_type,
            'entry_price_planned': planned_entry,
            'entry_price_actual': execution_entry_price,
            'stop_loss_planned': planned_stop_loss,
            'stop_loss_actual': execution_stop_loss,
            'take_profit_planned': planned_take_profit,
            'take_profit_actual': execution_take_profit,
            'lot_size': lot_size,
            'confidence': signal_data.get('confidence', 0),
            'execution_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'scalping_params': {},
            'risk_params': {
                'risk_amount': finalized.risk_amount_usd,
                'risk_percent': None,
                'actual_risk_percent': None,
                'sl_distance': abs(finalized.entry_price - finalized.stop_loss),
                'scalping_grade': scalping_grade,
                'max_holding_minutes': 60,
                'session': current_session or 'N/A',
                'session_multiplier': session_multiplier
            },
            'timeframe': timeframe,
            'signal_quality': signal_data.get('quality', 'MEDIUM'),
            'scalping_mode': True,
            'market_metrics': market_metrics,
            'real_time_data': {
                'bid_at_analysis': current_price_data.get('bid'),
                'ask_at_analysis': current_price_data.get('ask'),
                'bid_at_execution': signal_data.get('execution_bid'),
                'ask_at_execution': signal_data.get('execution_ask'),
                'price_deviation_pips': price_deviation_pips,
                'execution_source': current_price_data.get('source', 'unknown')
            }
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        execution_file = f"trade_reports/scalping_executions/{symbol}_scalping_{timestamp}.json"
        Path("trade_reports/scalping_executions").mkdir(parents=True, exist_ok=True)

        with open(execution_file, 'w', encoding='utf-8') as f:
            json.dump(execution_report, f, indent=2, ensure_ascii=False)

        logger.info(f"📝 گزارش خام معامله اسکلپینگ Real-Time در {execution_file} ذخیره شد")
        print(f"📝 گزارش خام معامله اسکلپینگ ذخیره شد")

        try:
            from src.reporting.report_generator import ReportGenerator
            if df is not None:
                report_gen = ReportGenerator(output_dir="trade_reports/scalping_reports")

                order_details = {
                    'signal': signal_data['signal'],
                    'side': signal_data['signal'],
                    'confidence': signal_data.get('confidence', 0),
                    'entry_planned': planned_entry,
                    'entry_actual': execution_entry_price,
                    'sl_planned': planned_stop_loss,
                    'sl_actual': execution_stop_loss,
                    'tp_planned': planned_take_profit,
                    'tp_actual': execution_take_profit,
                    'rr_ratio': finalized.rr_ratio,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'execution_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'lot': lot_size,
                    'scalping_grade': scalping_grade,
                    'scalping_mode': True,
                    'session': current_session,
                    'price_deviation_pips': price_deviation_pips,
                    'order_type': order_type
                }

                report_result = report_gen.generate_full_report(
                    df=df,
                    signal_data=signal_data,
                    order_details=order_details
                )

                if report_result['success']:
                    logger.info(f"📊 گزارش کامل معامله اسکلپینگ Real-Time ذخیره شد")
                    print(f"📊 گزارش کامل معامله اسکلپینگ تولید شد")
        except ImportError:
            logger.debug("ماژول گزارش‌گیری یافت نشد، فقط JSON ذخیره شد.")

    except Exception as e:
        logger.error(f"⚠️ خطا در فرآیند گزارش‌گیری اسکلپینگ: {e}")
        print(f"⚠️ خطا در گزارش‌گیری: {e}")
