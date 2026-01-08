"""
Legacy helpers moved out of analyzer layer to keep the analyzer purely analytical.

TODO: Move any remaining execution/risk logic into a dedicated execution module.
"""
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple


@dataclass
class EntryParameters:
    """
    Legacy execution-oriented parameters (kept for backward compatibility only).
    This is not used by the analyzer layer after the analysis-only refactor.
    """
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    risk_reward_ratio: float
    risk_amount: float
    reason: str
    confidence: float
    entry_type: str = "FVG"  # FVG, RANGE_BOUNCE, BREAKOUT, ORDER_BLOCK

    @property
    def is_valid(self) -> bool:
        return all([
            self.entry_price is not None,
            self.stop_loss is not None,
            self.take_profit is not None,
            self.risk_reward_ratio >= 1.0,
            self.risk_amount > 0,
        ])


class LegacyGoldNDSExecution:
    """
    Legacy execution-oriented analyzer methods preserved for reference.
    These are intentionally not used by the analysis-only analyzer layer.
    """

    def _clear_entry_parameters(self, result: Dict) -> Dict:
        raise NotImplementedError("Legacy execution method removed from analysis layer.")

    def _calculate_scalping_entry_parameters(self, signal, fvgs, order_blocks, structure,
                                             atr_value, entry_factor, current_price):
        raise NotImplementedError("Legacy execution method removed from analysis layer.")

    def _find_best_scalping_entry(self, fvgs, order_blocks, current_price, signal_type, atr_value=None):
        raise NotImplementedError("Legacy execution method removed from analysis layer.")

    def _adjust_scalping_entry_for_price_deviation(self, entry_params, current_price, atr_value, signal):
        raise NotImplementedError("Legacy execution method removed from analysis layer.")

    def _calculate_entry_parameters(self, signal: str, fvgs: List, order_blocks: List,
                                    structure, atr_value: float, entry_factor: float,
                                    adx_value: float = None) -> Optional[EntryParameters]:
        raise NotImplementedError("Legacy execution method removed from analysis layer.")

    def _calculate_trend_adjusted_entry(self, signal: str, fvgs: List, order_blocks: List,
                                        structure, atr_value: float, entry_factor: float,
                                        adx_value: float) -> Optional[EntryParameters]:
        raise NotImplementedError("Legacy execution method removed from analysis layer.")

    def _calculate_mixed_market_entry(self, signal: str, fvgs: List, order_blocks: List,
                                      structure, atr_value: float, entry_factor: float,
                                      adx_value: float) -> Optional[EntryParameters]:
        raise NotImplementedError("Legacy execution method removed from analysis layer.")

    def _calculate_range_entry(self, signal: str, structure, atr_value: float,
                               adx_value: float = None) -> EntryParameters:
        raise NotImplementedError("Legacy execution method removed from analysis layer.")

    def _calculate_ob_entry(self, order_block, signal: str, atr_value: float,
                            adx_value: float = None) -> EntryParameters:
        raise NotImplementedError("Legacy execution method removed from analysis layer.")

    def _calculate_entry_confidence(self, params: EntryParameters, atr_value: float) -> float:
        raise NotImplementedError("Legacy execution method removed from analysis layer.")

    def _calculate_risk_amount(self, risk_distance: float, scalping: bool = True) -> float:
        raise NotImplementedError("Legacy execution method removed from analysis layer.")

    def _validate_scalping_risk(self, entry_params) -> bool:
        raise NotImplementedError("Legacy execution method removed from analysis layer.")

    def _validate_regular_risk(self, entry_params) -> bool:
        raise NotImplementedError("Legacy execution method removed from analysis layer.")

    def _calculate_max_holding_time(self, volatility_state):
        raise NotImplementedError("Legacy execution method removed from analysis layer.")

    def _adjust_scalping_confidence(self, result, entry_params):
        raise NotImplementedError("Legacy execution method removed from analysis layer.")

    def _display_final_result(self, result: Dict):
        raise NotImplementedError("Legacy execution method removed from analysis layer.")

    def _create_error_result(self, error_message: str) -> Dict:
        raise NotImplementedError("Legacy execution method removed from analysis layer.")
