"""
专业套利风险管理模块
提供实时风险监控、资金管理、止损控制等功能
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """风险指标数据类"""
    total_capital: float  # 总资金
    used_capital: float   # 已用资金
    available_capital: float  # 可用资金
    utilization_rate: float   # 资金利用率
    max_drawdown: float       # 最大回撤
    current_drawdown: float   # 当前回撤
    var_1d: float            # 1日风险价值
    sharpe_ratio: float      # 夏普比率
    risk_score: int          # 风险评分(1-10)
    exposure_by_asset: Dict[str, float]  # 各资产敞口

@dataclass
class PositionInfo:
    """持仓信息"""
    symbol: str
    exchange: str
    amount: float
    value_usd: float
    entry_price: float
    current_price: float
    pnl: float
    pnl_percent: float
    risk_level: str  # low, medium, high

@dataclass
class ArbitrageOpportunity:
    """套利机会评估"""
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    spread_percent: float
    expected_profit: float
    execution_difficulty: str  # easy, medium, hard
    risk_level: str
    recommended_amount: float
    confidence_score: float

class RiskManager:
    """专业风险管理器"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, PositionInfo] = {}
        self.pnl_history: List[Tuple[datetime, float]] = []
        self.max_drawdown_limit = 0.15  # 15%最大回撤限制
        self.max_position_size = 0.20   # 单个持仓不超过20%
        self.max_utilization = 0.80     # 最大资金利用率80%
        
        # 风险参数
        self.risk_free_rate = 0.02  # 无风险利率2%
        self.confidence_level = 0.95  # VaR置信度95%
        
    def calculate_risk_metrics(self) -> RiskMetrics:
        """计算综合风险指标"""
        try:
            # 计算资金利用率
            total_position_value = sum(pos.value_usd for pos in self.positions.values())
            utilization_rate = total_position_value / self.current_capital
            
            # 计算最大回撤
            max_drawdown, current_drawdown = self._calculate_drawdown()
            
            # 计算VaR
            var_1d = self._calculate_var()
            
            # 计算夏普比率
            sharpe_ratio = self._calculate_sharpe_ratio()
            
            # 计算各资产敞口
            exposure_by_asset = self._calculate_exposure_by_asset()
            
            # 计算风险评分
            risk_score = self._calculate_risk_score(
                utilization_rate, max_drawdown, var_1d
            )
            
            return RiskMetrics(
                total_capital=self.current_capital,
                used_capital=total_position_value,
                available_capital=self.current_capital - total_position_value,
                utilization_rate=utilization_rate,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                var_1d=var_1d,
                sharpe_ratio=sharpe_ratio,
                risk_score=risk_score,
                exposure_by_asset=exposure_by_asset
            )
            
        except Exception as e:
            logger.error(f"计算风险指标失败: {e}")
            return self._get_default_risk_metrics()
    
    def evaluate_arbitrage_opportunity(
        self, 
        symbol: str,
        buy_exchange: str,
        sell_exchange: str,
        buy_price: float,
        sell_price: float,
        volume_24h: float = 0,
        liquidity_score: float = 0.5
    ) -> ArbitrageOpportunity:
        """评估套利机会"""
        try:
            # 计算价差
            spread_percent = ((sell_price - buy_price) / buy_price) * 100
            
            # 评估执行难度
            execution_difficulty = self._assess_execution_difficulty(
                volume_24h, liquidity_score, spread_percent
            )
            
            # 评估风险等级
            risk_level = self._assess_risk_level(symbol, spread_percent)
            
            # 计算推荐金额
            recommended_amount = self._calculate_recommended_amount(
                symbol, risk_level, spread_percent
            )
            
            # 计算预期利润
            expected_profit = recommended_amount * (spread_percent / 100) * 0.95  # 扣除手续费
            
            # 计算信心评分
            confidence_score = self._calculate_confidence_score(
                spread_percent, liquidity_score, execution_difficulty
            )
            
            return ArbitrageOpportunity(
                symbol=symbol,
                buy_exchange=buy_exchange,
                sell_exchange=sell_exchange,
                buy_price=buy_price,
                sell_price=sell_price,
                spread_percent=spread_percent,
                expected_profit=expected_profit,
                execution_difficulty=execution_difficulty,
                risk_level=risk_level,
                recommended_amount=recommended_amount,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"评估套利机会失败: {e}")
            return None
    
    def check_risk_limits(self, symbol: str, amount: float) -> Tuple[bool, str]:
        """检查风险限制"""
        try:
            # 检查资金利用率
            position_value = amount
            current_utilization = sum(pos.value_usd for pos in self.positions.values())
            new_utilization = (current_utilization + position_value) / self.current_capital
            
            if new_utilization > self.max_utilization:
                return False, f"超过最大资金利用率限制 ({self.max_utilization*100:.1f}%)"
            
            # 检查单个持仓大小
            position_ratio = position_value / self.current_capital
            if position_ratio > self.max_position_size:
                return False, f"单个持仓超过限制 ({self.max_position_size*100:.1f}%)"
            
            # 检查最大回撤
            _, current_drawdown = self._calculate_drawdown()
            if current_drawdown > self.max_drawdown_limit:
                return False, f"当前回撤超过限制 ({self.max_drawdown_limit*100:.1f}%)"
            
            return True, "风险检查通过"
            
        except Exception as e:
            logger.error(f"风险检查失败: {e}")
            return False, f"风险检查错误: {str(e)}"
    
    def _calculate_drawdown(self) -> Tuple[float, float]:
        """计算最大回撤和当前回撤"""
        if len(self.pnl_history) < 2:
            return 0.0, 0.0
        
        # 计算累计收益曲线
        values = [self.initial_capital + pnl for _, pnl in self.pnl_history]
        peak = values[0]
        max_drawdown = 0.0
        current_drawdown = 0.0
        
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # 当前回撤
        current_value = values[-1]
        current_peak = max(values)
        current_drawdown = (current_peak - current_value) / current_peak
        
        return max_drawdown, current_drawdown
    
    def _calculate_var(self, days: int = 1) -> float:
        """计算风险价值(VaR)"""
        if len(self.pnl_history) < 30:
            return 0.0
        
        # 计算日收益率
        returns = []
        for i in range(1, len(self.pnl_history)):
            prev_value = self.initial_capital + self.pnl_history[i-1][1]
            curr_value = self.initial_capital + self.pnl_history[i][1]
            daily_return = (curr_value - prev_value) / prev_value
            returns.append(daily_return)
        
        # 计算VaR
        returns_array = np.array(returns)
        var_percentile = (1 - self.confidence_level) * 100
        var = np.percentile(returns_array, var_percentile)
        
        return abs(var * self.current_capital * np.sqrt(days))
    
    def _calculate_sharpe_ratio(self) -> float:
        """计算夏普比率"""
        if len(self.pnl_history) < 30:
            return 0.0
        
        # 计算日收益率
        returns = []
        for i in range(1, len(self.pnl_history)):
            prev_value = self.initial_capital + self.pnl_history[i-1][1]
            curr_value = self.initial_capital + self.pnl_history[i][1]
            daily_return = (curr_value - prev_value) / prev_value
            returns.append(daily_return)
        
        returns_array = np.array(returns)
        excess_returns = returns_array - (self.risk_free_rate / 365)
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(365)
    
    def _calculate_exposure_by_asset(self) -> Dict[str, float]:
        """计算各资产敞口"""
        exposure = {}
        for symbol, position in self.positions.items():
            asset = symbol.split('/')[0]  # 提取基础资产
            if asset not in exposure:
                exposure[asset] = 0
            exposure[asset] += position.value_usd
        
        return exposure
    
    def _calculate_risk_score(self, utilization: float, drawdown: float, var: float) -> int:
        """计算综合风险评分(1-10, 10为最高风险)"""
        score = 1
        
        # 资金利用率评分
        if utilization > 0.8:
            score += 3
        elif utilization > 0.6:
            score += 2
        elif utilization > 0.4:
            score += 1
        
        # 回撤评分
        if drawdown > 0.15:
            score += 3
        elif drawdown > 0.10:
            score += 2
        elif drawdown > 0.05:
            score += 1
        
        # VaR评分
        var_ratio = var / self.current_capital
        if var_ratio > 0.05:
            score += 3
        elif var_ratio > 0.03:
            score += 2
        elif var_ratio > 0.01:
            score += 1
        
        return min(score, 10)
    
    def _assess_execution_difficulty(self, volume_24h: float, liquidity_score: float, spread: float) -> str:
        """评估执行难度"""
        if volume_24h > 10000000 and liquidity_score > 0.8 and spread < 5:
            return "easy"
        elif volume_24h > 1000000 and liquidity_score > 0.5 and spread < 10:
            return "medium"
        else:
            return "hard"
    
    def _assess_risk_level(self, symbol: str, spread: float) -> str:
        """评估风险等级"""
        if spread > 10:
            return "high"  # 价差过大可能有问题
        elif spread > 2:
            return "medium"
        else:
            return "low"
    
    def _calculate_recommended_amount(self, symbol: str, risk_level: str, spread: float) -> float:
        """计算推荐交易金额"""
        base_amount = self.current_capital * 0.05  # 基础5%
        
        # 根据风险等级调整
        if risk_level == "low":
            multiplier = 2.0
        elif risk_level == "medium":
            multiplier = 1.0
        else:
            multiplier = 0.5
        
        # 根据价差调整
        if spread > 5:
            multiplier *= 0.5
        elif spread > 2:
            multiplier *= 0.8
        
        return min(base_amount * multiplier, self.current_capital * self.max_position_size)
    
    def _calculate_confidence_score(self, spread: float, liquidity: float, difficulty: str) -> float:
        """计算信心评分"""
        score = 0.5
        
        # 价差评分
        if 1 < spread < 5:
            score += 0.3
        elif spread >= 5:
            score += 0.1
        
        # 流动性评分
        score += liquidity * 0.3
        
        # 执行难度评分
        if difficulty == "easy":
            score += 0.2
        elif difficulty == "medium":
            score += 0.1
        
        return min(score, 1.0)
    
    def _get_default_risk_metrics(self) -> RiskMetrics:
        """获取默认风险指标"""
        return RiskMetrics(
            total_capital=self.current_capital,
            used_capital=0,
            available_capital=self.current_capital,
            utilization_rate=0,
            max_drawdown=0,
            current_drawdown=0,
            var_1d=0,
            sharpe_ratio=0,
            risk_score=1,
            exposure_by_asset={}
        )
    
    def update_position(self, symbol: str, exchange: str, amount: float, 
                       entry_price: float, current_price: float):
        """更新持仓信息"""
        value_usd = amount * current_price
        pnl = amount * (current_price - entry_price)
        pnl_percent = ((current_price - entry_price) / entry_price) * 100
        
        # 评估风险等级
        if abs(pnl_percent) > 10:
            risk_level = "high"
        elif abs(pnl_percent) > 5:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        self.positions[f"{symbol}_{exchange}"] = PositionInfo(
            symbol=symbol,
            exchange=exchange,
            amount=amount,
            value_usd=value_usd,
            entry_price=entry_price,
            current_price=current_price,
            pnl=pnl,
            pnl_percent=pnl_percent,
            risk_level=risk_level
        )
    
    def update_pnl_history(self, total_pnl: float):
        """更新盈亏历史"""
        self.pnl_history.append((datetime.now(), total_pnl))
        
        # 只保留最近1000条记录
        if len(self.pnl_history) > 1000:
            self.pnl_history = self.pnl_history[-1000:]