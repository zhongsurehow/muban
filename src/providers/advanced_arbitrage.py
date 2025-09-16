"""
高级套利策略模块
实现三角套利、跨链套利、期现套利等专业策略
"""

import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from itertools import combinations, permutations

logger = logging.getLogger(__name__)

@dataclass
class TriangularArbitrageOpportunity:
    """三角套利机会"""
    path: List[str]  # 交易路径，如 ['BTC/USDT', 'ETH/BTC', 'ETH/USDT']
    exchanges: List[str]  # 对应的交易所
    prices: List[float]  # 对应的价格
    profit_rate: float  # 利润率
    required_capital: float  # 所需资金
    expected_profit: float  # 预期利润
    execution_time: float  # 预计执行时间(秒)
    risk_score: float  # 风险评分(0-1)
    confidence: float  # 信心度(0-1)

@dataclass
class CrossChainOpportunity:
    """跨链套利机会"""
    token: str  # 代币名称
    source_chain: str  # 源链
    target_chain: str  # 目标链
    source_price: float  # 源链价格
    target_price: float  # 目标链价格
    price_diff: float  # 价差
    bridge_fee: float  # 跨链手续费
    bridge_time: int  # 跨链时间(分钟)
    net_profit_rate: float  # 净利润率
    liquidity_score: float  # 流动性评分

@dataclass
class FuturesSpotOpportunity:
    """期现套利机会"""
    symbol: str  # 交易对
    spot_price: float  # 现货价格
    futures_price: float  # 期货价格
    spread: float  # 价差
    funding_rate: float  # 资金费率
    time_to_expiry: int  # 到期时间(天)
    annual_return: float  # 年化收益率
    strategy_type: str  # 策略类型: 'contango' 或 'backwardation'

class AdvancedArbitrageEngine:
    """高级套利策略引擎"""
    
    def __init__(self):
        self.supported_chains = ['ETH', 'BSC', 'POLYGON', 'ARBITRUM', 'OPTIMISM']
        self.bridge_fees = {
            ('ETH', 'BSC'): 0.001,
            ('ETH', 'POLYGON'): 0.0005,
            ('BSC', 'POLYGON'): 0.0008,
            # 更多跨链费用配置
        }
        self.min_profit_threshold = 0.005  # 最小利润阈值 0.5%
        
    async def find_triangular_arbitrage(
        self, 
        market_data: Dict[str, Dict[str, float]],
        base_currency: str = 'USDT',
        min_profit: float = 0.01
    ) -> List[TriangularArbitrageOpportunity]:
        """寻找三角套利机会"""
        opportunities = []
        
        try:
            # 获取所有可用的交易对
            symbols = list(market_data.keys())
            
            # 构建货币图
            currencies = set()
            for symbol in symbols:
                if '/' in symbol:
                    base, quote = symbol.split('/')
                    currencies.add(base)
                    currencies.add(quote)
            
            currencies = list(currencies)
            
            # 寻找三角套利路径
            for curr1 in currencies:
                if curr1 == base_currency:
                    continue
                    
                for curr2 in currencies:
                    if curr2 in [curr1, base_currency]:
                        continue
                    
                    # 构建三角路径: base_currency -> curr1 -> curr2 -> base_currency
                    path1 = f"{curr1}/{base_currency}"  # 买入curr1
                    path2 = f"{curr2}/{curr1}"         # 用curr1买curr2
                    path3 = f"{curr2}/{base_currency}"  # 卖出curr2
                    
                    if all(p in market_data for p in [path1, path2, path3]):
                        opportunity = await self._calculate_triangular_profit(
                            path1, path2, path3, market_data, min_profit
                        )
                        if opportunity:
                            opportunities.append(opportunity)
            
            # 按利润率排序
            opportunities.sort(key=lambda x: x.profit_rate, reverse=True)
            return opportunities[:10]  # 返回前10个机会
            
        except Exception as e:
            logger.error(f"三角套利计算失败: {e}")
            return []
    
    async def _calculate_triangular_profit(
        self,
        path1: str, path2: str, path3: str,
        market_data: Dict[str, Dict[str, float]],
        min_profit: float
    ) -> Optional[TriangularArbitrageOpportunity]:
        """计算三角套利利润"""
        try:
            # 获取价格数据
            price1 = market_data[path1].get('price', 0)
            price2 = market_data[path2].get('price', 0)
            price3 = market_data[path3].get('price', 0)
            
            if not all([price1, price2, price3]):
                return None
            
            # 计算套利路径
            # 假设初始资金为1000 USDT
            initial_amount = 1000
            
            # 第一步: USDT -> curr1
            amount1 = initial_amount / price1
            
            # 第二步: curr1 -> curr2
            amount2 = amount1 * price2
            
            # 第三步: curr2 -> USDT
            final_amount = amount2 * price3
            
            # 计算利润率
            profit_rate = (final_amount - initial_amount) / initial_amount
            
            if profit_rate < min_profit:
                return None
            
            # 计算风险评分和信心度
            risk_score = self._calculate_triangular_risk(price1, price2, price3)
            confidence = self._calculate_confidence(profit_rate, risk_score)
            
            return TriangularArbitrageOpportunity(
                path=[path1, path2, path3],
                exchanges=['Exchange1', 'Exchange2', 'Exchange3'],  # 实际应用中从数据获取
                prices=[price1, price2, price3],
                profit_rate=profit_rate,
                required_capital=initial_amount,
                expected_profit=final_amount - initial_amount,
                execution_time=30,  # 预计30秒执行
                risk_score=risk_score,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"三角套利计算错误: {e}")
            return None
    
    def _calculate_triangular_risk(self, price1: float, price2: float, price3: float) -> float:
        """计算三角套利风险评分"""
        # 基于价格波动性计算风险
        prices = [price1, price2, price3]
        volatility = np.std(prices) / np.mean(prices)
        
        # 风险评分: 0-1, 1为最高风险
        risk_score = min(volatility * 10, 1.0)
        return risk_score
    
    def _calculate_confidence(self, profit_rate: float, risk_score: float) -> float:
        """计算信心度"""
        # 基于利润率和风险的信心度计算
        base_confidence = min(profit_rate * 20, 1.0)  # 利润率越高信心越大
        risk_penalty = risk_score * 0.5  # 风险越高信心越低
        
        confidence = max(base_confidence - risk_penalty, 0.1)
        return min(confidence, 1.0)
    
    async def find_cross_chain_arbitrage(
        self,
        token_prices: Dict[str, Dict[str, float]]  # {chain: {token: price}}
    ) -> List[CrossChainOpportunity]:
        """寻找跨链套利机会"""
        opportunities = []
        
        try:
            # 获取所有代币
            all_tokens = set()
            for chain_data in token_prices.values():
                all_tokens.update(chain_data.keys())
            
            # 对每个代币寻找跨链套利机会
            for token in all_tokens:
                chain_prices = {}
                for chain, prices in token_prices.items():
                    if token in prices:
                        chain_prices[chain] = prices[token]
                
                if len(chain_prices) < 2:
                    continue
                
                # 寻找价差最大的链对
                for source_chain, target_chain in combinations(chain_prices.keys(), 2):
                    source_price = chain_prices[source_chain]
                    target_price = chain_prices[target_chain]
                    
                    # 计算价差
                    price_diff = abs(target_price - source_price) / source_price
                    
                    if price_diff > self.min_profit_threshold:
                        # 获取跨链费用
                        bridge_fee = self.bridge_fees.get((source_chain, target_chain), 0.002)
                        
                        # 计算净利润
                        if target_price > source_price:
                            net_profit_rate = (target_price - source_price) / source_price - bridge_fee
                            direction = (source_chain, target_chain)
                        else:
                            net_profit_rate = (source_price - target_price) / target_price - bridge_fee
                            direction = (target_chain, source_chain)
                        
                        if net_profit_rate > 0:
                            opportunity = CrossChainOpportunity(
                                token=token,
                                source_chain=direction[0],
                                target_chain=direction[1],
                                source_price=chain_prices[direction[0]],
                                target_price=chain_prices[direction[1]],
                                price_diff=price_diff,
                                bridge_fee=bridge_fee,
                                bridge_time=self._get_bridge_time(direction[0], direction[1]),
                                net_profit_rate=net_profit_rate,
                                liquidity_score=0.7  # 默认流动性评分
                            )
                            opportunities.append(opportunity)
            
            # 按净利润率排序
            opportunities.sort(key=lambda x: x.net_profit_rate, reverse=True)
            return opportunities[:5]
            
        except Exception as e:
            logger.error(f"跨链套利计算失败: {e}")
            return []
    
    def _get_bridge_time(self, source_chain: str, target_chain: str) -> int:
        """获取跨链时间(分钟)"""
        bridge_times = {
            ('ETH', 'BSC'): 15,
            ('ETH', 'POLYGON'): 30,
            ('BSC', 'POLYGON'): 10,
        }
        return bridge_times.get((source_chain, target_chain), 20)
    
    async def find_futures_spot_arbitrage(
        self,
        spot_prices: Dict[str, float],
        futures_data: Dict[str, Dict[str, Any]]  # {symbol: {price, funding_rate, expiry}}
    ) -> List[FuturesSpotOpportunity]:
        """寻找期现套利机会"""
        opportunities = []
        
        try:
            for symbol in spot_prices:
                if symbol not in futures_data:
                    continue
                
                spot_price = spot_prices[symbol]
                futures_info = futures_data[symbol]
                futures_price = futures_info.get('price', 0)
                funding_rate = futures_info.get('funding_rate', 0)
                expiry_days = futures_info.get('expiry_days', 30)
                
                if not futures_price:
                    continue
                
                # 计算价差
                spread = (futures_price - spot_price) / spot_price
                
                # 判断策略类型
                if spread > 0:
                    strategy_type = 'contango'  # 期货溢价，做空期货买入现货
                    annual_return = (spread - funding_rate * expiry_days / 365) * (365 / expiry_days)
                else:
                    strategy_type = 'backwardation'  # 期货贴水，买入期货卖出现货
                    annual_return = (-spread + funding_rate * expiry_days / 365) * (365 / expiry_days)
                
                # 只考虑年化收益率大于5%的机会
                if annual_return > 0.05:
                    opportunity = FuturesSpotOpportunity(
                        symbol=symbol,
                        spot_price=spot_price,
                        futures_price=futures_price,
                        spread=spread,
                        funding_rate=funding_rate,
                        time_to_expiry=expiry_days,
                        annual_return=annual_return,
                        strategy_type=strategy_type
                    )
                    opportunities.append(opportunity)
            
            # 按年化收益率排序
            opportunities.sort(key=lambda x: x.annual_return, reverse=True)
            return opportunities[:5]
            
        except Exception as e:
            logger.error(f"期现套利计算失败: {e}")
            return []
    
    def calculate_optimal_position_size(
        self,
        opportunity: Any,
        available_capital: float,
        risk_tolerance: float = 0.02
    ) -> float:
        """计算最优仓位大小"""
        try:
            if isinstance(opportunity, TriangularArbitrageOpportunity):
                # 三角套利仓位计算
                max_position = available_capital * 0.1  # 最大10%资金
                risk_adjusted = max_position * (1 - opportunity.risk_score)
                return min(risk_adjusted, opportunity.required_capital)
                
            elif isinstance(opportunity, CrossChainOpportunity):
                # 跨链套利仓位计算
                max_position = available_capital * 0.05  # 最大5%资金(风险较高)
                return max_position * opportunity.liquidity_score
                
            elif isinstance(opportunity, FuturesSpotOpportunity):
                # 期现套利仓位计算
                max_position = available_capital * 0.2  # 最大20%资金
                volatility_factor = 1 / (1 + abs(opportunity.spread))
                return max_position * volatility_factor
                
            return 0
            
        except Exception as e:
            logger.error(f"仓位计算失败: {e}")
            return 0
    
    def generate_execution_plan(self, opportunity: Any) -> Dict[str, Any]:
        """生成执行计划"""
        try:
            if isinstance(opportunity, TriangularArbitrageOpportunity):
                return {
                    'type': 'triangular',
                    'steps': [
                        {'action': 'buy', 'symbol': opportunity.path[0], 'price': opportunity.prices[0]},
                        {'action': 'sell', 'symbol': opportunity.path[1], 'price': opportunity.prices[1]},
                        {'action': 'sell', 'symbol': opportunity.path[2], 'price': opportunity.prices[2]}
                    ],
                    'estimated_time': opportunity.execution_time,
                    'risk_level': 'high' if opportunity.risk_score > 0.7 else 'medium' if opportunity.risk_score > 0.3 else 'low'
                }
                
            elif isinstance(opportunity, CrossChainOpportunity):
                return {
                    'type': 'cross_chain',
                    'steps': [
                        {'action': 'buy', 'chain': opportunity.source_chain, 'token': opportunity.token},
                        {'action': 'bridge', 'from': opportunity.source_chain, 'to': opportunity.target_chain},
                        {'action': 'sell', 'chain': opportunity.target_chain, 'token': opportunity.token}
                    ],
                    'estimated_time': opportunity.bridge_time * 60,  # 转换为秒
                    'risk_level': 'high'  # 跨链风险较高
                }
                
            elif isinstance(opportunity, FuturesSpotOpportunity):
                return {
                    'type': 'futures_spot',
                    'steps': [
                        {'action': 'buy_spot' if opportunity.strategy_type == 'contango' else 'sell_spot', 'symbol': opportunity.symbol},
                        {'action': 'sell_futures' if opportunity.strategy_type == 'contango' else 'buy_futures', 'symbol': opportunity.symbol}
                    ],
                    'estimated_time': 60,  # 1分钟执行
                    'risk_level': 'medium'
                }
                
            return {}
            
        except Exception as e:
            logger.error(f"执行计划生成失败: {e}")
            return {}

# 全局实例
advanced_arbitrage_engine = AdvancedArbitrageEngine()