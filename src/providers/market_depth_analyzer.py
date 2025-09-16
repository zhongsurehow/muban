"""
市场深度分析模块
提供订单簿分析、流动性评估、冲击成本计算等功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import logging

logger = logging.getLogger(__name__)

@dataclass
class OrderBookLevel:
    """订单簿层级数据"""
    price: float
    volume: float
    cumulative_volume: float
    
@dataclass
class OrderBookSnapshot:
    """订单簿快照"""
    symbol: str
    exchange: str
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    spread: float
    mid_price: float

@dataclass
class LiquidityMetrics:
    """流动性指标"""
    symbol: str
    exchange: str
    spread_bps: float  # 价差基点
    depth_1pct: float  # 1%深度
    depth_5pct: float  # 5%深度
    depth_10pct: float  # 10%深度
    volume_imbalance: float  # 成交量不平衡
    price_impact_1k: float  # 1000 USDT冲击成本
    price_impact_10k: float  # 10000 USDT冲击成本
    price_impact_100k: float  # 100000 USDT冲击成本
    liquidity_score: float  # 流动性评分

@dataclass
class MarketImpactAnalysis:
    """市场冲击分析"""
    symbol: str
    exchange: str
    side: str  # 'buy' or 'sell'
    trade_size_usd: float
    expected_price: float
    actual_price: float
    slippage_bps: float
    impact_cost_usd: float
    confidence_level: float

class MarketDepthAnalyzer:
    """市场深度分析器"""
    
    def __init__(self):
        self.orderbook_cache = {}
        self.liquidity_history = {}
        self.impact_models = {}
        
    async def analyze_orderbook(self, orderbook_data: Dict, symbol: str, exchange: str) -> OrderBookSnapshot:
        """分析订单簿数据"""
        try:
            # 解析订单簿数据
            bids_data = orderbook_data.get('bids', [])
            asks_data = orderbook_data.get('asks', [])
            
            if not bids_data or not asks_data:
                raise ValueError(f"订单簿数据不完整: {symbol}@{exchange}")
            
            # 处理买单数据
            bids = []
            cumulative_bid_volume = 0
            for price, volume in bids_data[:50]:  # 取前50档
                cumulative_bid_volume += volume
                bids.append(OrderBookLevel(
                    price=float(price),
                    volume=float(volume),
                    cumulative_volume=cumulative_bid_volume
                ))
            
            # 处理卖单数据
            asks = []
            cumulative_ask_volume = 0
            for price, volume in asks_data[:50]:  # 取前50档
                cumulative_ask_volume += volume
                asks.append(OrderBookLevel(
                    price=float(price),
                    volume=float(volume),
                    cumulative_volume=cumulative_ask_volume
                ))
            
            # 计算基本指标
            best_bid = bids[0].price if bids else 0
            best_ask = asks[0].price if asks else 0
            spread = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2
            
            snapshot = OrderBookSnapshot(
                symbol=symbol,
                exchange=exchange,
                timestamp=datetime.now(),
                bids=bids,
                asks=asks,
                spread=spread,
                mid_price=mid_price
            )
            
            # 缓存订单簿数据
            cache_key = f"{exchange}:{symbol}"
            self.orderbook_cache[cache_key] = snapshot
            
            return snapshot
            
        except Exception as e:
            logger.error(f"分析订单簿失败 {symbol}@{exchange}: {str(e)}")
            raise
    
    def calculate_liquidity_metrics(self, snapshot: OrderBookSnapshot) -> LiquidityMetrics:
        """计算流动性指标"""
        try:
            # 价差基点
            spread_bps = (snapshot.spread / snapshot.mid_price) * 10000
            
            # 计算不同深度的流动性
            depth_1pct = self._calculate_depth_at_percentage(snapshot, 0.01)
            depth_5pct = self._calculate_depth_at_percentage(snapshot, 0.05)
            depth_10pct = self._calculate_depth_at_percentage(snapshot, 0.10)
            
            # 成交量不平衡
            total_bid_volume = sum(level.volume for level in snapshot.bids[:10])
            total_ask_volume = sum(level.volume for level in snapshot.asks[:10])
            volume_imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
            
            # 价格冲击计算
            impact_1k = self._calculate_price_impact(snapshot, 1000, 'buy')
            impact_10k = self._calculate_price_impact(snapshot, 10000, 'buy')
            impact_100k = self._calculate_price_impact(snapshot, 100000, 'buy')
            
            # 流动性评分 (0-100)
            liquidity_score = self._calculate_liquidity_score(
                spread_bps, depth_1pct, depth_5pct, impact_10k
            )
            
            return LiquidityMetrics(
                symbol=snapshot.symbol,
                exchange=snapshot.exchange,
                spread_bps=spread_bps,
                depth_1pct=depth_1pct,
                depth_5pct=depth_5pct,
                depth_10pct=depth_10pct,
                volume_imbalance=volume_imbalance,
                price_impact_1k=impact_1k,
                price_impact_10k=impact_10k,
                price_impact_100k=impact_100k,
                liquidity_score=liquidity_score
            )
            
        except Exception as e:
            logger.error(f"计算流动性指标失败: {str(e)}")
            raise
    
    def _calculate_depth_at_percentage(self, snapshot: OrderBookSnapshot, percentage: float) -> float:
        """计算指定百分比深度的流动性"""
        target_price_bid = snapshot.mid_price * (1 - percentage)
        target_price_ask = snapshot.mid_price * (1 + percentage)
        
        # 计算买单深度
        bid_depth = 0
        for level in snapshot.bids:
            if level.price >= target_price_bid:
                bid_depth += level.volume * level.price
            else:
                break
        
        # 计算卖单深度
        ask_depth = 0
        for level in snapshot.asks:
            if level.price <= target_price_ask:
                ask_depth += level.volume * level.price
            else:
                break
        
        return bid_depth + ask_depth
    
    def _calculate_price_impact(self, snapshot: OrderBookSnapshot, trade_size_usd: float, side: str) -> float:
        """计算价格冲击"""
        if side == 'buy':
            # 买入冲击 - 消耗卖单
            remaining_size = trade_size_usd
            weighted_price = 0
            total_volume = 0
            
            for level in snapshot.asks:
                level_value = level.volume * level.price
                if remaining_size <= level_value:
                    # 部分消耗这一档
                    volume_needed = remaining_size / level.price
                    weighted_price += volume_needed * level.price
                    total_volume += volume_needed
                    break
                else:
                    # 完全消耗这一档
                    weighted_price += level.volume * level.price
                    total_volume += level.volume
                    remaining_size -= level_value
            
            if total_volume > 0:
                avg_price = weighted_price / total_volume
                impact = (avg_price - snapshot.mid_price) / snapshot.mid_price
                return impact * 10000  # 转换为基点
        
        else:  # sell
            # 卖出冲击 - 消耗买单
            remaining_size = trade_size_usd
            weighted_price = 0
            total_volume = 0
            
            for level in snapshot.bids:
                level_value = level.volume * level.price
                if remaining_size <= level_value:
                    volume_needed = remaining_size / level.price
                    weighted_price += volume_needed * level.price
                    total_volume += volume_needed
                    break
                else:
                    weighted_price += level.volume * level.price
                    total_volume += level.volume
                    remaining_size -= level_value
            
            if total_volume > 0:
                avg_price = weighted_price / total_volume
                impact = (snapshot.mid_price - avg_price) / snapshot.mid_price
                return impact * 10000  # 转换为基点
        
        return 0
    
    def _calculate_liquidity_score(self, spread_bps: float, depth_1pct: float, 
                                 depth_5pct: float, impact_10k: float) -> float:
        """计算流动性评分"""
        # 价差评分 (越小越好)
        spread_score = max(0, 100 - spread_bps * 2)
        
        # 深度评分 (越大越好)
        depth_score = min(100, (depth_1pct + depth_5pct) / 10000)
        
        # 冲击评分 (越小越好)
        impact_score = max(0, 100 - impact_10k * 5)
        
        # 综合评分
        total_score = (spread_score * 0.3 + depth_score * 0.4 + impact_score * 0.3)
        return min(100, max(0, total_score))
    
    async def analyze_market_impact(self, symbol: str, exchange: str, 
                                  trade_size_usd: float, side: str) -> MarketImpactAnalysis:
        """分析市场冲击"""
        try:
            cache_key = f"{exchange}:{symbol}"
            if cache_key not in self.orderbook_cache:
                raise ValueError(f"没有找到订单簿数据: {symbol}@{exchange}")
            
            snapshot = self.orderbook_cache[cache_key]
            
            # 计算预期价格和实际价格
            expected_price = snapshot.mid_price
            slippage_bps = self._calculate_price_impact(snapshot, trade_size_usd, side)
            
            if side == 'buy':
                actual_price = expected_price * (1 + slippage_bps / 10000)
            else:
                actual_price = expected_price * (1 - slippage_bps / 10000)
            
            impact_cost_usd = abs(actual_price - expected_price) * (trade_size_usd / actual_price)
            
            # 计算信心水平
            confidence_level = self._calculate_confidence_level(snapshot, trade_size_usd)
            
            return MarketImpactAnalysis(
                symbol=symbol,
                exchange=exchange,
                side=side,
                trade_size_usd=trade_size_usd,
                expected_price=expected_price,
                actual_price=actual_price,
                slippage_bps=slippage_bps,
                impact_cost_usd=impact_cost_usd,
                confidence_level=confidence_level
            )
            
        except Exception as e:
            logger.error(f"分析市场冲击失败: {str(e)}")
            raise
    
    def _calculate_confidence_level(self, snapshot: OrderBookSnapshot, trade_size_usd: float) -> float:
        """计算信心水平"""
        # 基于订单簿深度和交易规模计算信心水平
        total_depth = sum(level.volume * level.price for level in snapshot.bids[:10])
        total_depth += sum(level.volume * level.price for level in snapshot.asks[:10])
        
        depth_ratio = total_depth / trade_size_usd
        
        if depth_ratio >= 10:
            return 0.95
        elif depth_ratio >= 5:
            return 0.85
        elif depth_ratio >= 2:
            return 0.70
        elif depth_ratio >= 1:
            return 0.50
        else:
            return 0.30
    
    def compare_liquidity_across_exchanges(self, symbol: str, exchanges: List[str]) -> Dict[str, LiquidityMetrics]:
        """比较不同交易所的流动性"""
        results = {}
        
        for exchange in exchanges:
            cache_key = f"{exchange}:{symbol}"
            if cache_key in self.orderbook_cache:
                snapshot = self.orderbook_cache[cache_key]
                metrics = self.calculate_liquidity_metrics(snapshot)
                results[exchange] = metrics
        
        return results
    
    def get_best_execution_venue(self, symbol: str, trade_size_usd: float, 
                                side: str, exchanges: List[str]) -> Tuple[str, float]:
        """获取最佳执行场所"""
        best_exchange = None
        best_cost = float('inf')
        
        for exchange in exchanges:
            try:
                cache_key = f"{exchange}:{symbol}"
                if cache_key not in self.orderbook_cache:
                    continue
                
                snapshot = self.orderbook_cache[cache_key]
                impact_bps = self._calculate_price_impact(snapshot, trade_size_usd, side)
                
                # 考虑手续费 (简化为0.1%)
                fee_bps = 10
                total_cost = impact_bps + fee_bps
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_exchange = exchange
                    
            except Exception as e:
                logger.warning(f"评估交易所 {exchange} 失败: {str(e)}")
                continue
        
        return best_exchange, best_cost
    
    def generate_execution_strategy(self, symbol: str, total_size_usd: float, 
                                  side: str, exchanges: List[str]) -> List[Dict]:
        """生成执行策略"""
        strategy = []
        
        # 获取所有交易所的流动性数据
        liquidity_data = self.compare_liquidity_across_exchanges(symbol, exchanges)
        
        if not liquidity_data:
            return strategy
        
        # 按流动性评分排序
        sorted_exchanges = sorted(
            liquidity_data.items(),
            key=lambda x: x[1].liquidity_score,
            reverse=True
        )
        
        remaining_size = total_size_usd
        
        for exchange, metrics in sorted_exchanges:
            if remaining_size <= 0:
                break
            
            # 计算在该交易所的最优执行规模
            optimal_size = min(remaining_size, metrics.depth_1pct * 0.5)
            
            if optimal_size > 100:  # 最小执行规模
                strategy.append({
                    'exchange': exchange,
                    'size_usd': optimal_size,
                    'percentage': (optimal_size / total_size_usd) * 100,
                    'expected_impact_bps': self._calculate_price_impact(
                        self.orderbook_cache[f"{exchange}:{symbol}"],
                        optimal_size,
                        side
                    ),
                    'liquidity_score': metrics.liquidity_score
                })
                
                remaining_size -= optimal_size
        
        return strategy

# 全局实例
market_depth_analyzer = MarketDepthAnalyzer()