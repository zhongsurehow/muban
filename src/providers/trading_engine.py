"""
交易执行模块
包含自动下单、订单管理、滑点控制等功能
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

@dataclass
class Order:
    """订单对象"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    exchange: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: float = 0.0
    commission: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    execution_time: Optional[float] = None
    error_message: Optional[str] = None

@dataclass
class ExecutionResult:
    """执行结果"""
    success: bool
    order_id: str
    filled_quantity: float = 0.0
    average_price: float = 0.0
    commission: float = 0.0
    execution_time: float = 0.0
    error_message: Optional[str] = None
    slippage: float = 0.0

@dataclass
class SlippageConfig:
    """滑点配置"""
    max_slippage_percent: float = 0.5  # 最大滑点百分比
    price_impact_threshold: float = 0.1  # 价格冲击阈值
    adaptive_slippage: bool = True  # 自适应滑点
    volatility_multiplier: float = 2.0  # 波动率乘数

class TradingEngine:
    """交易执行引擎"""
    
    def __init__(self):
        self.orders: Dict[str, Order] = {}
        self.execution_history: List[ExecutionResult] = []
        self.slippage_config = SlippageConfig()
        self.commission_rates = {
            'binance': 0.001,
            'okx': 0.001,
            'bybit': 0.001,
            'kucoin': 0.001,
            'gate': 0.002,
            'mexc': 0.002,
            'bitget': 0.001,
            'coinex': 0.001
        }
        self.min_order_sizes = {
            'BTC': 0.00001,
            'ETH': 0.0001,
            'BNB': 0.001,
            'default': 0.01
        }
        
    async def execute_arbitrage_strategy(
        self,
        buy_order: Dict[str, Any],
        sell_order: Dict[str, Any],
        max_slippage: float = 0.005
    ) -> Tuple[ExecutionResult, ExecutionResult]:
        """执行套利策略"""
        try:
            # 创建买入和卖出订单
            buy_order_obj = self._create_order_from_dict(buy_order)
            sell_order_obj = self._create_order_from_dict(sell_order)
            
            # 验证订单
            buy_validation = self._validate_order(buy_order_obj)
            sell_validation = self._validate_order(sell_order_obj)
            
            if not buy_validation['valid']:
                return ExecutionResult(
                    success=False,
                    order_id=buy_order_obj.id,
                    error_message=f"买入订单验证失败: {buy_validation['error']}"
                ), ExecutionResult(
                    success=False,
                    order_id=sell_order_obj.id,
                    error_message="买入订单验证失败，跳过卖出"
                )
            
            if not sell_validation['valid']:
                return ExecutionResult(
                    success=False,
                    order_id=buy_order_obj.id,
                    error_message="卖出订单验证失败，跳过买入"
                ), ExecutionResult(
                    success=False,
                    order_id=sell_order_obj.id,
                    error_message=f"卖出订单验证失败: {sell_validation['error']}"
                )
            
            # 并行执行订单
            start_time = time.time()
            
            buy_task = asyncio.create_task(self._execute_single_order(buy_order_obj, max_slippage))
            sell_task = asyncio.create_task(self._execute_single_order(sell_order_obj, max_slippage))
            
            buy_result, sell_result = await asyncio.gather(buy_task, sell_task, return_exceptions=True)
            
            execution_time = time.time() - start_time
            
            # 处理异常
            if isinstance(buy_result, Exception):
                buy_result = ExecutionResult(
                    success=False,
                    order_id=buy_order_obj.id,
                    error_message=str(buy_result)
                )
            
            if isinstance(sell_result, Exception):
                sell_result = ExecutionResult(
                    success=False,
                    order_id=sell_order_obj.id,
                    error_message=str(sell_result)
                )
            
            # 更新执行时间
            buy_result.execution_time = execution_time
            sell_result.execution_time = execution_time
            
            # 记录执行历史
            self.execution_history.extend([buy_result, sell_result])
            
            return buy_result, sell_result
            
        except Exception as e:
            logger.error(f"套利策略执行失败: {e}")
            error_result = ExecutionResult(
                success=False,
                order_id="unknown",
                error_message=str(e)
            )
            return error_result, error_result
    
    def _create_order_from_dict(self, order_data: Dict[str, Any]) -> Order:
        """从字典创建订单对象"""
        return Order(
            symbol=order_data.get('symbol', ''),
            exchange=order_data.get('exchange', ''),
            side=OrderSide(order_data.get('side', 'buy')),
            order_type=OrderType(order_data.get('type', 'market')),
            quantity=float(order_data.get('quantity', 0)),
            price=order_data.get('price'),
            stop_price=order_data.get('stop_price')
        )
    
    def _validate_order(self, order: Order) -> Dict[str, Any]:
        """验证订单"""
        try:
            # 检查基本字段
            if not order.symbol:
                return {'valid': False, 'error': '交易对不能为空'}
            
            if not order.exchange:
                return {'valid': False, 'error': '交易所不能为空'}
            
            if order.quantity <= 0:
                return {'valid': False, 'error': '数量必须大于0'}
            
            # 检查最小订单大小
            base_currency = order.symbol.split('/')[0] if '/' in order.symbol else order.symbol
            min_size = self.min_order_sizes.get(base_currency, self.min_order_sizes['default'])
            
            if order.quantity < min_size:
                return {'valid': False, 'error': f'订单数量小于最小值 {min_size}'}
            
            # 检查限价单价格
            if order.order_type == OrderType.LIMIT and not order.price:
                return {'valid': False, 'error': '限价单必须指定价格'}
            
            # 检查止损单价格
            if order.order_type == OrderType.STOP_LOSS and not order.stop_price:
                return {'valid': False, 'error': '止损单必须指定止损价格'}
            
            return {'valid': True, 'error': None}
            
        except Exception as e:
            return {'valid': False, 'error': f'验证过程出错: {str(e)}'}
    
    async def _execute_single_order(self, order: Order, max_slippage: float) -> ExecutionResult:
        """执行单个订单"""
        try:
            start_time = time.time()
            
            # 更新订单状态
            order.status = OrderStatus.SUBMITTED
            order.updated_at = datetime.now()
            self.orders[order.id] = order
            
            # 模拟订单执行延迟
            await asyncio.sleep(0.1 + np.random.uniform(0, 0.2))
            
            # 计算滑点
            slippage = self._calculate_slippage(order)
            
            if slippage > max_slippage:
                order.status = OrderStatus.FAILED
                order.error_message = f"滑点过大: {slippage:.3%} > {max_slippage:.3%}"
                return ExecutionResult(
                    success=False,
                    order_id=order.id,
                    error_message=order.error_message,
                    slippage=slippage
                )
            
            # 模拟订单成交
            execution_price = self._calculate_execution_price(order, slippage)
            commission = self._calculate_commission(order, execution_price)
            
            # 更新订单状态
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.average_price = execution_price
            order.commission = commission
            order.execution_time = time.time() - start_time
            order.updated_at = datetime.now()
            
            return ExecutionResult(
                success=True,
                order_id=order.id,
                filled_quantity=order.filled_quantity,
                average_price=order.average_price,
                commission=commission,
                execution_time=order.execution_time,
                slippage=slippage
            )
            
        except Exception as e:
            order.status = OrderStatus.FAILED
            order.error_message = str(e)
            logger.error(f"订单执行失败 {order.id}: {e}")
            
            return ExecutionResult(
                success=False,
                order_id=order.id,
                error_message=str(e)
            )
    
    def _calculate_slippage(self, order: Order) -> float:
        """计算滑点"""
        try:
            # 基础滑点（模拟）
            base_slippage = np.random.uniform(0.0001, 0.003)
            
            # 根据订单大小调整滑点
            size_factor = min(order.quantity / 1000, 2.0)  # 订单越大滑点越大
            
            # 根据市场波动性调整滑点
            volatility_factor = np.random.uniform(0.5, 1.5)
            
            # 根据交易所调整滑点
            exchange_factor = {
                'binance': 0.8,
                'okx': 0.9,
                'bybit': 1.0,
                'kucoin': 1.1,
                'gate': 1.2,
                'mexc': 1.3,
                'bitget': 1.0,
                'coinex': 1.1
            }.get(order.exchange.lower(), 1.0)
            
            total_slippage = base_slippage * size_factor * volatility_factor * exchange_factor
            
            return min(total_slippage, self.slippage_config.max_slippage_percent / 100)
            
        except Exception as e:
            logger.error(f"滑点计算失败: {e}")
            return 0.001  # 默认滑点
    
    def _calculate_execution_price(self, order: Order, slippage: float) -> float:
        """计算执行价格"""
        try:
            if order.order_type == OrderType.MARKET:
                # 市价单：基于当前市价加上滑点
                base_price = order.price or 1000  # 模拟价格
                
                if order.side == OrderSide.BUY:
                    return base_price * (1 + slippage)
                else:
                    return base_price * (1 - slippage)
            
            elif order.order_type == OrderType.LIMIT:
                # 限价单：使用指定价格
                return order.price
            
            else:
                # 其他订单类型
                return order.price or 1000
                
        except Exception as e:
            logger.error(f"执行价格计算失败: {e}")
            return order.price or 1000
    
    def _calculate_commission(self, order: Order, execution_price: float) -> float:
        """计算手续费"""
        try:
            commission_rate = self.commission_rates.get(order.exchange.lower(), 0.001)
            return order.quantity * execution_price * commission_rate
        except Exception as e:
            logger.error(f"手续费计算失败: {e}")
            return 0
    
    async def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        try:
            if order_id not in self.orders:
                return False
            
            order = self.orders[order_id]
            
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.FAILED]:
                return False
            
            # 模拟取消延迟
            await asyncio.sleep(0.05)
            
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"取消订单失败 {order_id}: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """获取订单状态"""
        return self.orders.get(order_id)
    
    def get_active_orders(self) -> List[Order]:
        """获取活跃订单"""
        return [
            order for order in self.orders.values()
            if order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]
        ]
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """获取执行统计"""
        try:
            if not self.execution_history:
                return {}
            
            successful_executions = [r for r in self.execution_history if r.success]
            failed_executions = [r for r in self.execution_history if not r.success]
            
            total_executions = len(self.execution_history)
            success_rate = len(successful_executions) / total_executions if total_executions > 0 else 0
            
            avg_execution_time = np.mean([r.execution_time for r in successful_executions]) if successful_executions else 0
            avg_slippage = np.mean([r.slippage for r in successful_executions if r.slippage > 0]) if successful_executions else 0
            total_commission = sum([r.commission for r in successful_executions])
            
            return {
                'total_executions': total_executions,
                'successful_executions': len(successful_executions),
                'failed_executions': len(failed_executions),
                'success_rate': success_rate,
                'average_execution_time': avg_execution_time,
                'average_slippage': avg_slippage,
                'total_commission': total_commission,
                'last_24h_executions': len([
                    r for r in self.execution_history 
                    if hasattr(r, 'timestamp') and 
                    datetime.now() - getattr(r, 'timestamp', datetime.now()) < timedelta(days=1)
                ])
            }
            
        except Exception as e:
            logger.error(f"统计计算失败: {e}")
            return {}
    
    def optimize_execution_parameters(self) -> Dict[str, Any]:
        """优化执行参数"""
        try:
            stats = self.get_execution_statistics()
            
            recommendations = {}
            
            # 基于成功率优化
            if stats.get('success_rate', 0) < 0.9:
                recommendations['increase_slippage_tolerance'] = True
                recommendations['suggested_max_slippage'] = min(
                    self.slippage_config.max_slippage_percent * 1.2, 1.0
                )
            
            # 基于执行时间优化
            if stats.get('average_execution_time', 0) > 1.0:
                recommendations['use_market_orders'] = True
                recommendations['reduce_order_size'] = True
            
            # 基于滑点优化
            if stats.get('average_slippage', 0) > 0.005:
                recommendations['split_large_orders'] = True
                recommendations['use_limit_orders'] = True
            
            return recommendations
            
        except Exception as e:
            logger.error(f"参数优化失败: {e}")
            return {}

# 全局实例
trading_engine = TradingEngine()