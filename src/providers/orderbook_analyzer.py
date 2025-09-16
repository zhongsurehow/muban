import asyncio
import aiohttp
import time
from typing import Dict, List, Optional, Tuple
from cachetools import TTLCache
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrderBookAnalyzer:
    """订单簿深度与滑点分析器"""
    
    def __init__(self):
        self.last_request_time = {}
        self.rate_limit_delay = 0.5  # 500ms间隔
        self.cache = TTLCache(maxsize=200, ttl=30)  # 30秒缓存
        
        # 支持的交易所API端点
        self.exchanges = {
            'binance': {
                'orderbook_url': 'https://api.binance.com/api/v3/depth',
                'params': {'limit': 1000}
            },
            'okx': {
                'orderbook_url': 'https://www.okx.com/api/v5/market/books',
                'params': {'sz': 400}
            },
            'bybit': {
                'orderbook_url': 'https://api.bybit.com/v5/market/orderbook',
                'params': {'category': 'spot', 'limit': 200}
            },
            'gate': {
                'orderbook_url': 'https://api.gateio.ws/api/v4/spot/order_book',
                'params': {'limit': 100}
            },
            'kucoin': {
                'orderbook_url': 'https://api.kucoin.com/api/v1/market/orderbook/level2_100',
                'params': {}
            }
        }
    
    async def _check_rate_limit(self, exchange):
        """检查速率限制"""
        current_time = time.time()
        if exchange in self.last_request_time:
            time_diff = current_time - self.last_request_time[exchange]
            if time_diff < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - time_diff)
        
        self.last_request_time[exchange] = current_time
    
    def _normalize_symbol(self, symbol: str, exchange: str) -> str:
        """标准化交易对格式"""
        if exchange == 'binance':
            return symbol.replace('/', '').upper()
        elif exchange == 'okx':
            return symbol.replace('/', '-').upper()
        elif exchange == 'bybit':
            return symbol.replace('/', '').upper()
        elif exchange == 'gate':
            return symbol.replace('/', '_').lower()
        elif exchange == 'kucoin':
            return symbol.replace('/', '-').upper()
        return symbol
    
    async def get_orderbook(self, exchange: str, symbol: str) -> Optional[Dict]:
        """获取指定交易所的订单簿数据"""
        cache_key = f"{exchange}_{symbol}_orderbook"
        
        # 检查缓存
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if exchange not in self.exchanges:
            logger.warning(f"不支持的交易所: {exchange}")
            return None
        
        await self._check_rate_limit(exchange)
        
        try:
            config = self.exchanges[exchange]
            normalized_symbol = self._normalize_symbol(symbol, exchange)
            
            async with aiohttp.ClientSession() as session:
                params = config['params'].copy()
                
                if exchange == 'binance':
                    params['symbol'] = normalized_symbol
                elif exchange == 'okx':
                    params['instId'] = normalized_symbol
                elif exchange == 'bybit':
                    params['symbol'] = normalized_symbol
                elif exchange == 'gate':
                    url = f"{config['orderbook_url']}/{normalized_symbol}"
                elif exchange == 'kucoin':
                    url = f"{config['orderbook_url']}/{normalized_symbol}"
                
                if exchange not in ['gate', 'kucoin']:
                    url = config['orderbook_url']
                
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        orderbook = self._parse_orderbook(data, exchange)
                        
                        if orderbook:
                            self.cache[cache_key] = orderbook
                            return orderbook
                    else:
                        logger.error(f"获取{exchange}订单簿失败: HTTP {response.status}")
        
        except Exception as e:
            logger.error(f"获取{exchange}订单簿异常: {str(e)}")
        
        return None
    
    def _parse_orderbook(self, data: Dict, exchange: str) -> Optional[Dict]:
        """解析不同交易所的订单簿数据格式"""
        try:
            if exchange == 'binance':
                return {
                    'bids': [[float(price), float(qty)] for price, qty in data['bids']],
                    'asks': [[float(price), float(qty)] for price, qty in data['asks']],
                    'timestamp': int(time.time() * 1000)
                }
            
            elif exchange == 'okx':
                if 'data' in data and len(data['data']) > 0:
                    book_data = data['data'][0]
                    return {
                        'bids': [[float(price), float(qty)] for price, qty, _, _ in book_data['bids']],
                        'asks': [[float(price), float(qty)] for price, qty, _, _ in book_data['asks']],
                        'timestamp': int(book_data['ts'])
                    }
            
            elif exchange == 'bybit':
                if 'result' in data:
                    result = data['result']
                    return {
                        'bids': [[float(price), float(qty)] for price, qty in result['b']],
                        'asks': [[float(price), float(qty)] for price, qty in result['a']],
                        'timestamp': int(result['ts'])
                    }
            
            elif exchange == 'gate':
                return {
                    'bids': [[float(price), float(qty)] for price, qty in data['bids']],
                    'asks': [[float(price), float(qty)] for price, qty in data['asks']],
                    'timestamp': int(time.time() * 1000)
                }
            
            elif exchange == 'kucoin':
                if 'data' in data:
                    book_data = data['data']
                    return {
                        'bids': [[float(price), float(qty)] for price, qty in book_data['bids']],
                        'asks': [[float(price), float(qty)] for price, qty in book_data['asks']],
                        'timestamp': int(book_data['time'])
                    }
        
        except Exception as e:
            logger.error(f"解析{exchange}订单簿数据失败: {str(e)}")
        
        return None
    
    def calculate_slippage(self, orderbook: Dict, side: str, amount: float) -> Dict:
        """计算指定交易量的滑点
        
        Args:
            orderbook: 订单簿数据
            side: 'buy' 或 'sell'
            amount: 交易金额(USDT)
        
        Returns:
            包含滑点分析的字典
        """
        if not orderbook or side not in ['buy', 'sell']:
            return {}
        
        # 选择对应的订单簿一侧
        orders = orderbook['asks'] if side == 'buy' else orderbook['bids']
        
        if not orders:
            return {}
        
        # 计算加权平均价格和滑点
        total_cost = 0
        total_quantity = 0
        remaining_amount = amount
        filled_orders = []
        
        best_price = orders[0][0]  # 最优价格
        
        for price, quantity in orders:
            if remaining_amount <= 0:
                break
            
            # 计算这个价位能交易的数量
            max_qty_at_price = remaining_amount / price
            actual_qty = min(quantity, max_qty_at_price)
            
            cost = actual_qty * price
            total_cost += cost
            total_quantity += actual_qty
            remaining_amount -= cost
            
            filled_orders.append({
                'price': price,
                'quantity': actual_qty,
                'cost': cost
            })
        
        if total_quantity == 0:
            return {
                'error': '订单簿深度不足',
                'available_liquidity': sum(price * qty for price, qty in orders[:10])
            }
        
        # 计算平均成交价
        avg_price = total_cost / total_quantity
        
        # 计算滑点
        slippage_pct = abs(avg_price - best_price) / best_price * 100
        
        # 计算价格影响
        worst_price = filled_orders[-1]['price'] if filled_orders else best_price
        price_impact_pct = abs(worst_price - best_price) / best_price * 100
        
        return {
            'best_price': best_price,
            'average_price': avg_price,
            'worst_price': worst_price,
            'slippage_pct': slippage_pct,
            'price_impact_pct': price_impact_pct,
            'total_quantity': total_quantity,
            'total_cost': total_cost,
            'filled_orders': filled_orders,
            'unfilled_amount': remaining_amount,
            'fill_rate': (amount - remaining_amount) / amount * 100
        }
    
    async def analyze_cross_exchange_slippage(self, symbol: str, amount: float) -> Dict:
        """分析跨交易所的滑点情况"""
        results = {}
        
        # 并发获取多个交易所的订单簿
        tasks = []
        for exchange in self.exchanges.keys():
            tasks.append(self.get_orderbook(exchange, symbol))
        
        orderbooks = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 分析每个交易所的买卖滑点
        for i, exchange in enumerate(self.exchanges.keys()):
            orderbook = orderbooks[i]
            
            if isinstance(orderbook, Exception) or not orderbook:
                results[exchange] = {'error': '获取订单簿失败'}
                continue
            
            # 计算买入和卖出滑点
            buy_analysis = self.calculate_slippage(orderbook, 'buy', amount)
            sell_analysis = self.calculate_slippage(orderbook, 'sell', amount)
            
            results[exchange] = {
                'buy': buy_analysis,
                'sell': sell_analysis,
                'timestamp': orderbook.get('timestamp', int(time.time() * 1000))
            }
        
        return results
    
    def find_optimal_execution_strategy(self, slippage_analysis: Dict, total_amount: float) -> Dict:
        """寻找最优执行策略"""
        strategies = []
        
        # 单一交易所策略
        for exchange, data in slippage_analysis.items():
            if 'error' in data:
                continue
            
            for side in ['buy', 'sell']:
                if side in data and 'error' not in data[side]:
                    analysis = data[side]
                    strategies.append({
                        'type': 'single_exchange',
                        'exchange': exchange,
                        'side': side,
                        'amount': total_amount,
                        'avg_price': analysis['average_price'],
                        'slippage_pct': analysis['slippage_pct'],
                        'fill_rate': analysis['fill_rate']
                    })
        
        # 分割执行策略（简化版）
        valid_exchanges = [ex for ex, data in slippage_analysis.items() if 'error' not in data]
        
        if len(valid_exchanges) >= 2:
            # 按最优价格排序，分配交易量
            for side in ['buy', 'sell']:
                exchange_prices = []
                for exchange in valid_exchanges:
                    if side in slippage_analysis[exchange] and 'error' not in slippage_analysis[exchange][side]:
                        price = slippage_analysis[exchange][side]['best_price']
                        exchange_prices.append((exchange, price))
                
                if len(exchange_prices) >= 2:
                    # 按价格排序（买入按升序，卖出按降序）
                    exchange_prices.sort(key=lambda x: x[1], reverse=(side == 'sell'))
                    
                    # 简单的50-50分割策略
                    split_amount = total_amount / 2
                    total_cost = 0
                    weighted_slippage = 0
                    
                    for i, (exchange, _) in enumerate(exchange_prices[:2]):
                        analysis = slippage_analysis[exchange][side]
                        split_analysis = self.calculate_slippage(
                            {'asks' if side == 'buy' else 'bids': 
                             slippage_analysis[exchange][side].get('filled_orders', [])},
                            side, split_amount
                        )
                        
                        if split_analysis:
                            total_cost += split_analysis.get('total_cost', 0)
                            weighted_slippage += split_analysis.get('slippage_pct', 0) * 0.5
                    
                    if total_cost > 0:
                        strategies.append({
                            'type': 'split_execution',
                            'exchanges': [ex for ex, _ in exchange_prices[:2]],
                            'side': side,
                            'amount': total_amount,
                            'avg_price': total_cost / (total_amount / exchange_prices[0][1]),  # 近似计算
                            'slippage_pct': weighted_slippage,
                            'split_ratio': '50-50'
                        })
        
        # 按滑点排序，返回最优策略
        strategies.sort(key=lambda x: x['slippage_pct'])
        
        return {
            'optimal_strategy': strategies[0] if strategies else None,
            'all_strategies': strategies
        }

# 全局实例
orderbook_analyzer = OrderBookAnalyzer()