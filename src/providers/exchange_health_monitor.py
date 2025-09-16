import asyncio
import aiohttp
import time
from typing import Dict, List, Optional, Tuple
from cachetools import TTLCache
import logging
import json
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExchangeHealthMonitor:
    """交易所健康状态监控器"""
    
    def __init__(self):
        self.last_request_time = {}
        self.rate_limit_delay = 1.0  # 1秒间隔
        self.cache = TTLCache(maxsize=500, ttl=300)  # 5分钟缓存
        
        # 支持的交易所配置
        self.exchanges = {
            'binance': {
                'name': 'Binance',
                'api_url': 'https://api.binance.com',
                'status_endpoint': '/api/v3/ping',
                'server_time_endpoint': '/api/v3/time',
                'exchange_info_endpoint': '/api/v3/exchangeInfo',
                'ticker_endpoint': '/api/v3/ticker/24hr',
                'orderbook_endpoint': '/api/v3/depth',
                'klines_endpoint': '/api/v3/klines',
                'weight_limit': 1200,
                'order_limit': 10,
                'raw_request_limit': 6000
            },
            'okx': {
                'name': 'OKX',
                'api_url': 'https://www.okx.com',
                'status_endpoint': '/api/v5/public/time',
                'server_time_endpoint': '/api/v5/public/time',
                'exchange_info_endpoint': '/api/v5/public/instruments',
                'ticker_endpoint': '/api/v5/market/tickers',
                'orderbook_endpoint': '/api/v5/market/books',
                'klines_endpoint': '/api/v5/market/candles',
                'weight_limit': 20,
                'order_limit': 60,
                'raw_request_limit': 100
            },
            'bybit': {
                'name': 'Bybit',
                'api_url': 'https://api.bybit.com',
                'status_endpoint': '/v5/market/time',
                'server_time_endpoint': '/v5/market/time',
                'exchange_info_endpoint': '/v5/market/instruments-info',
                'ticker_endpoint': '/v5/market/tickers',
                'orderbook_endpoint': '/v5/market/orderbook',
                'klines_endpoint': '/v5/market/kline',
                'weight_limit': 120,
                'order_limit': 10,
                'raw_request_limit': 600
            },
            'gate': {
                'name': 'Gate.io',
                'api_url': 'https://api.gateio.ws',
                'status_endpoint': '/api/v4/spot/time',
                'server_time_endpoint': '/api/v4/spot/time',
                'exchange_info_endpoint': '/api/v4/spot/currency_pairs',
                'ticker_endpoint': '/api/v4/spot/tickers',
                'orderbook_endpoint': '/api/v4/spot/order_book',
                'klines_endpoint': '/api/v4/spot/candlesticks',
                'weight_limit': 900,
                'order_limit': 10,
                'raw_request_limit': 1000
            },
            'kucoin': {
                'name': 'KuCoin',
                'api_url': 'https://api.kucoin.com',
                'status_endpoint': '/api/v1/timestamp',
                'server_time_endpoint': '/api/v1/timestamp',
                'exchange_info_endpoint': '/api/v1/symbols',
                'ticker_endpoint': '/api/v1/market/allTickers',
                'orderbook_endpoint': '/api/v1/market/orderbook/level2_20',
                'klines_endpoint': '/api/v1/market/candles',
                'weight_limit': 100,
                'order_limit': 45,
                'raw_request_limit': 1800
            },
            'huobi': {
                'name': 'Huobi',
                'api_url': 'https://api.huobi.pro',
                'status_endpoint': '/v1/common/timestamp',
                'server_time_endpoint': '/v1/common/timestamp',
                'exchange_info_endpoint': '/v1/common/symbols',
                'ticker_endpoint': '/market/tickers',
                'orderbook_endpoint': '/market/depth',
                'klines_endpoint': '/market/history/kline',
                'weight_limit': 100,
                'order_limit': 10,
                'raw_request_limit': 1000
            },
            'mexc': {
                'name': 'MEXC',
                'api_url': 'https://api.mexc.com',
                'status_endpoint': '/api/v3/ping',
                'server_time_endpoint': '/api/v3/time',
                'exchange_info_endpoint': '/api/v3/exchangeInfo',
                'ticker_endpoint': '/api/v3/ticker/24hr',
                'orderbook_endpoint': '/api/v3/depth',
                'klines_endpoint': '/api/v3/klines',
                'weight_limit': 1200,
                'order_limit': 10,
                'raw_request_limit': 6000
            },
            'bitget': {
                'name': 'Bitget',
                'api_url': 'https://api.bitget.com',
                'status_endpoint': '/api/spot/v1/public/time',
                'server_time_endpoint': '/api/spot/v1/public/time',
                'exchange_info_endpoint': '/api/spot/v1/public/products',
                'ticker_endpoint': '/api/spot/v1/market/tickers',
                'orderbook_endpoint': '/api/spot/v1/market/depth',
                'klines_endpoint': '/api/spot/v1/market/candles',
                'weight_limit': 40,
                'order_limit': 20,
                'raw_request_limit': 1000
            }
        }
        
        # 健康状态指标
        self.health_metrics = {
            'api_status': 'API连接状态',
            'response_time': '响应时间',
            'server_time_sync': '服务器时间同步',
            'trading_pairs': '交易对数量',
            'volume_24h': '24小时交易量',
            'orderbook_depth': '订单簿深度',
            'price_accuracy': '价格精度',
            'rate_limit_status': '速率限制状态',
            'websocket_status': 'WebSocket连接状态',
            'maintenance_status': '维护状态'
        }
    
    async def _check_rate_limit(self, exchange):
        """检查速率限制"""
        current_time = time.time()
        if exchange in self.last_request_time:
            time_diff = current_time - self.last_request_time[exchange]
            if time_diff < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - time_diff)
        
        self.last_request_time[exchange] = current_time
    
    async def check_api_status(self, exchange: str) -> Dict:
        """检查交易所API状态"""
        if exchange not in self.exchanges:
            return {'status': 'error', 'message': '不支持的交易所'}
        
        exchange_config = self.exchanges[exchange]
        
        try:
            await self._check_rate_limit(exchange)
            
            start_time = time.time()
            
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = exchange_config['api_url'] + exchange_config['status_endpoint']
                
                async with session.get(url) as response:
                    response_time = (time.time() - start_time) * 1000  # 转换为毫秒
                    
                    if response.status == 200:
                        return {
                            'status': 'healthy',
                            'response_time': response_time,
                            'status_code': response.status,
                            'timestamp': time.time()
                        }
                    else:
                        return {
                            'status': 'unhealthy',
                            'response_time': response_time,
                            'status_code': response.status,
                            'timestamp': time.time()
                        }
        
        except asyncio.TimeoutError:
            return {
                'status': 'timeout',
                'response_time': 10000,  # 超时设为10秒
                'error': 'Request timeout',
                'timestamp': time.time()
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': time.time()
            }
    
    async def check_server_time_sync(self, exchange: str) -> Dict:
        """检查服务器时间同步"""
        if exchange not in self.exchanges:
            return {'status': 'error', 'message': '不支持的交易所'}
        
        exchange_config = self.exchanges[exchange]
        
        try:
            await self._check_rate_limit(exchange)
            
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = exchange_config['api_url'] + exchange_config['server_time_endpoint']
                
                local_time = time.time() * 1000  # 本地时间（毫秒）
                
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # 不同交易所的时间字段不同
                        if exchange == 'binance' or exchange == 'mexc':
                            server_time = data.get('serverTime', 0)
                        elif exchange == 'okx':
                            server_time = int(data['data'][0]['ts']) if data.get('data') else 0
                        elif exchange == 'bybit':
                            server_time = int(data['result']['timeSecond']) * 1000 if data.get('result') else 0
                        elif exchange == 'gate':
                            server_time = int(data.get('server_time', 0)) * 1000
                        elif exchange == 'kucoin':
                            server_time = data.get('data', 0)
                        elif exchange == 'huobi':
                            server_time = data.get('data', 0)
                        elif exchange == 'bitget':
                            server_time = int(data['data']) if data.get('data') else 0
                        else:
                            server_time = 0
                        
                        time_diff = abs(local_time - server_time)
                        
                        return {
                            'status': 'synced' if time_diff < 5000 else 'out_of_sync',  # 5秒容差
                            'time_diff': time_diff,
                            'local_time': local_time,
                            'server_time': server_time,
                            'timestamp': time.time()
                        }
                    else:
                        return {
                            'status': 'error',
                            'status_code': response.status,
                            'timestamp': time.time()
                        }
        
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': time.time()
            }
    
    async def check_trading_pairs(self, exchange: str) -> Dict:
        """检查交易对信息"""
        if exchange not in self.exchanges:
            return {'status': 'error', 'message': '不支持的交易所'}
        
        exchange_config = self.exchanges[exchange]
        
        try:
            await self._check_rate_limit(exchange)
            
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = exchange_config['api_url'] + exchange_config['exchange_info_endpoint']
                
                # 为不同交易所添加必要的参数
                params = {}
                if exchange == 'okx':
                    params['instType'] = 'SPOT'
                elif exchange == 'bybit':
                    params['category'] = 'spot'
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # 解析不同交易所的响应格式
                        if exchange == 'binance' or exchange == 'mexc':
                            symbols = data.get('symbols', [])
                            active_pairs = len([s for s in symbols if s.get('status') == 'TRADING'])
                            total_pairs = len(symbols)
                        elif exchange == 'okx':
                            symbols = data.get('data', [])
                            active_pairs = len([s for s in symbols if s.get('state') == 'live'])
                            total_pairs = len(symbols)
                        elif exchange == 'bybit':
                            symbols = data.get('result', {}).get('list', [])
                            active_pairs = len([s for s in symbols if s.get('status') == 'Trading'])
                            total_pairs = len(symbols)
                        elif exchange == 'gate':
                            symbols = data if isinstance(data, list) else []
                            active_pairs = len([s for s in symbols if s.get('trade_status') == 'tradable'])
                            total_pairs = len(symbols)
                        elif exchange == 'kucoin':
                            symbols = data.get('data', [])
                            active_pairs = len([s for s in symbols if s.get('enableTrading')])
                            total_pairs = len(symbols)
                        elif exchange == 'huobi':
                            symbols = data.get('data', [])
                            active_pairs = len([s for s in symbols if s.get('state') == 'online'])
                            total_pairs = len(symbols)
                        elif exchange == 'bitget':
                            symbols = data.get('data', [])
                            active_pairs = len([s for s in symbols if s.get('status') == 'online'])
                            total_pairs = len(symbols)
                        else:
                            active_pairs = 0
                            total_pairs = 0
                        
                        return {
                            'status': 'healthy',
                            'total_pairs': total_pairs,
                            'active_pairs': active_pairs,
                            'inactive_pairs': total_pairs - active_pairs,
                            'timestamp': time.time()
                        }
                    else:
                        return {
                            'status': 'error',
                            'status_code': response.status,
                            'timestamp': time.time()
                        }
        
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': time.time()
            }
    
    async def check_orderbook_depth(self, exchange: str, symbol: str = 'BTCUSDT') -> Dict:
        """检查订单簿深度"""
        if exchange not in self.exchanges:
            return {'status': 'error', 'message': '不支持的交易所'}
        
        exchange_config = self.exchanges[exchange]
        
        try:
            await self._check_rate_limit(exchange)
            
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = exchange_config['api_url'] + exchange_config['orderbook_endpoint']
                
                # 为不同交易所设置参数
                params = {}
                if exchange == 'binance' or exchange == 'mexc':
                    params = {'symbol': symbol, 'limit': 100}
                elif exchange == 'okx':
                    params = {'instId': symbol.replace('USDT', '-USDT'), 'sz': 100}
                elif exchange == 'bybit':
                    params = {'category': 'spot', 'symbol': symbol, 'limit': 50}
                elif exchange == 'gate':
                    params = {'currency_pair': symbol.replace('USDT', '_USDT'), 'limit': 100}
                elif exchange == 'kucoin':
                    params = {'symbol': symbol.replace('USDT', '-USDT')}
                elif exchange == 'huobi':
                    params = {'symbol': symbol.lower(), 'type': 'step0'}
                elif exchange == 'bitget':
                    params = {'symbol': symbol.replace('USDT', 'USDT_SPBL'), 'type': 'step0'}
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # 解析不同交易所的订单簿格式
                        bids = []
                        asks = []
                        
                        if exchange == 'binance' or exchange == 'mexc':
                            bids = data.get('bids', [])
                            asks = data.get('asks', [])
                        elif exchange == 'okx':
                            order_data = data.get('data', [{}])[0]
                            bids = order_data.get('bids', [])
                            asks = order_data.get('asks', [])
                        elif exchange == 'bybit':
                            result = data.get('result', {})
                            bids = result.get('b', [])
                            asks = result.get('a', [])
                        elif exchange == 'gate':
                            bids = data.get('bids', [])
                            asks = data.get('asks', [])
                        elif exchange == 'kucoin':
                            order_data = data.get('data', {})
                            bids = order_data.get('bids', [])
                            asks = order_data.get('asks', [])
                        elif exchange == 'huobi':
                            tick = data.get('tick', {})
                            bids = tick.get('bids', [])
                            asks = tick.get('asks', [])
                        elif exchange == 'bitget':
                            order_data = data.get('data', {})
                            bids = order_data.get('bids', [])
                            asks = order_data.get('asks', [])
                        
                        # 计算深度指标
                        bid_depth = len(bids)
                        ask_depth = len(asks)
                        
                        # 计算价差
                        spread = 0
                        if bids and asks:
                            try:
                                best_bid = float(bids[0][0])
                                best_ask = float(asks[0][0])
                                spread = ((best_ask - best_bid) / best_bid) * 100
                            except (IndexError, ValueError, TypeError):
                                spread = 0
                        
                        return {
                            'status': 'healthy',
                            'bid_depth': bid_depth,
                            'ask_depth': ask_depth,
                            'total_depth': bid_depth + ask_depth,
                            'spread_percentage': spread,
                            'symbol': symbol,
                            'timestamp': time.time()
                        }
                    else:
                        return {
                            'status': 'error',
                            'status_code': response.status,
                            'timestamp': time.time()
                        }
        
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': time.time()
            }
    
    async def get_comprehensive_health_check(self, exchange: str) -> Dict:
        """获取交易所综合健康检查"""
        cache_key = f"health_check_{exchange}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 并行执行多个健康检查
        tasks = [
            self.check_api_status(exchange),
            self.check_server_time_sync(exchange),
            self.check_trading_pairs(exchange),
            self.check_orderbook_depth(exchange)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        api_status, time_sync, trading_pairs, orderbook = results
        
        # 计算综合健康分数
        health_score = 0
        max_score = 100
        
        # API状态 (30分)
        if isinstance(api_status, dict) and api_status.get('status') == 'healthy':
            if api_status.get('response_time', 0) < 1000:  # 小于1秒
                health_score += 30
            elif api_status.get('response_time', 0) < 3000:  # 小于3秒
                health_score += 20
            else:
                health_score += 10
        
        # 时间同步 (20分)
        if isinstance(time_sync, dict) and time_sync.get('status') == 'synced':
            health_score += 20
        elif isinstance(time_sync, dict) and time_sync.get('time_diff', 0) < 10000:  # 小于10秒
            health_score += 10
        
        # 交易对状态 (25分)
        if isinstance(trading_pairs, dict) and trading_pairs.get('status') == 'healthy':
            active_ratio = trading_pairs.get('active_pairs', 0) / max(trading_pairs.get('total_pairs', 1), 1)
            if active_ratio > 0.9:
                health_score += 25
            elif active_ratio > 0.8:
                health_score += 20
            elif active_ratio > 0.7:
                health_score += 15
            else:
                health_score += 10
        
        # 订单簿深度 (25分)
        if isinstance(orderbook, dict) and orderbook.get('status') == 'healthy':
            total_depth = orderbook.get('total_depth', 0)
            spread = orderbook.get('spread_percentage', 100)
            
            if total_depth > 50 and spread < 0.1:
                health_score += 25
            elif total_depth > 30 and spread < 0.2:
                health_score += 20
            elif total_depth > 20 and spread < 0.5:
                health_score += 15
            else:
                health_score += 10
        
        # 确定健康等级
        if health_score >= 90:
            health_level = 'excellent'
            health_status = '优秀'
        elif health_score >= 75:
            health_level = 'good'
            health_status = '良好'
        elif health_score >= 60:
            health_level = 'fair'
            health_status = '一般'
        elif health_score >= 40:
            health_level = 'poor'
            health_status = '较差'
        else:
            health_level = 'critical'
            health_status = '严重'
        
        result = {
            'exchange': exchange,
            'exchange_name': self.exchanges[exchange]['name'],
            'health_score': health_score,
            'health_level': health_level,
            'health_status': health_status,
            'checks': {
                'api_status': api_status,
                'time_sync': time_sync,
                'trading_pairs': trading_pairs,
                'orderbook': orderbook
            },
            'timestamp': time.time(),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self.cache[cache_key] = result
        return result
    
    async def monitor_multiple_exchanges(self, exchanges: List[str]) -> Dict:
        """监控多个交易所的健康状态"""
        tasks = []
        for exchange in exchanges:
            if exchange in self.exchanges:
                tasks.append(self.get_comprehensive_health_check(exchange))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        exchange_results = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                exchange_results[exchanges[i]] = {
                    'status': 'error',
                    'error': str(result),
                    'timestamp': time.time()
                }
            else:
                exchange_results[exchanges[i]] = result
        
        # 计算整体统计
        healthy_count = 0
        total_score = 0
        
        for result in exchange_results.values():
            if isinstance(result, dict) and 'health_score' in result:
                total_score += result['health_score']
                if result['health_level'] in ['excellent', 'good']:
                    healthy_count += 1
        
        avg_score = total_score / len(exchange_results) if exchange_results else 0
        
        return {
            'exchanges': exchange_results,
            'summary': {
                'total_exchanges': len(exchanges),
                'healthy_exchanges': healthy_count,
                'average_score': avg_score,
                'timestamp': time.time()
            }
        }
    
    def get_supported_exchanges(self) -> List[str]:
        """获取支持的交易所列表"""
        return list(self.exchanges.keys())
    
    def get_exchange_info(self, exchange: str) -> Optional[Dict]:
        """获取交易所信息"""
        return self.exchanges.get(exchange)

# 全局实例
exchange_health_monitor = ExchangeHealthMonitor()