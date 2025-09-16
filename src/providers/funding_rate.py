import asyncio
import aiohttp
import time
from typing import Dict, List, Optional, Any
from cachetools import TTLCache
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FundingRateProvider:
    """资金费率数据提供者，支持多个交易所的永续合约资金费率"""
    
    def __init__(self):
        # 缓存设置：最多500个条目，TTL为300秒（5分钟）
        self.cache = TTLCache(maxsize=500, ttl=300)
        
        # 热门交易对列表
        self.popular_symbols = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',
            'XRP/USDT', 'DOT/USDT', 'DOGE/USDT', 'AVAX/USDT', 'MATIC/USDT',
            'LINK/USDT', 'UNI/USDT', 'LTC/USDT', 'BCH/USDT', 'ATOM/USDT'
        ]
        
        # 支持的交易所API端点
        self.exchanges = {
            'binance': {
                'base_url': 'https://fapi.binance.com',
                'funding_rate_endpoint': '/fapi/v1/fundingRate',
                'premium_index_endpoint': '/fapi/v1/premiumIndex',
                'rate_limit': 1200,  # 每分钟1200次请求
                'last_request': 0
            },
            'okx': {
                'base_url': 'https://www.okx.com',
                'funding_rate_endpoint': '/api/v5/public/funding-rate',
                'rate_limit': 20,  # 每2秒20次请求
                'last_request': 0
            },
            'bybit': {
                'base_url': 'https://api.bybit.com',
                'funding_rate_endpoint': '/v5/market/funding/history',
                'rate_limit': 120,  # 每分钟120次请求
                'last_request': 0
            },
            'gate': {
                'base_url': 'https://api.gateio.ws',
                'funding_rate_endpoint': '/api/v4/futures/usdt/funding_rate',
                'rate_limit': 900,  # 每分钟900次请求
                'last_request': 0
            }
        }
    
    def _check_rate_limit(self, exchange: str) -> bool:
        """检查API速率限制"""
        config = self.exchanges.get(exchange)
        if not config:
            return False
        
        now = time.time()
        if exchange == 'binance':
            # Binance: 1200次/分钟
            if now - config['last_request'] < 0.05:  # 0.05秒间隔
                return False
        elif exchange == 'okx':
            # OKX: 20次/2秒
            if now - config['last_request'] < 0.1:  # 0.1秒间隔
                return False
        elif exchange == 'bybit':
            # Bybit: 120次/分钟
            if now - config['last_request'] < 0.5:  # 0.5秒间隔
                return False
        elif exchange == 'gate':
            # Gate.io: 900次/分钟
            if now - config['last_request'] < 0.067:  # 0.067秒间隔
                return False
        
        config['last_request'] = now
        return True
    
    async def get_binance_funding_rates(self, symbols: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """获取Binance永续合约资金费率"""
        cache_key = f"binance_funding_{','.join(sorted(symbols or []))}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if not self._check_rate_limit('binance'):
            logger.warning("Binance API rate limit exceeded")
            return {}
        
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            
            # 获取当前资金费率和预测费率
            premium_url = f"{self.exchanges['binance']['base_url']}{self.exchanges['binance']['premium_index_endpoint']}"
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(premium_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        result = {}
                        for item in data:
                            symbol = item.get('symbol', '')
                            if symbols and symbol not in [s.replace('/', '') for s in symbols]:
                                continue
                            
                            # 格式化交易对名称
                            formatted_symbol = f"{symbol[:-4]}/{symbol[-4:]}" if symbol.endswith('USDT') else symbol
                            
                            result[formatted_symbol] = {
                                'exchange': 'binance',
                                'symbol': formatted_symbol,
                                'funding_rate': float(item.get('lastFundingRate', 0)),
                                'predicted_rate': float(item.get('interestRate', 0)),
                                'next_funding_time': item.get('nextFundingTime'),
                                'mark_price': float(item.get('markPrice', 0)),
                                'index_price': float(item.get('indexPrice', 0)),
                                'timestamp': item.get('time', int(time.time() * 1000))
                            }
                        
                        self.cache[cache_key] = result
                        return result
                    else:
                        logger.error(f"Binance API error: {response.status}")
                        return {}
        
        except Exception as e:
            logger.error(f"Error fetching Binance funding rates: {e}")
            return {}
    
    async def get_okx_funding_rates(self, symbols: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """获取OKX永续合约资金费率"""
        cache_key = f"okx_funding_{','.join(sorted(symbols or []))}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if not self._check_rate_limit('okx'):
            logger.warning("OKX API rate limit exceeded")
            return {}
        
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            url = f"{self.exchanges['okx']['base_url']}{self.exchanges['okx']['funding_rate_endpoint']}"
            
            result = {}
            
            # OKX需要逐个查询每个交易对
            target_symbols = symbols or ['BTC-USDT-SWAP', 'ETH-USDT-SWAP', 'SOL-USDT-SWAP']
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                for symbol in target_symbols:
                    okx_symbol = symbol.replace('/', '-') + '-SWAP' if '/' in symbol else symbol
                    params = {'instId': okx_symbol}
                    
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            if data.get('code') == '0' and data.get('data'):
                                item = data['data'][0]
                                formatted_symbol = okx_symbol.replace('-SWAP', '').replace('-', '/')
                                
                                result[formatted_symbol] = {
                                    'exchange': 'okx',
                                    'symbol': formatted_symbol,
                                    'funding_rate': float(item.get('fundingRate', 0)),
                                    'predicted_rate': float(item.get('nextFundingRate', 0)),
                                    'next_funding_time': item.get('nextFundingTime'),
                                    'funding_time': item.get('fundingTime'),
                                    'timestamp': int(item.get('ts', time.time() * 1000))
                                }
                    
                    # 避免请求过快
                    await asyncio.sleep(0.1)
            
            self.cache[cache_key] = result
            return result
        
        except Exception as e:
            logger.error(f"Error fetching OKX funding rates: {e}")
            return {}
    
    async def get_aggregated_funding_rates(self, symbols: List[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """获取聚合的资金费率数据"""
        cache_key = f"aggregated_funding_{','.join(sorted(symbols or []))}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # 并发获取多个交易所的数据
            tasks = [
                self.get_binance_funding_rates(symbols),
                self.get_okx_funding_rates(symbols)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            aggregated = {}
            
            for result in results:
                if isinstance(result, dict):
                    for symbol, data in result.items():
                        if symbol not in aggregated:
                            aggregated[symbol] = []
                        aggregated[symbol].append(data)
            
            # 按资金费率排序
            for symbol in aggregated:
                aggregated[symbol].sort(key=lambda x: x.get('funding_rate', 0), reverse=True)
            
            self.cache[cache_key] = aggregated
            return aggregated
        
        except Exception as e:
            logger.error(f"Error fetching aggregated funding rates: {e}")
            return {}
    
    def calculate_funding_arbitrage_opportunity(self, funding_data: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """计算资金费率套利机会"""
        opportunities = []
        
        for symbol, rates in funding_data.items():
            if len(rates) < 2:
                continue
            
            # 找到最高和最低资金费率
            highest = max(rates, key=lambda x: x.get('funding_rate', 0))
            lowest = min(rates, key=lambda x: x.get('funding_rate', 0))
            
            rate_diff = highest['funding_rate'] - lowest['funding_rate']
            
            # 只考虑费率差异大于0.01%的机会
            if rate_diff > 0.0001:
                # 计算年化收益率（假设每8小时收取一次费率）
                annual_return = rate_diff * 3 * 365 * 100  # 转换为百分比
                
                opportunities.append({
                    'symbol': symbol,
                    'strategy': 'funding_rate_arbitrage',
                    'long_exchange': lowest['exchange'],
                    'short_exchange': highest['exchange'],
                    'long_rate': lowest['funding_rate'],
                    'short_rate': highest['funding_rate'],
                    'rate_difference': rate_diff,
                    'annual_return_pct': annual_return,
                    'next_funding_time': highest.get('next_funding_time'),
                    'risk_level': 'low' if rate_diff < 0.001 else 'medium',
                    'timestamp': int(time.time() * 1000)
                })
        
        # 按年化收益率排序
        opportunities.sort(key=lambda x: x['annual_return_pct'], reverse=True)
        return opportunities
    
    def get_supported_exchanges(self) -> List[str]:
        """获取支持的交易所列表"""
        return list(self.exchanges.keys())
    
    def get_popular_symbols(self) -> List[str]:
        """获取热门交易对列表"""
        return self.popular_symbols

# 全局实例
funding_rate_provider = FundingRateProvider()