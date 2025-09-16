import asyncio
import aiohttp
import time
from typing import Dict, List, Optional, Tuple
from cachetools import TTLCache
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrossChainAnalyzer:
    """跨链转账效率与成本分析器"""
    
    def __init__(self):
        self.last_request_time = {}
        self.rate_limit_delay = 1.0  # 1秒间隔
        self.cache = TTLCache(maxsize=300, ttl=600)  # 10分钟缓存
        
        # 支持的区块链网络
        self.networks = {
            'ethereum': {
                'name': 'Ethereum',
                'chain_id': 1,
                'native_token': 'ETH',
                'rpc_url': 'https://eth.llamarpc.com',
                'gas_price_api': 'https://api.etherscan.io/api?module=gastracker&action=gasoracle'
            },
            'bsc': {
                'name': 'BSC',
                'chain_id': 56,
                'native_token': 'BNB',
                'rpc_url': 'https://bsc-dataseed.binance.org',
                'gas_price_api': 'https://api.bscscan.com/api?module=gastracker&action=gasoracle'
            },
            'polygon': {
                'name': 'Polygon',
                'chain_id': 137,
                'native_token': 'MATIC',
                'rpc_url': 'https://polygon-rpc.com',
                'gas_price_api': 'https://api.polygonscan.com/api?module=gastracker&action=gasoracle'
            },
            'arbitrum': {
                'name': 'Arbitrum',
                'chain_id': 42161,
                'native_token': 'ETH',
                'rpc_url': 'https://arb1.arbitrum.io/rpc',
                'gas_price_api': None
            },
            'optimism': {
                'name': 'Optimism',
                'chain_id': 10,
                'native_token': 'ETH',
                'rpc_url': 'https://mainnet.optimism.io',
                'gas_price_api': None
            },
            'avalanche': {
                'name': 'Avalanche',
                'chain_id': 43114,
                'native_token': 'AVAX',
                'rpc_url': 'https://api.avax.network/ext/bc/C/rpc',
                'gas_price_api': None
            }
        }
        
        # 跨链桥配置
        self.bridges = {
            'stargate': {
                'name': 'Stargate Finance',
                'supported_networks': ['ethereum', 'bsc', 'polygon', 'arbitrum', 'optimism', 'avalanche'],
                'api_url': 'https://api.stargate.finance',
                'fee_rate': 0.0006,  # 0.06%
                'estimated_time': 300  # 5分钟
            },
            'multichain': {
                'name': 'Multichain',
                'supported_networks': ['ethereum', 'bsc', 'polygon', 'arbitrum', 'optimism', 'avalanche'],
                'api_url': 'https://bridgeapi.anyswap.exchange',
                'fee_rate': 0.001,  # 0.1%
                'estimated_time': 600  # 10分钟
            },
            'cbridge': {
                'name': 'Celer cBridge',
                'supported_networks': ['ethereum', 'bsc', 'polygon', 'arbitrum', 'optimism', 'avalanche'],
                'api_url': 'https://cbridge-prod2.celer.app',
                'fee_rate': 0.0005,  # 0.05%
                'estimated_time': 180  # 3分钟
            },
            'hop': {
                'name': 'Hop Protocol',
                'supported_networks': ['ethereum', 'polygon', 'arbitrum', 'optimism'],
                'api_url': 'https://api.hop.exchange',
                'fee_rate': 0.0004,  # 0.04%
                'estimated_time': 240  # 4分钟
            },
            'synapse': {
                'name': 'Synapse Protocol',
                'supported_networks': ['ethereum', 'bsc', 'polygon', 'arbitrum', 'optimism', 'avalanche'],
                'api_url': 'https://api.synapseprotocol.com',
                'fee_rate': 0.0008,  # 0.08%
                'estimated_time': 420  # 7分钟
            }
        }
        
        # 常见代币配置
        self.tokens = {
            'USDT': {
                'name': 'Tether USD',
                'decimals': 6,
                'addresses': {
                    'ethereum': '0xdAC17F958D2ee523a2206206994597C13D831ec7',
                    'bsc': '0x55d398326f99059fF775485246999027B3197955',
                    'polygon': '0xc2132D05D31c914a87C6611C10748AEb04B58e8F',
                    'arbitrum': '0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9',
                    'optimism': '0x94b008aA00579c1307B0EF2c499aD98a8ce58e58',
                    'avalanche': '0x9702230A8Ea53601f5cD2dc00fDBc13d4dF4A8c7'
                }
            },
            'USDC': {
                'name': 'USD Coin',
                'decimals': 6,
                'addresses': {
                    'ethereum': '0xA0b86a33E6441b8435b662303c0f479c7e2b6b6',
                    'bsc': '0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d',
                    'polygon': '0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174',
                    'arbitrum': '0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8',
                    'optimism': '0x7F5c764cBc14f9669B88837ca1490cCa17c31607',
                    'avalanche': '0xB97EF9Ef8734C71904D8002F8b6Bc66Dd9c48a6E'
                }
            },
            'ETH': {
                'name': 'Ethereum',
                'decimals': 18,
                'addresses': {
                    'ethereum': '0x0000000000000000000000000000000000000000',
                    'bsc': '0x2170Ed0880ac9A755fd29B2688956BD959F933F8',
                    'polygon': '0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619',
                    'arbitrum': '0x0000000000000000000000000000000000000000',
                    'optimism': '0x0000000000000000000000000000000000000000'
                }
            }
        }
    
    async def _check_rate_limit(self, service):
        """检查速率限制"""
        current_time = time.time()
        if service in self.last_request_time:
            time_diff = current_time - self.last_request_time[service]
            if time_diff < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - time_diff)
        
        self.last_request_time[service] = current_time
    
    async def get_gas_price(self, network: str) -> Optional[Dict]:
        """获取网络Gas价格"""
        cache_key = f"gas_price_{network}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if network not in self.networks:
            return None
        
        network_config = self.networks[network]
        
        try:
            await self._check_rate_limit(f"gas_{network}")
            
            async with aiohttp.ClientSession() as session:
                if network_config['gas_price_api']:
                    # 使用区块链浏览器API
                    async with session.get(network_config['gas_price_api'], timeout=10) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            if network in ['ethereum', 'bsc', 'polygon']:
                                result = {
                                    'slow': float(data['result']['SafeGasPrice']),
                                    'standard': float(data['result']['ProposeGasPrice']),
                                    'fast': float(data['result']['FastGasPrice']),
                                    'unit': 'gwei',
                                    'timestamp': time.time()
                                }
                            else:
                                result = {
                                    'slow': 1.0,
                                    'standard': 2.0,
                                    'fast': 3.0,
                                    'unit': 'gwei',
                                    'timestamp': time.time()
                                }
                else:
                    # 使用默认值或RPC调用
                    if network == 'arbitrum':
                        result = {
                            'slow': 0.1,
                            'standard': 0.2,
                            'fast': 0.3,
                            'unit': 'gwei',
                            'timestamp': time.time()
                        }
                    elif network == 'optimism':
                        result = {
                            'slow': 0.001,
                            'standard': 0.002,
                            'fast': 0.003,
                            'unit': 'gwei',
                            'timestamp': time.time()
                        }
                    elif network == 'avalanche':
                        result = {
                            'slow': 25,
                            'standard': 30,
                            'fast': 35,
                            'unit': 'nAVAX',
                            'timestamp': time.time()
                        }
                    else:
                        result = {
                            'slow': 5,
                            'standard': 10,
                            'fast': 15,
                            'unit': 'gwei',
                            'timestamp': time.time()
                        }
                
                self.cache[cache_key] = result
                return result
        
        except Exception as e:
            logger.error(f"获取{network} Gas价格失败: {str(e)}")
        
        return None
    
    async def get_bridge_quote(self, bridge: str, from_network: str, to_network: str, 
                              token: str, amount: float) -> Optional[Dict]:
        """获取跨链桥报价"""
        cache_key = f"bridge_quote_{bridge}_{from_network}_{to_network}_{token}_{amount}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if bridge not in self.bridges:
            return None
        
        bridge_config = self.bridges[bridge]
        
        if from_network not in bridge_config['supported_networks'] or \
           to_network not in bridge_config['supported_networks']:
            return None
        
        try:
            await self._check_rate_limit(f"bridge_{bridge}")
            
            # 模拟跨链桥报价（实际应用中需要调用真实API）
            bridge_fee = amount * bridge_config['fee_rate']
            
            # 获取源链和目标链的Gas费用
            from_gas = await self.get_gas_price(from_network)
            to_gas = await self.get_gas_price(to_network)
            
            # 估算Gas费用（简化计算）
            from_gas_cost = 0
            to_gas_cost = 0
            
            if from_gas:
                gas_limit = 150000 if token == 'ETH' else 200000
                from_gas_cost = (from_gas['standard'] * gas_limit) / 1e9  # 转换为ETH单位
            
            if to_gas:
                gas_limit = 100000
                to_gas_cost = (to_gas['standard'] * gas_limit) / 1e9
            
            result = {
                'bridge': bridge,
                'bridge_name': bridge_config['name'],
                'from_network': from_network,
                'to_network': to_network,
                'token': token,
                'amount': amount,
                'bridge_fee': bridge_fee,
                'from_gas_cost': from_gas_cost,
                'to_gas_cost': to_gas_cost,
                'total_cost': bridge_fee + from_gas_cost + to_gas_cost,
                'estimated_time': bridge_config['estimated_time'],
                'fee_rate': bridge_config['fee_rate'],
                'timestamp': time.time()
            }
            
            self.cache[cache_key] = result
            return result
        
        except Exception as e:
            logger.error(f"获取{bridge}跨链报价失败: {str(e)}")
        
        return None
    
    async def analyze_cross_chain_routes(self, from_network: str, to_network: str, 
                                       token: str, amount: float) -> Dict:
        """分析跨链转账路由"""
        routes = []
        
        # 获取所有支持的桥的报价
        tasks = []
        for bridge_name, bridge_config in self.bridges.items():
            if (from_network in bridge_config['supported_networks'] and 
                to_network in bridge_config['supported_networks']):
                tasks.append(self.get_bridge_quote(bridge_name, from_network, to_network, token, amount))
        
        quotes = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理报价结果
        for quote in quotes:
            if isinstance(quote, Exception) or not quote:
                continue
            
            routes.append({
                'bridge': quote['bridge_name'],
                'total_cost': quote['total_cost'],
                'bridge_fee': quote['bridge_fee'],
                'gas_cost': quote['from_gas_cost'] + quote['to_gas_cost'],
                'estimated_time': quote['estimated_time'],
                'fee_rate': quote['fee_rate'],
                'cost_percentage': (quote['total_cost'] / amount) * 100,
                'details': quote
            })
        
        # 按成本排序
        routes.sort(key=lambda x: x['total_cost'])
        
        # 计算统计信息
        if routes:
            costs = [route['total_cost'] for route in routes]
            times = [route['estimated_time'] for route in routes]
            
            analysis = {
                'routes': routes,
                'best_cost_route': routes[0],
                'fastest_route': min(routes, key=lambda x: x['estimated_time']),
                'statistics': {
                    'min_cost': min(costs),
                    'max_cost': max(costs),
                    'avg_cost': sum(costs) / len(costs),
                    'min_time': min(times),
                    'max_time': max(times),
                    'avg_time': sum(times) / len(times)
                },
                'total_routes': len(routes)
            }
        else:
            analysis = {
                'routes': [],
                'error': '没有找到可用的跨链路由',
                'total_routes': 0
            }
        
        return analysis
    
    def get_supported_networks(self) -> List[str]:
        """获取支持的网络列表"""
        return list(self.networks.keys())
    
    def get_supported_tokens(self) -> List[str]:
        """获取支持的代币列表"""
        return list(self.tokens.keys())
    
    def get_supported_bridges(self) -> List[str]:
        """获取支持的跨链桥列表"""
        return list(self.bridges.keys())
    
    def calculate_arbitrage_opportunity(self, token: str, amount: float, 
                                      price_data: Dict) -> List[Dict]:
        """计算跨链套利机会"""
        opportunities = []
        
        if not price_data:
            return opportunities
        
        networks = list(price_data.keys())
        
        for i, from_net in enumerate(networks):
            for j, to_net in enumerate(networks):
                if i >= j:  # 避免重复和自己对自己
                    continue
                
                from_price = price_data[from_net].get('price', 0)
                to_price = price_data[to_net].get('price', 0)
                
                if from_price == 0 or to_price == 0:
                    continue
                
                # 计算价差
                price_diff = to_price - from_price
                price_diff_pct = (price_diff / from_price) * 100
                
                if price_diff_pct > 0.1:  # 价差大于0.1%才考虑
                    # 估算转账成本（简化）
                    estimated_cost = amount * 0.001  # 假设0.1%的转账成本
                    
                    profit = (price_diff * amount) - estimated_cost
                    profit_pct = (profit / (from_price * amount)) * 100
                    
                    if profit > 0:
                        opportunities.append({
                            'from_network': from_net,
                            'to_network': to_net,
                            'token': token,
                            'amount': amount,
                            'from_price': from_price,
                            'to_price': to_price,
                            'price_diff': price_diff,
                            'price_diff_pct': price_diff_pct,
                            'estimated_cost': estimated_cost,
                            'estimated_profit': profit,
                            'profit_pct': profit_pct,
                            'risk_level': 'Medium' if profit_pct > 1 else 'High'
                        })
        
        # 按利润率排序
        opportunities.sort(key=lambda x: x['profit_pct'], reverse=True)
        
        return opportunities

# 全局实例
cross_chain_analyzer = CrossChainAnalyzer()