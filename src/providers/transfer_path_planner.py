"""转账路径规划器

提供最优转账路径规划，包括跨链转账、交易所间转账等
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from enum import Enum

class TransferType(Enum):
    """转账类型"""
    DIRECT = "direct"  # 直接转账
    CROSS_CHAIN = "cross_chain"  # 跨链转账
    EXCHANGE_TRANSFER = "exchange_transfer"  # 交易所间转账
    MULTI_HOP = "multi_hop"  # 多跳转账

@dataclass
class TransferStep:
    """转账步骤"""
    step_id: int
    from_platform: str
    to_platform: str
    from_token: str
    to_token: str
    amount: float
    estimated_fee: float
    estimated_time: int  # 分钟
    transfer_type: TransferType
    bridge_name: Optional[str] = None
    gas_fee: Optional[float] = None
    slippage: Optional[float] = None

@dataclass
class TransferPath:
    """转账路径"""
    path_id: str
    steps: List[TransferStep]
    total_fee: float
    total_time: int  # 分钟
    success_rate: float  # 成功率
    risk_level: str
    final_amount: float
    efficiency_score: float
    created_at: datetime

class TransferPathPlanner:
    """转账路径规划器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 支持的平台配置
        self.platforms = {
            'ethereum': {
                'name': 'Ethereum',
                'type': 'blockchain',
                'native_token': 'ETH',
                'avg_gas_fee': 0.005,  # ETH
                'avg_confirmation_time': 2,  # 分钟
                'supported_tokens': ['ETH', 'USDT', 'USDC', 'DAI', 'WBTC']
            },
            'bsc': {
                'name': 'BSC',
                'type': 'blockchain',
                'native_token': 'BNB',
                'avg_gas_fee': 0.001,  # BNB
                'avg_confirmation_time': 1,
                'supported_tokens': ['BNB', 'USDT', 'USDC', 'BUSD', 'BTCB']
            },
            'polygon': {
                'name': 'Polygon',
                'type': 'blockchain',
                'native_token': 'MATIC',
                'avg_gas_fee': 0.01,  # MATIC
                'avg_confirmation_time': 1,
                'supported_tokens': ['MATIC', 'USDT', 'USDC', 'DAI', 'WBTC']
            },
            'arbitrum': {
                'name': 'Arbitrum',
                'type': 'blockchain',
                'native_token': 'ETH',
                'avg_gas_fee': 0.001,  # ETH
                'avg_confirmation_time': 1,
                'supported_tokens': ['ETH', 'USDT', 'USDC', 'DAI', 'WBTC']
            },
            'binance': {
                'name': 'Binance',
                'type': 'exchange',
                'withdrawal_fee': {'USDT': 1.0, 'BTC': 0.0005, 'ETH': 0.005},
                'avg_withdrawal_time': 10,
                'supported_tokens': ['BTC', 'ETH', 'USDT', 'USDC', 'BNB']
            },
            'okx': {
                'name': 'OKX',
                'type': 'exchange',
                'withdrawal_fee': {'USDT': 1.0, 'BTC': 0.0004, 'ETH': 0.005},
                'avg_withdrawal_time': 15,
                'supported_tokens': ['BTC', 'ETH', 'USDT', 'USDC', 'OKB']
            },
            'bybit': {
                'name': 'Bybit',
                'type': 'exchange',
                'withdrawal_fee': {'USDT': 1.0, 'BTC': 0.0005, 'ETH': 0.005},
                'avg_withdrawal_time': 20,
                'supported_tokens': ['BTC', 'ETH', 'USDT', 'USDC', 'BIT']
            }
        }
        
        # 跨链桥配置
        self.bridges = {
            'stargate': {
                'name': 'Stargate',
                'supported_chains': ['ethereum', 'bsc', 'polygon', 'arbitrum'],
                'supported_tokens': ['USDT', 'USDC'],
                'fee_rate': 0.0006,  # 0.06%
                'avg_time': 15,  # 分钟
                'success_rate': 0.99
            },
            'multichain': {
                'name': 'Multichain',
                'supported_chains': ['ethereum', 'bsc', 'polygon'],
                'supported_tokens': ['USDT', 'USDC', 'DAI', 'WBTC'],
                'fee_rate': 0.001,  # 0.1%
                'avg_time': 20,
                'success_rate': 0.98
            },
            'cbridge': {
                'name': 'cBridge',
                'supported_chains': ['ethereum', 'bsc', 'polygon', 'arbitrum'],
                'supported_tokens': ['USDT', 'USDC', 'ETH', 'WBTC'],
                'fee_rate': 0.0005,  # 0.05%
                'avg_time': 10,
                'success_rate': 0.97
            }
        }
        
        # 代币价格缓存
        self.token_prices = {}
        self.price_cache_time = None
    
    async def get_token_prices(self) -> Dict[str, float]:
        """获取代币价格"""
        # 如果缓存有效（5分钟内），直接返回
        if (self.price_cache_time and 
            datetime.now() - self.price_cache_time < timedelta(minutes=5)):
            return self.token_prices
        
        try:
            # 使用CoinGecko API获取价格
            url = "https://api.coingecko.com/api/v3/simple/price"
            params = {
                'ids': 'bitcoin,ethereum,tether,usd-coin,binancecoin,matic-network,dai',
                'vs_currencies': 'usd'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        self.token_prices = {
                            'BTC': data.get('bitcoin', {}).get('usd', 50000),
                            'ETH': data.get('ethereum', {}).get('usd', 3000),
                            'USDT': data.get('tether', {}).get('usd', 1),
                            'USDC': data.get('usd-coin', {}).get('usd', 1),
                            'BNB': data.get('binancecoin', {}).get('usd', 300),
                            'MATIC': data.get('matic-network', {}).get('usd', 1),
                            'DAI': data.get('dai', {}).get('usd', 1)
                        }
                        
                        self.price_cache_time = datetime.now()
                        return self.token_prices
            
            # 如果API失败，使用默认价格
            self.token_prices = {
                'BTC': 50000, 'ETH': 3000, 'USDT': 1, 'USDC': 1,
                'BNB': 300, 'MATIC': 1, 'DAI': 1
            }
            return self.token_prices
            
        except Exception as e:
            self.logger.error(f"获取代币价格失败: {e}")
            # 返回默认价格
            return {
                'BTC': 50000, 'ETH': 3000, 'USDT': 1, 'USDC': 1,
                'BNB': 300, 'MATIC': 1, 'DAI': 1
            }
    
    def calculate_gas_fee_usd(self, platform: str, token: str, amount: float) -> float:
        """计算Gas费用（USD）"""
        if platform not in self.platforms:
            return 0
        
        platform_info = self.platforms[platform]
        
        if platform_info['type'] == 'blockchain':
            gas_fee_native = platform_info['avg_gas_fee']
            native_token = platform_info['native_token']
            native_price = self.token_prices.get(native_token, 1)
            return gas_fee_native * native_price
        
        elif platform_info['type'] == 'exchange':
            withdrawal_fees = platform_info.get('withdrawal_fee', {})
            fee_amount = withdrawal_fees.get(token, 0)
            token_price = self.token_prices.get(token, 1)
            return fee_amount * token_price
        
        return 0
    
    def find_direct_path(self, from_platform: str, to_platform: str, 
                        token: str, amount: float) -> Optional[TransferPath]:
        """寻找直接转账路径"""
        try:
            # 检查平台是否支持该代币
            if (from_platform not in self.platforms or 
                to_platform not in self.platforms):
                return None
            
            from_info = self.platforms[from_platform]
            to_info = self.platforms[to_platform]
            
            if (token not in from_info.get('supported_tokens', []) or
                token not in to_info.get('supported_tokens', [])):
                return None
            
            # 计算费用和时间
            gas_fee = self.calculate_gas_fee_usd(from_platform, token, amount)
            withdrawal_fee = 0
            
            if from_info['type'] == 'exchange':
                withdrawal_fees = from_info.get('withdrawal_fee', {})
                withdrawal_fee = withdrawal_fees.get(token, 0) * self.token_prices.get(token, 1)
            
            total_fee = gas_fee + withdrawal_fee
            
            # 估算时间
            if from_info['type'] == 'exchange':
                transfer_time = from_info.get('avg_withdrawal_time', 10)
            else:
                transfer_time = from_info.get('avg_confirmation_time', 5)
            
            # 创建转账步骤
            step = TransferStep(
                step_id=1,
                from_platform=from_platform,
                to_platform=to_platform,
                from_token=token,
                to_token=token,
                amount=amount,
                estimated_fee=total_fee,
                estimated_time=transfer_time,
                transfer_type=TransferType.DIRECT,
                gas_fee=gas_fee
            )
            
            # 计算最终金额
            token_price = self.token_prices.get(token, 1)
            fee_in_token = total_fee / token_price
            final_amount = amount - fee_in_token
            
            # 计算效率分数
            efficiency_score = (final_amount / amount) * 100 - (transfer_time / 60) * 5
            
            path = TransferPath(
                path_id=f"direct_{from_platform}_{to_platform}",
                steps=[step],
                total_fee=total_fee,
                total_time=transfer_time,
                success_rate=0.95,
                risk_level="低",
                final_amount=final_amount,
                efficiency_score=efficiency_score,
                created_at=datetime.now()
            )
            
            return path
            
        except Exception as e:
            self.logger.error(f"寻找直接路径失败: {e}")
            return None
    
    def find_cross_chain_path(self, from_chain: str, to_chain: str, 
                             token: str, amount: float) -> List[TransferPath]:
        """寻找跨链转账路径"""
        paths = []
        
        try:
            # 检查是否为区块链平台
            if (self.platforms.get(from_chain, {}).get('type') != 'blockchain' or
                self.platforms.get(to_chain, {}).get('type') != 'blockchain'):
                return paths
            
            # 遍历所有跨链桥
            for bridge_name, bridge_info in self.bridges.items():
                if (from_chain in bridge_info['supported_chains'] and
                    to_chain in bridge_info['supported_chains'] and
                    token in bridge_info['supported_tokens']):
                    
                    # 计算跨链费用
                    bridge_fee_rate = bridge_info['fee_rate']
                    bridge_fee = amount * bridge_fee_rate * self.token_prices.get(token, 1)
                    
                    # Gas费用
                    from_gas = self.calculate_gas_fee_usd(from_chain, token, amount)
                    to_gas = self.calculate_gas_fee_usd(to_chain, token, amount)
                    
                    total_fee = bridge_fee + from_gas + to_gas
                    total_time = bridge_info['avg_time']
                    
                    # 创建转账步骤
                    step = TransferStep(
                        step_id=1,
                        from_platform=from_chain,
                        to_platform=to_chain,
                        from_token=token,
                        to_token=token,
                        amount=amount,
                        estimated_fee=total_fee,
                        estimated_time=total_time,
                        transfer_type=TransferType.CROSS_CHAIN,
                        bridge_name=bridge_name,
                        gas_fee=from_gas + to_gas,
                        slippage=bridge_fee_rate
                    )
                    
                    # 计算最终金额
                    token_price = self.token_prices.get(token, 1)
                    fee_in_token = total_fee / token_price
                    final_amount = amount - fee_in_token
                    
                    # 风险评估
                    if bridge_info['success_rate'] > 0.98:
                        risk_level = "低"
                    elif bridge_info['success_rate'] > 0.95:
                        risk_level = "中"
                    else:
                        risk_level = "高"
                    
                    # 计算效率分数
                    efficiency_score = (
                        (final_amount / amount) * 100 - 
                        (total_time / 60) * 3 +
                        bridge_info['success_rate'] * 10
                    )
                    
                    path = TransferPath(
                        path_id=f"crosschain_{bridge_name}_{from_chain}_{to_chain}",
                        steps=[step],
                        total_fee=total_fee,
                        total_time=total_time,
                        success_rate=bridge_info['success_rate'],
                        risk_level=risk_level,
                        final_amount=final_amount,
                        efficiency_score=efficiency_score,
                        created_at=datetime.now()
                    )
                    
                    paths.append(path)
            
            return paths
            
        except Exception as e:
            self.logger.error(f"寻找跨链路径失败: {e}")
            return paths
    
    def find_multi_hop_path(self, from_platform: str, to_platform: str, 
                           token: str, amount: float) -> List[TransferPath]:
        """寻找多跳转账路径"""
        paths = []
        
        try:
            # 寻找中间平台
            intermediate_platforms = []
            
            # 添加主要交易所作为中间平台
            for platform_name, platform_info in self.platforms.items():
                if (platform_info['type'] == 'exchange' and 
                    platform_name not in [from_platform, to_platform] and
                    token in platform_info.get('supported_tokens', [])):
                    intermediate_platforms.append(platform_name)
            
            # 为每个中间平台创建路径
            for intermediate in intermediate_platforms:
                try:
                    # 第一步：从源到中间平台
                    step1_path = self.find_direct_path(from_platform, intermediate, token, amount)
                    if not step1_path:
                        continue
                    
                    # 第二步：从中间平台到目标
                    step1_final_amount = step1_path.final_amount
                    step2_path = self.find_direct_path(intermediate, to_platform, token, step1_final_amount)
                    if not step2_path:
                        continue
                    
                    # 合并步骤
                    step1 = step1_path.steps[0]
                    step1.step_id = 1
                    
                    step2 = step2_path.steps[0]
                    step2.step_id = 2
                    step2.amount = step1_final_amount
                    
                    # 计算总费用和时间
                    total_fee = step1_path.total_fee + step2_path.total_fee
                    total_time = step1_path.total_time + step2_path.total_time
                    final_amount = step2_path.final_amount
                    
                    # 计算成功率（两步相乘）
                    success_rate = step1_path.success_rate * step2_path.success_rate
                    
                    # 风险评估
                    if success_rate > 0.9:
                        risk_level = "中"
                    elif success_rate > 0.8:
                        risk_level = "高"
                    else:
                        risk_level = "极高"
                    
                    # 计算效率分数
                    efficiency_score = (
                        (final_amount / amount) * 100 - 
                        (total_time / 60) * 8 +
                        success_rate * 5
                    )
                    
                    path = TransferPath(
                        path_id=f"multihop_{from_platform}_{intermediate}_{to_platform}",
                        steps=[step1, step2],
                        total_fee=total_fee,
                        total_time=total_time,
                        success_rate=success_rate,
                        risk_level=risk_level,
                        final_amount=final_amount,
                        efficiency_score=efficiency_score,
                        created_at=datetime.now()
                    )
                    
                    paths.append(path)
                    
                except Exception as e:
                    self.logger.error(f"创建多跳路径失败 {intermediate}: {e}")
                    continue
            
            return paths
            
        except Exception as e:
            self.logger.error(f"寻找多跳路径失败: {e}")
            return paths
    
    async def plan_transfer_paths(self, from_platform: str, to_platform: str, 
                                 token: str, amount: float) -> List[TransferPath]:
        """规划转账路径"""
        # 更新代币价格
        await self.get_token_prices()
        
        all_paths = []
        
        try:
            # 1. 寻找直接转账路径
            direct_path = self.find_direct_path(from_platform, to_platform, token, amount)
            if direct_path:
                all_paths.append(direct_path)
            
            # 2. 如果是区块链间转账，寻找跨链路径
            from_type = self.platforms.get(from_platform, {}).get('type')
            to_type = self.platforms.get(to_platform, {}).get('type')
            
            if from_type == 'blockchain' and to_type == 'blockchain':
                cross_chain_paths = self.find_cross_chain_path(from_platform, to_platform, token, amount)
                all_paths.extend(cross_chain_paths)
            
            # 3. 寻找多跳转账路径
            multi_hop_paths = self.find_multi_hop_path(from_platform, to_platform, token, amount)
            all_paths.extend(multi_hop_paths)
            
            # 按效率分数排序
            all_paths.sort(key=lambda x: x.efficiency_score, reverse=True)
            
            return all_paths[:10]  # 返回前10个最优路径
            
        except Exception as e:
            self.logger.error(f"规划转账路径失败: {e}")
            return all_paths
    
    def generate_path_comparison(self, paths: List[TransferPath]) -> Dict:
        """生成路径对比分析"""
        if not paths:
            return {
                'total_paths': 0,
                'best_path': None,
                'avg_fee': 0,
                'avg_time': 0,
                'summary': '未找到可用路径'
            }
        
        # 统计分析
        total_paths = len(paths)
        best_path = paths[0]  # 效率分数最高的路径
        avg_fee = np.mean([path.total_fee for path in paths])
        avg_time = np.mean([path.total_time for path in paths])
        
        # 按类型分组
        path_types = {}
        for path in paths:
            if len(path.steps) == 1:
                if path.steps[0].transfer_type == TransferType.CROSS_CHAIN:
                    path_type = "跨链转账"
                else:
                    path_type = "直接转账"
            else:
                path_type = "多跳转账"
            
            path_types[path_type] = path_types.get(path_type, 0) + 1
        
        return {
            'total_paths': total_paths,
            'best_path': best_path,
            'avg_fee': avg_fee,
            'avg_time': avg_time,
            'path_types': path_types,
            'summary': f'找到{total_paths}条路径，最优路径费用${best_path.total_fee:.2f}，预计{best_path.total_time}分钟'
        }

# 创建全局实例
transfer_path_planner = TransferPathPlanner()