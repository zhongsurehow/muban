"""期现套利机会分析器

提供期货与现货价格差异分析，识别套利机会
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

@dataclass
class ArbitrageOpportunity:
    """套利机会数据结构"""
    symbol: str
    spot_price: float
    futures_price: float
    spread: float
    spread_percentage: float
    funding_rate: float
    expected_return: float
    risk_level: str
    exchange_spot: str
    exchange_futures: str
    timestamp: datetime

class ArbitrageAnalyzer:
    """期现套利机会分析器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 支持的交易所配置
        self.exchanges = {
            'binance': {
                'spot_api': 'https://api.binance.com/api/v3',
                'futures_api': 'https://fapi.binance.com/fapi/v1',
                'name': 'Binance'
            },
            'okx': {
                'spot_api': 'https://www.okx.com/api/v5',
                'futures_api': 'https://www.okx.com/api/v5',
                'name': 'OKX'
            },
            'bybit': {
                'spot_api': 'https://api.bybit.com/v5',
                'futures_api': 'https://api.bybit.com/v5',
                'name': 'Bybit'
            }
        }
        
        # 主要交易对
        self.major_symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT',
            'XRPUSDT', 'DOTUSDT', 'LINKUSDT', 'LTCUSDT', 'BCHUSDT'
        ]
        
        # 套利阈值配置
        self.thresholds = {
            'min_spread': 0.1,  # 最小价差百分比
            'max_spread': 10.0,  # 最大价差百分比
            'min_volume': 10000,  # 最小交易量
            'max_funding_rate': 0.1  # 最大资金费率
        }
    
    async def get_spot_price(self, exchange: str, symbol: str) -> Optional[float]:
        """获取现货价格"""
        try:
            if exchange == 'binance':
                url = f"{self.exchanges[exchange]['spot_api']}/ticker/price?symbol={symbol}"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            return float(data['price'])
            
            elif exchange == 'okx':
                url = f"{self.exchanges[exchange]['spot_api']}/market/ticker?instId={symbol}"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data['code'] == '0' and data['data']:
                                return float(data['data'][0]['last'])
            
            elif exchange == 'bybit':
                url = f"{self.exchanges[exchange]['spot_api']}/market/tickers?category=spot&symbol={symbol}"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data['retCode'] == 0 and data['result']['list']:
                                return float(data['result']['list'][0]['lastPrice'])
            
            return None
            
        except Exception as e:
            self.logger.error(f"获取{exchange}现货价格失败: {e}")
            return None
    
    async def get_futures_price(self, exchange: str, symbol: str) -> Optional[float]:
        """获取期货价格"""
        try:
            if exchange == 'binance':
                url = f"{self.exchanges[exchange]['futures_api']}/ticker/price?symbol={symbol}"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            return float(data['price'])
            
            elif exchange == 'okx':
                # OKX期货合约格式不同
                futures_symbol = symbol.replace('USDT', '-USDT-SWAP')
                url = f"{self.exchanges[exchange]['futures_api']}/market/ticker?instId={futures_symbol}"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data['code'] == '0' and data['data']:
                                return float(data['data'][0]['last'])
            
            elif exchange == 'bybit':
                url = f"{self.exchanges[exchange]['futures_api']}/market/tickers?category=linear&symbol={symbol}"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data['retCode'] == 0 and data['result']['list']:
                                return float(data['result']['list'][0]['lastPrice'])
            
            return None
            
        except Exception as e:
            self.logger.error(f"获取{exchange}期货价格失败: {e}")
            return None
    
    async def get_funding_rate(self, exchange: str, symbol: str) -> Optional[float]:
        """获取资金费率"""
        try:
            if exchange == 'binance':
                url = f"{self.exchanges[exchange]['futures_api']}/premiumIndex?symbol={symbol}"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            return float(data['lastFundingRate'])
            
            elif exchange == 'okx':
                futures_symbol = symbol.replace('USDT', '-USDT-SWAP')
                url = f"{self.exchanges[exchange]['futures_api']}/public/funding-rate?instId={futures_symbol}"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data['code'] == '0' and data['data']:
                                return float(data['data'][0]['fundingRate'])
            
            elif exchange == 'bybit':
                url = f"{self.exchanges[exchange]['futures_api']}/market/funding/history?category=linear&symbol={symbol}&limit=1"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data['retCode'] == 0 and data['result']['list']:
                                return float(data['result']['list'][0]['fundingRate'])
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"获取{exchange}资金费率失败: {e}")
            return 0.0
    
    def calculate_arbitrage_opportunity(self, spot_price: float, futures_price: float, 
                                      funding_rate: float, symbol: str, 
                                      exchange_spot: str, exchange_futures: str) -> Optional[ArbitrageOpportunity]:
        """计算套利机会"""
        try:
            # 计算价差
            spread = futures_price - spot_price
            spread_percentage = (spread / spot_price) * 100
            
            # 计算预期收益（考虑资金费率）
            # 正向套利：买现货，卖期货
            # 反向套利：卖现货，买期货
            if spread > 0:
                # 正向套利机会
                expected_return = spread_percentage - (funding_rate * 100 * 8)  # 8小时资金费率
                risk_level = self._assess_risk_level(spread_percentage, funding_rate)
            else:
                # 反向套利机会
                expected_return = abs(spread_percentage) + (funding_rate * 100 * 8)
                risk_level = self._assess_risk_level(abs(spread_percentage), funding_rate)
            
            # 检查是否满足套利阈值
            if abs(spread_percentage) < self.thresholds['min_spread']:
                return None
            
            if abs(spread_percentage) > self.thresholds['max_spread']:
                risk_level = "极高"
            
            return ArbitrageOpportunity(
                symbol=symbol,
                spot_price=spot_price,
                futures_price=futures_price,
                spread=spread,
                spread_percentage=spread_percentage,
                funding_rate=funding_rate,
                expected_return=expected_return,
                risk_level=risk_level,
                exchange_spot=exchange_spot,
                exchange_futures=exchange_futures,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"计算套利机会失败: {e}")
            return None
    
    def _assess_risk_level(self, spread_percentage: float, funding_rate: float) -> str:
        """评估风险等级"""
        if abs(spread_percentage) > 5.0 or abs(funding_rate) > 0.05:
            return "高"
        elif abs(spread_percentage) > 2.0 or abs(funding_rate) > 0.02:
            return "中"
        else:
            return "低"
    
    async def scan_arbitrage_opportunities(self, symbols: Optional[List[str]] = None, 
                                         exchanges: Optional[List[str]] = None) -> List[ArbitrageOpportunity]:
        """扫描套利机会"""
        if symbols is None:
            symbols = self.major_symbols
        
        if exchanges is None:
            exchanges = list(self.exchanges.keys())
        
        opportunities = []
        
        for symbol in symbols:
            for exchange in exchanges:
                try:
                    # 获取现货和期货价格
                    spot_price, futures_price, funding_rate = await asyncio.gather(
                        self.get_spot_price(exchange, symbol),
                        self.get_futures_price(exchange, symbol),
                        self.get_funding_rate(exchange, symbol),
                        return_exceptions=True
                    )
                    
                    # 检查数据有效性
                    if (isinstance(spot_price, float) and isinstance(futures_price, float) and 
                        isinstance(funding_rate, float)):
                        
                        opportunity = self.calculate_arbitrage_opportunity(
                            spot_price, futures_price, funding_rate, symbol, exchange, exchange
                        )
                        
                        if opportunity:
                            opportunities.append(opportunity)
                    
                    # 添加延迟避免API限制
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    self.logger.error(f"扫描{exchange}-{symbol}套利机会失败: {e}")
                    continue
        
        # 按预期收益排序
        opportunities.sort(key=lambda x: abs(x.expected_return), reverse=True)
        
        return opportunities
    
    async def get_cross_exchange_opportunities(self, symbols: Optional[List[str]] = None) -> List[ArbitrageOpportunity]:
        """获取跨交易所套利机会"""
        if symbols is None:
            symbols = self.major_symbols
        
        opportunities = []
        exchanges = list(self.exchanges.keys())
        
        for symbol in symbols:
            # 获取所有交易所的现货价格
            spot_prices = {}
            futures_prices = {}
            funding_rates = {}
            
            for exchange in exchanges:
                try:
                    spot_price = await self.get_spot_price(exchange, symbol)
                    futures_price = await self.get_futures_price(exchange, symbol)
                    funding_rate = await self.get_funding_rate(exchange, symbol)
                    
                    if spot_price is not None:
                        spot_prices[exchange] = spot_price
                    if futures_price is not None:
                        futures_prices[exchange] = futures_price
                    if funding_rate is not None:
                        funding_rates[exchange] = funding_rate
                    
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    self.logger.error(f"获取{exchange}-{symbol}价格失败: {e}")
                    continue
            
            # 寻找跨交易所套利机会
            for spot_exchange, spot_price in spot_prices.items():
                for futures_exchange, futures_price in futures_prices.items():
                    if spot_exchange != futures_exchange:
                        funding_rate = funding_rates.get(futures_exchange, 0.0)
                        
                        opportunity = self.calculate_arbitrage_opportunity(
                            spot_price, futures_price, funding_rate, symbol,
                            spot_exchange, futures_exchange
                        )
                        
                        if opportunity:
                            opportunities.append(opportunity)
        
        # 按预期收益排序
        opportunities.sort(key=lambda x: abs(x.expected_return), reverse=True)
        
        return opportunities
    
    def generate_arbitrage_report(self, opportunities: List[ArbitrageOpportunity]) -> Dict:
        """生成套利报告"""
        if not opportunities:
            return {
                'total_opportunities': 0,
                'avg_expected_return': 0,
                'risk_distribution': {},
                'top_symbols': [],
                'summary': '未发现套利机会'
            }
        
        # 统计分析
        total_opportunities = len(opportunities)
        avg_expected_return = np.mean([abs(op.expected_return) for op in opportunities])
        
        # 风险分布
        risk_distribution = {}
        for op in opportunities:
            risk_distribution[op.risk_level] = risk_distribution.get(op.risk_level, 0) + 1
        
        # 热门交易对
        symbol_counts = {}
        for op in opportunities:
            symbol_counts[op.symbol] = symbol_counts.get(op.symbol, 0) + 1
        
        top_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_opportunities': total_opportunities,
            'avg_expected_return': avg_expected_return,
            'risk_distribution': risk_distribution,
            'top_symbols': top_symbols,
            'summary': f'发现{total_opportunities}个套利机会，平均预期收益{avg_expected_return:.2f}%'
        }

# 创建全局实例
arbitrage_analyzer = ArbitrageAnalyzer()