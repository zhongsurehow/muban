"""动态风险仪表盘

提供实时风险监控、风险评估和预警功能
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
import json

class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"  # 低风险
    MEDIUM = "medium"  # 中风险
    HIGH = "high"  # 高风险
    CRITICAL = "critical"  # 极高风险

class AlertType(Enum):
    """警报类型"""
    PRICE_VOLATILITY = "price_volatility"  # 价格波动
    LIQUIDITY_RISK = "liquidity_risk"  # 流动性风险
    COUNTERPARTY_RISK = "counterparty_risk"  # 交易对手风险
    MARKET_RISK = "market_risk"  # 市场风险
    OPERATIONAL_RISK = "operational_risk"  # 操作风险
    REGULATORY_RISK = "regulatory_risk"  # 监管风险

@dataclass
class RiskMetric:
    """风险指标"""
    name: str
    value: float
    threshold: float
    risk_level: RiskLevel
    description: str
    timestamp: datetime
    unit: str = ""
    trend: str = "stable"  # up, down, stable

@dataclass
class RiskAlert:
    """风险警报"""
    alert_id: str
    alert_type: AlertType
    risk_level: RiskLevel
    title: str
    message: str
    affected_assets: List[str]
    timestamp: datetime
    is_active: bool = True
    resolution_time: Optional[datetime] = None

@dataclass
class PortfolioRisk:
    """投资组合风险"""
    total_value: float
    var_1d: float  # 1日风险价值
    var_7d: float  # 7日风险价值
    max_drawdown: float  # 最大回撤
    sharpe_ratio: float  # 夏普比率
    beta: float  # 贝塔系数
    correlation_btc: float  # 与BTC相关性
    diversification_score: float  # 多样化分数
    timestamp: datetime

class RiskDashboard:
    """动态风险仪表盘"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 风险阈值配置
        self.risk_thresholds = {
            'volatility_24h': {'low': 5, 'medium': 15, 'high': 30},  # %
            'volume_change': {'low': 20, 'medium': 50, 'high': 80},  # %
            'price_change_1h': {'low': 2, 'medium': 5, 'high': 10},  # %
            'liquidity_ratio': {'low': 0.8, 'medium': 0.5, 'high': 0.3},  # 流动性比率
            'correlation_spike': {'low': 0.7, 'medium': 0.85, 'high': 0.95},  # 相关性
            'funding_rate': {'low': 0.01, 'medium': 0.05, 'high': 0.1},  # %
            'open_interest_change': {'low': 10, 'medium': 25, 'high': 50}  # %
        }
        
        # 监控的交易所
        self.exchanges = [
            'binance', 'okx', 'bybit', 'coinbase', 'kraken',
            'huobi', 'kucoin', 'gate', 'mexc', 'bitget'
        ]
        
        # 主要监控币种
        self.major_symbols = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',
            'XRP/USDT', 'DOT/USDT', 'DOGE/USDT', 'AVAX/USDT', 'MATIC/USDT'
        ]
        
        # 数据缓存
        self.market_data_cache = {}
        self.risk_metrics_cache = {}
        self.alerts_cache = []
        self.cache_timestamp = None
        
        # 历史数据存储
        self.price_history = {}
        self.volume_history = {}
        self.volatility_history = {}
    
    async def fetch_market_data(self) -> Dict:
        """获取市场数据"""
        try:
            # 使用CoinGecko API获取市场数据
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': 50,
                'page': 1,
                'sparkline': 'true',
                'price_change_percentage': '1h,24h,7d'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        market_data = {}
                        for coin in data:
                            symbol = coin['symbol'].upper()
                            market_data[symbol] = {
                                'price': coin['current_price'],
                                'market_cap': coin['market_cap'],
                                'volume_24h': coin['total_volume'],
                                'price_change_1h': coin.get('price_change_percentage_1h_in_currency', 0),
                                'price_change_24h': coin.get('price_change_percentage_24h_in_currency', 0),
                                'price_change_7d': coin.get('price_change_percentage_7d_in_currency', 0),
                                'sparkline': coin.get('sparkline_in_7d', {}).get('price', []),
                                'last_updated': coin['last_updated']
                            }
                        
                        self.market_data_cache = market_data
                        self.cache_timestamp = datetime.now()
                        return market_data
            
            return {}
            
        except Exception as e:
            self.logger.error(f"获取市场数据失败: {e}")
            return {}
    
    def calculate_volatility(self, prices: List[float]) -> float:
        """计算波动率"""
        if len(prices) < 2:
            return 0
        
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] != 0:
                returns.append((prices[i] - prices[i-1]) / prices[i-1])
        
        if not returns:
            return 0
        
        return np.std(returns) * np.sqrt(24) * 100  # 年化波动率
    
    def calculate_var(self, returns: List[float], confidence: float = 0.95) -> float:
        """计算风险价值(VaR)"""
        if len(returns) < 10:
            return 0
        
        sorted_returns = sorted(returns)
        index = int((1 - confidence) * len(sorted_returns))
        return abs(sorted_returns[index]) * 100
    
    def assess_risk_level(self, metric_name: str, value: float) -> RiskLevel:
        """评估风险等级"""
        thresholds = self.risk_thresholds.get(metric_name, {})
        
        if metric_name in ['liquidity_ratio']:  # 数值越小风险越高
            if value >= thresholds.get('low', 0.8):
                return RiskLevel.LOW
            elif value >= thresholds.get('medium', 0.5):
                return RiskLevel.MEDIUM
            elif value >= thresholds.get('high', 0.3):
                return RiskLevel.HIGH
            else:
                return RiskLevel.CRITICAL
        else:  # 数值越大风险越高
            if value <= thresholds.get('low', 5):
                return RiskLevel.LOW
            elif value <= thresholds.get('medium', 15):
                return RiskLevel.MEDIUM
            elif value <= thresholds.get('high', 30):
                return RiskLevel.HIGH
            else:
                return RiskLevel.CRITICAL
    
    async def calculate_risk_metrics(self) -> Dict[str, RiskMetric]:
        """计算风险指标"""
        # 确保有市场数据
        if not self.market_data_cache or not self.cache_timestamp:
            await self.fetch_market_data()
        
        risk_metrics = {}
        current_time = datetime.now()
        
        try:
            # 1. 市场波动率风险
            volatilities = []
            for symbol, data in self.market_data_cache.items():
                if 'sparkline' in data and data['sparkline']:
                    volatility = self.calculate_volatility(data['sparkline'])
                    volatilities.append(volatility)
            
            avg_volatility = np.mean(volatilities) if volatilities else 0
            risk_metrics['market_volatility'] = RiskMetric(
                name="市场波动率",
                value=avg_volatility,
                threshold=self.risk_thresholds['volatility_24h']['medium'],
                risk_level=self.assess_risk_level('volatility_24h', avg_volatility),
                description=f"当前市场平均波动率为 {avg_volatility:.2f}%",
                timestamp=current_time,
                unit="%",
                trend="up" if avg_volatility > 20 else "stable"
            )
            
            # 2. 流动性风险
            btc_data = self.market_data_cache.get('BTC', {})
            eth_data = self.market_data_cache.get('ETH', {})
            
            if btc_data and eth_data:
                btc_volume = btc_data.get('volume_24h', 0)
                eth_volume = eth_data.get('volume_24h', 0)
                total_volume = btc_volume + eth_volume
                
                # 简化的流动性比率计算
                liquidity_ratio = min(1.0, total_volume / 50000000000)  # 500亿作为基准
                
                risk_metrics['liquidity_risk'] = RiskMetric(
                    name="流动性风险",
                    value=liquidity_ratio,
                    threshold=self.risk_thresholds['liquidity_ratio']['medium'],
                    risk_level=self.assess_risk_level('liquidity_ratio', liquidity_ratio),
                    description=f"当前市场流动性比率为 {liquidity_ratio:.3f}",
                    timestamp=current_time,
                    unit="",
                    trend="down" if liquidity_ratio < 0.5 else "stable"
                )
            
            # 3. 价格冲击风险
            price_changes_1h = []
            for symbol, data in self.market_data_cache.items():
                change_1h = abs(data.get('price_change_1h', 0))
                price_changes_1h.append(change_1h)
            
            max_price_change = max(price_changes_1h) if price_changes_1h else 0
            risk_metrics['price_shock'] = RiskMetric(
                name="价格冲击风险",
                value=max_price_change,
                threshold=self.risk_thresholds['price_change_1h']['medium'],
                risk_level=self.assess_risk_level('price_change_1h', max_price_change),
                description=f"1小时内最大价格变动为 {max_price_change:.2f}%",
                timestamp=current_time,
                unit="%",
                trend="up" if max_price_change > 5 else "stable"
            )
            
            # 4. 市场相关性风险
            if len(self.market_data_cache) >= 3:
                # 计算主要币种间的相关性
                major_coins = ['BTC', 'ETH', 'BNB']
                correlations = []
                
                for i, coin1 in enumerate(major_coins):
                    for coin2 in major_coins[i+1:]:
                        if coin1 in self.market_data_cache and coin2 in self.market_data_cache:
                            data1 = self.market_data_cache[coin1].get('sparkline', [])
                            data2 = self.market_data_cache[coin2].get('sparkline', [])
                            
                            if len(data1) > 10 and len(data2) > 10:
                                corr = np.corrcoef(data1[-10:], data2[-10:])[0, 1]
                                if not np.isnan(corr):
                                    correlations.append(abs(corr))
                
                avg_correlation = np.mean(correlations) if correlations else 0.5
                risk_metrics['correlation_risk'] = RiskMetric(
                    name="相关性风险",
                    value=avg_correlation,
                    threshold=self.risk_thresholds['correlation_spike']['medium'],
                    risk_level=self.assess_risk_level('correlation_spike', avg_correlation),
                    description=f"主要币种平均相关性为 {avg_correlation:.3f}",
                    timestamp=current_time,
                    unit="",
                    trend="up" if avg_correlation > 0.8 else "stable"
                )
            
            # 5. 交易量异常风险
            volume_changes = []
            for symbol, data in self.market_data_cache.items():
                # 简化的交易量变化计算
                volume_24h = data.get('volume_24h', 0)
                market_cap = data.get('market_cap', 1)
                volume_ratio = (volume_24h / market_cap) * 100 if market_cap > 0 else 0
                volume_changes.append(volume_ratio)
            
            avg_volume_ratio = np.mean(volume_changes) if volume_changes else 0
            risk_metrics['volume_anomaly'] = RiskMetric(
                name="交易量异常",
                value=avg_volume_ratio,
                threshold=self.risk_thresholds['volume_change']['medium'],
                risk_level=self.assess_risk_level('volume_change', avg_volume_ratio),
                description=f"平均成交量比率为 {avg_volume_ratio:.2f}%",
                timestamp=current_time,
                unit="%",
                trend="up" if avg_volume_ratio > 30 else "stable"
            )
            
            self.risk_metrics_cache = risk_metrics
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"计算风险指标失败: {e}")
            return {}
    
    def generate_risk_alerts(self, risk_metrics: Dict[str, RiskMetric]) -> List[RiskAlert]:
        """生成风险警报"""
        alerts = []
        current_time = datetime.now()
        
        try:
            for metric_name, metric in risk_metrics.items():
                if metric.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                    alert_type = {
                        'market_volatility': AlertType.MARKET_RISK,
                        'liquidity_risk': AlertType.LIQUIDITY_RISK,
                        'price_shock': AlertType.PRICE_VOLATILITY,
                        'correlation_risk': AlertType.MARKET_RISK,
                        'volume_anomaly': AlertType.MARKET_RISK
                    }.get(metric_name, AlertType.MARKET_RISK)
                    
                    alert = RiskAlert(
                        alert_id=f"{metric_name}_{current_time.strftime('%Y%m%d_%H%M%S')}",
                        alert_type=alert_type,
                        risk_level=metric.risk_level,
                        title=f"{metric.name}风险警报",
                        message=f"{metric.description}，已超过{metric.risk_level.value}风险阈值",
                        affected_assets=list(self.market_data_cache.keys())[:5],
                        timestamp=current_time
                    )
                    alerts.append(alert)
            
            # 更新警报缓存
            self.alerts_cache.extend(alerts)
            
            # 保留最近100条警报
            self.alerts_cache = self.alerts_cache[-100:]
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"生成风险警报失败: {e}")
            return []
    
    def calculate_portfolio_risk(self, portfolio_data: Dict) -> PortfolioRisk:
        """计算投资组合风险"""
        try:
            # 模拟投资组合数据
            total_value = portfolio_data.get('total_value', 100000)
            
            # 获取BTC价格历史用于计算相关性
            btc_data = self.market_data_cache.get('BTC', {})
            btc_prices = btc_data.get('sparkline', [])
            
            if len(btc_prices) > 10:
                btc_returns = []
                for i in range(1, len(btc_prices)):
                    if btc_prices[i-1] != 0:
                        btc_returns.append((btc_prices[i] - btc_prices[i-1]) / btc_prices[i-1])
                
                # 计算风险指标
                var_1d = self.calculate_var(btc_returns[-24:]) if len(btc_returns) >= 24 else 2.5
                var_7d = self.calculate_var(btc_returns) if len(btc_returns) >= 10 else 5.0
                
                # 模拟其他指标
                max_drawdown = abs(min(btc_returns)) * 100 if btc_returns else 5.0
                sharpe_ratio = (np.mean(btc_returns) / np.std(btc_returns)) if btc_returns and np.std(btc_returns) > 0 else 1.0
                beta = 1.0  # 相对于市场的贝塔系数
                correlation_btc = 1.0  # 与BTC的相关性
                diversification_score = 0.7  # 多样化分数
            else:
                # 默认值
                var_1d = 2.5
                var_7d = 5.0
                max_drawdown = 5.0
                sharpe_ratio = 1.0
                beta = 1.0
                correlation_btc = 1.0
                diversification_score = 0.7
            
            return PortfolioRisk(
                total_value=total_value,
                var_1d=var_1d,
                var_7d=var_7d,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                beta=beta,
                correlation_btc=correlation_btc,
                diversification_score=diversification_score,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"计算投资组合风险失败: {e}")
            # 返回默认风险数据
            return PortfolioRisk(
                total_value=100000,
                var_1d=2.5,
                var_7d=5.0,
                max_drawdown=5.0,
                sharpe_ratio=1.0,
                beta=1.0,
                correlation_btc=1.0,
                diversification_score=0.7,
                timestamp=datetime.now()
            )
    
    async def get_risk_dashboard_data(self, portfolio_data: Optional[Dict] = None) -> Dict:
        """获取风险仪表盘数据"""
        try:
            # 获取市场数据
            await self.fetch_market_data()
            
            # 计算风险指标
            risk_metrics = await self.calculate_risk_metrics()
            
            # 生成风险警报
            new_alerts = self.generate_risk_alerts(risk_metrics)
            
            # 计算投资组合风险
            portfolio_risk = self.calculate_portfolio_risk(portfolio_data or {})
            
            # 计算整体风险评分
            risk_scores = []
            for metric in risk_metrics.values():
                if metric.risk_level == RiskLevel.LOW:
                    risk_scores.append(25)
                elif metric.risk_level == RiskLevel.MEDIUM:
                    risk_scores.append(50)
                elif metric.risk_level == RiskLevel.HIGH:
                    risk_scores.append(75)
                else:  # CRITICAL
                    risk_scores.append(100)
            
            overall_risk_score = np.mean(risk_scores) if risk_scores else 25
            
            if overall_risk_score <= 30:
                overall_risk_level = RiskLevel.LOW
            elif overall_risk_score <= 60:
                overall_risk_level = RiskLevel.MEDIUM
            elif overall_risk_score <= 85:
                overall_risk_level = RiskLevel.HIGH
            else:
                overall_risk_level = RiskLevel.CRITICAL
            
            return {
                'risk_metrics': risk_metrics,
                'portfolio_risk': portfolio_risk,
                'alerts': self.alerts_cache[-10:],  # 最近10条警报
                'new_alerts': new_alerts,
                'overall_risk_score': overall_risk_score,
                'overall_risk_level': overall_risk_level,
                'market_data': self.market_data_cache,
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"获取风险仪表盘数据失败: {e}")
            return {
                'risk_metrics': {},
                'portfolio_risk': None,
                'alerts': [],
                'new_alerts': [],
                'overall_risk_score': 50,
                'overall_risk_level': RiskLevel.MEDIUM,
                'market_data': {},
                'last_updated': datetime.now()
            }
    
    def get_risk_recommendations(self, dashboard_data: Dict) -> List[str]:
        """获取风险管理建议"""
        recommendations = []
        
        try:
            overall_risk_level = dashboard_data.get('overall_risk_level', RiskLevel.MEDIUM)
            risk_metrics = dashboard_data.get('risk_metrics', {})
            
            # 基于整体风险等级的建议
            if overall_risk_level == RiskLevel.CRITICAL:
                recommendations.append("🚨 市场风险极高，建议立即减仓或停止交易")
                recommendations.append("💰 考虑将资金转移到稳定币或现金")
            elif overall_risk_level == RiskLevel.HIGH:
                recommendations.append("⚠️ 市场风险较高，建议降低仓位")
                recommendations.append("🛡️ 加强止损设置，控制单笔损失")
            elif overall_risk_level == RiskLevel.MEDIUM:
                recommendations.append("📊 市场风险适中，保持谨慎交易")
                recommendations.append("⚖️ 注意仓位管理和风险分散")
            else:
                recommendations.append("✅ 市场风险较低，可适度增加仓位")
                recommendations.append("📈 关注优质投资机会")
            
            # 基于具体风险指标的建议
            for metric_name, metric in risk_metrics.items():
                if metric.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                    if metric_name == 'market_volatility':
                        recommendations.append(f"📉 {metric.name}过高，建议减少高风险资产配置")
                    elif metric_name == 'liquidity_risk':
                        recommendations.append(f"💧 {metric.name}较高，避免大额交易")
                    elif metric_name == 'correlation_risk':
                        recommendations.append(f"🔗 {metric.name}过高，增加资产多样化")
            
            # 通用建议
            recommendations.append("📱 建议启用实时风险监控和警报")
            recommendations.append("📚 定期回顾和调整风险管理策略")
            
            return recommendations[:8]  # 返回最多8条建议
            
        except Exception as e:
            self.logger.error(f"获取风险建议失败: {e}")
            return ["📊 建议定期监控市场风险状况"]

# 创建全局实例
risk_dashboard = RiskDashboard()