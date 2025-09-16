"""
多账户管理系统模块
提供资金分配、账户监控、统一管理等功能
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import pandas as pd
from decimal import Decimal

logger = logging.getLogger(__name__)

class AccountType(Enum):
    """账户类型"""
    SPOT = "spot"
    FUTURES = "futures"
    MARGIN = "margin"
    OPTIONS = "options"

class AccountStatus(Enum):
    """账户状态"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    MAINTENANCE = "maintenance"

class AllocationStrategy(Enum):
    """资金分配策略"""
    EQUAL = "equal"  # 平均分配
    WEIGHTED = "weighted"  # 权重分配
    RISK_BASED = "risk_based"  # 基于风险分配
    PERFORMANCE_BASED = "performance_based"  # 基于表现分配

@dataclass
class AccountBalance:
    """账户余额"""
    total: Decimal
    available: Decimal
    frozen: Decimal
    currency: str
    timestamp: datetime

@dataclass
class AccountInfo:
    """账户信息"""
    account_id: str
    exchange: str
    account_type: AccountType
    status: AccountStatus
    balances: Dict[str, AccountBalance]
    api_key: str
    api_secret: str
    passphrase: Optional[str] = None
    sandbox: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class AllocationRule:
    """资金分配规则"""
    id: str
    name: str
    strategy: AllocationStrategy
    target_accounts: List[str]
    weights: Dict[str, float] = field(default_factory=dict)
    min_allocation: Decimal = Decimal('0')
    max_allocation: Decimal = Decimal('1000000')
    rebalance_threshold: float = 0.05  # 5%
    enabled: bool = True

@dataclass
class AccountMetrics:
    """账户指标"""
    account_id: str
    total_value_usd: Decimal
    daily_pnl: Decimal
    daily_pnl_percentage: float
    weekly_pnl: Decimal
    monthly_pnl: Decimal
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    avg_trade_size: Decimal
    last_trade_time: Optional[datetime] = None

@dataclass
class RiskMetrics:
    """风险指标"""
    account_id: str
    var_95: Decimal  # 95% VaR
    var_99: Decimal  # 99% VaR
    volatility: float
    beta: float
    correlation_btc: float
    max_position_size: Decimal
    leverage_ratio: float
    margin_ratio: float

class AccountManager:
    """多账户管理系统"""
    
    def __init__(self):
        self.accounts: Dict[str, AccountInfo] = {}
        self.allocation_rules: Dict[str, AllocationRule] = {}
        self.metrics_cache: Dict[str, AccountMetrics] = {}
        self.risk_cache: Dict[str, RiskMetrics] = {}
        self.monitoring_enabled = True
        
        # 初始化默认分配规则
        self._init_default_rules()
    
    def _init_default_rules(self):
        """初始化默认分配规则"""
        
        # 平均分配规则
        equal_rule = AllocationRule(
            id="equal_allocation",
            name="平均分配",
            strategy=AllocationStrategy.EQUAL,
            target_accounts=[],
            min_allocation=Decimal('1000'),
            max_allocation=Decimal('100000')
        )
        
        # 风险分配规则
        risk_rule = AllocationRule(
            id="risk_based_allocation",
            name="风险基础分配",
            strategy=AllocationStrategy.RISK_BASED,
            target_accounts=[],
            rebalance_threshold=0.1
        )
        
        self.allocation_rules[equal_rule.id] = equal_rule
        self.allocation_rules[risk_rule.id] = risk_rule
    
    def add_account(self, account_info: AccountInfo) -> bool:
        """添加账户"""
        try:
            # 验证账户信息
            if not self._validate_account_info(account_info):
                return False
            
            # 测试API连接
            if not self._test_api_connection(account_info):
                logger.warning(f"API connection test failed for account {account_info.account_id}")
            
            self.accounts[account_info.account_id] = account_info
            logger.info(f"Added account: {account_info.account_id} on {account_info.exchange}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add account: {e}")
            return False
    
    def remove_account(self, account_id: str) -> bool:
        """删除账户"""
        try:
            if account_id in self.accounts:
                del self.accounts[account_id]
                
                # 清理相关数据
                if account_id in self.metrics_cache:
                    del self.metrics_cache[account_id]
                if account_id in self.risk_cache:
                    del self.risk_cache[account_id]
                
                logger.info(f"Removed account: {account_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to remove account: {e}")
            return False
    
    def update_account_status(self, account_id: str, status: AccountStatus) -> bool:
        """更新账户状态"""
        try:
            if account_id in self.accounts:
                self.accounts[account_id].status = status
                self.accounts[account_id].last_updated = datetime.now()
                logger.info(f"Updated account {account_id} status to {status.value}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to update account status: {e}")
            return False
    
    def get_account_balances(self, account_id: str) -> Optional[Dict[str, AccountBalance]]:
        """获取账户余额"""
        try:
            if account_id not in self.accounts:
                return None
            
            account = self.accounts[account_id]
            
            # 这里应该调用实际的交易所API获取余额
            # 现在返回模拟数据
            balances = self._fetch_account_balances(account)
            
            # 更新账户余额
            account.balances = balances
            account.last_updated = datetime.now()
            
            return balances
            
        except Exception as e:
            logger.error(f"Failed to get account balances: {e}")
            return None
    
    def _fetch_account_balances(self, account: AccountInfo) -> Dict[str, AccountBalance]:
        """获取账户余额（模拟实现）"""
        # 模拟余额数据
        import random
        
        currencies = ['USDT', 'BTC', 'ETH', 'BNB']
        balances = {}
        
        for currency in currencies:
            total = Decimal(str(random.uniform(100, 10000)))
            frozen = total * Decimal(str(random.uniform(0, 0.2)))
            available = total - frozen
            
            balances[currency] = AccountBalance(
                total=total,
                available=available,
                frozen=frozen,
                currency=currency,
                timestamp=datetime.now()
            )
        
        return balances
    
    def calculate_total_portfolio_value(self) -> Decimal:
        """计算总投资组合价值"""
        total_value = Decimal('0')
        
        for account_id in self.accounts:
            balances = self.get_account_balances(account_id)
            if balances:
                for balance in balances.values():
                    # 这里应该使用实际汇率转换为USD
                    if balance.currency == 'USDT':
                        total_value += balance.total
                    elif balance.currency == 'BTC':
                        total_value += balance.total * Decimal('43000')  # 模拟BTC价格
                    elif balance.currency == 'ETH':
                        total_value += balance.total * Decimal('2500')   # 模拟ETH价格
                    elif balance.currency == 'BNB':
                        total_value += balance.total * Decimal('300')    # 模拟BNB价格
        
        return total_value
    
    def allocate_funds(self, rule_id: str, total_amount: Decimal) -> Dict[str, Decimal]:
        """资金分配"""
        try:
            if rule_id not in self.allocation_rules:
                raise ValueError(f"Allocation rule {rule_id} not found")
            
            rule = self.allocation_rules[rule_id]
            if not rule.enabled:
                raise ValueError(f"Allocation rule {rule_id} is disabled")
            
            # 获取目标账户
            target_accounts = rule.target_accounts or list(self.accounts.keys())
            active_accounts = [
                acc_id for acc_id in target_accounts 
                if acc_id in self.accounts and self.accounts[acc_id].status == AccountStatus.ACTIVE
            ]
            
            if not active_accounts:
                raise ValueError("No active target accounts found")
            
            # 根据策略分配资金
            allocation = self._calculate_allocation(rule, active_accounts, total_amount)
            
            logger.info(f"Allocated {total_amount} using rule {rule.name}")
            return allocation
            
        except Exception as e:
            logger.error(f"Failed to allocate funds: {e}")
            return {}
    
    def _calculate_allocation(self, rule: AllocationRule, accounts: List[str], 
                            total_amount: Decimal) -> Dict[str, Decimal]:
        """计算资金分配"""
        allocation = {}
        
        if rule.strategy == AllocationStrategy.EQUAL:
            # 平均分配
            amount_per_account = total_amount / len(accounts)
            for account_id in accounts:
                allocation[account_id] = max(
                    min(amount_per_account, rule.max_allocation),
                    rule.min_allocation
                )
        
        elif rule.strategy == AllocationStrategy.WEIGHTED:
            # 权重分配
            total_weight = sum(rule.weights.get(acc_id, 1.0) for acc_id in accounts)
            for account_id in accounts:
                weight = rule.weights.get(account_id, 1.0)
                amount = total_amount * Decimal(str(weight / total_weight))
                allocation[account_id] = max(
                    min(amount, rule.max_allocation),
                    rule.min_allocation
                )
        
        elif rule.strategy == AllocationStrategy.RISK_BASED:
            # 基于风险分配
            risk_scores = self._calculate_risk_scores(accounts)
            total_risk_score = sum(risk_scores.values())
            
            for account_id in accounts:
                # 风险越低，分配越多
                risk_factor = 1.0 - (risk_scores.get(account_id, 0.5) / total_risk_score)
                amount = total_amount * Decimal(str(risk_factor / len(accounts)))
                allocation[account_id] = max(
                    min(amount, rule.max_allocation),
                    rule.min_allocation
                )
        
        elif rule.strategy == AllocationStrategy.PERFORMANCE_BASED:
            # 基于表现分配
            performance_scores = self._calculate_performance_scores(accounts)
            total_performance = sum(performance_scores.values())
            
            if total_performance > 0:
                for account_id in accounts:
                    performance_factor = performance_scores.get(account_id, 0) / total_performance
                    amount = total_amount * Decimal(str(performance_factor))
                    allocation[account_id] = max(
                        min(amount, rule.max_allocation),
                        rule.min_allocation
                    )
            else:
                # 如果没有表现数据，回退到平均分配
                amount_per_account = total_amount / len(accounts)
                for account_id in accounts:
                    allocation[account_id] = amount_per_account
        
        return allocation
    
    def _calculate_risk_scores(self, accounts: List[str]) -> Dict[str, float]:
        """计算风险评分"""
        risk_scores = {}
        
        for account_id in accounts:
            # 模拟风险评分计算
            import random
            risk_scores[account_id] = random.uniform(0.1, 0.9)
        
        return risk_scores
    
    def _calculate_performance_scores(self, accounts: List[str]) -> Dict[str, float]:
        """计算表现评分"""
        performance_scores = {}
        
        for account_id in accounts:
            metrics = self.get_account_metrics(account_id)
            if metrics:
                # 基于夏普比率和收益率计算表现评分
                score = max(0, metrics.sharpe_ratio * 0.5 + metrics.daily_pnl_percentage * 0.5)
                performance_scores[account_id] = score
            else:
                performance_scores[account_id] = 0.0
        
        return performance_scores
    
    def get_account_metrics(self, account_id: str) -> Optional[AccountMetrics]:
        """获取账户指标"""
        try:
            if account_id not in self.accounts:
                return None
            
            # 检查缓存
            if account_id in self.metrics_cache:
                cached_metrics = self.metrics_cache[account_id]
                # 如果缓存时间不超过5分钟，直接返回
                if datetime.now() - cached_metrics.last_trade_time < timedelta(minutes=5):
                    return cached_metrics
            
            # 计算新的指标
            metrics = self._calculate_account_metrics(account_id)
            
            # 更新缓存
            if metrics:
                self.metrics_cache[account_id] = metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get account metrics: {e}")
            return None
    
    def _calculate_account_metrics(self, account_id: str) -> Optional[AccountMetrics]:
        """计算账户指标（模拟实现）"""
        import random
        
        # 模拟指标数据
        total_value = Decimal(str(random.uniform(10000, 100000)))
        daily_pnl = Decimal(str(random.uniform(-1000, 1000)))
        daily_pnl_pct = float(daily_pnl / total_value * 100)
        
        metrics = AccountMetrics(
            account_id=account_id,
            total_value_usd=total_value,
            daily_pnl=daily_pnl,
            daily_pnl_percentage=daily_pnl_pct,
            weekly_pnl=daily_pnl * Decimal('7'),
            monthly_pnl=daily_pnl * Decimal('30'),
            max_drawdown=random.uniform(0.05, 0.25),
            sharpe_ratio=random.uniform(-1.0, 3.0),
            win_rate=random.uniform(0.4, 0.8),
            total_trades=random.randint(50, 500),
            avg_trade_size=Decimal(str(random.uniform(100, 5000))),
            last_trade_time=datetime.now()
        )
        
        return metrics
    
    def get_risk_metrics(self, account_id: str) -> Optional[RiskMetrics]:
        """获取风险指标"""
        try:
            if account_id not in self.accounts:
                return None
            
            # 模拟风险指标
            import random
            
            risk_metrics = RiskMetrics(
                account_id=account_id,
                var_95=Decimal(str(random.uniform(500, 2000))),
                var_99=Decimal(str(random.uniform(1000, 3000))),
                volatility=random.uniform(0.1, 0.5),
                beta=random.uniform(0.5, 1.5),
                correlation_btc=random.uniform(-0.5, 0.9),
                max_position_size=Decimal(str(random.uniform(5000, 20000))),
                leverage_ratio=random.uniform(1.0, 5.0),
                margin_ratio=random.uniform(0.1, 0.8)
            )
            
            self.risk_cache[account_id] = risk_metrics
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Failed to get risk metrics: {e}")
            return None
    
    def check_rebalancing_needed(self, rule_id: str) -> bool:
        """检查是否需要重新平衡"""
        try:
            if rule_id not in self.allocation_rules:
                return False
            
            rule = self.allocation_rules[rule_id]
            if not rule.enabled:
                return False
            
            # 获取当前分配
            current_allocation = self._get_current_allocation(rule.target_accounts)
            
            # 计算理想分配
            total_value = sum(current_allocation.values())
            ideal_allocation = self._calculate_allocation(rule, list(current_allocation.keys()), total_value)
            
            # 检查偏差
            for account_id in current_allocation:
                current_pct = float(current_allocation[account_id] / total_value)
                ideal_pct = float(ideal_allocation.get(account_id, Decimal('0')) / total_value)
                
                if abs(current_pct - ideal_pct) > rule.rebalance_threshold:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check rebalancing: {e}")
            return False
    
    def _get_current_allocation(self, target_accounts: List[str]) -> Dict[str, Decimal]:
        """获取当前资金分配"""
        allocation = {}
        
        accounts_to_check = target_accounts or list(self.accounts.keys())
        
        for account_id in accounts_to_check:
            if account_id in self.accounts:
                balances = self.get_account_balances(account_id)
                if balances:
                    total_value = Decimal('0')
                    for balance in balances.values():
                        # 转换为USD价值
                        if balance.currency == 'USDT':
                            total_value += balance.total
                        elif balance.currency == 'BTC':
                            total_value += balance.total * Decimal('43000')
                        elif balance.currency == 'ETH':
                            total_value += balance.total * Decimal('2500')
                        elif balance.currency == 'BNB':
                            total_value += balance.total * Decimal('300')
                    
                    allocation[account_id] = total_value
        
        return allocation
    
    def _validate_account_info(self, account_info: AccountInfo) -> bool:
        """验证账户信息"""
        if not account_info.account_id:
            logger.error("Account ID is required")
            return False
        
        if not account_info.exchange:
            logger.error("Exchange is required")
            return False
        
        if not account_info.api_key:
            logger.error("API key is required")
            return False
        
        if not account_info.api_secret:
            logger.error("API secret is required")
            return False
        
        return True
    
    def _test_api_connection(self, account_info: AccountInfo) -> bool:
        """测试API连接"""
        try:
            # 这里应该实际测试API连接
            # 现在返回模拟结果
            import random
            return random.choice([True, False])
            
        except Exception as e:
            logger.error(f"API connection test failed: {e}")
            return False
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """获取投资组合摘要"""
        try:
            total_accounts = len(self.accounts)
            active_accounts = len([
                acc for acc in self.accounts.values() 
                if acc.status == AccountStatus.ACTIVE
            ])
            
            total_value = self.calculate_total_portfolio_value()
            
            # 计算总体指标
            total_daily_pnl = Decimal('0')
            total_trades = 0
            
            for account_id in self.accounts:
                metrics = self.get_account_metrics(account_id)
                if metrics:
                    total_daily_pnl += metrics.daily_pnl
                    total_trades += metrics.total_trades
            
            daily_pnl_pct = float(total_daily_pnl / total_value * 100) if total_value > 0 else 0
            
            return {
                "total_accounts": total_accounts,
                "active_accounts": active_accounts,
                "total_value_usd": float(total_value),
                "daily_pnl_usd": float(total_daily_pnl),
                "daily_pnl_percentage": daily_pnl_pct,
                "total_trades": total_trades,
                "allocation_rules": len(self.allocation_rules),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get portfolio summary: {e}")
            return {}

# 全局账户管理器实例
account_manager = AccountManager()