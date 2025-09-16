"""
智能预警系统模块
提供价差预警、异常监控、机会推送等功能
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)

class AlertType(Enum):
    """预警类型"""
    SPREAD_ALERT = "spread_alert"
    VOLUME_ALERT = "volume_alert"
    PRICE_ALERT = "price_alert"
    ARBITRAGE_OPPORTUNITY = "arbitrage_opportunity"
    SYSTEM_ERROR = "system_error"
    MARKET_ANOMALY = "market_anomaly"

class AlertSeverity(Enum):
    """预警严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class NotificationChannel(Enum):
    """通知渠道"""
    EMAIL = "email"
    WEBHOOK = "webhook"
    DESKTOP = "desktop"
    MOBILE = "mobile"

@dataclass
class AlertRule:
    """预警规则"""
    id: str
    name: str
    alert_type: AlertType
    conditions: Dict[str, Any]
    severity: AlertSeverity
    enabled: bool = True
    cooldown_minutes: int = 5
    channels: List[NotificationChannel] = field(default_factory=list)
    last_triggered: Optional[datetime] = None

@dataclass
class Alert:
    """预警信息"""
    id: str
    rule_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    data: Dict[str, Any]
    timestamp: datetime
    acknowledged: bool = False
    resolved: bool = False

@dataclass
class NotificationConfig:
    """通知配置"""
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    webhook_url: str = ""
    webhook_headers: Dict[str, str] = field(default_factory=dict)

class AlertSystem:
    """智能预警系统"""
    
    def __init__(self, config: NotificationConfig = None):
        self.config = config or NotificationConfig()
        self.rules: Dict[str, AlertRule] = {}
        self.alerts: List[Alert] = []
        self.subscribers: Dict[AlertType, List[Callable]] = {}
        self.running = False
        
        # 初始化默认规则
        self._init_default_rules()
    
    def _init_default_rules(self):
        """初始化默认预警规则"""
        
        # 价差预警规则
        spread_rule = AlertRule(
            id="spread_alert_1",
            name="高价差机会预警",
            alert_type=AlertType.SPREAD_ALERT,
            conditions={
                "min_spread_percentage": 0.5,
                "min_volume_usd": 10000,
                "exchanges": ["binance", "okx", "bybit"]
            },
            severity=AlertSeverity.MEDIUM,
            channels=[NotificationChannel.DESKTOP, NotificationChannel.EMAIL]
        )
        
        # 套利机会预警
        arbitrage_rule = AlertRule(
            id="arbitrage_alert_1", 
            name="套利机会预警",
            alert_type=AlertType.ARBITRAGE_OPPORTUNITY,
            conditions={
                "min_profit_percentage": 1.0,
                "max_execution_time_seconds": 30,
                "min_liquidity_usd": 50000
            },
            severity=AlertSeverity.HIGH,
            channels=[NotificationChannel.DESKTOP, NotificationChannel.WEBHOOK]
        )
        
        # 市场异常预警
        anomaly_rule = AlertRule(
            id="anomaly_alert_1",
            name="市场异常监控",
            alert_type=AlertType.MARKET_ANOMALY,
            conditions={
                "price_change_threshold": 5.0,
                "volume_spike_multiplier": 3.0,
                "time_window_minutes": 5
            },
            severity=AlertSeverity.HIGH,
            channels=[NotificationChannel.EMAIL, NotificationChannel.WEBHOOK]
        )
        
        self.rules[spread_rule.id] = spread_rule
        self.rules[arbitrage_rule.id] = arbitrage_rule
        self.rules[anomaly_rule.id] = anomaly_rule
    
    def add_rule(self, rule: AlertRule) -> bool:
        """添加预警规则"""
        try:
            self.rules[rule.id] = rule
            logger.info(f"Added alert rule: {rule.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add alert rule: {e}")
            return False
    
    def remove_rule(self, rule_id: str) -> bool:
        """删除预警规则"""
        try:
            if rule_id in self.rules:
                del self.rules[rule_id]
                logger.info(f"Removed alert rule: {rule_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove alert rule: {e}")
            return False
    
    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """更新预警规则"""
        try:
            if rule_id not in self.rules:
                return False
            
            rule = self.rules[rule_id]
            for key, value in updates.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)
            
            logger.info(f"Updated alert rule: {rule_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update alert rule: {e}")
            return False
    
    def check_spread_alert(self, spread_data: Dict[str, Any]) -> Optional[Alert]:
        """检查价差预警"""
        for rule in self.rules.values():
            if (rule.alert_type == AlertType.SPREAD_ALERT and 
                rule.enabled and 
                self._can_trigger(rule)):
                
                conditions = rule.conditions
                spread_pct = spread_data.get('spread_percentage', 0)
                volume_usd = spread_data.get('volume_usd', 0)
                
                if (spread_pct >= conditions.get('min_spread_percentage', 0) and
                    volume_usd >= conditions.get('min_volume_usd', 0)):
                    
                    alert = Alert(
                        id=f"alert_{datetime.now().timestamp()}",
                        rule_id=rule.id,
                        alert_type=rule.alert_type,
                        severity=rule.severity,
                        title=f"高价差机会: {spread_data.get('symbol', 'Unknown')}",
                        message=f"发现{spread_pct:.2f}%的价差机会，交易量${volume_usd:,.2f}",
                        data=spread_data,
                        timestamp=datetime.now()
                    )
                    
                    self._trigger_alert(rule, alert)
                    return alert
        
        return None
    
    def check_arbitrage_opportunity(self, opportunity_data: Dict[str, Any]) -> Optional[Alert]:
        """检查套利机会预警"""
        for rule in self.rules.values():
            if (rule.alert_type == AlertType.ARBITRAGE_OPPORTUNITY and 
                rule.enabled and 
                self._can_trigger(rule)):
                
                conditions = rule.conditions
                profit_pct = opportunity_data.get('profit_percentage', 0)
                execution_time = opportunity_data.get('execution_time_seconds', 0)
                liquidity = opportunity_data.get('liquidity_usd', 0)
                
                if (profit_pct >= conditions.get('min_profit_percentage', 0) and
                    execution_time <= conditions.get('max_execution_time_seconds', 999) and
                    liquidity >= conditions.get('min_liquidity_usd', 0)):
                    
                    alert = Alert(
                        id=f"alert_{datetime.now().timestamp()}",
                        rule_id=rule.id,
                        alert_type=rule.alert_type,
                        severity=rule.severity,
                        title=f"套利机会: {opportunity_data.get('strategy', 'Unknown')}",
                        message=f"发现{profit_pct:.2f}%的套利机会，预计执行时间{execution_time}秒",
                        data=opportunity_data,
                        timestamp=datetime.now()
                    )
                    
                    self._trigger_alert(rule, alert)
                    return alert
        
        return None
    
    def check_market_anomaly(self, market_data: Dict[str, Any]) -> Optional[Alert]:
        """检查市场异常预警"""
        for rule in self.rules.values():
            if (rule.alert_type == AlertType.MARKET_ANOMALY and 
                rule.enabled and 
                self._can_trigger(rule)):
                
                conditions = rule.conditions
                price_change = abs(market_data.get('price_change_percentage', 0))
                volume_spike = market_data.get('volume_spike_multiplier', 1)
                
                if (price_change >= conditions.get('price_change_threshold', 0) or
                    volume_spike >= conditions.get('volume_spike_multiplier', 1)):
                    
                    alert = Alert(
                        id=f"alert_{datetime.now().timestamp()}",
                        rule_id=rule.id,
                        alert_type=rule.alert_type,
                        severity=rule.severity,
                        title=f"市场异常: {market_data.get('symbol', 'Unknown')}",
                        message=f"价格变动{price_change:.2f}%，交易量激增{volume_spike:.1f}倍",
                        data=market_data,
                        timestamp=datetime.now()
                    )
                    
                    self._trigger_alert(rule, alert)
                    return alert
        
        return None
    
    def _can_trigger(self, rule: AlertRule) -> bool:
        """检查规则是否可以触发（考虑冷却时间）"""
        if rule.last_triggered is None:
            return True
        
        cooldown_delta = timedelta(minutes=rule.cooldown_minutes)
        return datetime.now() - rule.last_triggered > cooldown_delta
    
    def _trigger_alert(self, rule: AlertRule, alert: Alert):
        """触发预警"""
        try:
            # 更新规则触发时间
            rule.last_triggered = datetime.now()
            
            # 添加到预警列表
            self.alerts.append(alert)
            
            # 发送通知
            self._send_notifications(rule, alert)
            
            # 通知订阅者
            self._notify_subscribers(alert)
            
            logger.info(f"Alert triggered: {alert.title}")
            
        except Exception as e:
            logger.error(f"Failed to trigger alert: {e}")
    
    def _send_notifications(self, rule: AlertRule, alert: Alert):
        """发送通知"""
        for channel in rule.channels:
            try:
                if channel == NotificationChannel.EMAIL:
                    self._send_email_notification(alert)
                elif channel == NotificationChannel.WEBHOOK:
                    self._send_webhook_notification(alert)
                elif channel == NotificationChannel.DESKTOP:
                    self._send_desktop_notification(alert)
                    
            except Exception as e:
                logger.error(f"Failed to send {channel.value} notification: {e}")
    
    def _send_email_notification(self, alert: Alert):
        """发送邮件通知"""
        if not self.config.email_username or not self.config.email_password:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.email_username
            msg['To'] = self.config.email_username  # 发送给自己
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            body = f"""
            预警时间: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
            预警类型: {alert.alert_type.value}
            严重程度: {alert.severity.value}
            
            {alert.message}
            
            详细数据:
            {json.dumps(alert.data, indent=2, ensure_ascii=False)}
            """
            
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            server = smtplib.SMTP(self.config.email_smtp_server, self.config.email_smtp_port)
            server.starttls()
            server.login(self.config.email_username, self.config.email_password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email notification sent for alert: {alert.id}")
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
    
    def _send_webhook_notification(self, alert: Alert):
        """发送Webhook通知"""
        if not self.config.webhook_url:
            return
        
        try:
            import requests
            
            payload = {
                "alert_id": alert.id,
                "title": alert.title,
                "message": alert.message,
                "severity": alert.severity.value,
                "type": alert.alert_type.value,
                "timestamp": alert.timestamp.isoformat(),
                "data": alert.data
            }
            
            response = requests.post(
                self.config.webhook_url,
                json=payload,
                headers=self.config.webhook_headers,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Webhook notification sent for alert: {alert.id}")
            else:
                logger.error(f"Webhook notification failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
    
    def _send_desktop_notification(self, alert: Alert):
        """发送桌面通知"""
        try:
            # 这里可以集成桌面通知库，如plyer
            logger.info(f"Desktop notification: {alert.title}")
            
        except Exception as e:
            logger.error(f"Failed to send desktop notification: {e}")
    
    def _notify_subscribers(self, alert: Alert):
        """通知订阅者"""
        subscribers = self.subscribers.get(alert.alert_type, [])
        for callback in subscribers:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Subscriber callback failed: {e}")
    
    def subscribe(self, alert_type: AlertType, callback: Callable[[Alert], None]):
        """订阅预警类型"""
        if alert_type not in self.subscribers:
            self.subscribers[alert_type] = []
        self.subscribers[alert_type].append(callback)
    
    def unsubscribe(self, alert_type: AlertType, callback: Callable[[Alert], None]):
        """取消订阅"""
        if alert_type in self.subscribers:
            try:
                self.subscribers[alert_type].remove(callback)
            except ValueError:
                pass
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """确认预警"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                logger.info(f"Alert acknowledged: {alert_id}")
                return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """解决预警"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                logger.info(f"Alert resolved: {alert_id}")
                return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃预警"""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """按严重程度获取预警"""
        return [alert for alert in self.alerts if alert.severity == severity]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """获取预警统计"""
        total_alerts = len(self.alerts)
        active_alerts = len(self.get_active_alerts())
        
        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = len(self.get_alerts_by_severity(severity))
        
        type_counts = {}
        for alert_type in AlertType:
            type_counts[alert_type.value] = len([
                alert for alert in self.alerts 
                if alert.alert_type == alert_type
            ])
        
        return {
            "total_alerts": total_alerts,
            "active_alerts": active_alerts,
            "resolved_alerts": total_alerts - active_alerts,
            "severity_distribution": severity_counts,
            "type_distribution": type_counts,
            "rules_count": len(self.rules),
            "enabled_rules": len([rule for rule in self.rules.values() if rule.enabled])
        }

# 全局预警系统实例
alert_system = AlertSystem()