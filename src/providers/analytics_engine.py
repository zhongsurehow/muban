"""
专业级数据分析引擎
包含收益分析、历史回测、策略优化等功能
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """性能指标"""
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0
    var_95: float = 0.0  # 95% VaR
    cvar_95: float = 0.0  # 95% CVaR

@dataclass
class BacktestResult:
    """回测结果"""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    performance_metrics: PerformanceMetrics
    daily_returns: pd.Series
    equity_curve: pd.Series
    drawdown_series: pd.Series
    trade_log: pd.DataFrame

@dataclass
class StrategyOptimization:
    """策略优化结果"""
    parameter_name: str
    optimal_value: Any
    performance_score: float
    optimization_results: pd.DataFrame

class AnalyticsEngine:
    """数据分析引擎"""
    
    def __init__(self):
        self.risk_free_rate = 0.02  # 无风险利率 2%
        self.trading_days_per_year = 365
        
    def calculate_performance_metrics(
        self, 
        returns: pd.Series, 
        benchmark_returns: Optional[pd.Series] = None
    ) -> PerformanceMetrics:
        """计算性能指标"""
        try:
            if len(returns) == 0:
                return PerformanceMetrics()
            
            # 基础指标
            total_return = (1 + returns).prod() - 1
            annualized_return = (1 + total_return) ** (self.trading_days_per_year / len(returns)) - 1
            volatility = returns.std() * np.sqrt(self.trading_days_per_year)
            
            # 夏普比率
            excess_returns = returns - self.risk_free_rate / self.trading_days_per_year
            sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(self.trading_days_per_year) if returns.std() > 0 else 0
            
            # 最大回撤
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # 胜率
            win_rate = (returns > 0).mean()
            
            # 盈亏比
            winning_returns = returns[returns > 0]
            losing_returns = returns[returns < 0]
            avg_win = winning_returns.mean() if len(winning_returns) > 0 else 0
            avg_loss = abs(losing_returns.mean()) if len(losing_returns) > 0 else 1
            profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
            
            # 卡尔玛比率
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # 索提诺比率
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(self.trading_days_per_year) if len(downside_returns) > 0 else 0
            sortino_ratio = (annualized_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
            # VaR和CVaR (95%置信度)
            var_95 = np.percentile(returns, 5)
            cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
            
            return PerformanceMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                calmar_ratio=calmar_ratio,
                sortino_ratio=sortino_ratio,
                var_95=var_95,
                cvar_95=cvar_95
            )
            
        except Exception as e:
            logger.error(f"性能指标计算失败: {e}")
            return PerformanceMetrics()
    
    def run_backtest(
        self,
        strategy_name: str,
        price_data: pd.DataFrame,
        signals: pd.DataFrame,
        initial_capital: float = 100000,
        commission_rate: float = 0.001
    ) -> BacktestResult:
        """运行回测"""
        try:
            if price_data.empty or signals.empty:
                raise ValueError("价格数据或信号数据为空")
            
            # 确保数据对齐
            common_index = price_data.index.intersection(signals.index)
            price_data = price_data.loc[common_index]
            signals = signals.loc[common_index]
            
            # 计算收益率
            returns = price_data.pct_change().fillna(0)
            
            # 生成交易信号
            positions = signals.shift(1).fillna(0)  # 信号延迟一期执行
            
            # 计算策略收益
            strategy_returns = (returns * positions).sum(axis=1)
            
            # 扣除交易成本
            position_changes = positions.diff().abs().sum(axis=1)
            transaction_costs = position_changes * commission_rate
            strategy_returns -= transaction_costs
            
            # 计算权益曲线
            equity_curve = initial_capital * (1 + strategy_returns).cumprod()
            
            # 计算回撤
            running_max = equity_curve.expanding().max()
            drawdown_series = (equity_curve - running_max) / running_max
            
            # 生成交易记录
            trade_log = self._generate_trade_log(positions, price_data, commission_rate)
            
            # 计算性能指标
            performance_metrics = self.calculate_performance_metrics(strategy_returns)
            
            return BacktestResult(
                strategy_name=strategy_name,
                start_date=price_data.index[0],
                end_date=price_data.index[-1],
                initial_capital=initial_capital,
                final_capital=equity_curve.iloc[-1],
                performance_metrics=performance_metrics,
                daily_returns=strategy_returns,
                equity_curve=equity_curve,
                drawdown_series=drawdown_series,
                trade_log=trade_log
            )
            
        except Exception as e:
            logger.error(f"回测运行失败: {e}")
            return BacktestResult(
                strategy_name=strategy_name,
                start_date=datetime.now(),
                end_date=datetime.now(),
                initial_capital=initial_capital,
                final_capital=initial_capital,
                performance_metrics=PerformanceMetrics(),
                daily_returns=pd.Series(),
                equity_curve=pd.Series(),
                drawdown_series=pd.Series(),
                trade_log=pd.DataFrame()
            )
    
    def _generate_trade_log(
        self, 
        positions: pd.DataFrame, 
        price_data: pd.DataFrame, 
        commission_rate: float
    ) -> pd.DataFrame:
        """生成交易记录"""
        try:
            trades = []
            
            for symbol in positions.columns:
                position_changes = positions[symbol].diff()
                
                for date, change in position_changes.items():
                    if abs(change) > 0.001:  # 忽略微小变化
                        price = price_data.loc[date, symbol] if symbol in price_data.columns else 0
                        commission = abs(change) * price * commission_rate
                        
                        trades.append({
                            'date': date,
                            'symbol': symbol,
                            'action': 'BUY' if change > 0 else 'SELL',
                            'quantity': abs(change),
                            'price': price,
                            'value': abs(change) * price,
                            'commission': commission
                        })
            
            return pd.DataFrame(trades)
            
        except Exception as e:
            logger.error(f"交易记录生成失败: {e}")
            return pd.DataFrame()
    
    def optimize_strategy_parameters(
        self,
        strategy_func: callable,
        price_data: pd.DataFrame,
        parameter_ranges: Dict[str, List],
        optimization_metric: str = 'sharpe_ratio',
        initial_capital: float = 100000
    ) -> List[StrategyOptimization]:
        """优化策略参数"""
        try:
            optimization_results = []
            
            for param_name, param_values in parameter_ranges.items():
                results = []
                
                for value in param_values:
                    try:
                        # 运行策略
                        signals = strategy_func(price_data, **{param_name: value})
                        
                        # 运行回测
                        backtest_result = self.run_backtest(
                            f"Strategy_{param_name}_{value}",
                            price_data,
                            signals,
                            initial_capital
                        )
                        
                        # 获取优化指标
                        metric_value = getattr(backtest_result.performance_metrics, optimization_metric, 0)
                        
                        results.append({
                            'parameter_value': value,
                            'metric_value': metric_value,
                            'total_return': backtest_result.performance_metrics.total_return,
                            'max_drawdown': backtest_result.performance_metrics.max_drawdown,
                            'win_rate': backtest_result.performance_metrics.win_rate
                        })
                        
                    except Exception as e:
                        logger.error(f"参数优化失败 {param_name}={value}: {e}")
                        continue
                
                if results:
                    results_df = pd.DataFrame(results)
                    best_result = results_df.loc[results_df['metric_value'].idxmax()]
                    
                    optimization_results.append(StrategyOptimization(
                        parameter_name=param_name,
                        optimal_value=best_result['parameter_value'],
                        performance_score=best_result['metric_value'],
                        optimization_results=results_df
                    ))
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"策略参数优化失败: {e}")
            return []
    
    def analyze_correlation_matrix(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """分析相关性矩阵"""
        try:
            return returns_data.corr()
        except Exception as e:
            logger.error(f"相关性分析失败: {e}")
            return pd.DataFrame()
    
    def calculate_portfolio_metrics(
        self, 
        weights: np.ndarray, 
        returns: pd.DataFrame
    ) -> Dict[str, float]:
        """计算投资组合指标"""
        try:
            # 投资组合收益率
            portfolio_returns = (returns * weights).sum(axis=1)
            
            # 年化收益率
            annual_return = portfolio_returns.mean() * self.trading_days_per_year
            
            # 年化波动率
            annual_volatility = portfolio_returns.std() * np.sqrt(self.trading_days_per_year)
            
            # 夏普比率
            sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
            
            return {
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': sharpe_ratio
            }
            
        except Exception as e:
            logger.error(f"投资组合指标计算失败: {e}")
            return {}
    
    def monte_carlo_simulation(
        self,
        returns: pd.Series,
        initial_capital: float = 100000,
        days: int = 252,
        simulations: int = 1000
    ) -> Dict[str, Any]:
        """蒙特卡洛模拟"""
        try:
            mean_return = returns.mean()
            std_return = returns.std()
            
            # 生成随机收益率
            random_returns = np.random.normal(mean_return, std_return, (simulations, days))
            
            # 计算权益曲线
            equity_curves = initial_capital * np.cumprod(1 + random_returns, axis=1)
            
            # 最终价值分布
            final_values = equity_curves[:, -1]
            
            # 统计结果
            percentiles = np.percentile(final_values, [5, 25, 50, 75, 95])
            
            return {
                'final_values': final_values,
                'equity_curves': equity_curves,
                'percentiles': {
                    '5%': percentiles[0],
                    '25%': percentiles[1],
                    '50%': percentiles[2],
                    '75%': percentiles[3],
                    '95%': percentiles[4]
                },
                'probability_of_loss': (final_values < initial_capital).mean(),
                'expected_return': final_values.mean() / initial_capital - 1
            }
            
        except Exception as e:
            logger.error(f"蒙特卡洛模拟失败: {e}")
            return {}
    
    def create_performance_dashboard(self, backtest_result: BacktestResult) -> Dict[str, go.Figure]:
        """创建性能仪表盘"""
        try:
            figures = {}
            
            # 权益曲线图
            fig_equity = go.Figure()
            fig_equity.add_trace(go.Scatter(
                x=backtest_result.equity_curve.index,
                y=backtest_result.equity_curve.values,
                mode='lines',
                name='权益曲线',
                line=dict(color='blue', width=2)
            ))
            fig_equity.update_layout(
                title='权益曲线',
                xaxis_title='日期',
                yaxis_title='资产价值',
                hovermode='x unified'
            )
            figures['equity_curve'] = fig_equity
            
            # 回撤图
            fig_drawdown = go.Figure()
            fig_drawdown.add_trace(go.Scatter(
                x=backtest_result.drawdown_series.index,
                y=backtest_result.drawdown_series.values * 100,
                mode='lines',
                name='回撤',
                fill='tonexty',
                line=dict(color='red', width=1)
            ))
            fig_drawdown.update_layout(
                title='回撤分析',
                xaxis_title='日期',
                yaxis_title='回撤 (%)',
                hovermode='x unified'
            )
            figures['drawdown'] = fig_drawdown
            
            # 收益率分布
            fig_returns = go.Figure()
            fig_returns.add_trace(go.Histogram(
                x=backtest_result.daily_returns * 100,
                nbinsx=50,
                name='日收益率分布',
                opacity=0.7
            ))
            fig_returns.update_layout(
                title='收益率分布',
                xaxis_title='日收益率 (%)',
                yaxis_title='频次',
                showlegend=False
            )
            figures['returns_distribution'] = fig_returns
            
            # 滚动夏普比率
            if len(backtest_result.daily_returns) > 30:
                rolling_sharpe = backtest_result.daily_returns.rolling(30).apply(
                    lambda x: x.mean() / x.std() * np.sqrt(self.trading_days_per_year) if x.std() > 0 else 0
                )
                
                fig_sharpe = go.Figure()
                fig_sharpe.add_trace(go.Scatter(
                    x=rolling_sharpe.index,
                    y=rolling_sharpe.values,
                    mode='lines',
                    name='30日滚动夏普比率',
                    line=dict(color='green', width=2)
                ))
                fig_sharpe.update_layout(
                    title='滚动夏普比率',
                    xaxis_title='日期',
                    yaxis_title='夏普比率',
                    hovermode='x unified'
                )
                figures['rolling_sharpe'] = fig_sharpe
            
            return figures
            
        except Exception as e:
            logger.error(f"性能仪表盘创建失败: {e}")
            return {}
    
    def generate_performance_report(self, backtest_result: BacktestResult) -> str:
        """生成性能报告"""
        try:
            metrics = backtest_result.performance_metrics
            
            report = f"""
# 策略性能报告: {backtest_result.strategy_name}

## 基本信息
- **回测期间**: {backtest_result.start_date.strftime('%Y-%m-%d')} 至 {backtest_result.end_date.strftime('%Y-%m-%d')}
- **初始资金**: ${backtest_result.initial_capital:,.2f}
- **最终资金**: ${backtest_result.final_capital:,.2f}

## 收益指标
- **总收益率**: {metrics.total_return:.2%}
- **年化收益率**: {metrics.annualized_return:.2%}
- **年化波动率**: {metrics.volatility:.2%}

## 风险指标
- **最大回撤**: {metrics.max_drawdown:.2%}
- **95% VaR**: {metrics.var_95:.2%}
- **95% CVaR**: {metrics.cvar_95:.2%}

## 风险调整收益
- **夏普比率**: {metrics.sharpe_ratio:.3f}
- **卡尔玛比率**: {metrics.calmar_ratio:.3f}
- **索提诺比率**: {metrics.sortino_ratio:.3f}

## 交易统计
- **胜率**: {metrics.win_rate:.2%}
- **盈亏比**: {metrics.profit_factor:.3f}
- **总交易次数**: {len(backtest_result.trade_log)}

## 评级
"""
            
            # 策略评级
            if metrics.sharpe_ratio >= 2.0:
                rating = "优秀 ⭐⭐⭐⭐⭐"
            elif metrics.sharpe_ratio >= 1.5:
                rating = "良好 ⭐⭐⭐⭐"
            elif metrics.sharpe_ratio >= 1.0:
                rating = "一般 ⭐⭐⭐"
            elif metrics.sharpe_ratio >= 0.5:
                rating = "较差 ⭐⭐"
            else:
                rating = "很差 ⭐"
            
            report += f"- **策略评级**: {rating}\n"
            
            return report
            
        except Exception as e:
            logger.error(f"性能报告生成失败: {e}")
            return "报告生成失败"

# 全局实例
analytics_engine = AnalyticsEngine()