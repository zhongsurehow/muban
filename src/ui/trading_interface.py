"""
专业交易界面组件
提供自定义布局、快捷操作和专业交易功能
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
from .styles import apply_trading_theme, get_price_color, get_alert_class, render_metric_card, render_status_indicator

class TradingInterface:
    """专业交易界面"""
    
    def __init__(self):
        self.layout_configs = {
            "classic": {
                "name": "经典布局",
                "description": "传统的交易界面布局",
                "layout": [
                    {"type": "market_overview", "size": "full"},
                    {"type": "price_chart", "size": "large"},
                    {"type": "orderbook", "size": "medium"},
                    {"type": "trade_history", "size": "medium"}
                ]
            },
            "professional": {
                "name": "专业布局",
                "description": "适合专业交易员的多窗口布局",
                "layout": [
                    {"type": "watchlist", "size": "small"},
                    {"type": "price_chart", "size": "large"},
                    {"type": "orderbook", "size": "medium"},
                    {"type": "portfolio", "size": "medium"},
                    {"type": "alerts", "size": "small"},
                    {"type": "trade_history", "size": "medium"}
                ]
            },
            "analysis": {
                "name": "分析布局",
                "description": "专注于市场分析和数据展示",
                "layout": [
                    {"type": "market_overview", "size": "full"},
                    {"type": "analytics_dashboard", "size": "large"},
                    {"type": "arbitrage_opportunities", "size": "large"},
                    {"type": "risk_metrics", "size": "medium"}
                ]
            }
        }
        
        self.widget_configs = {
            "market_overview": {
                "name": "市场概览",
                "icon": "📊",
                "description": "显示主要市场指标和概览信息"
            },
            "price_chart": {
                "name": "价格图表",
                "icon": "📈",
                "description": "实时价格图表和技术分析"
            },
            "orderbook": {
                "name": "订单簿",
                "icon": "📋",
                "description": "实时买卖订单深度"
            },
            "trade_history": {
                "name": "交易历史",
                "icon": "📜",
                "description": "最近的交易记录"
            },
            "portfolio": {
                "name": "投资组合",
                "icon": "💼",
                "description": "账户余额和持仓信息"
            },
            "alerts": {
                "name": "预警中心",
                "icon": "🚨",
                "description": "实时预警和通知"
            },
            "watchlist": {
                "name": "关注列表",
                "icon": "⭐",
                "description": "自定义交易对关注列表"
            },
            "analytics_dashboard": {
                "name": "分析仪表盘",
                "icon": "🔬",
                "description": "高级分析和指标"
            },
            "arbitrage_opportunities": {
                "name": "套利机会",
                "icon": "🎯",
                "description": "实时套利机会展示"
            },
            "risk_metrics": {
                "name": "风险指标",
                "icon": "⚠️",
                "description": "风险管理和指标监控"
            }
        }
    
    def render_layout_selector(self):
        """渲染布局选择器"""
        st.sidebar.markdown("### 🎨 界面布局")
        
        # 获取当前布局
        current_layout = st.session_state.get('trading_layout', 'classic')
        
        # 布局选择
        layout_options = list(self.layout_configs.keys())
        layout_names = [self.layout_configs[key]["name"] for key in layout_options]
        
        selected_index = layout_options.index(current_layout) if current_layout in layout_options else 0
        
        new_layout = st.sidebar.selectbox(
            "选择布局",
            options=layout_options,
            format_func=lambda x: self.layout_configs[x]["name"],
            index=selected_index
        )
        
        if new_layout != current_layout:
            st.session_state.trading_layout = new_layout
            st.rerun()
        
        # 显示布局描述
        st.sidebar.info(self.layout_configs[new_layout]["description"])
        
        # 自定义布局选项
        if st.sidebar.button("🛠️ 自定义布局"):
            st.session_state.show_layout_customizer = True
        
        return new_layout
    
    def render_layout_customizer(self):
        """渲染布局自定义器"""
        if not st.session_state.get('show_layout_customizer', False):
            return
        
        with st.expander("🛠️ 自定义布局", expanded=True):
            st.write("### 拖拽组件来自定义您的交易界面")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write("**可用组件**")
                for widget_id, config in self.widget_configs.items():
                    if st.button(f"{config['icon']} {config['name']}", key=f"add_{widget_id}"):
                        self._add_widget_to_layout(widget_id)
                        st.rerun()
            
            with col2:
                st.write("**当前布局**")
                current_layout = st.session_state.get('custom_layout', [])
                
                if current_layout:
                    for i, widget in enumerate(current_layout):
                        widget_config = self.widget_configs.get(widget['type'], {})
                        
                        widget_col1, widget_col2, widget_col3 = st.columns([3, 1, 1])
                        
                        with widget_col1:
                            st.write(f"{widget_config.get('icon', '📦')} {widget_config.get('name', widget['type'])}")
                        
                        with widget_col2:
                            new_size = st.selectbox(
                                "大小",
                                ["small", "medium", "large", "full"],
                                index=["small", "medium", "large", "full"].index(widget.get('size', 'medium')),
                                key=f"size_{i}"
                            )
                            if new_size != widget.get('size'):
                                st.session_state.custom_layout[i]['size'] = new_size
                                st.rerun()
                        
                        with widget_col3:
                            if st.button("🗑️", key=f"remove_{i}"):
                                st.session_state.custom_layout.pop(i)
                                st.rerun()
                else:
                    st.info("点击左侧组件来添加到布局中")
            
            # 保存和取消按钮
            save_col1, save_col2 = st.columns(2)
            
            with save_col1:
                if st.button("💾 保存布局"):
                    st.session_state.trading_layout = 'custom'
                    st.session_state.show_layout_customizer = False
                    st.success("布局已保存！")
                    st.rerun()
            
            with save_col2:
                if st.button("❌ 取消"):
                    st.session_state.show_layout_customizer = False
                    st.rerun()
    
    def _add_widget_to_layout(self, widget_id: str):
        """添加组件到自定义布局"""
        if 'custom_layout' not in st.session_state:
            st.session_state.custom_layout = []
        
        new_widget = {
            "type": widget_id,
            "size": "medium"
        }
        
        st.session_state.custom_layout.append(new_widget)
    
    def render_trading_interface(self, layout_name: str, engine, providers):
        """渲染交易界面"""
        # 应用专业交易主题
        apply_trading_theme()
        
        if layout_name == 'custom':
            layout = st.session_state.get('custom_layout', [])
        else:
            layout = self.layout_configs.get(layout_name, {}).get('layout', [])
        
        if not layout:
            st.warning("⚠️ 布局配置为空，请选择其他布局或自定义布局")
            return
        
        # 渲染快捷操作栏
        self._render_quick_actions()
        
        # 根据布局渲染组件
        for widget in layout:
            self._render_widget(widget, engine, providers)
    
    def _render_quick_actions(self):
        """渲染快捷操作栏"""
        st.markdown('<div class="trading-widget">', unsafe_allow_html=True)
        st.markdown("### ⚡ 快捷操作")
        
        # 使用自定义HTML按钮样式
        st.markdown("""
        <div style="display: flex; gap: 1rem; flex-wrap: wrap; justify-content: center; margin: 1rem 0;">
        """, unsafe_allow_html=True)
        
        action_col1, action_col2, action_col3, action_col4, action_col5 = st.columns(5)
        
        with action_col1:
            if st.button("🔄 刷新数据", help="刷新所有数据", key="refresh_data"):
                st.cache_data.clear()
                st.success("数据已刷新")
                st.rerun()
        
        with action_col2:
            if st.button("📊 快速分析", help="执行快速市场分析", key="quick_analysis"):
                st.session_state.show_quick_analysis = True
        
        with action_col3:
            if st.button("🎯 寻找套利", help="搜索套利机会", key="find_arbitrage"):
                st.session_state.show_arbitrage_search = True
        
        with action_col4:
            if st.button("⚠️ 风险检查", help="执行风险检查", key="risk_check"):
                st.session_state.show_risk_check = True
        
        with action_col5:
            if st.button("📈 技术分析", help="打开技术分析工具", key="technical_analysis"):
                st.session_state.show_technical_analysis = True
        
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")
    
    def _render_widget(self, widget_config: Dict, engine, providers):
        """渲染单个组件"""
        widget_type = widget_config.get('type')
        widget_size = widget_config.get('size', 'medium')
        
        # 根据大小确定列数
        if widget_size == 'small':
            cols = st.columns([1, 3])
            container = cols[0]
        elif widget_size == 'medium':
            cols = st.columns([1, 1])
            container = cols[0] if len(st.session_state.get('current_row_widgets', [])) % 2 == 0 else cols[1]
        elif widget_size == 'large':
            cols = st.columns([2, 1])
            container = cols[0]
        else:  # full
            container = st.container()
        
        with container:
            if widget_type == 'market_overview':
                self._render_market_overview(engine, providers)
            elif widget_type == 'price_chart':
                self._render_price_chart(engine, providers)
            elif widget_type == 'orderbook':
                self._render_orderbook(engine, providers)
            elif widget_type == 'trade_history':
                self._render_trade_history(engine, providers)
            elif widget_type == 'portfolio':
                self._render_portfolio(engine, providers)
            elif widget_type == 'alerts':
                self._render_alerts(engine, providers)
            elif widget_type == 'watchlist':
                self._render_watchlist(engine, providers)
            elif widget_type == 'analytics_dashboard':
                self._render_analytics_dashboard(engine, providers)
            elif widget_type == 'arbitrage_opportunities':
                self._render_arbitrage_opportunities(engine, providers)
            elif widget_type == 'risk_metrics':
                self._render_risk_metrics(engine, providers)
    
    def _render_market_overview(self, engine, providers):
        """渲染市场概览"""
        st.markdown('<div class="trading-widget">', unsafe_allow_html=True)
        st.write("### 📊 市场概览")
        
        # 模拟市场数据
        market_data = {
            'BTC/USDT': {'price': 43250.50, 'change': 2.34, 'volume': 1234567890},
            'ETH/USDT': {'price': 2580.75, 'change': -1.23, 'volume': 987654321},
            'BNB/USDT': {'price': 315.20, 'change': 0.89, 'volume': 456789123}
        }
        
        # 使用列布局显示市场数据
        cols = st.columns(len(market_data))
        
        for i, (symbol, data) in enumerate(market_data.items()):
            with cols[i]:
                st.markdown('<div class="market-card">', unsafe_allow_html=True)
                
                # 价格和变化
                price_color = get_price_color(data['change'])
                st.markdown(f"""
                <div style="text-align: center;">
                    <h4 style="color: white; margin: 0;">{symbol}</h4>
                    <h2 style="color: white; margin: 0.5rem 0;">${data['price']:,.2f}</h2>
                    <div class="{price_color}">
                        {"+" if data['change'] > 0 else ""}{data['change']:.2f}%
                    </div>
                    <small style="color: rgba(255,255,255,0.7);">
                        24h 成交量: ${data['volume']:,.0f}
                    </small>
                </div>
                """, unsafe_allow_html=True)
                
                # 简单的价格趋势图
                trend_data = np.random.randn(20).cumsum() + data['price']
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=trend_data,
                    mode='lines',
                    name=symbol,
                    line=dict(
                        width=2,
                        color='#00ff88' if data['change'] > 0 else '#ff4757'
                    )
                ))
                fig.update_layout(
                    height=80,
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=False,
                    xaxis=dict(showticklabels=False, showgrid=False),
                    yaxis=dict(showticklabels=False, showgrid=False),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_price_chart(self, engine, providers):
        """渲染价格图表"""
        st.write("### 📈 价格图表")
        
        # 选择交易对
        symbol = st.selectbox("选择交易对", ["BTC/USDT", "ETH/USDT", "BNB/USDT"], key="chart_symbol")
        
        # 生成模拟K线数据
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='H')
        np.random.seed(42)
        
        price_base = 43000 if symbol == "BTC/USDT" else (2500 if symbol == "ETH/USDT" else 300)
        prices = price_base + np.random.randn(len(dates)).cumsum() * 100
        
        df = pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': prices * (1 + np.random.rand(len(dates)) * 0.02),
            'low': prices * (1 - np.random.rand(len(dates)) * 0.02),
            'close': prices + np.random.randn(len(dates)) * 50,
            'volume': np.random.randint(1000, 10000, len(dates))
        })
        
        # 创建K线图
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f'{symbol} 价格', '成交量'),
            row_width=[0.7, 0.3]
        )
        
        # K线图
        fig.add_trace(
            go.Candlestick(
                x=df['datetime'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="价格"
            ),
            row=1, col=1
        )
        
        # 成交量
        fig.add_trace(
            go.Bar(
                x=df['datetime'],
                y=df['volume'],
                name="成交量",
                marker_color='rgba(158,202,225,0.8)'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=400,
            xaxis_rangeslider_visible=False,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_orderbook(self, engine, providers):
        """渲染订单簿"""
        st.write("### 📋 订单簿")
        
        # 生成模拟订单簿数据
        base_price = 43250.50
        
        # 买单
        buy_orders = []
        for i in range(10):
            price = base_price - (i + 1) * 10
            quantity = np.random.uniform(0.1, 5.0)
            buy_orders.append({'价格': price, '数量': quantity, '总额': price * quantity})
        
        # 卖单
        sell_orders = []
        for i in range(10):
            price = base_price + (i + 1) * 10
            quantity = np.random.uniform(0.1, 5.0)
            sell_orders.append({'价格': price, '数量': quantity, '总额': price * quantity})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**买单 (Bids)**")
            buy_df = pd.DataFrame(buy_orders)
            st.dataframe(
                buy_df.style.format({
                    '价格': '${:.2f}',
                    '数量': '{:.4f}',
                    '总额': '${:.2f}'
                }),
                use_container_width=True
            )
        
        with col2:
            st.write("**卖单 (Asks)**")
            sell_df = pd.DataFrame(sell_orders)
            st.dataframe(
                sell_df.style.format({
                    '价格': '${:.2f}',
                    '数量': '{:.4f}',
                    '总额': '${:.2f}'
                }),
                use_container_width=True
            )
    
    def _render_trade_history(self, engine, providers):
        """渲染交易历史"""
        st.write("### 📜 交易历史")
        
        # 生成模拟交易数据
        trades = []
        for i in range(20):
            trade = {
                '时间': datetime.now() - timedelta(minutes=i*5),
                '交易对': np.random.choice(['BTC/USDT', 'ETH/USDT', 'BNB/USDT']),
                '类型': np.random.choice(['买入', '卖出']),
                '价格': np.random.uniform(40000, 45000),
                '数量': np.random.uniform(0.01, 1.0),
                '手续费': np.random.uniform(1, 10)
            }
            trade['总额'] = trade['价格'] * trade['数量']
            trades.append(trade)
        
        df = pd.DataFrame(trades)
        
        st.dataframe(
            df.style.format({
                '时间': lambda x: x.strftime('%H:%M:%S'),
                '价格': '${:.2f}',
                '数量': '{:.4f}',
                '总额': '${:.2f}',
                '手续费': '${:.2f}'
            }),
            use_container_width=True
        )
    
    def _render_portfolio(self, engine, providers):
        """渲染投资组合"""
        st.write("### 💼 投资组合")
        
        # 模拟投资组合数据
        portfolio = {
            'USDT': {'余额': 10000.00, '价值': 10000.00, '占比': 45.5},
            'BTC': {'余额': 0.2500, '价值': 10812.50, '占比': 49.1},
            'ETH': {'余额': 0.4200, '价值': 1083.90, '占比': 4.9},
            'BNB': {'余额': 0.3500, '价值': 110.32, '占比': 0.5}
        }
        
        total_value = sum(data['价值'] for data in portfolio.values())
        
        st.metric("总资产价值", f"${total_value:,.2f}", delta="+2.34%")
        
        # 资产分布饼图
        labels = list(portfolio.keys())
        values = [data['价值'] for data in portfolio.values()]
        
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
        fig.update_layout(height=300, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # 详细余额表
        portfolio_df = pd.DataFrame.from_dict(portfolio, orient='index')
        st.dataframe(
            portfolio_df.style.format({
                '余额': '{:.4f}',
                '价值': '${:.2f}',
                '占比': '{:.1f}%'
            }),
            use_container_width=True
        )
    
    def _render_alerts(self, engine, providers):
        """渲染预警中心"""
        st.markdown('<div class="trading-widget">', unsafe_allow_html=True)
        st.write("### 🚨 预警中心")
        
        # 模拟预警数据
        alerts = [
            {'时间': '10:30:25', '类型': '价差预警', '消息': 'BTC/USDT 价差超过阈值', '严重程度': 'high'},
            {'时间': '10:25:12', '类型': '套利机会', '消息': '发现ETH跨交易所套利机会', '严重程度': 'medium'},
            {'时间': '10:20:08', '类型': '市场异常', '消息': 'BNB交易量异常增长', '严重程度': 'low'}
        ]
        
        for alert in alerts:
            alert_class = get_alert_class(alert['严重程度'])
            severity_icon = {
                'high': '🔴',
                'medium': '🟡', 
                'low': '🟢'
            }.get(alert['严重程度'], '⚪')
            
            st.markdown(f"""
            <div class="{alert_class} fade-in">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong>{severity_icon} {alert['类型']}</strong>
                        <div style="margin-top: 0.5rem;">{alert['消息']}</div>
                    </div>
                    <div style="font-size: 0.9em; opacity: 0.8;">
                        {alert['时间']}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_watchlist(self, engine, providers):
        """渲染关注列表"""
        st.write("### ⭐ 关注列表")
        
        # 获取用户关注列表
        watchlist = st.session_state.get('watchlist', ['BTC/USDT', 'ETH/USDT'])
        
        # 添加新关注
        new_symbol = st.text_input("添加交易对", placeholder="例如: BNB/USDT")
        if st.button("➕ 添加") and new_symbol:
            if new_symbol not in watchlist:
                watchlist.append(new_symbol)
                st.session_state.watchlist = watchlist
                st.success(f"已添加 {new_symbol}")
                st.rerun()
        
        # 显示关注列表
        for symbol in watchlist:
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**{symbol}**")
            
            with col2:
                # 模拟价格
                price = np.random.uniform(100, 50000)
                change = np.random.uniform(-5, 5)
                st.metric("", f"${price:.2f}", delta=f"{change:+.2f}%")
            
            with col3:
                if st.button("🗑️", key=f"remove_{symbol}"):
                    watchlist.remove(symbol)
                    st.session_state.watchlist = watchlist
                    st.rerun()
    
    def _render_analytics_dashboard(self, engine, providers):
        """渲染分析仪表盘"""
        st.write("### 🔬 分析仪表盘")
        st.info("高级分析功能 - 请参考数据分析中心页面")
    
    def _render_arbitrage_opportunities(self, engine, providers):
        """渲染套利机会"""
        st.write("### 🎯 套利机会")
        st.info("套利机会展示 - 请参考实时仪表盘页面")
    
    def _render_risk_metrics(self, engine, providers):
        """渲染风险指标"""
        st.write("### ⚠️ 风险指标")
        st.info("风险指标监控 - 请参考风险管理页面")

# 全局交易界面实例
trading_interface = TradingInterface()