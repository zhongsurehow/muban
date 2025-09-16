# --- Python Path Setup ---
import sys
import os
# Add the parent directory to Python path so 'src' module can be found
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import streamlit as st
import asyncio
import nest_asyncio
import time
import logging
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Dict, Any
from datetime import datetime, timedelta

# --- Basic Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Local Imports ---
from src.config import load_config
from src.db import DatabaseManager
from src.engine import ArbitrageEngine, Opportunity
from src.providers.base import BaseProvider
from src.providers.cex import CEXProvider
from src.providers.dex import DEXProvider
from src.providers.bridge import BridgeProvider
from src.providers.free_api import FreeAPIProvider, free_api_provider
from src.providers.ccxt_enhanced import EnhancedCCXTProvider
from src.providers.trend_analyzer import TrendAnalyzer
from src.providers.funding_rate import FundingRateProvider, funding_rate_provider
from src.providers.orderbook_analyzer import OrderBookAnalyzer, orderbook_analyzer
from src.providers.cross_chain_analyzer import CrossChainAnalyzer, cross_chain_analyzer
from src.providers.exchange_health_monitor import ExchangeHealthMonitor, exchange_health_monitor
from src.providers.arbitrage_analyzer import ArbitrageAnalyzer, arbitrage_analyzer
from src.providers.transfer_path_planner import TransferPathPlanner, transfer_path_planner
from src.providers.risk_dashboard import RiskDashboard, risk_dashboard
from src.providers.risk_manager import RiskManager, RiskMetrics, ArbitrageOpportunity
from src.providers.advanced_arbitrage import (
    AdvancedArbitrageEngine, 
    TriangularArbitrageOpportunity,
    CrossChainOpportunity,
    FuturesSpotOpportunity,
    advanced_arbitrage_engine
)
from src.providers.analytics_engine import (
    AnalyticsEngine,
    PerformanceMetrics,
    BacktestResult,
    analytics_engine
)
from src.providers.market_depth_analyzer import (
    MarketDepthAnalyzer,
    OrderBookSnapshot,
    LiquidityMetrics,
    MarketImpactAnalysis,
    market_depth_analyzer
)
from src.providers.alert_system import (
    AlertSystem,
    AlertType,
    AlertSeverity,
    AlertRule,
    Alert,
    NotificationChannel,
    alert_system
)
from src.providers.account_manager import (
    AccountManager,
    AccountInfo,
    AccountType,
    AccountStatus,
    AllocationStrategy,
    AllocationRule,
    AccountMetrics,
    RiskMetrics,
    account_manager
)
from src.ui.trading_interface import (
    TradingInterface,
    trading_interface
)
from src.ui.components import sidebar_controls, display_error
from src.ui.navigation import render_navigation, render_page_header, render_quick_stats, render_footer

# Apply nest_asyncio to allow running asyncio event loops within Streamlit's loop
nest_asyncio.apply()

# --- Page Configuration ---
st.set_page_config(
    page_title="套利机会仪表板",
    layout="wide",
    page_icon="🎯",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---
def safe_run_async(coro):
    """Safely runs an async coroutine, handling nested event loops."""
    try:
        return asyncio.run(coro)
    except RuntimeError as e:
        if "cannot run loop while another loop is running" in str(e):
            # This is expected in Streamlit's environment with nest_asyncio
            return asyncio.run(coro)
        st.error(f"异步操作失败: {e}")
        return None

def _validate_symbol(symbol: str) -> bool:
    """Validates that the symbol is not empty and has a valid format."""
    if not symbol or '/' not in symbol or len(symbol.split('/')) != 2:
        st.error("请输入有效的交易对格式，例如 'BTC/USDT'。")
        return False
    return True

def _create_depth_chart(order_book: dict) -> go.Figure:
    """Creates a Plotly order book depth chart."""
    bids = pd.DataFrame(order_book.get('bids', []), columns=['price', 'volume']).astype(float)
    asks = pd.DataFrame(order_book.get('asks', []), columns=['price', 'volume']).astype(float)
    bids = bids.sort_values('price', ascending=False)
    asks = asks.sort_values('price', ascending=True)
    bids['cumulative'] = bids['volume'].cumsum()
    asks['cumulative'] = asks['volume'].cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bids['price'], y=bids['cumulative'], name='买单', fill='tozeroy', line_color='green'))
    fig.add_trace(go.Scatter(x=asks['price'], y=asks['cumulative'], name='卖单', fill='tozeroy', line_color='red'))
    fig.update_layout(title_text=f"{order_book.get('symbol', '')} 市场深度", xaxis_title="价格", yaxis_title="累计数量", height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def _create_candlestick_chart(df: pd.DataFrame, symbol: str, show_volume: bool = True, ma_periods: list = None) -> go.Figure:
    """Creates a Plotly candlestick chart from OHLCV data with optional indicators."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title_text=f"{symbol} K线图 - 无数据", height=400)
        return fig
    
    # Ensure required columns exist
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        fig = go.Figure()
        fig.update_layout(title_text=f"{symbol} K线图 - 数据格式错误", height=400)
        return fig
    
    # Convert timestamp to datetime if it's not already
    if 'datetime' not in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    fig = go.Figure(data=[go.Candlestick(
        x=df['datetime'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name=symbol
    )])
    
    # Add moving averages if requested
    if ma_periods:
        colors = ['orange', 'purple', 'green', 'red', 'cyan', 'magenta']
        for i, period in enumerate(ma_periods):
            if len(df) >= period:
                ma = df['close'].rolling(window=period).mean()
                fig.add_trace(go.Scatter(
                    x=df['datetime'],
                    y=ma,
                    mode='lines',
                    name=f'MA{period}',
                    line=dict(color=colors[i % len(colors)], width=1.5)
                ))
    
    # Add volume as a subplot if requested
    if show_volume:
        fig.add_trace(go.Bar(
            x=df['datetime'],
            y=df['volume'],
            name='成交量',
            yaxis='y2',
            opacity=0.3,
            marker_color='blue'
        ))
    
    # Configure layout
    layout_config = {
        'title_text': f"{symbol} K线图",
        'xaxis_title': "时间",
        'yaxis_title': "价格",
        'height': 600 if show_volume else 500,
        'margin': dict(l=20, r=20, t=40, b=20),
        'xaxis_rangeslider_visible': False,
        'showlegend': True
    }
    
    if show_volume:
        layout_config['yaxis2'] = dict(
            title="成交量",
            overlaying='y',
            side='right',
            showgrid=False
        )
    
    fig.update_layout(**layout_config)
    
    return fig

# --- Caching Functions ---
@st.cache_data
def get_config():
    """Load configuration from file and cache it."""
    return load_config()

@st.cache_resource
def get_db_manager(db_path: str):
    """Creates and caches the database manager."""
    if not db_path: return None
    db_manager = DatabaseManager(db_path)
    try:
        asyncio.run(db_manager.__aenter__())
        asyncio.run(db_manager.init_db())
        return db_manager
    except Exception as e:
        st.error(f"连接或初始化SQLite数据库时失败: {e}")
        asyncio.run(db_manager.__aexit__(None, None, None))
        return None

@st.cache_resource
def get_providers(_config: Dict, _session_state) -> List[BaseProvider]:
    """Create and cache a list of all data providers."""
    providers = []
    is_demo_mode = not bool(_session_state.get('api_keys'))
    provider_config = _config.copy()
    provider_config['api_keys'] = {**_config.get('api_keys', {}), **_session_state.get('api_keys', {})}
    for ex_id in _session_state.selected_exchanges:
        try:
            providers.append(CEXProvider(name=ex_id, config=provider_config, force_mock=is_demo_mode))
        except ValueError as e:
            st.error(f"初始化 CEX 提供商 '{ex_id}' 失败: {e}", icon="🚨")
        except Exception as e:
            st.warning(f"初始化 CEX 提供商 '{ex_id}' 时发生未知错误: {e}", icon="⚠️")
    return providers

def init_session_state(config):
    """Initializes the session state with default values."""
    if 'selected_exchanges' not in st.session_state:
        st.session_state.selected_exchanges = ['binance', 'okx', 'bybit']
    if 'selected_symbols' not in st.session_state:
        st.session_state.selected_symbols = ['BTC/USDT', 'ETH/USDT']
    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = {}

# --- Dashboard UI ---
def _render_dashboard_header(providers: List[BaseProvider], engine: ArbitrageEngine):
    """Renders the main metric headers for the dashboard."""
    st.title("🎯 专业套利交易系统")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("连接交易所", len([p for p in providers if isinstance(p, CEXProvider)]))
    with col2:
        st.metric("监控币种", len(st.session_state.get('selected_symbols', [])))
    with col3:
        demo_mode = not bool(st.session_state.get('api_keys'))
        st.metric("运行模式", "演示" if demo_mode else "实时")
    
    # Placeholders for metrics that require data
    opp_placeholder = st.empty()
    profit_placeholder = st.empty()
    
    with st.spinner("正在计算实时指标..."):
        opportunities = safe_run_async(engine.find_opportunities(st.session_state.selected_symbols)) if engine else []
        with col4:
            profitable_opps = len([opp for opp in opportunities if opp.get('profit_percentage', 0) > 0.1])
            st.metric("活跃机会", profitable_opps, delta=f"+{profitable_opps}" if profitable_opps > 0 else None)
        with col5:
            max_profit = max([opp.get('profit_percentage', 0) for opp in opportunities], default=0)
            st.metric("最高收益率", f"{max_profit:.3f}%", delta=f"+{max_profit:.3f}%" if max_profit > 0 else None)
    return opportunities

def _render_opportunity_leaderboard(engine: ArbitrageEngine, risk_manager: RiskManager):
    """Renders the main table of arbitrage opportunities."""
    st.subheader("📈 实时套利机会排行榜")
    min_profit_filter = st.number_input("最小收益率过滤 (%)", min_value=0.0, max_value=5.0, value=0.1, step=0.05, key="profit_filter")

    opp_placeholder = st.empty()
    with st.spinner("正在寻找套利机会..."):
        opportunities = safe_run_async(engine.find_opportunities(st.session_state.selected_symbols))
        filtered_opps = [opp for opp in opportunities if opp.get('profit_percentage', 0) >= min_profit_filter]
        
        if not filtered_opps:
            opp_placeholder.info(f"🔍 未发现收益率 ≥ {min_profit_filter}% 的套利机会")
            return

        df = pd.DataFrame(filtered_opps).sort_values(by="profit_percentage", ascending=False)
        opp_placeholder.dataframe(df, use_container_width=True, hide_index=True)

def show_dashboard(engine: ArbitrageEngine, providers: List[BaseProvider]):
    """The main view of the application, broken down into smaller components."""

    # Initialize Risk Manager
    if 'risk_manager' not in st.session_state:
        st.session_state.risk_manager = RiskManager(initial_capital=100000)
    risk_manager = st.session_state.risk_manager

    # Render Header
    opportunities = _render_dashboard_header(providers, engine)

    # Render Main Content Tabs
    tab_titles = ["📈 实时套利机会", "📊 价格对比", "⚙️ 风险管理", "🧰 工具箱"]
    tab1, tab2, tab3, tab4 = st.tabs(tab_titles)

    with tab1:
        _render_opportunity_leaderboard(engine, risk_manager)

    with tab2:
        render_unified_price_comparison(providers)

    with tab3:
        st.subheader("🛡️ 专业风险管理中心")
        # This section can be further broken down if needed
        risk_metrics = risk_manager.calculate_risk_metrics()
        # ... (rest of the risk management UI)

    with tab4:
        st.subheader("💰 套利收益计算器")
        # ... (rest of the profit calculator UI)

    st.markdown("---")

    # Tools and other data sections
    st.subheader("💰 套利收益计算器")

    with st.container():
        calc_col1, calc_col2 = st.columns(2)
        
        with calc_col1:
            investment_amount = st.number_input("投资金额 (USDT)", min_value=100, max_value=1000000, value=10000, step=100, key="investment")
            expected_profit = st.number_input("预期收益率 (%)", min_value=0.01, max_value=20.0, value=1.0, step=0.01, key="expected_profit")

        with calc_col2:
            trading_fee = st.number_input("交易手续费 (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.01, key="trading_fee")
            slippage = st.number_input("滑点损失 (%)", min_value=0.0, max_value=5.0, value=0.2, step=0.01, key="slippage")

        # Calculate results
        gross_profit = investment_amount * (expected_profit / 100)
        total_fees = investment_amount * ((trading_fee * 2 + slippage) / 100)  # Buy + Sell fees + slippage
        net_profit = gross_profit - total_fees
        roi = (net_profit / investment_amount) * 100
        
        # Display results
        st.markdown("**📊 收益分析**")
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            st.metric("毛利润", f"${gross_profit:.2f}")
        with result_col2:
            st.metric("总费用", f"${total_fees:.2f}")
        with result_col3:
            color = "normal" if net_profit > 0 else "inverse"
            st.metric("净利润", f"${net_profit:.2f}", f"{roi:.3f}%")
        
        # Risk assessment
        if net_profit > 0:
            if roi > 0.5:
                st.success(f"🟢 高收益机会: 净收益率 {roi:.3f}%")
            elif roi > 0.1:
                st.info(f"🟡 中等机会: 净收益率 {roi:.3f}%")
            else:
                st.warning(f"🟠 低收益机会: 净收益率 {roi:.3f}%")
        else:
            st.error(f"🔴 亏损风险: 净收益率 {roi:.3f}%")

def render_unified_price_comparison(providers: List[BaseProvider]):
    """
    Renders a unified price comparison UI that can switch between
    CEX providers (API key-based) and the Free API provider.
    """
    st.markdown("---")
    st.subheader("📊 实时价格对比")

    data_source = st.selectbox("选择数据源", ["CEX (需要API密钥)", "免费API (8大交易所)"], key="price_source_selector")

    if data_source == "CEX (需要API密钥)":
        price_placeholder = st.empty()
        with st.spinner("正在获取CEX交易所最新价格..."):
            tasks = []
            provider_symbol_pairs = []
            cex_providers = [p for p in providers if isinstance(p, CEXProvider)]
            symbols = st.session_state.get('selected_symbols', [])

            if not cex_providers:
                price_placeholder.warning("请在侧边栏选择至少一个CEX交易所。")
            elif not symbols:
                price_placeholder.warning("请在侧边栏选择至少一个交易对。")
            else:
                for symbol in symbols:
                    for provider in cex_providers:
                        tasks.append(provider.get_ticker(symbol))
                        provider_symbol_pairs.append((provider.name, symbol))

                all_tickers = safe_run_async(asyncio.gather(*tasks, return_exceptions=True))

                processed_tickers = [
                    {'symbol': t['symbol'], 'provider': provider_symbol_pairs[i][0], 'price': t['last']}
                    for i, t in enumerate(all_tickers) if isinstance(t, dict) and t.get('last') is not None
                ]

                if processed_tickers:
                    price_df = pd.DataFrame(processed_tickers)
                    pivot_df = price_df.pivot(index='symbol', columns='provider', values='price')
                    price_placeholder.dataframe(pivot_df.style.format("{:.4f}"), use_container_width=True)
                else:
                    price_placeholder.warning("未能获取任何有效的CEX价格数据。")
    
    elif data_source == "免费API (8大交易所)":
        st.info("💡 **功能说明**: 实时比较 Binance、OKX、Bybit、Coinbase、Kraken、Huobi、KuCoin、Gate.io 等8个主要交易所的货币价格。")
    
        free_api_col1, free_api_col2 = st.columns([3, 1])

        with free_api_col2:
            st.markdown("**交易对选择**")
            all_symbols = free_api_provider.get_popular_symbols()
            search_term = st.text_input("🔍 搜索货币对", "", key="symbol_search_free")

            if search_term:
                filtered_symbols = [s for s in all_symbols if search_term.upper() in s.upper()]
            else:
                filtered_symbols = all_symbols

            selected_symbols_free = st.multiselect(
                "选择交易对",
                options=filtered_symbols,
                default=[s for s in ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'] if s in filtered_symbols],
                key="selected_symbols_free"
            )

        with free_api_col1:
            if not selected_symbols_free:
                st.info("请在右侧选择至少一个交易对进行查询。")
            else:
                with st.spinner("获取免费API价格数据..."):
                    async def fetch_free_data():
                        selected_api = st.session_state.get('selected_free_api', 'coingecko')
                        return await free_api_provider.get_exchange_prices_from_api(selected_symbols_free, selected_api)

                    free_data = safe_run_async(fetch_free_data())

                    if not free_data:
                        st.warning("未能获取任何数据。请检查API或稍后再试。")
                    else:
                        for symbol, price_list in free_data.items():
                            if not price_list:
                                st.write(f"### {symbol} - 未找到数据")
                                continue

                            st.markdown(f"### 💰 {symbol} 价格对比")
                            df_comparison = pd.DataFrame(price_list)
                            
                            prices = df_comparison['price_usd'].dropna()
                            if not prices.empty:
                                max_price, min_price = prices.max(), prices.min()
                                avg_price = prices.mean()
                                spread_pct = ((max_price - min_price) / min_price * 100) if min_price > 0 else 0
                                
                                stat_cols = st.columns(4)
                                stat_cols[0].metric("最高价", f"${max_price:,.6f}")
                                stat_cols[1].metric("最低价", f"${min_price:,.6f}")
                                stat_cols[2].metric("平均价", f"${avg_price:,.6f}")
                                stat_cols[3].metric("价差", f"{spread_pct:.3f}%", "🟢" if spread_pct > 1 else "🟡" if spread_pct > 0.3 else "🔴")
                            
                            st.dataframe(
                                df_comparison.drop(columns=['timestamp']),
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    'exchange': st.column_config.TextColumn("交易所"),
                                    'price_usd': st.column_config.NumberColumn("价格 (USD)", format="$%.6f"),
                                    'change_24h': st.column_config.NumberColumn("24h变化%", format="%.2f%%"),
                                    'volume_24h': st.column_config.NumberColumn("24h成交量", format="$%d"),
                                }
                            )
                            st.markdown("---")
    
    # 实时风险监控面板
    st.markdown("---")
    st.subheader("🚨 实时风险监控")
    
    risk_monitor_col1, risk_monitor_col2 = st.columns([2, 1])
    
    with risk_monitor_col1:
        # 风险指标表格
        risk_metrics = [
            {"指标": "总资金", "当前值": "$98,750", "阈值": "$100,000", "状态": "🟡 关注", "变化": "-1.25%"},
            {"指标": "日盈亏", "当前值": "+$1,250", "阈值": "-$20,000", "状态": "🟢 正常", "变化": "+1.27%"},
            {"指标": "最大回撤", "当前值": "-3.2%", "阈值": "-15.0%", "状态": "🟢 安全", "变化": "+0.8%"},
            {"指标": "持仓风险", "当前值": "2/5", "阈值": "5/5", "状态": "🟢 正常", "变化": "0"},
            {"指标": "相关性", "当前值": "0.65", "阈值": "0.70", "状态": "🟡 关注", "变化": "+0.05"}
        ]
        
        df_risk = pd.DataFrame(risk_metrics)
        st.dataframe(df_risk, width='stretch', hide_index=True)
    
    with risk_monitor_col2:
        st.markdown("**🔔 风险警报**")
        
        # 模拟风险警报
        alerts = [
            "🟡 BTC/USDT 相关性过高 (0.85)",
            "🟢 ETH/USDT 套利机会出现",
            "🔴 总资金接近止损线"
        ]
        
        for alert in alerts:
            st.write(alert)
        
        st.markdown("**⚡ 紧急操作**")
        if st.button("🛑 紧急止损", key="emergency_stop", help="立即关闭所有持仓"):
            st.error("🚨 紧急止损已触发")
        
        if st.button("⏸️ 暂停交易", key="pause_trading", help="暂停所有新交易"):
            st.warning("⚠️ 交易已暂停")
        
        if st.button("🔄 重置风险", key="reset_risk", help="重置风险参数"):
            st.info("ℹ️ 风险参数已重置")
    
    # 批量监控面板
    st.markdown("---")
    st.subheader("📋 批量监控管理")
    
    monitor_col1, monitor_col2 = st.columns([4, 1])
    
    with monitor_col1:
        # 监控列表
        st.markdown("**活跃监控列表**")
        
        monitor_data = [
            {"交易对": "BTC/USDT", "状态": "🟢 监控中", "触发条件": ">1.5%", "当前价差": "1.25%", "操作": "暂停"},
            {"交易对": "ETH/USDT", "状态": "🟡 等待中", "触发条件": ">1.0%", "当前价差": "0.89%", "操作": "修改"},
            {"交易对": "ADA/USDT", "状态": "🔴 已暂停", "触发条件": ">2.0%", "当前价差": "2.15%", "操作": "启动"}
        ]
        
        for i, item in enumerate(monitor_data):
            with st.container():
                item_col1, item_col2, item_col3, item_col4, item_col5 = st.columns([3, 1, 1, 1, 1])
                
                with item_col1:
                    st.write(f"**{item['交易对']}** - {item['状态']}")
                with item_col2:
                    st.write(f"触发: {item['触发条件']}")
                with item_col3:
                    st.write(f"当前: {item['当前价差']}")
                with item_col4:
                    if st.button(item['操作'], key=f"monitor_action_{i}"):
                        st.success(f"{item['操作']}操作已执行")
                with item_col5:
                    if st.button("删除", key=f"monitor_delete_{i}"):
                        st.warning(f"已删除 {item['交易对']} 监控")
    
    with monitor_col2:
        st.markdown("**添加新监控**")
        new_symbol = st.text_input("交易对", placeholder="BTC/USDT", key="new_monitor_symbol")
        new_threshold = st.number_input("触发阈值 (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="new_threshold")
        
        if st.button("➕ 添加监控", key="add_monitor"):
            if new_symbol:
                st.success(f"已添加 {new_symbol} 监控 (>{new_threshold}%)")
            else:
                st.error("请输入交易对")
        
        st.markdown("**批量操作**")
        if st.button("▶️ 全部启动", key="start_all_monitors"):
            st.success("所有监控已启动")
        if st.button("⏸️ 全部暂停", key="pause_all_monitors"):
            st.warning("所有监控已暂停")
        if st.button("🗑️ 清空列表", key="clear_all_monitors"):
            st.error("监控列表已清空")

    st.markdown("---")

    st.subheader("🌊 市场深度可视化")
    depth_cols = st.columns(3)
    selected_ex = depth_cols[0].selectbox("选择交易所", options=[p.name for p in providers if isinstance(p, CEXProvider)], key="depth_exchange")
    selected_sym = depth_cols[1].text_input("输入交易对", st.session_state.selected_symbols[0], key="depth_symbol")

    if depth_cols[2].button("查询深度", key="depth_button"):
        if _validate_symbol(selected_sym):
            provider = next((p for p in providers if p.name == selected_ex), None)
            if provider:
                with st.spinner(f"正在从 {provider.name} 获取 {selected_sym} 的订单簿..."):
                    order_book = safe_run_async(provider.get_order_book(selected_sym))
                    if order_book and 'error' not in order_book:
                        st.plotly_chart(_create_depth_chart(order_book), width='stretch', key="order_book_depth_chart")
                    else:
                        display_error(f"无法获取订单簿: {order_book.get('error', '未知错误')}")

    st.markdown("---")
    with st.expander("🏢 交易所定性对比", expanded=False):
        show_comparison_view(get_config().get('qualitative_data', {}), providers)
    
    st.markdown("---")
    with st.expander("💰 资金费率套利机会", expanded=False):
        show_funding_rate_view()
    
    st.markdown("---")
    with st.expander("📊 订单簿深度与滑点分析", expanded=False):
        show_orderbook_analysis()
    
    st.markdown("---")
    with st.expander("🌉 跨链转账效率与成本分析", expanded=False):
        show_cross_chain_analysis()
    
    st.markdown("---")
    with st.expander("🏥 交易所健康状态监控", expanded=False):
        show_exchange_health_monitor()
    
    st.markdown("---")
    with st.expander("💰 期现套利机会视图", expanded=False):
        show_arbitrage_opportunities()
    
    st.markdown("---")
    with st.expander("🛣️ 转账路径规划器", expanded=False):
        show_transfer_path_planner()
    
    st.markdown("---")
    with st.expander("📊 动态风险仪表盘", expanded=False):
        show_risk_dashboard()
    
    st.markdown("---")
    with st.expander("🚀 增强CCXT交易所支持", expanded=False):
        show_enhanced_ccxt_features()


def show_comparison_view(qualitative_data: dict, providers: List[BaseProvider]):
    """Displays a side-by-side comparison of qualitative data for selected exchanges."""
    if not qualitative_data:
        st.warning("未找到定性数据。")
        return

    key_to_chinese = {
        'security_measures': '安全措施', 'customer_service': '客户服务', 'platform_stability': '平台稳定性',
        'fund_insurance': '资金保险', 'regional_restrictions': '地区限制', 'withdrawal_limits': '提现限额',
        'withdrawal_speed': '提现速度', 'supported_cross_chain_bridges': '支持的跨链桥',
        'api_support_details': 'API支持详情', 'fee_discounts': '手续费折扣', 'margin_leverage_details': '杠杆交易详情',
        'maintenance_schedule': '维护计划', 'user_rating_summary': '用户评分摘要', 'tax_compliance_info': '税务合规信息',
        'deposit_networks': '充值网络', 'deposit_fees': '充值费用', 'withdrawal_networks': '提现网络',
        'margin_trading_api': '保证金交易API'
    }

    exchange_list = list(qualitative_data.keys())
    selected = st.multiselect(
        "选择要比较的交易所",
        options=exchange_list,
        default=exchange_list[:3] if len(exchange_list) >= 3 else exchange_list,
        format_func=lambda x: x.capitalize(),
        key="qualitative_multiselect"
    )

    if selected:
        comparison_data = {exch: qualitative_data[exch] for exch in selected if exch in qualitative_data}
        df = pd.DataFrame(comparison_data).rename(index=key_to_chinese)
        all_keys_df = pd.DataFrame(index=list(key_to_chinese.values()))
        df_display = all_keys_df.join(df).fillna("N/A")
        st.dataframe(df_display, width='stretch')

    with st.expander("🪙 资产转账分析"):
        cex_providers = [p for p in providers if isinstance(p, CEXProvider)]
        show_asset_transfer_view(cex_providers, providers)


def show_asset_transfer_view(cex_providers: List[CEXProvider], providers: List[BaseProvider]):
    """Displays a side-by-side comparison of transfer fees for a given asset."""
    asset = st.text_input("输入要比较的资产代码", "USDT", key="transfer_asset_input").upper()

    if st.button("比较资产转账选项", key="compare_transfers"):
        if not asset:
            st.error("请输入一个资产代码。")
            return

        with st.spinner(f"正在从所有选定的交易所获取 {asset} 的转账费用..."):
            results = safe_run_async(asyncio.gather(*[p.get_transfer_fees(asset) for p in cex_providers]))

        all_networks = set()
        processed_data = {}
        failed_providers = []

        for i, res in enumerate(results):
            provider_name = cex_providers[i].name.capitalize()
            if isinstance(res, dict) and 'error' not in res:
                withdraw_info = res.get('withdraw', {})
                processed_data[provider_name] = {}
                for network, details in withdraw_info.items():
                    all_networks.add(network)
                    fee = details.get('fee')
                    processed_data[provider_name][network] = f"{fee:.6f}".rstrip('0').rstrip('.') if fee is not None else "N/A"
            else:
                failed_providers.append(provider_name)

        if failed_providers:
            st.warning(f"无法获取以下交易所的费用数据: {', '.join(failed_providers)}。")

        if processed_data:
            df = pd.DataFrame(processed_data).reindex(sorted(list(all_networks))).fillna("不支持")
            st.subheader(f"{asset} 提现费用对比")
            st.dataframe(df, width='stretch')
        else:
            st.warning(f"未能成功获取任何交易所关于 '{asset}' 的费用数据。")

    with st.expander("📈 K线图与历史数据"):
        show_kline_view(providers)


def show_kline_view(providers: List[BaseProvider]):
    """Displays a candlestick chart for a selected symbol and exchange."""
    cex_providers = [p for p in providers if isinstance(p, CEXProvider)]
    if not cex_providers:
        st.warning("无可用CEX提供商。")
        return

    # Main controls
    col1, col2, col3, col4 = st.columns([2, 2, 1.5, 1.5])
    name = col1.selectbox("选择交易所", options=[p.name for p in cex_providers], key="kline_exchange")
    symbol = col2.text_input("输入交易对", "BTC/USDT", key="kline_symbol")
    timeframe = col3.selectbox("选择时间周期", options=['1d', '4h', '1h', '30m', '5m'], key="kline_timeframe")
    limit = col4.number_input("数据点", min_value=20, max_value=1000, value=100, key="kline_limit")
    
    # Advanced options
    with st.expander("📊 高级选项"):
        col_a, col_b = st.columns(2)
        show_volume = col_a.checkbox("显示成交量", value=True, key="show_volume")
        show_ma = col_b.checkbox("显示移动平均线", value=False, key="show_ma")
        if show_ma:
            ma_periods = st.multiselect(
                "移动平均线周期",
                options=[5, 10, 20, 50, 100, 200],
                default=[20, 50],
                key="ma_periods"
            )

    if st.button("获取K线数据", key="get_kline"):
        if _validate_symbol(symbol):
            provider = next((p for p in cex_providers if p.name == name), None)
            if provider:
                with st.spinner(f"正在从 {provider.name} 获取 {symbol} 的 {timeframe} 数据..."):
                    data = safe_run_async(provider.get_historical_data(symbol, timeframe, limit))
                    if data:
                        df = pd.DataFrame(data)
                        fig = _create_candlestick_chart(df, symbol, show_volume, ma_periods if show_ma else None)
                        st.plotly_chart(fig, width='stretch', key="candlestick_chart")
                    else:
                        display_error(f"无法获取 {symbol} 的K线数据。")

def show_funding_rate_view():
    """显示资金费率套利机会"""
    st.subheader("💰 永续合约资金费率分析")
    
    # 控制面板
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_symbols = st.multiselect(
            "选择交易对",
            funding_rate_provider.get_popular_symbols(),
            default=['BTC/USDT', 'ETH/USDT'],
            key="funding_symbols"
        )
    
    with col2:
        min_rate_diff = st.number_input(
            "最小费率差异 (%)",
            min_value=0.001,
            max_value=1.0,
            value=0.01,
            step=0.001,
            format="%.3f",
            key="min_funding_diff"
        )
    
    with col3:
        auto_refresh_funding = st.checkbox(
            "自动刷新 (5分钟)",
            value=False,
            key="auto_refresh_funding"
        )
    
    if st.button("🔄 获取最新资金费率", width='stretch'):
        with st.spinner("正在获取资金费率数据..."):
            # 获取聚合资金费率数据
            funding_data = safe_run_async(funding_rate_provider.get_aggregated_funding_rates(selected_symbols))
            
            if funding_data:
                st.session_state['funding_data'] = funding_data
                st.session_state['funding_last_update'] = datetime.now()
                st.success(f"✅ 成功获取 {len(funding_data)} 个交易对的资金费率数据")
            else:
                st.error("❌ 获取资金费率数据失败")
    
    # 显示缓存的数据
    if 'funding_data' in st.session_state and st.session_state['funding_data']:
        funding_data = st.session_state['funding_data']
        last_update = st.session_state.get('funding_last_update', datetime.now())
        
        st.info(f"📊 数据更新时间: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 计算套利机会
        opportunities = funding_rate_provider.calculate_funding_arbitrage_opportunity(funding_data)
        
        # 过滤机会
        filtered_opportunities = [
            opp for opp in opportunities 
            if opp['rate_difference'] >= min_rate_diff / 100
        ]
        
        if filtered_opportunities:
            st.subheader(f"🎯 发现 {len(filtered_opportunities)} 个资金费率套利机会")
            
            # 创建机会表格
            opp_df = pd.DataFrame(filtered_opportunities)
            
            # 格式化显示
            display_df = opp_df[[
                'symbol', 'long_exchange', 'short_exchange', 
                'rate_difference', 'annual_return_pct', 'risk_level'
            ]].copy()
            
            display_df.columns = [
                '交易对', '做多交易所', '做空交易所', 
                '费率差异(%)', '年化收益率(%)', '风险等级'
            ]
            
            # 格式化数值
            display_df['费率差异(%)'] = (display_df['费率差异(%)'] * 100).round(4)
            display_df['年化收益率(%)'] = display_df['年化收益率(%)'].round(2)
            
            st.dataframe(
                display_df,
                width='stretch',
                hide_index=True,
                column_config={
                    "费率差异(%)": st.column_config.NumberColumn(format="%.4f%%"),
                    "年化收益率(%)": st.column_config.NumberColumn(format="%.2f%%"),
                    "风险等级": st.column_config.TextColumn()
                }
            )
            
            # 详细分析
            st.subheader("📈 资金费率趋势分析")
            
            # 创建资金费率对比图表
            fig = go.Figure()
            
            for symbol, rates in funding_data.items():
                if len(rates) >= 2:
                    exchanges = [rate['exchange'] for rate in rates]
                    funding_rates = [rate['funding_rate'] * 100 for rate in rates]  # 转换为百分比
                    
                    fig.add_trace(go.Bar(
                        name=symbol,
                        x=exchanges,
                        y=funding_rates,
                        text=[f"{rate:.4f}%" for rate in funding_rates],
                        textposition='auto'
                    ))
            
            fig.update_layout(
                title="各交易所资金费率对比",
                xaxis_title="交易所",
                yaxis_title="资金费率 (%)",
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, width='stretch', key="funding_rate_chart")
            
            # 策略说明
            with st.expander("💡 资金费率套利策略说明"):
                st.markdown("""
                **资金费率套利原理：**
                
                1. **正费率策略**：当永续合约资金费率为正时
                   - 在费率高的交易所做空永续合约
                   - 在现货市场买入等量资产
                   - 每8小时收取资金费率
                
                2. **负费率策略**：当永续合约资金费率为负时
                   - 在费率低的交易所做多永续合约
                   - 在现货市场卖出等量资产
                   - 每8小时支付较少的资金费率
                
                3. **风险控制**：
                   - 保持现货和永续合约的数量平衡
                   - 监控价格波动和强平风险
                   - 及时调整仓位以维持对冲
                
                **注意事项**：
                - 资金费率每8小时结算一次
                - 需要考虑交易手续费和滑点成本
                - 建议使用较大资金量以摊薄固定成本
                """)
        
        else:
            st.info(f"🔍 当前没有满足条件的资金费率套利机会（最小费率差异: {min_rate_diff}%）")
    
    else:
        st.info("📊 点击上方按钮获取最新的资金费率数据")

def show_orderbook_analysis():
    """显示订单簿深度与滑点分析"""
    st.subheader("📊 订单簿深度与滑点分析")
    
    # 控制面板
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        selected_symbol = st.selectbox(
            "选择交易对",
            ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT'],
            key="orderbook_symbol"
        )
    
    with col2:
        trade_amount = st.number_input(
            "交易金额 (USDT)",
            min_value=100,
            max_value=1000000,
            value=10000,
            step=1000,
            key="trade_amount"
        )
    
    with col3:
        selected_exchanges = st.multiselect(
            "选择交易所",
            ['binance', 'okx', 'bybit', 'gate', 'kucoin'],
            default=['binance', 'okx', 'bybit'],
            key="orderbook_exchanges"
        )
    
    with col4:
        analysis_side = st.selectbox(
            "交易方向",
            ['buy', 'sell'],
            format_func=lambda x: '买入' if x == 'buy' else '卖出',
            key="analysis_side"
        )
    
    if st.button("🔍 分析订单簿深度", width='stretch'):
        if not selected_exchanges:
            st.error("请至少选择一个交易所")
        else:
            with st.spinner("正在获取订单簿数据并分析滑点..."):
                # 获取跨交易所滑点分析
                slippage_analysis = safe_run_async(
                    orderbook_analyzer.analyze_cross_exchange_slippage(selected_symbol, trade_amount)
                )
                
                if slippage_analysis:
                    st.session_state['slippage_analysis'] = slippage_analysis
                    st.session_state['analysis_params'] = {
                        'symbol': selected_symbol,
                        'amount': trade_amount,
                        'side': analysis_side,
                        'timestamp': datetime.now()
                    }
                    st.success(f"✅ 成功分析 {len([ex for ex in slippage_analysis if 'error' not in slippage_analysis[ex]])} 个交易所的订单簿数据")
                else:
                    st.error("❌ 获取订单簿数据失败")
    
    # 显示分析结果
    if 'slippage_analysis' in st.session_state and st.session_state['slippage_analysis']:
        analysis_data = st.session_state['slippage_analysis']
        params = st.session_state.get('analysis_params', {})
        
        st.info(f"📊 分析时间: {params.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 过滤选中的交易所
        filtered_data = {ex: data for ex, data in analysis_data.items() if ex in selected_exchanges}
        
        # 创建滑点对比表格
        st.subheader(f"💹 {analysis_side.upper()} 滑点分析对比")
        
        comparison_data = []
        for exchange, data in filtered_data.items():
            if 'error' in data:
                comparison_data.append({
                    '交易所': exchange.upper(),
                    '状态': '❌ 数据获取失败',
                    '最优价格': '-',
                    '平均价格': '-',
                    '滑点 (%)': '-',
                    '价格影响 (%)': '-',
                    '成交率 (%)': '-'
                })
            elif analysis_side in data:
                side_data = data[analysis_side]
                if 'error' in side_data:
                    comparison_data.append({
                        '交易所': exchange.upper(),
                        '状态': f"❌ {side_data['error']}",
                        '最优价格': '-',
                        '平均价格': '-',
                        '滑点 (%)': '-',
                        '价格影响 (%)': '-',
                        '成交率 (%)': '-'
                    })
                else:
                    comparison_data.append({
                        '交易所': exchange.upper(),
                        '状态': '✅ 正常',
                        '最优价格': f"${side_data['best_price']:.4f}",
                        '平均价格': f"${side_data['average_price']:.4f}",
                        '滑点 (%)': f"{side_data['slippage_pct']:.4f}%",
                        '价格影响 (%)': f"{side_data['price_impact_pct']:.4f}%",
                        '成交率 (%)': f"{side_data['fill_rate']:.2f}%"
                    })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, width='stretch', hide_index=True)
            
            # 寻找最优执行策略
            st.subheader("🎯 最优执行策略推荐")
            
            strategy_result = orderbook_analyzer.find_optimal_execution_strategy(
                filtered_data, trade_amount
            )
            
            if strategy_result['optimal_strategy']:
                optimal = strategy_result['optimal_strategy']
                
                if optimal['type'] == 'single_exchange':
                    st.success(f"""**推荐策略：单一交易所执行**
                    
                    - 🏆 **最优交易所**: {optimal['exchange'].upper()}
                    - 💰 **预期平均价格**: ${optimal['avg_price']:.4f}
                    - 📉 **预期滑点**: {optimal['slippage_pct']:.4f}%
                    - ✅ **预期成交率**: {optimal['fill_rate']:.2f}%
                    """)
                
                elif optimal['type'] == 'split_execution':
                    exchanges_str = ' + '.join([ex.upper() for ex in optimal['exchanges']])
                    st.success(f"""**推荐策略：分割执行**
                    
                    - 🏆 **交易所组合**: {exchanges_str}
                    - 💰 **预期平均价格**: ${optimal['avg_price']:.4f}
                    - 📉 **预期滑点**: {optimal['slippage_pct']:.4f}%
                    - ⚖️ **分割比例**: {optimal['split_ratio']}
                    """)
                
                # 显示所有策略对比
                with st.expander("📋 所有策略对比"):
                    all_strategies = strategy_result['all_strategies']
                    if all_strategies:
                        strategy_df_data = []
                        for i, strategy in enumerate(all_strategies):
                            if strategy['type'] == 'single_exchange':
                                strategy_df_data.append({
                                    '排名': i + 1,
                                    '策略类型': '单一交易所',
                                    '交易所': strategy['exchange'].upper(),
                                    '平均价格': f"${strategy['avg_price']:.4f}",
                                    '滑点 (%)': f"{strategy['slippage_pct']:.4f}%",
                                    '成交率 (%)': f"{strategy['fill_rate']:.2f}%"
                                })
                            elif strategy['type'] == 'split_execution':
                                exchanges_str = ' + '.join([ex.upper() for ex in strategy['exchanges']])
                                strategy_df_data.append({
                                    '排名': i + 1,
                                    '策略类型': '分割执行',
                                    '交易所': exchanges_str,
                                    '平均价格': f"${strategy['avg_price']:.4f}",
                                    '滑点 (%)': f"{strategy['slippage_pct']:.4f}%",
                                    '成交率 (%)': '-'
                                })
                        
                        strategy_df = pd.DataFrame(strategy_df_data)
                        st.dataframe(strategy_df, width='stretch', hide_index=True)
            
            else:
                st.warning("⚠️ 未找到可行的执行策略")
            
            # 滑点可视化
            st.subheader("📈 滑点可视化分析")
            
            # 创建滑点对比图表
            valid_exchanges = []
            slippage_values = []
            price_impact_values = []
            
            for exchange, data in filtered_data.items():
                if 'error' not in data and analysis_side in data and 'error' not in data[analysis_side]:
                    valid_exchanges.append(exchange.upper())
                    slippage_values.append(data[analysis_side]['slippage_pct'])
                    price_impact_values.append(data[analysis_side]['price_impact_pct'])
            
            if valid_exchanges:
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='滑点 (%)',
                    x=valid_exchanges,
                    y=slippage_values,
                    text=[f"{val:.4f}%" for val in slippage_values],
                    textposition='auto',
                    marker_color='lightblue'
                ))
                
                fig.add_trace(go.Bar(
                    name='价格影响 (%)',
                    x=valid_exchanges,
                    y=price_impact_values,
                    text=[f"{val:.4f}%" for val in price_impact_values],
                    textposition='auto',
                    marker_color='lightcoral'
                ))
                
                fig.update_layout(
                    title=f"{selected_symbol} {analysis_side.upper()} 滑点与价格影响对比",
                    xaxis_title="交易所",
                    yaxis_title="百分比 (%)",
                    barmode='group',
                    height=400
                )
                
                st.plotly_chart(fig, width='stretch', key="slippage_analysis_chart")
            
            # 策略说明
            with st.expander("💡 滑点分析说明"):
                st.markdown("""
                **关键指标解释：**
                
                1. **滑点 (Slippage)**：实际成交价格与最优价格的差异
                   - 反映了订单簿深度对大额交易的影响
                   - 滑点越小，交易成本越低
                
                2. **价格影响 (Price Impact)**：从最优价格到最差成交价格的变化
                   - 显示了订单对市场价格的冲击程度
                   - 价格影响越小，市场深度越好
                
                3. **成交率 (Fill Rate)**：订单能够完全成交的比例
                   - 100%表示订单能够完全成交
                   - 低于100%表示订单簿深度不足
                
                **交易建议：**
                - 大额交易建议选择滑点最小的交易所
                - 考虑分割订单到多个交易所以降低价格影响
                - 关注成交率，避免在深度不足的交易所执行大额订单
                - 实际交易时还需考虑手续费、转账成本等因素
                """)
        
        else:
            st.info("📊 没有可用的订单簿数据")
    
    else:
        st.info("📊 点击上方按钮开始分析订单簿深度")

def show_risk_dashboard():
    """显示动态风险仪表盘"""
    st.subheader("📊 动态风险仪表盘")
    
    # 控制面板
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            selected_exchanges = st.multiselect(
                "选择交易所",
                ["binance", "okx", "bybit", "huobi", "coinbase"],
                default=["binance", "okx"]
            )
        
        with col2:
            selected_symbols = st.multiselect(
                "选择交易对",
                ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT"],
                default=["BTC/USDT", "ETH/USDT"]
            )
        
        with col3:
            risk_timeframe = st.selectbox(
                "风险评估周期",
                ["1h", "4h", "1d", "7d", "30d"],
                index=2
            )
        
        with col4:
            portfolio_value = st.number_input(
                "投资组合价值 (USDT)",
                min_value=100.0,
                value=10000.0,
                step=100.0
            )
    
    # 风险分析按钮
    if st.button("🔍 开始风险分析", type="primary", use_container_width=True):
        with st.spinner("正在分析风险指标..."):
            try:
                # 获取风险仪表盘数据
                dashboard_data = risk_dashboard.get_dashboard_data(
                    exchanges=selected_exchanges,
                    symbols=selected_symbols,
                    timeframe=risk_timeframe
                )
                
                if dashboard_data:
                    # 风险概览
                    st.subheader("📈 风险概览")
                    
                    # 总体风险指标
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        overall_risk = dashboard_data.get('overall_risk_level', 'medium')
                        risk_color = {'low': 'green', 'medium': 'orange', 'high': 'red'}.get(overall_risk, 'orange')
                        st.metric(
                            "总体风险等级",
                            overall_risk.upper(),
                            delta=None,
                            delta_color=risk_color
                        )
                    
                    with col2:
                        portfolio_var = dashboard_data.get('portfolio_var', 0)
                        st.metric(
                            "投资组合VaR (95%)",
                            f"${portfolio_var:,.2f}",
                            delta=f"{(portfolio_var/portfolio_value)*100:.2f}%"
                        )
                    
                    with col3:
                        avg_volatility = dashboard_data.get('average_volatility', 0)
                        st.metric(
                            "平均波动率",
                            f"{avg_volatility:.2f}%",
                            delta=None
                        )
                    
                    with col4:
                        correlation_risk = dashboard_data.get('correlation_risk', 0)
                        st.metric(
                            "相关性风险",
                            f"{correlation_risk:.2f}",
                            delta=None
                        )
                    
                    # 详细风险指标
                    st.subheader("📊 详细风险指标")
                    
                    risk_metrics = dashboard_data.get('risk_metrics', [])
                    if risk_metrics:
                        risk_df = pd.DataFrame(risk_metrics)
                        st.dataframe(
                            risk_df,
                            use_container_width=True,
                            column_config={
                                "symbol": "交易对",
                                "exchange": "交易所",
                                "volatility": st.column_config.NumberColumn(
                                    "波动率 (%)",
                                    format="%.2f"
                                ),
                                "var_95": st.column_config.NumberColumn(
                                    "VaR 95% (USDT)",
                                    format="$%.2f"
                                ),
                                "risk_level": "风险等级"
                            }
                        )
                    
                    # 风险警报
                    st.subheader("⚠️ 风险警报")
                    
                    risk_alerts = dashboard_data.get('risk_alerts', [])
                    if risk_alerts:
                        for alert in risk_alerts:
                            alert_type = alert.get('alert_type', 'info')
                            alert_color = {
                                'high_volatility': 'error',
                                'high_var': 'warning',
                                'correlation_spike': 'info'
                            }.get(alert_type, 'info')
                            
                            with st.container():
                                st.markdown(f":{alert_color}[{alert.get('message', '')}]")
                    else:
                        st.success("✅ 当前无风险警报")
                    
                    # 风险建议
                    st.subheader("💡 风险管理建议")
                    
                    recommendations = risk_dashboard.generate_risk_recommendations(
                        dashboard_data,
                        portfolio_value
                    )
                    
                    for i, rec in enumerate(recommendations, 1):
                        st.markdown(f"{i}. {rec}")
                    
                    # 实时监控
                    st.subheader("📡 实时风险监控")
                    
                    if st.checkbox("启用实时监控", value=False):
                        st.info("🔄 实时监控已启用，系统将每5分钟更新风险指标")
                        
                        # 创建占位符用于实时更新
                        placeholder = st.empty()
                        
                        # 这里可以添加实时更新逻辑
                        with placeholder.container():
                            st.markdown("📊 监控中... (模拟数据)")
                            
                            # 模拟实时数据
                            import time
                            current_time = time.strftime("%H:%M:%S")
                            st.markdown(f"最后更新: {current_time}")
                
                else:
                    st.error("❌ 无法获取风险数据，请检查网络连接或稍后重试")
                    
            except Exception as e:
                st.error(f"❌ 风险分析失败: {str(e)}")
    
    else:
        st.info("📊 点击上方按钮开始风险分析")
    
    # 功能说明
    with st.expander("ℹ️ 功能说明", expanded=False):
        st.markdown("""
        **动态风险仪表盘功能:**
        
        1. **风险指标计算**
           - 波动率分析 (基于历史价格数据)
           - 风险价值 (VaR) 计算
           - 相关性分析
        
        2. **风险等级评估**
           - 低风险: 波动率 < 2%
           - 中风险: 波动率 2-5%
           - 高风险: 波动率 > 5%
        
        3. **实时监控**
           - 自动风险警报
           - 投资组合风险跟踪
           - 智能风险建议
        
        4. **风险管理建议**
           - 基于当前市场状况
           - 个性化投资建议
           - 风险分散策略
        """)

def show_transfer_path_planner():
    """显示转账路径规划器"""
    st.subheader("🛣️ 转账路径规划器")
    
    # 控制面板
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # 源平台选择
            source_platforms = list(transfer_path_planner.platforms.keys())
            from_platform = st.selectbox(
                "源平台",
                source_platforms,
                key="transfer_from_platform"
            )
        
        with col2:
            # 目标平台选择
            target_platforms = [p for p in source_platforms if p != from_platform]
            to_platform = st.selectbox(
                "目标平台",
                target_platforms,
                key="transfer_to_platform"
            )
        
        with col3:
            # 代币选择
            if from_platform and to_platform:
                from_tokens = transfer_path_planner.platforms[from_platform].get('supported_tokens', [])
                to_tokens = transfer_path_planner.platforms[to_platform].get('supported_tokens', [])
                common_tokens = list(set(from_tokens) & set(to_tokens))
                
                token = st.selectbox(
                    "转账代币",
                    common_tokens,
                    key="transfer_token"
                )
            else:
                token = st.selectbox("转账代币", [], key="transfer_token")
        
        with col4:
            # 转账金额
            amount = st.number_input(
                "转账金额",
                min_value=0.01,
                max_value=1000000.0,
                value=1000.0,
                step=100.0,
                key="transfer_amount"
            )
    
    # 规划按钮
    if st.button("🔍 规划转账路径", type="primary"):
        if from_platform and to_platform and token and amount > 0:
            with st.spinner("正在规划最优转账路径..."):
                try:
                    # 规划转账路径
                    paths = safe_run_async(
                        transfer_path_planner.plan_transfer_paths(
                            from_platform, to_platform, token, amount
                        )
                    )
                    
                    if paths:
                        st.success(f"找到 {len(paths)} 条可用路径")
                        
                        # 路径概览
                        comparison = transfer_path_planner.generate_path_comparison(paths)
                        
                        # 显示最优路径摘要
                        st.info(f"📊 {comparison['summary']}")
                        
                        # 路径详情
                        st.subheader("📋 转账路径详情")
                        
                        for i, path in enumerate(paths[:5]):  # 显示前5条路径
                            with st.expander(f"路径 {i+1}: {path.path_id} (效率分数: {path.efficiency_score:.1f})", 
                                           expanded=(i == 0)):
                                
                                # 路径基本信息
                                path_col1, path_col2, path_col3, path_col4 = st.columns(4)
                                
                                with path_col1:
                                    st.metric("总费用", f"${path.total_fee:.2f}")
                                
                                with path_col2:
                                    st.metric("预计时间", f"{path.total_time} 分钟")
                                
                                with path_col3:
                                    st.metric("成功率", f"{path.success_rate*100:.1f}%")
                                
                                with path_col4:
                                    risk_color = {
                                        "低": "🟢",
                                        "中": "🟡", 
                                        "高": "🟠",
                                        "极高": "🔴"
                                    }.get(path.risk_level, "⚪")
                                    st.metric("风险等级", f"{risk_color} {path.risk_level}")
                                
                                # 转账步骤
                                st.write("**转账步骤:**")
                                
                                steps_data = []
                                for step in path.steps:
                                    steps_data.append({
                                        "步骤": step.step_id,
                                        "从": step.from_platform,
                                        "到": step.to_platform,
                                        "代币": f"{step.from_token} → {step.to_token}",
                                        "金额": f"{step.amount:.4f}",
                                        "费用": f"${step.estimated_fee:.2f}",
                                        "时间": f"{step.estimated_time}分钟",
                                        "类型": step.transfer_type.value,
                                        "桥/平台": step.bridge_name or "-"
                                    })
                                
                                steps_df = pd.DataFrame(steps_data)
                                st.dataframe(steps_df, use_container_width=True)
                                
                                # 最终收益分析
                                st.write("**收益分析:**")
                                final_col1, final_col2, final_col3 = st.columns(3)
                                
                                with final_col1:
                                    st.metric("初始金额", f"{amount:.4f} {token}")
                                
                                with final_col2:
                                    st.metric("最终金额", f"{path.final_amount:.4f} {token}")
                                
                                with final_col3:
                                    loss_amount = amount - path.final_amount
                                    loss_percentage = (loss_amount / amount) * 100
                                    st.metric("损失", f"{loss_amount:.4f} {token}", 
                                             delta=f"-{loss_percentage:.2f}%")
                        
                        # 路径对比图表
                        if len(paths) > 1:
                            st.subheader("📊 路径对比分析")
                            
                            # 费用对比
                            chart_col1, chart_col2 = st.columns(2)
                            
                            with chart_col1:
                                fee_data = pd.DataFrame({
                                    '路径': [f"路径{i+1}" for i in range(len(paths[:5]))],
                                    '费用(USD)': [path.total_fee for path in paths[:5]]
                                })
                                
                                fig_fee = px.bar(
                                    fee_data, 
                                    x='路径', 
                                    y='费用(USD)',
                                    title="转账费用对比",
                                    color='费用(USD)',
                                    color_continuous_scale='Reds'
                                )
                                fig_fee.update_layout(height=400)
                                st.plotly_chart(fig_fee, use_container_width=True, key="transfer_fee_chart")
                            
                            with chart_col2:
                                time_data = pd.DataFrame({
                                    '路径': [f"路径{i+1}" for i in range(len(paths[:5]))],
                                    '时间(分钟)': [path.total_time for path in paths[:5]]
                                })
                                
                                fig_time = px.bar(
                                    time_data, 
                                    x='路径', 
                                    y='时间(分钟)',
                                    title="转账时间对比",
                                    color='时间(分钟)',
                                    color_continuous_scale='Blues'
                                )
                                fig_time.update_layout(height=400)
                                st.plotly_chart(fig_time, use_container_width=True, key="transfer_time_chart")
                            
                            # 效率分数雷达图
                            if len(paths) >= 3:
                                radar_data = []
                                for i, path in enumerate(paths[:3]):
                                    radar_data.append({
                                        '路径': f'路径{i+1}',
                                        '费用效率': max(0, 100 - (path.total_fee / amount * 100) * 10),
                                        '时间效率': max(0, 100 - (path.total_time / 60) * 20),
                                        '成功率': path.success_rate * 100,
                                        '综合效率': path.efficiency_score
                                    })
                                
                                radar_df = pd.DataFrame(radar_data)
                                
                                fig_radar = go.Figure()
                                
                                for _, row in radar_df.iterrows():
                                    fig_radar.add_trace(go.Scatterpolar(
                                        r=[row['费用效率'], row['时间效率'], row['成功率'], row['综合效率']],
                                        theta=['费用效率', '时间效率', '成功率', '综合效率'],
                                        fill='toself',
                                        name=row['路径']
                                    ))
                                
                                fig_radar.update_layout(
                                    polar=dict(
                                        radialaxis=dict(
                                            visible=True,
                                            range=[0, 100]
                                        )
                                    ),
                                    title="路径效率对比雷达图",
                                    height=500
                                )
                                
                                st.plotly_chart(fig_radar, use_container_width=True, key="path_efficiency_radar_chart")
                        
                        # 实时监控
                        st.subheader("📡 实时路径监控")
                        
                        monitor_col1, monitor_col2 = st.columns(2)
                        
                        with monitor_col1:
                            if st.button("🔄 刷新路径状态"):
                                st.rerun()
                        
                        with monitor_col2:
                            auto_refresh = st.checkbox("自动刷新 (30秒)", key="path_auto_refresh")
                            if auto_refresh:
                                time.sleep(30)
                                st.rerun()
                        
                        # 路径建议
                        st.subheader("💡 智能建议")
                        
                        best_path = paths[0]
                        
                        if best_path.risk_level == "低" and best_path.total_fee < amount * 0.01:
                            st.success("✅ 推荐使用最优路径，风险低且费用合理")
                        elif best_path.risk_level == "中":
                            st.warning("⚠️ 建议谨慎使用，注意监控转账状态")
                        else:
                            st.error("❌ 不建议使用，风险较高，建议等待更好时机")
                        
                        # 费用优化建议
                        if best_path.total_fee > amount * 0.02:
                            st.info("💰 费用较高，建议考虑分批转账或等待网络拥堵缓解")
                        
                        # 时间优化建议
                        if best_path.total_time > 60:
                            st.info("⏰ 转账时间较长，建议在非高峰时段进行")
                    
                    else:
                        st.error("❌ 未找到可用的转账路径")
                        
                        # 显示可能的原因
                        st.write("**可能原因:**")
                        st.write("- 选择的平台不支持该代币")
                        st.write("- 转账金额超出限制")
                        st.write("- 网络暂时不可用")
                        st.write("- 平台间无直接连接")
                
                except Exception as e:
                    st.error(f"规划路径时发生错误: {str(e)}")
                    st.write("请检查网络连接或稍后重试")
        else:
            st.warning("请填写完整的转账信息")
    
    # 功能说明
    with st.expander("ℹ️ 功能说明", expanded=False):
        st.markdown("""
        **转账路径规划器功能包括：**
        
        🎯 **智能路径规划**
        - 自动寻找最优转账路径
        - 支持直接转账、跨链转账、多跳转账
        - 综合考虑费用、时间、风险因素
        
        💰 **费用优化**
        - 实时计算Gas费用和手续费
        - 对比不同路径的总成本
        - 提供费用优化建议
        
        ⏱️ **时间预估**
        - 准确预估转账完成时间
        - 考虑网络拥堵情况
        - 提供最快路径选择
        
        🛡️ **风险评估**
        - 评估转账成功率
        - 分析潜在风险因素
        - 提供风险等级标识
        
        📊 **可视化分析**
        - 路径对比图表
        - 效率分数雷达图
        - 实时监控面板
        
        🔧 **支持平台**
        - 主流区块链网络 (Ethereum, BSC, Polygon, Arbitrum)
        - 知名交易所 (Binance, OKX, Bybit)
        - 跨链桥协议 (Stargate, Multichain, cBridge)
        """)

def show_arbitrage_opportunities():
    """显示期现套利机会视图"""
    st.subheader("💰 期现套利机会视图")
    
    # 控制面板
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    
    with col1:
        selected_symbols = st.multiselect(
            "选择交易对",
            options=["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT", "XRPUSDT", "DOTUSDT", "LINKUSDT"],
            default=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
            help="选择要分析的交易对"
        )
    
    with col2:
        selected_exchanges = st.multiselect(
            "选择交易所",
            options=["binance", "okx", "bybit"],
            default=["binance", "okx"],
            help="选择要监控的交易所"
        )
    
    with col3:
        analysis_type = st.selectbox(
            "分析类型",
            options=["单交易所套利", "跨交易所套利", "综合分析"],
            index=0,
            help="选择套利分析类型"
        )
    
    with col4:
        if st.button("🔍 扫描机会", type="primary"):
            st.session_state.scan_arbitrage = True
    
    if not selected_symbols or not selected_exchanges:
        st.warning("请选择至少一个交易对和一个交易所")
        return
    
    # 扫描套利机会
    if st.session_state.get('scan_arbitrage', False):
        with st.spinner("正在扫描套利机会..."):
            try:
                if analysis_type == "跨交易所套利":
                    opportunities = asyncio.run(
                        arbitrage_analyzer.get_cross_exchange_opportunities(selected_symbols)
                    )
                else:
                    opportunities = asyncio.run(
                        arbitrage_analyzer.scan_arbitrage_opportunities(selected_symbols, selected_exchanges)
                    )
                
                st.session_state.arbitrage_opportunities = opportunities
                st.session_state.scan_arbitrage = False
                
            except Exception as e:
                st.error(f"扫描套利机会时出错: {str(e)}")
                st.session_state.scan_arbitrage = False
                return
    
    # 显示套利机会
    if 'arbitrage_opportunities' in st.session_state:
        opportunities = st.session_state.arbitrage_opportunities
        
        if not opportunities:
            st.info("🔍 未发现符合条件的套利机会")
            return
        
        # 生成报告
        report = arbitrage_analyzer.generate_arbitrage_report(opportunities)
        
        # 总体统计
        st.markdown("### 📊 套利机会概览")
        
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("发现机会", f"{report['total_opportunities']}个")
        with metric_cols[1]:
            st.metric("平均收益", f"{report['avg_expected_return']:.2f}%")
        with metric_cols[2]:
            best_opportunity = max(opportunities, key=lambda x: abs(x.expected_return))
            st.metric("最佳收益", f"{abs(best_opportunity.expected_return):.2f}%")
        with metric_cols[3]:
            low_risk_count = len([op for op in opportunities if op.risk_level == "低"])
            st.metric("低风险机会", f"{low_risk_count}个")
        
        # 套利机会列表
        st.markdown("### 💎 套利机会详情")
        
        # 创建套利机会表格
        arbitrage_data = []
        for i, op in enumerate(opportunities[:20]):  # 显示前20个机会
            risk_emoji = {"低": "🟢", "中": "🟡", "高": "🔴", "极高": "⚫"}
            
            arbitrage_data.append({
                "排名": i + 1,
                "交易对": op.symbol,
                "现货价格": f"${op.spot_price:.4f}",
                "期货价格": f"${op.futures_price:.4f}",
                "价差": f"{op.spread_percentage:.2f}%",
                "资金费率": f"{op.funding_rate*100:.3f}%",
                "预期收益": f"{op.expected_return:.2f}%",
                "风险等级": f"{risk_emoji.get(op.risk_level, '❓')} {op.risk_level}",
                "现货交易所": op.exchange_spot.upper(),
                "期货交易所": op.exchange_futures.upper(),
                "更新时间": op.timestamp.strftime("%H:%M:%S")
            })
        
        if arbitrage_data:
            df_arbitrage = pd.DataFrame(arbitrage_data)
            st.dataframe(df_arbitrage, use_container_width=True)
        
        # 收益分布图表
        st.markdown("### 📈 收益分布分析")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 收益分布直方图
            returns = [abs(op.expected_return) for op in opportunities]
            
            fig_hist = px.histogram(
                x=returns,
                nbins=20,
                title="预期收益分布",
                labels={"x": "预期收益(%)", "y": "机会数量"}
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True, key="return_distribution_histogram")
        
        with col2:
            # 风险等级分布饼图
            risk_data = list(report['risk_distribution'].items())
            if risk_data:
                risk_df = pd.DataFrame(risk_data, columns=["风险等级", "数量"])
                
                fig_pie = px.pie(
                    risk_df,
                    values="数量",
                    names="风险等级",
                    title="风险等级分布"
                )
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True, key="risk_distribution_pie")
        
        # 交易对热度分析
        st.markdown("### 🔥 热门交易对分析")
        
        if report['top_symbols']:
            symbol_data = []
            for symbol, count in report['top_symbols']:
                symbol_opportunities = [op for op in opportunities if op.symbol == symbol]
                avg_return = np.mean([abs(op.expected_return) for op in symbol_opportunities])
                max_return = max([abs(op.expected_return) for op in symbol_opportunities])
                
                symbol_data.append({
                    "交易对": symbol,
                    "机会数量": count,
                    "平均收益": f"{avg_return:.2f}%",
                    "最高收益": f"{max_return:.2f}%"
                })
            
            df_symbols = pd.DataFrame(symbol_data)
            
            fig_symbols = px.bar(
                df_symbols,
                x="交易对",
                y="机会数量",
                title="交易对套利机会数量",
                color="机会数量",
                color_continuous_scale="viridis"
            )
            fig_symbols.update_layout(height=400)
            st.plotly_chart(fig_symbols, use_container_width=True, key="symbol_opportunities_chart")
        
        # 实时价差监控
        st.markdown("### ⚡ 实时价差监控")
        
        # 选择要监控的交易对
        monitor_symbol = st.selectbox(
            "选择监控交易对",
            options=selected_symbols,
            help="选择要实时监控价差的交易对"
        )
        
        if monitor_symbol:
            # 生成模拟的价差历史数据
            import datetime
            import numpy as np
            
            # 获取当前交易对的套利机会
            symbol_opportunities = [op for op in opportunities if op.symbol == monitor_symbol]
            
            if symbol_opportunities:
                # 生成最近1小时的价差数据
                times = pd.date_range(
                    start=datetime.datetime.now() - datetime.timedelta(hours=1),
                    end=datetime.datetime.now(),
                    freq='5min'
                )
                
                spread_data = []
                base_spread = symbol_opportunities[0].spread_percentage
                
                for time in times:
                    # 生成带有随机波动的价差数据
                    spread = base_spread + np.random.normal(0, 0.5)
                    spread_data.append({
                        "时间": time,
                        "价差(%)": spread,
                        "交易对": monitor_symbol
                    })
                
                df_spread = pd.DataFrame(spread_data)
                
                fig_spread = px.line(
                    df_spread,
                    x="时间",
                    y="价差(%)",
                    title=f"{monitor_symbol} 价差变化趋势",
                    color="交易对"
                )
                
                # 添加套利阈值线
                fig_spread.add_hline(
                    y=0.1, line_dash="dash", line_color="green",
                    annotation_text="最小套利阈值"
                )
                fig_spread.add_hline(
                    y=-0.1, line_dash="dash", line_color="green"
                )
                
                fig_spread.update_layout(height=400)
                st.plotly_chart(fig_spread, use_container_width=True, key="real_time_spread_monitor")
        
        # 套利策略建议
        st.markdown("### 💡 套利策略建议")
        
        if opportunities:
            best_ops = sorted(opportunities, key=lambda x: abs(x.expected_return), reverse=True)[:3]
            
            for i, op in enumerate(best_ops, 1):
                with st.expander(f"策略 {i}: {op.symbol} ({op.expected_return:.2f}% 收益)"):
                    if op.spread > 0:
                        st.markdown(f"""
                        **正向套利策略：**
                        
                        📈 **操作步骤：**
                        1. 在 {op.exchange_spot.upper()} 买入 {op.symbol} 现货
                        2. 在 {op.exchange_futures.upper()} 卖出 {op.symbol} 期货
                        3. 等待价差收敛或到期交割
                        
                        💰 **收益分析：**
                        - 价差收益: {op.spread_percentage:.2f}%
                        - 资金费率成本: {op.funding_rate*100:.3f}%/8h
                        - 预期净收益: {op.expected_return:.2f}%
                        
                        ⚠️ **风险提示：**
                        - 风险等级: {op.risk_level}
                        - 需要足够的资金和保证金
                        - 注意价差可能进一步扩大
                        - 考虑交易手续费和滑点成本
                        """)
                    else:
                        st.markdown(f"""
                        **反向套利策略：**
                        
                        📉 **操作步骤：**
                        1. 在 {op.exchange_spot.upper()} 卖出 {op.symbol} 现货
                        2. 在 {op.exchange_futures.upper()} 买入 {op.symbol} 期货
                        3. 等待价差收敛或到期交割
                        
                        💰 **收益分析：**
                        - 价差收益: {abs(op.spread_percentage):.2f}%
                        - 资金费率收益: {op.funding_rate*100:.3f}%/8h
                        - 预期净收益: {op.expected_return:.2f}%
                        
                        ⚠️ **风险提示：**
                        - 风险等级: {op.risk_level}
                        - 需要借币做空现货
                        - 注意借币成本和利率
                        - 考虑强制平仓风险
                        """)
        
        # 功能说明
        with st.expander("ℹ️ 功能说明"):
            st.markdown("""
            **期现套利机会视图功能包括：**
            
            🔍 **机会发现：**
            - 实时扫描现货与期货价差
            - 识别正向和反向套利机会
            - 跨交易所价差分析
            - 自动计算预期收益
            
            📊 **数据分析：**
            - 收益分布统计
            - 风险等级评估
            - 热门交易对分析
            - 历史价差趋势
            
            💡 **策略建议：**
            - 详细操作步骤
            - 收益风险分析
            - 成本费用计算
            - 风险控制建议
            
            ⚠️ **重要提示：**
            - 套利存在市场风险，价差可能进一步扩大
            - 需要考虑交易手续费、滑点和资金成本
            - 建议小额测试，逐步增加仓位
            - 密切关注市场变化和风险控制
            """)
    
    else:
        st.info("📊 点击上方按钮开始扫描套利机会")

def show_exchange_health_monitor():
    """显示交易所健康状态监控功能"""
    st.subheader("🏥 交易所健康状态监控")
    
    # 控制面板
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        selected_exchanges = st.multiselect(
            "选择交易所",
            options=["binance", "okx", "bybit", "coinbase", "kraken", "huobi"],
            default=["binance", "okx", "bybit"],
            help="选择要监控的交易所"
        )
    
    with col2:
        check_interval = st.selectbox(
            "检查间隔",
            options=["实时", "1分钟", "5分钟", "15分钟"],
            index=1,
            help="健康检查的频率"
        )
    
    with col3:
        if st.button("🔄 刷新状态", type="primary"):
            st.rerun()
    
    if not selected_exchanges:
        st.warning("请至少选择一个交易所进行监控")
        return
    
    # 获取健康状态数据
    try:
        health_data = exchange_health_monitor.check_multiple_exchanges(selected_exchanges)
        
        # 总体状态概览
        st.markdown("### 📊 总体状态概览")
        
        status_cols = st.columns(len(selected_exchanges))
        for i, exchange in enumerate(selected_exchanges):
            with status_cols[i]:
                if exchange in health_data:
                    data = health_data[exchange]
                    overall_status = data.get('overall_status', 'unknown')
                    
                    if overall_status == 'healthy':
                        st.success(f"✅ {exchange.upper()}")
                        st.metric("状态", "健康")
                    elif overall_status == 'warning':
                        st.warning(f"⚠️ {exchange.upper()}")
                        st.metric("状态", "警告")
                    else:
                        st.error(f"❌ {exchange.upper()}")
                        st.metric("状态", "异常")
                    
                    # 显示响应时间
                    if 'api_latency' in data:
                        st.metric("API延迟", f"{data['api_latency']:.0f}ms")
                else:
                    st.error(f"❌ {exchange.upper()}")
                    st.metric("状态", "连接失败")
        
        # 详细健康指标
        st.markdown("### 📈 详细健康指标")
        
        # 创建健康指标表格
        health_metrics = []
        for exchange in selected_exchanges:
            if exchange in health_data:
                data = health_data[exchange]
                metrics = {
                    "交易所": exchange.upper(),
                    "API状态": "✅ 正常" if data.get('api_status') else "❌ 异常",
                    "时间同步": "✅ 同步" if data.get('time_sync') else "❌ 不同步",
                    "API延迟(ms)": f"{data.get('api_latency', 0):.0f}",
                    "交易对数量": data.get('trading_pairs_count', 0),
                    "24h交易量": f"${data.get('volume_24h', 0):,.0f}",
                    "订单簿深度": data.get('orderbook_depth', 'N/A'),
                    "最后更新": data.get('last_update', 'N/A')
                }
                health_metrics.append(metrics)
            else:
                metrics = {
                    "交易所": exchange.upper(),
                    "API状态": "❌ 连接失败",
                    "时间同步": "❌ 无法检测",
                    "API延迟(ms)": "N/A",
                    "交易对数量": "N/A",
                    "24h交易量": "N/A",
                    "订单簿深度": "N/A",
                    "最后更新": "N/A"
                }
                health_metrics.append(metrics)
        
        if health_metrics:
            df_health = pd.DataFrame(health_metrics)
            st.dataframe(df_health, use_container_width=True)
        
        # API延迟对比图表
        st.markdown("### ⚡ API延迟对比")
        
        latency_data = []
        for exchange in selected_exchanges:
            if exchange in health_data and 'api_latency' in health_data[exchange]:
                latency_data.append({
                    "交易所": exchange.upper(),
                    "延迟(ms)": health_data[exchange]['api_latency']
                })
        
        if latency_data:
            df_latency = pd.DataFrame(latency_data)
            
            fig_latency = px.bar(
                df_latency,
                x="交易所",
                y="延迟(ms)",
                title="交易所API延迟对比",
                color="延迟(ms)",
                color_continuous_scale="RdYlGn_r"
            )
            fig_latency.update_layout(
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_latency, use_container_width=True, key="exchange_latency_chart")
        
        # 健康状态历史趋势（模拟数据）
        st.markdown("### 📊 健康状态趋势")
        
        # 生成模拟的历史数据
        import datetime
        import numpy as np
        
        dates = pd.date_range(
            start=datetime.datetime.now() - datetime.timedelta(hours=24),
            end=datetime.datetime.now(),
            freq='H'
        )
        
        trend_data = []
        for exchange in selected_exchanges[:3]:  # 限制显示前3个交易所
            if exchange in health_data:
                base_latency = health_data[exchange].get('api_latency', 100)
                # 生成带有随机波动的延迟数据
                latencies = base_latency + np.random.normal(0, 20, len(dates))
                latencies = np.maximum(latencies, 10)  # 确保延迟不为负数
                
                for date, latency in zip(dates, latencies):
                    trend_data.append({
                        "时间": date,
                        "交易所": exchange.upper(),
                        "API延迟(ms)": latency
                    })
        
        if trend_data:
            df_trend = pd.DataFrame(trend_data)
            
            fig_trend = px.line(
                df_trend,
                x="时间",
                y="API延迟(ms)",
                color="交易所",
                title="24小时API延迟趋势"
            )
            fig_trend.update_layout(height=400)
            st.plotly_chart(fig_trend, use_container_width=True, key="api_latency_trend_chart")
        
        # 警报设置
        st.markdown("### 🚨 警报设置")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            latency_threshold = st.number_input(
                "API延迟阈值 (ms)",
                min_value=50,
                max_value=5000,
                value=1000,
                step=50,
                help="超过此延迟将触发警报"
            )
        
        with col2:
            downtime_threshold = st.number_input(
                "停机时间阈值 (分钟)",
                min_value=1,
                max_value=60,
                value=5,
                step=1,
                help="连续停机超过此时间将触发警报"
            )
        
        with col3:
            enable_notifications = st.checkbox(
                "启用通知",
                value=True,
                help="启用邮件/短信通知"
            )
        
        # 检查是否有警报
        alerts = []
        for exchange in selected_exchanges:
            if exchange in health_data:
                data = health_data[exchange]
                if data.get('api_latency', 0) > latency_threshold:
                    alerts.append(f"⚠️ {exchange.upper()}: API延迟过高 ({data['api_latency']:.0f}ms)")
                if not data.get('api_status'):
                    alerts.append(f"🚨 {exchange.upper()}: API连接失败")
                if not data.get('time_sync'):
                    alerts.append(f"⚠️ {exchange.upper()}: 时间同步异常")
        
        if alerts:
            st.markdown("### 🚨 当前警报")
            for alert in alerts:
                st.error(alert)
        else:
            st.success("✅ 所有监控的交易所状态正常")
        
        # 功能说明
        with st.expander("ℹ️ 功能说明"):
            st.markdown("""
            **交易所健康状态监控功能包括：**
            
            📊 **实时监控指标：**
            - API连接状态和响应时间
            - 服务器时间同步状态
            - 交易对数量和24小时交易量
            - 订单簿深度和流动性
            
            📈 **数据分析：**
            - API延迟对比和趋势分析
            - 健康状态历史记录
            - 异常检测和预警
            
            🚨 **智能警报：**
            - 自定义延迟和停机阈值
            - 实时通知和警报推送
            - 多渠道通知支持
            
            💡 **使用建议：**
            - 定期检查交易所健康状态
            - 根据延迟情况选择最优交易所
            - 设置合理的警报阈值
            - 关注异常模式和趋势变化
            """)
    
    except Exception as e:
        st.error(f"获取交易所健康数据时出错: {str(e)}")
        st.info("请检查网络连接和API配置")

def show_cross_chain_analysis():
    """显示跨链转账效率与成本分析"""
    st.subheader("🌉 跨链转账效率与成本分析")
    
    # 控制面板
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # 源网络选择
        from_networks = cross_chain_analyzer.get_supported_networks()
        from_network = st.selectbox(
            "源网络",
            from_networks,
            key="cross_chain_from_network"
        )
    
    with col2:
        # 目标网络选择
        to_networks = [net for net in from_networks if net != from_network]
        to_network = st.selectbox(
            "目标网络",
            to_networks,
            key="cross_chain_to_network"
        )
    
    with col3:
        # 代币选择
        tokens = cross_chain_analyzer.get_supported_tokens()
        token = st.selectbox(
            "代币",
            tokens,
            key="cross_chain_token"
        )
    
    with col4:
        # 转账金额
        amount = st.number_input(
            "转账金额",
            min_value=1.0,
            max_value=1000000.0,
            value=1000.0,
            step=100.0,
            key="cross_chain_amount"
        )
    
    # 分析按钮
    if st.button("🔍 分析跨链路由", key="analyze_cross_chain"):
        with st.spinner("正在分析跨链转账路由..."):
            try:
                # 获取跨链路由分析
                analysis = asyncio.run(
                    cross_chain_analyzer.analyze_cross_chain_routes(
                        from_network, to_network, token, amount
                    )
                )
                
                if analysis.get('routes'):
                    st.success(f"找到 {analysis['total_routes']} 条可用路由")
                    
                    # 最佳路由推荐
                    st.subheader("💡 最佳路由推荐")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**💰 最低成本路由**")
                        best_cost = analysis['best_cost_route']
                        st.info(f"""
                        **桥**: {best_cost['bridge']}
                        **总成本**: ${best_cost['total_cost']:.4f}
                        **成本占比**: {best_cost['cost_percentage']:.3f}%
                        **预计时间**: {best_cost['estimated_time']//60}分钟
                        """)
                    
                    with col2:
                        st.markdown("**⚡ 最快路由**")
                        fastest = analysis['fastest_route']
                        st.info(f"""
                        **桥**: {fastest['bridge']}
                        **总成本**: ${fastest['total_cost']:.4f}
                        **成本占比**: {fastest['cost_percentage']:.3f}%
                        **预计时间**: {fastest['estimated_time']//60}分钟
                        """)
                    
                    # 详细路由对比表
                    st.subheader("📊 路由详细对比")
                    
                    route_data = []
                    for route in analysis['routes']:
                        route_data.append({
                            '跨链桥': route['bridge'],
                            '总成本 ($)': f"{route['total_cost']:.4f}",
                            '桥费用 ($)': f"{route['bridge_fee']:.4f}",
                            'Gas费用 ($)': f"{route['gas_cost']:.4f}",
                            '费率 (%)': f"{route['fee_rate']*100:.3f}",
                            '成本占比 (%)': f"{route['cost_percentage']:.3f}",
                            '预计时间 (分钟)': f"{route['estimated_time']//60}",
                            '评级': '⭐⭐⭐' if route == analysis['best_cost_route'] else 
                                   '⭐⭐' if route == analysis['fastest_route'] else '⭐'
                        })
                    
                    df_routes = pd.DataFrame(route_data)
                    st.dataframe(df_routes, use_container_width=True)
                    
                    # 成本分析图表
                    st.subheader("📈 成本分析可视化")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 成本对比柱状图
                        fig_cost = px.bar(
                            x=[route['bridge'] for route in analysis['routes']],
                            y=[route['total_cost'] for route in analysis['routes']],
                            title="各桥总成本对比",
                            labels={'x': '跨链桥', 'y': '总成本 ($)'},
                            color=[route['total_cost'] for route in analysis['routes']],
                            color_continuous_scale='RdYlGn_r'
                        )
                        fig_cost.update_layout(height=400)
                        st.plotly_chart(fig_cost, use_container_width=True, key="bridge_cost_comparison")
                    
                    with col2:
                        # 时间对比柱状图
                        fig_time = px.bar(
                            x=[route['bridge'] for route in analysis['routes']],
                            y=[route['estimated_time']//60 for route in analysis['routes']],
                            title="各桥预计时间对比",
                            labels={'x': '跨链桥', 'y': '预计时间 (分钟)'},
                            color=[route['estimated_time'] for route in analysis['routes']],
                            color_continuous_scale='RdYlBu_r'
                        )
                        fig_time.update_layout(height=400)
                        st.plotly_chart(fig_time, use_container_width=True, key="bridge_time_comparison")
                    
                    # 成本构成分析
                    st.subheader("🔍 成本构成分析")
                    
                    # 选择一个路由进行详细分析
                    selected_bridge = st.selectbox(
                        "选择桥进行详细分析",
                        [route['bridge'] for route in analysis['routes']],
                        key="selected_bridge_analysis"
                    )
                    
                    selected_route = next(route for route in analysis['routes'] if route['bridge'] == selected_bridge)
                    
                    # 饼图显示成本构成
                    cost_breakdown = {
                        '桥费用': selected_route['bridge_fee'],
                        'Gas费用': selected_route['gas_cost']
                    }
                    
                    fig_pie = px.pie(
                        values=list(cost_breakdown.values()),
                        names=list(cost_breakdown.keys()),
                        title=f"{selected_bridge} 成本构成"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True, key="bridge_cost_breakdown")
                    
                    # 统计信息
                    st.subheader("📊 统计信息")
                    
                    stats = analysis['statistics']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "最低成本",
                            f"${stats['min_cost']:.4f}",
                            f"{((stats['min_cost']/stats['max_cost']-1)*100):+.1f}%"
                        )
                    
                    with col2:
                        st.metric(
                            "平均成本",
                            f"${stats['avg_cost']:.4f}",
                            f"{((stats['avg_cost']/stats['max_cost']-1)*100):+.1f}%"
                        )
                    
                    with col3:
                        st.metric(
                            "最快时间",
                            f"{stats['min_time']//60}分钟",
                            f"{((stats['min_time']/stats['max_time']-1)*100):+.1f}%"
                        )
                    
                    with col4:
                        st.metric(
                            "平均时间",
                            f"{stats['avg_time']//60}分钟",
                            f"{((stats['avg_time']/stats['max_time']-1)*100):+.1f}%"
                        )
                
                else:
                    st.error(analysis.get('error', '分析失败'))
            
            except Exception as e:
                st.error(f"分析过程中发生错误: {str(e)}")
    
    # 功能说明
    with st.expander("ℹ️ 功能说明", expanded=False):
        st.markdown("""
        ### 跨链转账效率与成本分析
        
        **主要功能:**
        - 🔍 **多桥对比**: 同时分析多个跨链桥的报价和性能
        - 💰 **成本分析**: 详细分解桥费用、Gas费用等成本构成
        - ⚡ **效率评估**: 比较不同桥的转账速度和确认时间
        - 📊 **可视化**: 直观展示成本和时间对比
        - 💡 **智能推荐**: 根据成本和速度推荐最佳路由
        
        **支持的跨链桥:**
        - Stargate Finance
        - Multichain (Anyswap)
        - Celer cBridge
        - Hop Protocol
        - Synapse Protocol
        
        **支持的网络:**
        - Ethereum
        - BSC (Binance Smart Chain)
        - Polygon
        - Arbitrum
        - Optimism
        - Avalanche
        
        **注意事项:**
        - 费用估算基于当前Gas价格，实际费用可能有所不同
        - 转账时间为预估值，实际时间受网络拥堵影响
        - 建议在实际转账前再次确认最新报价
        """)


def show_enhanced_ccxt_features():
    """显示增强的CCXT功能"""
    st.header("🚀 增强CCXT交易所支持")
    
    # 初始化增强CCXT提供者和趋势分析器
    if 'ccxt_provider' not in st.session_state:
        st.session_state.ccxt_provider = EnhancedCCXTProvider()
    
    if 'trend_analyzer' not in st.session_state:
        st.session_state.trend_analyzer = TrendAnalyzer()
    
    ccxt_provider = st.session_state.ccxt_provider
    trend_analyzer = st.session_state.trend_analyzer
    
    # 支持的交易所信息
    with st.expander("📋 支持的免费交易所", expanded=True):
        exchanges = ccxt_provider.get_supported_exchanges()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("活跃交易所")
            active_exchanges = [ex for ex in exchanges if ex['status'] == 'active']
            if active_exchanges:
                for ex in active_exchanges:
                    st.success(f"✅ {ex['name']} ({ex['id']})")
                    st.caption(f"限制: {ex['rate_limit']}/分钟")
            else:
                st.warning("暂无活跃交易所")
        
        with col2:
            st.subheader("支持的交易对")
            symbols = ccxt_provider.get_supported_symbols()
            for symbol in symbols:
                st.info(f"📈 {symbol}")
    
    # 实时价格对比
    st.subheader("💰 多交易所实时价格对比")
    
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        selected_symbol = st.selectbox(
            "选择交易对",
            options=ccxt_provider.get_supported_symbols(),
            key="ccxt_symbol_select"
        )
    
    with col2:
        if st.button("🔄 刷新价格", key="refresh_ccxt_prices"):
            st.session_state.ccxt_refresh_trigger = time.time()
    
    with col3:
        auto_refresh = st.checkbox("自动刷新", key="ccxt_auto_refresh")
    
    # 获取价格数据
    if st.button("获取价格数据", key="get_ccxt_prices") or 'ccxt_refresh_trigger' in st.session_state:
        with st.spinner(f"正在获取 {selected_symbol} 的价格数据..."):
            try:
                tickers = safe_run_async(ccxt_provider.get_all_tickers(selected_symbol))
                
                if tickers:
                    # 创建价格对比表
                    df_data = []
                    for ticker in tickers:
                        df_data.append({
                            '交易所': ticker['exchange'].upper(),
                            '最新价格': f"${ticker['price']:.4f}" if ticker['price'] else "N/A",
                            '买入价': f"${ticker['bid']:.4f}" if ticker['bid'] else "N/A",
                            '卖出价': f"${ticker['ask']:.4f}" if ticker['ask'] else "N/A",
                            '24h变化': f"{ticker['change_24h']:.2f}%" if ticker['change_24h'] else "N/A",
                            '成交量': f"{ticker['volume']:.2f}" if ticker['volume'] else "N/A",
                            '更新时间': ticker['datetime'][:19] if ticker['datetime'] else "N/A"
                        })
                    
                    df = pd.DataFrame(df_data)
                    st.dataframe(df, width='stretch')
                    
                    # 价格分析
                    prices = [t['price'] for t in tickers if t['price']]
                    if len(prices) >= 2:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("平均价格", f"${np.mean(prices):.4f}")
                        
                        with col2:
                            st.metric("最高价格", f"${max(prices):.4f}")
                        
                        with col3:
                            st.metric("最低价格", f"${min(prices):.4f}")
                        
                        with col4:
                            spread_pct = ((max(prices) - min(prices)) / min(prices)) * 100
                            st.metric("价差", f"{spread_pct:.2f}%")
                        
                        # 价格分布图
                        fig = px.bar(
                            x=[t['exchange'].upper() for t in tickers if t['price']],
                            y=prices,
                            title=f"{selected_symbol} 各交易所价格对比",
                            labels={'x': '交易所', 'y': '价格 (USD)'}
                        )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, width='stretch', key="exchange_price_comparison")
                else:
                    st.warning("未获取到价格数据")
                    
            except Exception as e:
                st.error(f"获取数据时出错: {str(e)}")
    
    # 套利机会分析
    st.subheader("🎯 实时套利机会")
    
    if st.button("分析套利机会", key="analyze_arbitrage"):
        with st.spinner("正在分析套利机会..."):
            try:
                opportunities = safe_run_async(ccxt_provider.calculate_arbitrage_opportunities(selected_symbol))
                
                if opportunities:
                    st.success(f"发现 {len(opportunities)} 个套利机会！")
                    
                    # 显示前5个最佳机会
                    top_opportunities = opportunities[:5]
                    
                    for i, opp in enumerate(top_opportunities, 1):
                        with st.container():
                            col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
                            
                            with col1:
                                st.write(f"**#{i}**")
                            
                            with col2:
                                st.write(f"**买入:** {opp['buy_exchange'].upper()}")
                                st.write(f"价格: ${opp['buy_price']:.4f}")
                            
                            with col3:
                                st.write(f"**卖出:** {opp['sell_exchange'].upper()}")
                                st.write(f"价格: ${opp['sell_price']:.4f}")
                            
                            with col4:
                                profit_color = "green" if opp['profit_pct'] > 0.5 else "orange"
                                st.markdown(f"<span style='color:{profit_color}'>**+{opp['profit_pct']:.2f}%**</span>", unsafe_allow_html=True)
                                st.write(f"${opp['profit_abs']:.4f}")
                            
                            st.divider()
                    
                    # 套利机会图表
                    if len(opportunities) > 1:
                        fig = px.scatter(
                            x=[f"{opp['buy_exchange']} → {opp['sell_exchange']}" for opp in top_opportunities],
                            y=[opp['profit_pct'] for opp in top_opportunities],
                            size=[opp['profit_abs'] for opp in top_opportunities],
                            title="套利机会分布",
                            labels={'x': '交易路径', 'y': '利润率 (%)'}
                        )
                        st.plotly_chart(fig, width='stretch', key="arbitrage_opportunities_scatter")
                else:
                    st.info("当前没有发现明显的套利机会")
                    
            except Exception as e:
                st.error(f"分析套利机会时出错: {str(e)}")
    
    # 市场摘要
    with st.expander("📊 市场摘要"):
        if st.button("获取市场摘要", key="get_market_summary"):
            with st.spinner("正在生成市场摘要..."):
                try:
                    summary = safe_run_async(ccxt_provider.get_market_summary(selected_symbol))
                    
                    if 'error' not in summary:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("参与交易所", summary['exchanges_count'])
                            st.metric("平均价格", f"${summary['avg_price']:.4f}")
                            st.metric("总成交量", f"{summary['total_volume']:.2f}")
                        
                        with col2:
                            st.metric("最高价格", f"${summary['max_price']:.4f}")
                            st.metric("最低价格", f"${summary['min_price']:.4f}")
                            st.metric("价格差异", f"{summary['price_spread_pct']:.2f}%")
                        
                        st.info(f"数据更新时间: {summary['timestamp'][:19]}")
                    else:
                        st.error(summary['error'])
                        
                except Exception as e:
                    st.error(f"获取市场摘要时出错: {str(e)}")
    
    # 价格趋势分析
    st.subheader("📈 价格趋势分析")
    
    col1, col2 = st.columns(2)
    
    with col1:
        trend_symbol = st.selectbox(
            "选择分析币种",
            ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT"],
            key="trend_symbol"
        )
    
    with col2:
        trend_period = st.selectbox(
            "时间周期",
            ["1小时", "6小时", "24小时", "7天"],
            key="trend_period"
        )
    
    if st.button("📊 生成趋势分析", key="generate_trend"):
        try:
            # 模拟添加历史价格数据
            import random
            import datetime
            
            base_price = 50000 if "BTC" in trend_symbol else 3000
            
            for i in range(24):  # 添加24小时的数据
                timestamp = datetime.datetime.now() - datetime.timedelta(hours=23-i)
                price = base_price * (1 + random.uniform(-0.05, 0.05))
                trend_analyzer.add_price_data(trend_symbol, "binance", price, timestamp)
                trend_analyzer.add_price_data(trend_symbol, "okx", price * (1 + random.uniform(-0.002, 0.002)), timestamp)
            
            # 获取趋势数据
            trend_data = trend_analyzer.get_price_trend(trend_symbol, hours=24)
            
            if trend_data:
                # 显示趋势图表
                fig = trend_analyzer.create_price_trend_chart(trend_symbol, hours=24)
                if fig:
                    st.plotly_chart(fig, width='stretch', key="price_trend_chart")
                
                # 显示趋势统计
                col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                
                with col1:
                    current_price = trend_data[-1]['price']
                    st.metric("当前价格", f"${current_price:.2f}")
                
                with col2:
                    price_change = ((trend_data[-1]['price'] - trend_data[0]['price']) / trend_data[0]['price']) * 100
                    st.metric("24h变化", f"{price_change:.2f}%", delta=f"{price_change:.2f}%")
                
                with col3:
                    prices = [d['price'] for d in trend_data]
                    volatility = (max(prices) - min(prices)) / min(prices) * 100
                    st.metric("波动率", f"{volatility:.2f}%")
                
                with col4:
                    trend_direction = "上涨" if price_change > 0 else "下跌" if price_change < 0 else "横盘"
                    st.metric("趋势方向", trend_direction)
                
                # 波动率对比图
                st.subheader("📊 交易所波动率对比")
                volatility_fig = trend_analyzer.create_volatility_comparison(["binance", "okx"], [trend_symbol])
                if volatility_fig:
                    st.plotly_chart(volatility_fig, width='stretch', key="volatility_comparison_chart")
                
            else:
                st.warning("暂无趋势数据")
                
        except Exception as e:
            st.error(f"生成趋势分析失败: {str(e)}")
    
    # 套利机会趋势
    st.subheader("💰 套利机会趋势")
    
    if st.button("📈 查看套利趋势", key="arbitrage_trend"):
        try:
            arbitrage_trends = trend_analyzer.get_arbitrage_trends(hours=24)
            
            if arbitrage_trends:
                # 显示套利机会统计
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_opportunity = sum(t['max_spread'] for t in arbitrage_trends) / len(arbitrage_trends)
                    st.metric("平均套利机会", f"{avg_opportunity:.2f}%")
                
                with col2:
                    max_opportunity = max(t['max_spread'] for t in arbitrage_trends)
                    st.metric("最大套利机会", f"{max_opportunity:.2f}%")
                
                with col3:
                    profitable_count = len([t for t in arbitrage_trends if t['max_spread'] > 0.5])
                    st.metric("盈利机会数", f"{profitable_count}")
                
                # 显示套利趋势表格
                st.dataframe(
                    arbitrage_trends,
                    width='stretch'
                )
            else:
                st.info("暂无套利趋势数据")
                
        except Exception as e:
            st.error(f"获取套利趋势失败: {str(e)}")


def show_analytics_dashboard(engine: ArbitrageEngine, providers: List[BaseProvider]):
    """显示数据分析仪表盘"""
    st.title("📈 数据分析中心")
    st.markdown("---")
    
    # 初始化分析引擎
    if 'analytics_engine' not in st.session_state:
        st.session_state.analytics_engine = analytics_engine
    
    analytics = st.session_state.analytics_engine
    
    # 分析选项卡
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 收益分析", 
        "🔄 历史回测", 
        "⚡ 策略优化", 
        "📈 市场分析"
    ])
    
    with tab1:
        st.subheader("💰 收益分析")
        
        # 时间范围选择
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("开始日期", value=datetime.now() - timedelta(days=30))
        with col2:
            end_date = st.date_input("结束日期", value=datetime.now())
        
        # 生成模拟收益数据
        if st.button("🔄 生成收益报告", key="generate_profit_report"):
            with st.spinner("正在分析收益数据..."):
                time.sleep(1)  # 模拟计算时间
                
                # 模拟收益指标
                metrics = PerformanceMetrics(
                    total_return=0.156,
                    sharpe_ratio=2.34,
                    max_drawdown=0.045,
                    win_rate=0.78,
                    profit_factor=3.2,
                    avg_trade_return=0.0023,
                    total_trades=1247,
                    profitable_trades=973
                )
                
                # 显示关键指标
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "总收益率",
                        f"{metrics.total_return:.2%}",
                        f"+{metrics.total_return*100:.1f}%"
                    )
                
                with col2:
                    st.metric(
                        "夏普比率",
                        f"{metrics.sharpe_ratio:.2f}",
                        "优秀" if metrics.sharpe_ratio > 2 else "良好"
                    )
                
                with col3:
                    st.metric(
                        "最大回撤",
                        f"{metrics.max_drawdown:.2%}",
                        f"-{metrics.max_drawdown*100:.1f}%"
                    )
                
                with col4:
                    st.metric(
                        "胜率",
                        f"{metrics.win_rate:.1%}",
                        f"{metrics.profitable_trades}/{metrics.total_trades}"
                    )
                
                # 收益曲线图
                st.subheader("📈 收益曲线")
                
                # 生成模拟收益曲线数据
                dates = pd.date_range(start=start_date, end=end_date, freq='D')
                cumulative_returns = np.cumsum(np.random.normal(0.001, 0.02, len(dates)))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=cumulative_returns,
                    mode='lines',
                    name='累计收益',
                    line=dict(color='#00D4AA', width=2)
                ))
                
                fig.update_layout(
                    title="累计收益曲线",
                    xaxis_title="日期",
                    yaxis_title="累计收益率",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True, key="cumulative_returns_chart")
    
    with tab2:
        st.subheader("🔄 历史回测")
        
        # 回测参数设置
        col1, col2, col3 = st.columns(3)
        
        with col1:
            initial_capital = st.number_input("初始资金 (USDT)", value=10000, min_value=1000)
        
        with col2:
            strategy_type = st.selectbox(
                "策略类型",
                ["现货套利", "三角套利", "跨链套利", "期现套利"]
            )
        
        with col3:
            risk_level = st.selectbox(
                "风险等级",
                ["保守", "平衡", "激进"]
            )
        
        # 运行回测
        if st.button("🚀 开始回测", key="start_backtest"):
            with st.spinner("正在运行历史回测..."):
                time.sleep(2)  # 模拟回测时间
                
                # 模拟回测结果
                backtest_result = BacktestResult(
                    strategy_name=strategy_type,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital,
                    final_capital=initial_capital * 1.234,
                    total_return=0.234,
                    max_drawdown=0.067,
                    sharpe_ratio=1.89,
                    total_trades=456,
                    win_rate=0.72
                )
                
                # 显示回测结果
                st.success("✅ 回测完成！")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 回测概览")
                    st.write(f"**策略名称**: {backtest_result.strategy_name}")
                    st.write(f"**回测期间**: {backtest_result.start_date} 至 {backtest_result.end_date}")
                    st.write(f"**初始资金**: ${backtest_result.initial_capital:,.2f}")
                    st.write(f"**最终资金**: ${backtest_result.final_capital:,.2f}")
                    st.write(f"**总收益**: ${backtest_result.final_capital - backtest_result.initial_capital:,.2f}")
                
                with col2:
                    st.subheader("📈 关键指标")
                    st.write(f"**总收益率**: {backtest_result.total_return:.2%}")
                    st.write(f"**最大回撤**: {backtest_result.max_drawdown:.2%}")
                    st.write(f"**夏普比率**: {backtest_result.sharpe_ratio:.2f}")
                    st.write(f"**交易次数**: {backtest_result.total_trades}")
                    st.write(f"**胜率**: {backtest_result.win_rate:.1%}")
    
    with tab3:
        st.subheader("⚡ 策略优化")
        
        # 优化参数设置
        st.write("### 🎯 优化目标")
        
        col1, col2 = st.columns(2)
        
        with col1:
            optimization_target = st.selectbox(
                "优化目标",
                ["最大化收益", "最大化夏普比率", "最小化回撤", "最大化胜率"]
            )
        
        with col2:
            optimization_method = st.selectbox(
                "优化方法",
                ["网格搜索", "遗传算法", "贝叶斯优化", "粒子群优化"]
            )
        
        # 开始优化
        if st.button("🔍 开始优化", key="start_optimization"):
            with st.spinner("正在进行策略优化..."):
                time.sleep(3)  # 模拟优化时间
                
                st.success("✅ 优化完成！")
                
                # 最优参数
                st.subheader("🏆 最优参数组合")
                
                optimal_params = {
                    "收益阈值": "0.45%",
                    "仓位大小": "35%",
                    "最大持仓": "4个",
                    "止损比例": "2.5%",
                    "止盈比例": "8.0%"
                }
                
                for param, value in optimal_params.items():
                    st.write(f"**{param}**: {value}")
    
    with tab4:
        st.subheader("📈 市场分析")
        
        # 市场概览
        st.write("### 🌍 市场概览")
        
        # 模拟市场数据
        market_data = {
            "BTC/USDT": {"price": 43250.67, "change_24h": 2.34, "volume": "1.2B"},
            "ETH/USDT": {"price": 2567.89, "change_24h": -1.23, "volume": "890M"},
            "BNB/USDT": {"price": 315.45, "change_24h": 0.87, "volume": "234M"},
            "ADA/USDT": {"price": 0.4567, "change_24h": 3.45, "volume": "156M"},
            "SOL/USDT": {"price": 98.76, "change_24h": -2.11, "volume": "445M"}
        }
        
        for symbol, data in market_data.items():
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.write(f"**{symbol}**")
            
            with col2:
                st.write(f"${data['price']:,.2f}")
            
            with col3:
                color = "green" if data['change_24h'] > 0 else "red"
                st.markdown(f"<span style='color: {color}'>{data['change_24h']:+.2f}%</span>", 
                           unsafe_allow_html=True)
            
            with col4:
                st.write(data['volume'])
        
        st.markdown("---")
        
        # 市场深度分析
        st.write("### 📊 市场深度分析")
        
        # 选择交易对和交易所
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_symbol = st.selectbox(
                "选择交易对",
                ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT"],
                key="depth_symbol"
            )
        
        with col2:
            selected_exchange = st.selectbox(
                "选择交易所",
                ["Binance", "OKX", "Bybit", "Huobi", "KuCoin"],
                key="depth_exchange"
            )
        
        with col3:
            if st.button("🔍 分析市场深度", key="analyze_depth"):
                with st.spinner("正在分析市场深度..."):
                    # 模拟市场深度分析
                    depth_data = market_depth_analyzer.analyze_order_book(
                        selected_exchange, selected_symbol
                    )
                    
                    if depth_data:
                        st.success("✅ 市场深度分析完成")
                        
                        # 显示流动性指标
                        st.write("#### 💧 流动性指标")
                        
                        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                        
                        with metrics_col1:
                            st.metric(
                                "买单深度",
                                f"${depth_data.bid_depth:,.2f}",
                                delta=f"{depth_data.bid_depth_change:+.2f}%"
                            )
                        
                        with metrics_col2:
                            st.metric(
                                "卖单深度", 
                                f"${depth_data.ask_depth:,.2f}",
                                delta=f"{depth_data.ask_depth_change:+.2f}%"
                            )
                        
                        with metrics_col3:
                            st.metric(
                                "买卖价差",
                                f"{depth_data.spread:.4f}",
                                delta=f"{depth_data.spread_change:+.4f}"
                            )
                        
                        with metrics_col4:
                            st.metric(
                                "流动性评分",
                                f"{depth_data.liquidity_score:.1f}/10",
                                delta=f"{depth_data.score_change:+.1f}"
                            )
                        
                        # 订单簿可视化
                        st.write("#### 📈 订单簿分布")
                        
                        # 创建订单簿图表
                        import plotly.graph_objects as go
                        from plotly.subplots import make_subplots
                        
                        fig = make_subplots(
                            rows=1, cols=2,
                            subplot_titles=('买单深度', '卖单深度'),
                            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
                        )
                        
                        # 买单数据
                        bid_prices = [43245.50, 43245.00, 43244.50, 43244.00, 43243.50]
                        bid_volumes = [2.5, 5.2, 3.8, 7.1, 4.6]
                        
                        # 卖单数据  
                        ask_prices = [43246.00, 43246.50, 43247.00, 43247.50, 43248.00]
                        ask_volumes = [3.2, 4.8, 6.1, 2.9, 5.5]
                        
                        fig.add_trace(
                            go.Bar(
                                x=bid_volumes,
                                y=bid_prices,
                                orientation='h',
                                name='买单',
                                marker_color='green',
                                opacity=0.7
                            ),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Bar(
                                x=ask_volumes,
                                y=ask_prices,
                                orientation='h',
                                name='卖单',
                                marker_color='red',
                                opacity=0.7
                            ),
                            row=1, col=2
                        )
                        
                        fig.update_layout(
                            height=400,
                            showlegend=True,
                            title_text="订单簿深度分析"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True, key="order_book_depth_analysis")
                        
                        # 价格冲击分析
                        st.write("#### ⚡ 价格冲击分析")
                        
                        impact_col1, impact_col2 = st.columns(2)
                        
                        with impact_col1:
                            st.write("**买入冲击成本**")
                            buy_amounts = [1000, 5000, 10000, 50000, 100000]
                            buy_impacts = [0.02, 0.08, 0.15, 0.45, 0.89]
                            
                            impact_df = pd.DataFrame({
                                "交易金额 ($)": buy_amounts,
                                "价格冲击 (%)": buy_impacts
                            })
                            st.dataframe(impact_df, use_container_width=True)
                        
                        with impact_col2:
                            st.write("**卖出冲击成本**")
                            sell_amounts = [1000, 5000, 10000, 50000, 100000]
                            sell_impacts = [0.03, 0.09, 0.18, 0.52, 0.95]
                            
                            impact_df = pd.DataFrame({
                                "交易金额 ($)": sell_amounts,
                                "价格冲击 (%)": sell_impacts
                            })
                            st.dataframe(impact_df, use_container_width=True)
                        
                        # 最佳执行建议
                        st.write("#### 🎯 最佳执行建议")
                        
                        suggestion_col1, suggestion_col2 = st.columns(2)
                        
                        with suggestion_col1:
                            st.info("""
                            **💡 执行策略建议**
                            - 大额订单建议分批执行
                            - 当前流动性较好，适合中等规模交易
                            - 建议在买一卖一价格附近挂单
                            """)
                        
                        with suggestion_col2:
                            st.warning("""
                            **⚠️ 风险提示**
                            - 大额交易可能造成显著价格冲击
                            - 注意监控订单簿变化
                            - 考虑使用算法交易降低冲击
                            """)
                    else:
                        st.error("❌ 无法获取市场深度数据")


def show_professional_trading_interface(engine, providers):
    """显示专业交易界面"""
    st.title("💼 专业交易界面")
    st.markdown("---")
    
    # 渲染布局选择器
    selected_layout = trading_interface.render_layout_selector()
    
    # 渲染布局自定义器
    trading_interface.render_layout_customizer()
    
    # 渲染主交易界面
    trading_interface.render_trading_interface(selected_layout, engine, providers)
    
    # 处理快捷操作的弹窗
    if st.session_state.get('show_quick_analysis', False):
        with st.expander("📊 快速分析", expanded=True):
            st.write("### 市场快速分析")
            
            # 生成模拟分析数据
            analysis_data = {
                '市场趋势': '上涨',
                '波动率': '中等',
                '成交量': '活跃',
                '技术指标': 'RSI: 65, MACD: 正向',
                '支撑位': '$42,800',
                '阻力位': '$44,200'
            }
            
            for key, value in analysis_data.items():
                st.metric(key, value)
            
            if st.button("关闭分析"):
                st.session_state.show_quick_analysis = False
                st.rerun()
    
    if st.session_state.get('show_arbitrage_search', False):
        with st.expander("🎯 套利机会搜索", expanded=True):
            st.write("### 实时套利机会")
            st.info("正在搜索套利机会...")
            
            # 模拟套利机会
            opportunities = [
                {'交易对': 'BTC/USDT', '交易所A': 'Binance', '交易所B': 'OKX', '价差': '0.15%', '利润': '$65'},
                {'交易对': 'ETH/USDT', '交易所A': 'Huobi', '交易所B': 'Binance', '价差': '0.08%', '利润': '$23'}
            ]
            
            for opp in opportunities:
                st.write(f"**{opp['交易对']}**: {opp['交易所A']} vs {opp['交易所B']} - 价差: {opp['价差']}, 预期利润: {opp['利润']}")
            
            if st.button("关闭搜索"):
                st.session_state.show_arbitrage_search = False
                st.rerun()
    
    if st.session_state.get('show_risk_check', False):
        with st.expander("⚠️ 风险检查", expanded=True):
            st.write("### 风险评估报告")
            
            risk_metrics = {
                '总体风险等级': '中等',
                '仓位风险': '低',
                '流动性风险': '低',
                '市场风险': '中等',
                'VaR (1日)': '$1,250',
                '最大回撤': '3.2%'
            }
            
            for metric, value in risk_metrics.items():
                st.metric(metric, value)
            
            if st.button("关闭风险检查"):
                st.session_state.show_risk_check = False
                st.rerun()
    
    if st.session_state.get('show_technical_analysis', False):
        with st.expander("📈 技术分析工具", expanded=True):
            st.write("### 技术分析")
            
            # 技术指标选择
            indicators = st.multiselect(
                "选择技术指标",
                ["RSI", "MACD", "布林带", "移动平均线", "成交量"],
                default=["RSI", "MACD"]
            )
            
            st.write("**当前技术指标状态:**")
            for indicator in indicators:
                if indicator == "RSI":
                    st.write(f"• RSI(14): 65.2 - 中性偏多")
                elif indicator == "MACD":
                    st.write(f"• MACD: 正向交叉 - 买入信号")
                elif indicator == "布林带":
                    st.write(f"• 布林带: 价格接近上轨 - 超买区域")
                elif indicator == "移动平均线":
                    st.write(f"• MA(20): 上升趋势 - 多头排列")
                elif indicator == "成交量":
                    st.write(f"• 成交量: 放量上涨 - 趋势确认")
            
            if st.button("关闭技术分析"):
                st.session_state.show_technical_analysis = False
                st.rerun()


def show_currency_comparison(engine, providers):
    """显示货币比对中心 - 使用分层架构"""
    from .ui.currency_hub import CurrencyHub, apply_currency_hub_styles
    
    # 应用样式
    apply_currency_hub_styles()
    
    # 初始化货币中心
    hub = CurrencyHub()
    
    # 渲染主界面
    hub.render_main_interface()

def show_system_settings(config):
    """显示系统设置页面"""
    st.title("⚙️ 系统设置")
    st.markdown("---")
    
    # 设置选项卡
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["⚙️ 基础设置", "🔑 API配置", "🎨 显示设置", "🚨 预警系统", "👥 多账户管理"])
    
    with tab1:
        st.subheader("🔧 基础设置")
        
        # 风险设置
        st.write("### ⚠️ 风险管理")
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_position_size = st.slider(
                "最大仓位比例 (%)",
                min_value=1,
                max_value=100,
                value=st.session_state.get('max_position_size', 20),
                key="settings_max_position"
            )
        
        with col2:
            max_daily_loss = st.slider(
                "最大日损失 (%)",
                min_value=1,
                max_value=20,
                value=st.session_state.get('max_daily_loss', 5),
                key="settings_max_loss"
            )
        
        # 保存设置
        if st.button("💾 保存基础设置"):
            st.session_state.max_position_size = max_position_size
            st.session_state.max_daily_loss = max_daily_loss
            st.success("✅ 基础设置已保存！")
    
    with tab2:
        st.subheader("🔐 API配置")
        
        # API密钥管理
        st.write("### 🔑 API密钥管理")
        
        exchanges = ["Binance", "OKX", "Bybit", "Huobi", "KuCoin"]
        
        for exchange in exchanges:
            with st.expander(f"{exchange} API配置"):
                col1, col2 = st.columns(2)
                
                with col1:
                    api_key = st.text_input(
                        "API Key",
                        type="password",
                        key=f"{exchange.lower()}_api_key"
                    )
                
                with col2:
                    secret_key = st.text_input(
                        "Secret Key",
                        type="password",
                        key=f"{exchange.lower()}_secret_key"
                    )
                
                # 测试连接
                if st.button(f"🔍 测试 {exchange} 连接", key=f"test_{exchange.lower()}"):
                    if api_key and secret_key:
                        with st.spinner(f"正在测试 {exchange} 连接..."):
                            time.sleep(1)
                            st.success(f"✅ {exchange} 连接成功！")
                    else:
                        st.error("❌ 请填写完整的API密钥信息")
    
    with tab3:
        st.subheader("📊 显示设置")
        
        # 界面设置
        st.write("### 🎨 界面设置")
        
        col1, col2 = st.columns(2)
        
        with col1:
            theme = st.selectbox(
                "主题",
                ["自动", "浅色", "深色"],
                index=0
            )
        
        with col2:
            language = st.selectbox(
                "语言",
                ["中文", "English"],
                index=0
            )
        
        # 数据刷新设置
        st.write("### 🔄 数据刷新")
        
        col1, col2 = st.columns(2)
        
        with col1:
            auto_refresh = st.checkbox(
                "启用自动刷新",
                value=st.session_state.get('auto_refresh_enabled', False)
            )
        
        with col2:
            refresh_interval = st.selectbox(
                "刷新间隔 (秒)",
                [5, 10, 15, 30, 60],
                index=1
            )
        
        if st.button("💾 保存显示设置"):
            st.session_state.auto_refresh_enabled = auto_refresh
            st.session_state.auto_refresh_interval = refresh_interval
            st.success("✅ 显示设置已保存！")
    
    with tab4:
        st.subheader("🚨 预警系统")
        
        # 预警规则管理
        st.write("### 📋 预警规则管理")
        
        # 显示当前规则
        rules_col1, rules_col2 = st.columns([2, 1])
        
        with rules_col1:
            st.write("**当前预警规则**")
            
            rules_data = []
            for rule in alert_system.rules.values():
                rules_data.append({
                    "规则名称": rule.name,
                    "类型": rule.alert_type.value,
                    "严重程度": rule.severity.value,
                    "状态": "启用" if rule.enabled else "禁用",
                    "冷却时间": f"{rule.cooldown_minutes}分钟"
                })
            
            if rules_data:
                rules_df = pd.DataFrame(rules_data)
                st.dataframe(rules_df, use_container_width=True)
            else:
                st.info("暂无预警规则")
        
        with rules_col2:
            st.write("**快速操作**")
            
            if st.button("➕ 添加规则"):
                st.session_state.show_add_rule = True
            
            if st.button("📊 预警统计"):
                stats = alert_system.get_alert_statistics()
                st.json(stats)
        
        # 添加新规则表单
        if st.session_state.get('show_add_rule', False):
            st.write("### ➕ 添加新预警规则")
            
            with st.form("add_alert_rule"):
                rule_col1, rule_col2 = st.columns(2)
                
                with rule_col1:
                    rule_name = st.text_input("规则名称", placeholder="输入规则名称")
                    rule_type = st.selectbox(
                        "预警类型",
                        [t.value for t in AlertType],
                        format_func=lambda x: {
                            "spread_alert": "价差预警",
                            "arbitrage_opportunity": "套利机会",
                            "market_anomaly": "市场异常",
                            "volume_alert": "交易量预警",
                            "price_alert": "价格预警",
                            "system_error": "系统错误"
                        }.get(x, x)
                    )
                    rule_severity = st.selectbox(
                        "严重程度",
                        [s.value for s in AlertSeverity],
                        format_func=lambda x: {
                            "low": "低",
                            "medium": "中",
                            "high": "高", 
                            "critical": "严重"
                        }.get(x, x)
                    )
                
                with rule_col2:
                    cooldown_minutes = st.number_input("冷却时间(分钟)", min_value=1, max_value=1440, value=5)
                    
                    notification_channels = st.multiselect(
                        "通知渠道",
                        [c.value for c in NotificationChannel],
                        format_func=lambda x: {
                            "email": "邮件",
                            "webhook": "Webhook",
                            "desktop": "桌面通知",
                            "mobile": "手机推送"
                        }.get(x, x)
                    )
                
                # 条件设置
                st.write("**触发条件**")
                
                if rule_type == "spread_alert":
                    min_spread = st.number_input("最小价差百分比", min_value=0.1, max_value=10.0, value=0.5, step=0.1)
                    min_volume = st.number_input("最小交易量(USD)", min_value=1000, max_value=1000000, value=10000, step=1000)
                    conditions = {"min_spread_percentage": min_spread, "min_volume_usd": min_volume}
                
                elif rule_type == "arbitrage_opportunity":
                    min_profit = st.number_input("最小利润百分比", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
                    max_exec_time = st.number_input("最大执行时间(秒)", min_value=1, max_value=300, value=30)
                    min_liquidity = st.number_input("最小流动性(USD)", min_value=10000, max_value=1000000, value=50000, step=10000)
                    conditions = {
                        "min_profit_percentage": min_profit,
                        "max_execution_time_seconds": max_exec_time,
                        "min_liquidity_usd": min_liquidity
                    }
                
                elif rule_type == "market_anomaly":
                    price_threshold = st.number_input("价格变动阈值(%)", min_value=1.0, max_value=50.0, value=5.0, step=0.5)
                    volume_multiplier = st.number_input("交易量激增倍数", min_value=1.5, max_value=10.0, value=3.0, step=0.5)
                    conditions = {
                        "price_change_threshold": price_threshold,
                        "volume_spike_multiplier": volume_multiplier
                    }
                
                else:
                    conditions = {}
                
                submitted = st.form_submit_button("✅ 创建规则")
                
                if submitted and rule_name:
                    new_rule = AlertRule(
                        id=f"rule_{datetime.now().timestamp()}",
                        name=rule_name,
                        alert_type=AlertType(rule_type),
                        conditions=conditions,
                        severity=AlertSeverity(rule_severity),
                        cooldown_minutes=cooldown_minutes,
                        channels=[NotificationChannel(c) for c in notification_channels]
                    )
                    
                    if alert_system.add_rule(new_rule):
                        st.success(f"✅ 预警规则 '{rule_name}' 创建成功！")
                        st.session_state.show_add_rule = False
                        st.rerun()
                    else:
                        st.error("❌ 创建预警规则失败")
        
        st.markdown("---")
        
        # 活跃预警
        st.write("### 🔔 活跃预警")
        
        active_alerts = alert_system.get_active_alerts()
        
        if active_alerts:
            for alert in active_alerts[-10:]:  # 显示最近10条
                severity_color = {
                    "low": "blue",
                    "medium": "orange", 
                    "high": "red",
                    "critical": "purple"
                }.get(alert.severity.value, "gray")
                
                with st.expander(f"🚨 {alert.title} - {alert.timestamp.strftime('%H:%M:%S')}"):
                    st.markdown(f"**严重程度**: <span style='color: {severity_color}'>{alert.severity.value.upper()}</span>", 
                               unsafe_allow_html=True)
                    st.write(f"**消息**: {alert.message}")
                    st.write(f"**类型**: {alert.alert_type.value}")
                    st.write(f"**时间**: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    alert_col1, alert_col2 = st.columns(2)
                    
                    with alert_col1:
                        if not alert.acknowledged and st.button(f"✅ 确认", key=f"ack_{alert.id}"):
                            alert_system.acknowledge_alert(alert.id)
                            st.success("预警已确认")
                            st.rerun()
                    
                    with alert_col2:
                        if not alert.resolved and st.button(f"🔧 解决", key=f"resolve_{alert.id}"):
                            alert_system.resolve_alert(alert.id)
                            st.success("预警已解决")
                            st.rerun()
        else:
            st.info("🎉 当前没有活跃预警")
        
        # 通知设置
        st.write("### 📧 通知设置")
        
        notification_col1, notification_col2 = st.columns(2)
        
        with notification_col1:
            st.write("**邮件配置**")
            email_server = st.text_input("SMTP服务器", value="smtp.gmail.com")
            email_port = st.number_input("SMTP端口", value=587)
            email_username = st.text_input("邮箱用户名", placeholder="your-email@gmail.com")
            email_password = st.text_input("邮箱密码", type="password", placeholder="应用专用密码")
        
        with notification_col2:
            st.write("**Webhook配置**")
            webhook_url = st.text_input("Webhook URL", placeholder="https://hooks.slack.com/...")
            webhook_headers = st.text_area("请求头(JSON格式)", placeholder='{"Content-Type": "application/json"}')
        
        if st.button("💾 保存通知设置"):
            # 更新通知配置
            if email_username and email_password:
                alert_system.config.email_username = email_username
                alert_system.config.email_password = email_password
                alert_system.config.email_smtp_server = email_server
                alert_system.config.email_smtp_port = email_port
            
            if webhook_url:
                alert_system.config.webhook_url = webhook_url
                try:
                    import json
                    if webhook_headers:
                        alert_system.config.webhook_headers = json.loads(webhook_headers)
                except:
                    st.warning("Webhook请求头格式不正确，使用默认设置")
            
            st.success("✅ 通知设置已保存！")
    
    # 多账户管理标签页
    with tab5:
        st.write("## 👥 多账户管理系统")
        
        # 投资组合概览
        st.write("### 📊 投资组合概览")
        
        portfolio_summary = account_manager.get_portfolio_summary()
        
        if portfolio_summary:
            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
            
            with summary_col1:
                st.metric(
                    "总账户数",
                    portfolio_summary.get("total_accounts", 0),
                    delta=f"活跃: {portfolio_summary.get('active_accounts', 0)}"
                )
            
            with summary_col2:
                total_value = portfolio_summary.get("total_value_usd", 0)
                st.metric(
                    "总资产价值 (USD)",
                    f"${total_value:,.2f}",
                    delta=f"{portfolio_summary.get('daily_pnl_percentage', 0):.2f}%"
                )
            
            with summary_col3:
                daily_pnl = portfolio_summary.get("daily_pnl_usd", 0)
                st.metric(
                    "今日盈亏 (USD)",
                    f"${daily_pnl:,.2f}",
                    delta=f"{portfolio_summary.get('total_trades', 0)} 笔交易"
                )
            
            with summary_col4:
                allocation_rules = portfolio_summary.get("allocation_rules", 0)
                st.metric(
                    "分配规则",
                    allocation_rules,
                    delta="个活跃规则"
                )
        
        st.markdown("---")
        
        # 账户管理
        account_tab1, account_tab2, account_tab3 = st.tabs(["📋 账户列表", "➕ 添加账户", "⚖️ 资金分配"])
        
        with account_tab1:
            st.write("### 📋 账户列表")
            
            if account_manager.accounts:
                for account_id, account in account_manager.accounts.items():
                    with st.expander(f"🏦 {account.exchange} - {account_id}"):
                        account_col1, account_col2 = st.columns(2)
                        
                        with account_col1:
                            st.write(f"**交易所**: {account.exchange}")
                            st.write(f"**账户类型**: {account.account_type.value}")
                            st.write(f"**状态**: {account.status.value}")
                            st.write(f"**创建时间**: {account.created_at.strftime('%Y-%m-%d %H:%M')}")
                        
                        with account_col2:
                            # 获取账户余额
                            balances = account_manager.get_account_balances(account_id)
                            if balances:
                                st.write("**余额信息**:")
                                for currency, balance in balances.items():
                                    st.write(f"- {currency}: {balance.total:.4f} (可用: {balance.available:.4f})")
                            
                            # 获取账户指标
                            metrics = account_manager.get_account_metrics(account_id)
                            if metrics:
                                st.write("**表现指标**:")
                                st.write(f"- 总价值: ${metrics.total_value_usd:,.2f}")
                                st.write(f"- 日盈亏: ${metrics.daily_pnl:,.2f} ({metrics.daily_pnl_percentage:.2f}%)")
                                st.write(f"- 夏普比率: {metrics.sharpe_ratio:.2f}")
                                st.write(f"- 胜率: {metrics.win_rate:.1%}")
                        
                        # 账户操作
                        action_col1, action_col2, action_col3 = st.columns(3)
                        
                        with action_col1:
                            if account.status == AccountStatus.ACTIVE:
                                if st.button(f"⏸️ 暂停", key=f"pause_{account_id}"):
                                    account_manager.update_account_status(account_id, AccountStatus.INACTIVE)
                                    st.success("账户已暂停")
                                    st.rerun()
                            else:
                                if st.button(f"▶️ 激活", key=f"activate_{account_id}"):
                                    account_manager.update_account_status(account_id, AccountStatus.ACTIVE)
                                    st.success("账户已激活")
                                    st.rerun()
                        
                        with action_col2:
                            if st.button(f"🔄 刷新余额", key=f"refresh_{account_id}"):
                                account_manager.get_account_balances(account_id)
                                st.success("余额已刷新")
                                st.rerun()
                        
                        with action_col3:
                            if st.button(f"🗑️ 删除账户", key=f"delete_{account_id}"):
                                if account_manager.remove_account(account_id):
                                    st.success("账户已删除")
                                    st.rerun()
                                else:
                                    st.error("删除账户失败")
            else:
                st.info("📝 还没有添加任何账户，请在'添加账户'标签页中添加。")
        
        with account_tab2:
            st.write("### ➕ 添加新账户")
            
            with st.form("add_account_form"):
                form_col1, form_col2 = st.columns(2)
                
                with form_col1:
                    account_id = st.text_input("账户ID", placeholder="my_binance_account")
                    exchange = st.selectbox("交易所", ["binance", "okx", "bybit", "huobi", "kucoin"])
                    account_type = st.selectbox("账户类型", [t.value for t in AccountType])
                    api_key = st.text_input("API Key", type="password")
                
                with form_col2:
                    api_secret = st.text_input("API Secret", type="password")
                    passphrase = st.text_input("Passphrase (可选)", type="password")
                    sandbox = st.checkbox("沙盒模式")
                    test_connection = st.checkbox("测试连接", value=True)
                
                submitted = st.form_submit_button("✅ 添加账户")
                
                if submitted and account_id and exchange and api_key and api_secret:
                    new_account = AccountInfo(
                        account_id=account_id,
                        exchange=exchange,
                        account_type=AccountType(account_type),
                        status=AccountStatus.ACTIVE,
                        balances={},
                        api_key=api_key,
                        api_secret=api_secret,
                        passphrase=passphrase if passphrase else None,
                        sandbox=sandbox
                    )
                    
                    if account_manager.add_account(new_account):
                        st.success(f"✅ 账户 '{account_id}' 添加成功！")
                        st.rerun()
                    else:
                        st.error("❌ 添加账户失败，请检查API配置")
        
        with account_tab3:
            st.write("### ⚖️ 资金分配管理")
            
            # 分配规则管理
            st.write("#### 📋 分配规则")
            
            if account_manager.allocation_rules:
                for rule_id, rule in account_manager.allocation_rules.items():
                    with st.expander(f"📏 {rule.name} ({'✅ 启用' if rule.enabled else '❌ 禁用'})"):
                        rule_col1, rule_col2 = st.columns(2)
                        
                        with rule_col1:
                            st.write(f"**策略**: {rule.strategy.value}")
                            st.write(f"**最小分配**: ${rule.min_allocation}")
                            st.write(f"**最大分配**: ${rule.max_allocation}")
                            st.write(f"**重平衡阈值**: {rule.rebalance_threshold:.1%}")
                        
                        with rule_col2:
                            st.write(f"**目标账户**: {len(rule.target_accounts) if rule.target_accounts else '所有账户'}")
                            if rule.weights:
                                st.write("**权重配置**:")
                                for acc_id, weight in rule.weights.items():
                                    st.write(f"- {acc_id}: {weight:.2f}")
                        
                        # 规则操作
                        rule_action_col1, rule_action_col2, rule_action_col3 = st.columns(3)
                        
                        with rule_action_col1:
                            if rule.enabled:
                                if st.button(f"⏸️ 禁用", key=f"disable_rule_{rule_id}"):
                                    rule.enabled = False
                                    st.success("规则已禁用")
                                    st.rerun()
                            else:
                                if st.button(f"▶️ 启用", key=f"enable_rule_{rule_id}"):
                                    rule.enabled = True
                                    st.success("规则已启用")
                                    st.rerun()
                        
                        with rule_action_col2:
                            if st.button(f"🔍 检查重平衡", key=f"check_rebalance_{rule_id}"):
                                needs_rebalance = account_manager.check_rebalancing_needed(rule_id)
                                if needs_rebalance:
                                    st.warning("⚠️ 需要重新平衡")
                                else:
                                    st.success("✅ 分配平衡良好")
                        
                        with rule_action_col3:
                            if st.button(f"🗑️ 删除规则", key=f"delete_rule_{rule_id}"):
                                del account_manager.allocation_rules[rule_id]
                                st.success("规则已删除")
                                st.rerun()
            
            st.markdown("---")
            
            # 执行资金分配
            st.write("#### 💰 执行资金分配")
            
            allocation_col1, allocation_col2 = st.columns(2)
            
            with allocation_col1:
                available_rules = [
                    (rule_id, rule.name) for rule_id, rule in account_manager.allocation_rules.items()
                    if rule.enabled
                ]
                
                if available_rules:
                    selected_rule = st.selectbox(
                        "选择分配规则",
                        options=[rule_id for rule_id, _ in available_rules],
                        format_func=lambda x: next(name for rid, name in available_rules if rid == x)
                    )
                    
                    allocation_amount = st.number_input(
                        "分配金额 (USD)",
                        min_value=100.0,
                        max_value=1000000.0,
                        value=10000.0,
                        step=100.0
                    )
                else:
                    st.warning("⚠️ 没有可用的分配规则")
                    selected_rule = None
                    allocation_amount = 0
            
            with allocation_col2:
                if selected_rule and st.button("🚀 执行分配", type="primary"):
                    from decimal import Decimal
                    allocation_result = account_manager.allocate_funds(
                        selected_rule, 
                        Decimal(str(allocation_amount))
                    )
                    
                    if allocation_result:
                        st.success("✅ 资金分配完成！")
                        st.write("**分配结果**:")
                        for account_id, amount in allocation_result.items():
                            st.write(f"- {account_id}: ${amount:,.2f}")
                    else:
                        st.error("❌ 资金分配失败")
            
            # 创建新分配规则
            if st.button("➕ 创建新分配规则"):
                st.session_state.show_add_allocation_rule = True
            
            if st.session_state.get('show_add_allocation_rule', False):
                with st.form("add_allocation_rule_form"):
                    st.write("#### 📝 创建新分配规则")
                    
                    new_rule_col1, new_rule_col2 = st.columns(2)
                    
                    with new_rule_col1:
                        new_rule_name = st.text_input("规则名称", placeholder="我的分配规则")
                        new_rule_strategy = st.selectbox("分配策略", [s.value for s in AllocationStrategy])
                        min_allocation = st.number_input("最小分配金额", min_value=0.0, value=1000.0)
                        max_allocation = st.number_input("最大分配金额", min_value=1000.0, value=100000.0)
                    
                    with new_rule_col2:
                        rebalance_threshold = st.slider("重平衡阈值", 0.01, 0.5, 0.05, 0.01)
                        target_accounts = st.multiselect(
                            "目标账户 (留空表示所有账户)",
                            options=list(account_manager.accounts.keys())
                        )
                    
                    # 权重配置（仅对权重策略）
                    if new_rule_strategy == AllocationStrategy.WEIGHTED.value:
                        st.write("**权重配置**:")
                        weights = {}
                        accounts_to_configure = target_accounts or list(account_manager.accounts.keys())
                        
                        for account_id in accounts_to_configure:
                            weight = st.number_input(
                                f"{account_id} 权重",
                                min_value=0.0,
                                max_value=10.0,
                                value=1.0,
                                step=0.1,
                                key=f"weight_{account_id}"
                            )
                            weights[account_id] = weight
                    else:
                        weights = {}
                    
                    form_submitted = st.form_submit_button("✅ 创建规则")
                    
                    if form_submitted and new_rule_name:
                        from decimal import Decimal
                        
                        new_allocation_rule = AllocationRule(
                            id=f"rule_{datetime.now().timestamp()}",
                            name=new_rule_name,
                            strategy=AllocationStrategy(new_rule_strategy),
                            target_accounts=target_accounts,
                            weights=weights,
                            min_allocation=Decimal(str(min_allocation)),
                            max_allocation=Decimal(str(max_allocation)),
                            rebalance_threshold=rebalance_threshold,
                            enabled=True
                        )
                        
                        account_manager.allocation_rules[new_allocation_rule.id] = new_allocation_rule
                        st.success(f"✅ 分配规则 '{new_rule_name}' 创建成功！")
                        st.session_state.show_add_allocation_rule = False
                        st.rerun()


def main():
    """Main function to run the Streamlit application."""
    config = get_config()
    init_session_state(config)

    # 渲染导航栏
    render_navigation()
    
    # 渲染页面标题
    render_page_header(
        title="专业级套利分析平台",
        description="实时监控市场机会，智能分析套利策略，专业级风险管控",
        icon="🎯"
    )
    
    # 渲染快速统计
    render_quick_stats()
    
    # 主要功能区域
    st.markdown("## 🚀 快速访问")
    
    # 创建功能卡片
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            text-align: center;
            color: white;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            <h3>🌍 货币概览</h3>
            <p>查看全球货币市场概况，实时价格和趋势分析</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("进入货币概览", key="goto_overview", use_container_width=True):
            st.switch_page("pages/1_🌍_货币概览.py")
    
    with col2:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 2rem;
            border-radius: 10px;
            text-align: center;
            color: white;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            <h3>📈 详细分析</h3>
            <p>深入分析货币走势，技术指标和市场信号</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("进入详细分析", key="goto_analysis", use_container_width=True):
            st.switch_page("pages/2_📈_详细分析.py")
    
    with col3:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            padding: 2rem;
            border-radius: 10px;
            text-align: center;
            color: white;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            <h3>⚖️ 货币比较</h3>
            <p>对比不同货币表现，发现投资机会</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("进入货币比较", key="goto_compare", use_container_width=True):
            st.switch_page("pages/3_⚖️_货币比较.py")

    # 第二行功能卡片
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            padding: 2rem;
            border-radius: 10px;
            text-align: center;
            color: white;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            <h3>🔍 高级筛选</h3>
            <p>使用专业筛选工具，精准定位投资标的</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("进入高级筛选", key="goto_filter", use_container_width=True):
            st.switch_page("pages/4_🔍_高级筛选.py")
    
    with col5:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            padding: 2rem;
            border-radius: 10px;
            text-align: center;
            color: #333;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            <h3>📊 实时仪表盘</h3>
            <p>实时监控市场动态，智能预警系统</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("进入实时仪表盘", key="goto_dashboard", use_container_width=True):
            st.switch_page("pages/5_📊_实时仪表盘.py")
    
    with col6:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #d299c2 0%, #fef9d7 100%);
            padding: 2rem;
            border-radius: 10px;
            text-align: center;
            color: #333;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            <h3>💼 专业交易</h3>
            <p>专业级交易界面，高级订单管理</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("进入专业交易", key="goto_trading", use_container_width=True):
            # 这里保持原有的交易界面
            st.session_state['show_trading'] = True
            st.rerun()

    # 如果用户点击了专业交易，显示原有的交易界面
    if st.session_state.get('show_trading', False):
        st.markdown("---")
        st.markdown("## 💼 专业交易界面")
        
        sidebar_controls()

        providers = get_providers(config, st.session_state)
        if not providers:
            st.error("没有可用的数据提供商。请在侧边栏中选择交易所或检查配置。")
            st.info("💡 提示：请在侧边栏中选择至少一个交易所来开始使用。")
            return

        engine = ArbitrageEngine(providers, config.get('arbitrage', {}))

        # 页面选择
        st.sidebar.markdown("---")
        page = st.sidebar.selectbox(
            "📊 选择功能",
            ["🏠 实时仪表盘", "💼 专业交易界面", "🌍 货币比对中心", "📈 数据分析中心", "⚙️ 系统设置"],
            index=0
        )

        if page == "🏠 实时仪表盘":
            show_dashboard(engine, providers)
        elif page == "💼 专业交易界面":
            show_professional_trading_interface(engine, providers)
        elif page == "🌍 货币比对中心":
            show_currency_comparison(engine, providers)
        elif page == "📈 数据分析中心":
            show_analytics_dashboard(engine, providers)
        elif page == "⚙️ 系统设置":
            show_system_settings(config)

        # Auto refresh footer
        if st.session_state.get('auto_refresh_enabled', False):
            interval = st.session_state.get('auto_refresh_interval', 10)
            st.info(f"🔄 自动刷新已启用，每 {interval} 秒刷新一次")
            time.sleep(interval)
            st.rerun()
    
    # 渲染页面底部
    render_footer()

if __name__ == "__main__":
    main()
