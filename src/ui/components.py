import streamlit as st

def sidebar_controls():
    """
    Defines the controls in the sidebar and updates session_state.
    The main app will react to changes in st.session_state.
    """
    st.sidebar.header("⚙️ 配置")
    
    # --- Welcome Message for Demo Mode ---
    if st.session_state.get('demo_mode', True):
        st.sidebar.success(
            "🎯 **欢迎体验演示模式！**\n\n"
            "您可以立即探索所有功能，无需配置API密钥。\n\n"
            "💡 配置API密钥后可获取真实市场数据。",
            icon="🚀"
        )

    # --- Demo Mode Toggle ---
    # This toggle now acts as a read-only status indicator. It is always disabled,
    # and its state is determined automatically by the presence of API keys.
    keys_provided = bool(st.session_state.get('api_keys'))
    st.sidebar.toggle(
        "🚀 演示模式",
        value=not keys_provided,
        key='demo_mode',
        help="这是一个状态指示器。当您提供API密钥时，它会自动关闭。",
        disabled=True  # The toggle is always disabled, making it read-only.
    )
    st.sidebar.divider()

    # --- Exchange Selection ---
    EXCHANGES = ['binance', 'okx', 'bybit', 'kucoin', 'gate', 'mexc', 'bitget', 'htx']
    st.sidebar.multiselect(
        "选择中心化交易所",
        options=EXCHANGES,
        key='selected_exchanges', # This key links the widget to session_state
        help="选择用于行情和套利分析的中心化交易所。"
    )

    # --- API Key Management ---
    st.sidebar.subheader("🔑 API密钥管理")
    if st.session_state.get('demo_mode', True):
        st.sidebar.info(
            "🎯 **可选配置**\n\n"
            "演示模式下无需API密钥即可体验所有功能。\n\n"
            "配置后可获取真实市场数据进行分析。",
            icon="💡"
        )
    st.sidebar.caption("API密钥存储在会话状态中，不会被永久保存。")

    # Dynamically create input fields for selected exchanges
    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = {}

    for ex_id in st.session_state.selected_exchanges:
        with st.sidebar.expander(f"{ex_id.capitalize()} API密钥"):
            api_key = st.text_input(f"{ex_id} API Key", key=f"api_key_{ex_id}", value=st.session_state.api_keys.get(ex_id, {}).get('apiKey', ''))
            api_secret = st.text_input(f"{ex_id} API Secret", type="password", key=f"api_secret_{ex_id}", value=st.session_state.api_keys.get(ex_id, {}).get('secret', ''))

            # Update session state as user types
            if api_key and api_secret:
                 st.session_state.api_keys[ex_id] = {'apiKey': api_key, 'secret': api_secret}
            elif ex_id in st.session_state.api_keys:
                 # Clear keys if fields are emptied
                 del st.session_state.api_keys[ex_id]

    st.sidebar.divider()

    # --- Free API Selection ---
    st.sidebar.subheader("📊 免费API数据源")
    
    # 获取所有可用的免费API
    from ..providers.free_api import free_api_provider
    all_apis = free_api_provider.get_all_apis()
    
    # 默认选择的API
    default_api = 'coingecko'
    
    # 初始化session state
    if 'selected_free_api' not in st.session_state:
        st.session_state.selected_free_api = default_api
    
    # API单选器
    selected_api = st.sidebar.radio(
        "选择要使用的免费API",
        options=list(all_apis.keys()),
        index=list(all_apis.keys()).index(st.session_state.selected_free_api) if st.session_state.selected_free_api in all_apis else 0,
        format_func=lambda x: all_apis[x],
        key='selected_free_api',
        help="选择一个免费API数据源来获取8个交易所的价格数据进行对比。"
    )
    
    # 更新免费API提供者的配置（只启用选中的一个API）
    selected_apis = [selected_api] if selected_api else []
    if selected_apis != free_api_provider.get_enabled_apis():
        free_api_provider.update_enabled_apis(selected_apis)
    
    # 显示当前选择的API
    st.sidebar.caption(f"当前使用: {all_apis.get(selected_api, '未选择')}")
    
    st.sidebar.divider()

    # --- Symbol Selection ---
    st.sidebar.multiselect(
        "选择CEX/DEX交易对",
        options=['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ETH/BTC', 'WETH/USDC', 'WBTC/WETH'],
        default=['BTC/USDT', 'ETH/USDT'],
        key='selected_symbols',
        help="选择要在不同交易所之间跟踪的交易对。"
    )

    # --- Bridge Symbol Selection ---
    st.sidebar.text_input(
        "跨链桥交易对",
        value='BTC.BTC/ETH.ETH',
        key='bridge_symbol',
        help="为Thorchain输入一个跨链交易对，例如 'BTC.BTC/ETH.ETH'。"
    )

    # --- Arbitrage Settings ---
    st.sidebar.subheader("套利设置")
    st.sidebar.number_input(
        "利润阈值 (%)",
        min_value=0.01,
        max_value=10.0,
        value=0.2,
        step=0.01,
        key='arbitrage_threshold',
        help="设置触发警报的最低利润百分比。"
    )

    # --- Refresh Control ---
    st.sidebar.subheader("显示控制")
    if st.sidebar.button("🔄 强制刷新所有数据"):
        # Clearing cached resources will force them to rerun
        st.cache_resource.clear()
        st.rerun()

    st.sidebar.toggle("自动刷新", key='auto_refresh_enabled', value=False)
    st.sidebar.number_input(
        "刷新间隔 (秒)",
        min_value=5,
        max_value=120,
        value=10,
        step=5,
        key='auto_refresh_interval',
        disabled=not st.session_state.get('auto_refresh_enabled', False)
    )

def display_error(message: str):
    """A standardized way to display errors."""
    st.error(message, icon="🚨")

def display_warning(message: str):
    """A standardized way to display warnings."""
    st.warning(message, icon="⚠️")


# --- Charting and Utility Functions ---
import pandas as pd
import plotly.graph_objects as go
import asyncio
import nest_asyncio

# Apply nest_asyncio to allow running asyncio event loops within Streamlit's loop
nest_asyncio.apply()

def safe_run_async(coro):
    """
    Safely runs an async coroutine in a Streamlit environment.
    Handles the "cannot run loop while another loop is running" error
    by leveraging the existing event loop if one is present.
    """
    try:
        # Check if an event loop is already running
        loop = asyncio.get_running_loop()
        return loop.run_until_complete(coro)
    except RuntimeError:
        # If no loop is running, create a new one
        return asyncio.run(coro)
    except Exception as e:
        st.error(f"An unexpected error occurred during an async operation: {e}")
        return None


def validate_symbol(symbol: str) -> bool:
    """Validates that the symbol is not empty and has a valid format (e.g., 'BTC/USDT')."""
    if not symbol or '/' not in symbol or len(symbol.split('/')) != 2:
        st.error("请输入有效的交易对格式，例如 'BTC/USDT'。")
        return False
    return True


def create_depth_chart(order_book: dict) -> go.Figure:
    """Creates a Plotly order book depth chart."""
    if not order_book or 'bids' not in order_book or 'asks' not in order_book:
        fig = go.Figure()
        fig.update_layout(title_text="市场深度 - 无数据", height=300)
        return fig

    bids = pd.DataFrame(order_book.get('bids', []), columns=['price', 'volume']).astype(float)
    asks = pd.DataFrame(order_book.get('asks', []), columns=['price', 'volume']).astype(float)

    bids = bids.sort_values('price', ascending=False)
    asks = asks.sort_values('price', ascending=True)

    bids['cumulative'] = bids['volume'].cumsum()
    asks['cumulative'] = asks['volume'].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bids['price'], y=bids['cumulative'], name='买单', fill='tozeroy', line_color='green'))
    fig.add_trace(go.Scatter(x=asks['price'], y=asks['cumulative'], name='卖单', fill='tozeroy', line_color='red'))

    fig.update_layout(
        title_text=f"{order_book.get('symbol', '')} 市场深度",
        xaxis_title="价格",
        yaxis_title="累计数量",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_white"
    )
    return fig


def create_candlestick_chart(df: pd.DataFrame, symbol: str, show_volume: bool = True, ma_periods: list = None) -> go.Figure:
    """Creates a Plotly candlestick chart from OHLCV data with optional indicators."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title_text=f"{symbol} K线图 - 无数据", height=400)
        return fig

    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        display_error(f"K线图数据格式错误，缺少以下列: {', '.join(set(required_cols) - set(df.columns))}")
        return go.Figure()

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

    if ma_periods:
        colors = ['orange', 'purple', 'blue', 'red', 'cyan', 'magenta']
        for i, period in enumerate(ma_periods):
            if len(df) >= period:
                ma = df['close'].rolling(window=period).mean()
                fig.add_trace(go.Scatter(
                    x=df['datetime'], y=ma, mode='lines', name=f'MA{period}',
                    line=dict(color=colors[i % len(colors)], width=1.5)
                ))

    fig.update_layout(
        title_text=f"{symbol} K线图",
        xaxis_title="时间",
        yaxis_title="价格",
        height=600 if show_volume else 500,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_rangeslider_visible=False,
        showlegend=True,
        template="plotly_white"
    )

    if show_volume:
        fig.update_layout(
            yaxis2=dict(title="成交量", overlaying='y', side='right', showgrid=False),
            shapes=[
                dict(type='rect', xref='paper', yref='paper', x0=0, y0=0, x1=1, y1=1,
                     line=dict(color='rgba(0,0,0,0)'), fillcolor='rgba(0,0,0,0)'),
            ],
            yaxis=dict(domain=[0.3, 1]),
            yaxis2=dict(domain=[0, 0.25])
        )
        fig.add_trace(go.Bar(
            x=df['datetime'], y=df['volume'], name='成交量', yaxis='y2', opacity=0.4, marker_color='grey'
        ))

    return fig
