"""
详细分析页面
独立网页 - 可通过 /详细分析 直接访问
支持URL参数: ?symbol=BTC
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random
import sys
import os

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)  # Go up one level to src directory
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from src.ui.navigation import render_navigation, render_page_header, render_footer

# 页面配置
st.set_page_config(
    page_title="详细分析 - 加密货币分析平台",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义样式
st.markdown("""
<style>
.analysis-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 2rem;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}

.metric-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    border-left: 4px solid #667eea;
}

.indicator-positive {
    color: #00d4aa;
    font-weight: bold;
}

.indicator-negative {
    color: #ff6b6b;
    font-weight: bold;
}

.indicator-neutral {
    color: #ffa726;
    font-weight: bold;
}

.back-nav {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

def generate_price_data(days=30, base_price=45000):
    """生成价格数据"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='H')
    prices = []
    volumes = []
    
    current_price = base_price
    
    for i in range(days):
        # 价格变化
        change = random.uniform(-0.03, 0.03)
        current_price *= (1 + change)
        prices.append(current_price)
        
        # 交易量
        volume = random.uniform(1e6, 1e8)
        volumes.append(volume)
    
    return pd.DataFrame({
        'datetime': dates,
        'price': prices,
        'volume': volumes
    })

def generate_technical_indicators():
    """生成技术指标"""
    return {
        'RSI': {
            'value': random.uniform(20, 80),
            'signal': random.choice(['买入', '卖出', '中性'])
        },
        'MACD': {
            'value': random.uniform(-100, 100),
            'signal': random.choice(['买入', '卖出', '中性'])
        },
        'MA20': {
            'value': random.uniform(40000, 50000),
            'signal': random.choice(['多头', '空头', '中性'])
        },
        'MA50': {
            'value': random.uniform(38000, 48000),
            'signal': random.choice(['多头', '空头', '中性'])
        },
        'Bollinger': {
            'upper': random.uniform(48000, 52000),
            'lower': random.uniform(38000, 42000),
            'signal': random.choice(['突破上轨', '跌破下轨', '区间震荡'])
        }
    }

def create_price_chart(data, currency):
    """创建价格图表"""
    fig = go.Figure()
    
    # 价格线
    fig.add_trace(go.Scatter(
        x=data['datetime'],
        y=data['price'],
        mode='lines',
        name=f'{currency} 价格',
        line=dict(color='#00d4aa', width=2),
        hovertemplate='<b>%{y:$,.2f}</b><br>%{x}<extra></extra>'
    ))
    
    # 移动平均线
    ma20 = data['price'].rolling(window=20).mean()
    ma50 = data['price'].rolling(window=50).mean()
    
    fig.add_trace(go.Scatter(
        x=data['datetime'],
        y=ma20,
        mode='lines',
        name='MA20',
        line=dict(color='orange', width=1, dash='dash'),
        opacity=0.7
    ))
    
    fig.add_trace(go.Scatter(
        x=data['datetime'],
        y=ma50,
        mode='lines',
        name='MA50',
        line=dict(color='red', width=1, dash='dash'),
        opacity=0.7
    ))
    
    fig.update_layout(
        title=f'{currency} 价格走势图',
        xaxis_title='时间',
        yaxis_title='价格 (USD)',
        template='plotly_white',
        height=500,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def create_volume_chart(data):
    """创建交易量图表"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=data['datetime'],
        y=data['volume'],
        name='交易量',
        marker_color='rgba(102, 126, 234, 0.6)',
        hovertemplate='<b>%{y:,.0f}</b><br>%{x}<extra></extra>'
    ))
    
    fig.update_layout(
        title='交易量分布',
        xaxis_title='时间',
        yaxis_title='交易量',
        template='plotly_white',
        height=300
    )
    
    return fig

def main():
    # 渲染导航栏
    render_navigation()
    
    # 渲染页面标题
    render_page_header(
        title="货币详细分析",
        description="深入分析货币走势，技术指标和市场信号",
        icon="📈"
    )
    
    # 获取URL参数或session state中的货币
    query_params = st.query_params
    
    if 'symbol' in query_params:
        selected_currency = query_params['symbol']
        st.session_state['selected_currency'] = selected_currency
    elif 'selected_currency' in st.session_state:
        selected_currency = st.session_state['selected_currency']
    else:
        selected_currency = 'BTC'
    
    # 返回导航
    st.markdown("""
    <div class="back-nav">
        <h4>🔙 导航</h4>
    </div>
    """, unsafe_allow_html=True)
    
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
    
    with nav_col1:
        if st.button("← 返回概览", key="analysis_back_overview"):
            st.switch_page("pages/1_🌍_货币概览.py")
    
    with nav_col2:
        if st.button("⚖️ 货币比较", key="analysis_nav_compare"):
            st.switch_page("pages/3_⚖️_货币比较.py")
    
    with nav_col3:
        if st.button("🔍 高级筛选", key="analysis_nav_filter"):
            st.switch_page("pages/4_🔍_高级筛选.py")
        
    with nav_col4:
        if st.button("📊 主仪表盘", key="analysis_nav_main"):
            st.switch_page("src/app.py")
    
    # 主标题
    st.markdown(f"""
    <div class="analysis-header">
        <h1>📈 {selected_currency} 详细分析</h1>
        <p>深度技术分析和市场洞察</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 货币选择器
    currencies = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'AVAX', 'MATIC', 'LINK', 'UNI']
    
    col1, col2 = st.columns([2, 1])
    with col1:
        new_currency = st.selectbox(
            "选择要分析的货币",
            currencies,
            index=currencies.index(selected_currency) if selected_currency in currencies else 0,
            key="currency_selector"
        )
    
    with col2:
        if st.button("🔄 刷新数据", key="analysis_refresh_data"):
            st.rerun()
    
    if new_currency != selected_currency:
        st.session_state['selected_currency'] = new_currency
        # 更新URL参数
        st.query_params['symbol'] = new_currency
        st.rerun()
    
    # 基本信息指标
    st.header("📊 基本信息")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        current_price = random.uniform(40000, 50000)
        price_change = random.uniform(-5, 5)
        st.metric(
            "当前价格",
            f"${current_price:,.2f}",
            f"{price_change:+.2f}%"
        )
    
    with col2:
        volume_24h = random.uniform(20e9, 30e9)
        volume_change = random.uniform(-10, 10)
        st.metric(
            "24h交易量",
            f"${volume_24h/1e9:.1f}B",
            f"{volume_change:+.1f}%"
        )
    
    with col3:
        market_cap = random.uniform(800e9, 900e9)
        cap_change = random.uniform(-3, 3)
        st.metric(
            "市值",
            f"${market_cap/1e9:.0f}B",
            f"{cap_change:+.1f}%"
        )
    
    with col4:
        st.metric(
            "市值排名",
            "#1",
            "0"
        )
    
    with col5:
        supply = random.uniform(19e6, 21e6)
        st.metric(
            "流通供应量",
            f"{supply/1e6:.1f}M",
            "0"
        )
    
    # 价格图表
    st.header("📈 价格走势")
    
    # 时间范围选择
    time_col1, time_col2 = st.columns([3, 1])
    
    with time_col1:
        time_range = st.selectbox(
            "时间范围",
            ["1小时", "4小时", "1天", "7天", "30天", "90天", "1年"],
            index=4,
            key="time_range"
        )
    
    with time_col2:
        chart_type = st.selectbox(
            "图表类型",
            ["线图", "蜡烛图", "面积图"],
            key="chart_type"
        )
    
    # 生成和显示价格数据
    days_map = {"1小时": 1, "4小时": 4, "1天": 24, "7天": 168, "30天": 720, "90天": 2160, "1年": 8760}
    hours = days_map.get(time_range, 720)
    
    price_data = generate_price_data(hours, current_price)
    price_chart = create_price_chart(price_data, selected_currency)
    st.plotly_chart(price_chart, use_container_width=True)
    
    # 交易量图表
    volume_chart = create_volume_chart(price_data)
    st.plotly_chart(volume_chart, use_container_width=True)
    
    # 技术指标
    st.header("🔧 技术指标")
    
    indicators = generate_technical_indicators()
    
    ind_col1, ind_col2, ind_col3 = st.columns(3)
    
    with ind_col1:
        st.subheader("动量指标")
        
        rsi = indicators['RSI']
        rsi_color = "indicator-positive" if rsi['signal'] == '买入' else "indicator-negative" if rsi['signal'] == '卖出' else "indicator-neutral"
        st.markdown(f"**RSI (14)**: <span class='{rsi_color}'>{rsi['value']:.1f} ({rsi['signal']})</span>", unsafe_allow_html=True)
        
        macd = indicators['MACD']
        macd_color = "indicator-positive" if macd['signal'] == '买入' else "indicator-negative" if macd['signal'] == '卖出' else "indicator-neutral"
        st.markdown(f"**MACD**: <span class='{macd_color}'>{macd['signal']}</span>", unsafe_allow_html=True)
    
    with ind_col2:
        st.subheader("趋势指标")
        
        ma20 = indicators['MA20']
        ma20_color = "indicator-positive" if ma20['signal'] == '多头' else "indicator-negative" if ma20['signal'] == '空头' else "indicator-neutral"
        st.markdown(f"**MA20**: <span class='{ma20_color}'>${ma20['value']:,.0f} ({ma20['signal']})</span>", unsafe_allow_html=True)
        
        ma50 = indicators['MA50']
        ma50_color = "indicator-positive" if ma50['signal'] == '多头' else "indicator-negative" if ma50['signal'] == '空头' else "indicator-neutral"
        st.markdown(f"**MA50**: <span class='{ma50_color}'>${ma50['value']:,.0f} ({ma50['signal']})</span>", unsafe_allow_html=True)
    
    with ind_col3:
        st.subheader("波动性指标")
        
        bb = indicators['Bollinger']
        bb_color = "indicator-positive" if '突破' in bb['signal'] else "indicator-negative" if '跌破' in bb['signal'] else "indicator-neutral"
        st.markdown(f"**布林带上轨**: ${bb['upper']:,.0f}", unsafe_allow_html=True)
        st.markdown(f"**布林带下轨**: ${bb['lower']:,.0f}", unsafe_allow_html=True)
        st.markdown(f"**信号**: <span class='{bb_color}'>{bb['signal']}</span>", unsafe_allow_html=True)
    
    # 支撑阻力位
    st.header("📍 关键价位")
    
    support_col1, support_col2 = st.columns(2)
    
    with support_col1:
        st.subheader("支撑位")
        support_levels = [
            current_price * 0.95,
            current_price * 0.90,
            current_price * 0.85
        ]
        
        for i, level in enumerate(support_levels, 1):
            st.write(f"**S{i}**: ${level:,.0f}")
    
    with support_col2:
        st.subheader("阻力位")
        resistance_levels = [
            current_price * 1.05,
            current_price * 1.10,
            current_price * 1.15
        ]
        
        for i, level in enumerate(resistance_levels, 1):
            st.write(f"**R{i}**: ${level:,.0f}")
    
    # 侧边栏
    with st.sidebar:
        st.header("📋 分析工具")
        
        st.subheader("🎯 价格预警")
        alert_price = st.number_input(
            "设置价格预警",
            min_value=0.0,
            value=float(current_price * 1.1),
            step=100.0,
            format="%.2f"
        )
        
        alert_type = st.selectbox(
            "预警类型",
            ["价格突破", "价格跌破"]
        )
        
        if st.button("设置预警"):
            st.success(f"已设置{alert_type}预警: ${alert_price:,.2f}")
        
        st.subheader("📊 分析周期")
        analysis_period = st.selectbox(
            "分析周期",
            ["短期 (1-7天)", "中期 (1-4周)", "长期 (1-6月)"]
        )
        
        st.subheader("🔗 相关链接")
        st.markdown(f"""
        - [CoinGecko - {selected_currency}](https://coingecko.com)
        - [CoinMarketCap - {selected_currency}](https://coinmarketcap.com)
        - [TradingView - {selected_currency}](https://tradingview.com)
        """)
        
        st.subheader("📈 快速操作")
        if st.button("添加到比较列表"):
            if 'comparison_list' not in st.session_state:
                st.session_state['comparison_list'] = []
            
            if selected_currency not in st.session_state['comparison_list']:
                st.session_state['comparison_list'].append(selected_currency)
                st.success(f"已添加 {selected_currency}")
            else:
                st.warning("已在比较列表中")
        
        if st.button("分享此分析"):
            share_url = f"http://localhost:8501/详细分析?symbol={selected_currency}"
            st.code(share_url)
            st.info("复制上方链接即可分享此分析页面")
    
    # 渲染页面底部
    render_footer()

if __name__ == "__main__":
    main()
