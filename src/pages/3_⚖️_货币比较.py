"""
货币比较页面
独立网页 - 可通过 /货币比较 直接访问
支持URL参数: ?symbols=BTC,ETH,BNB
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

def generate_currency_data(symbol):
    """生成货币数据"""
    base_prices = {
        'BTC': 45000, 'ETH': 3000, 'BNB': 300, 'ADA': 0.5, 'SOL': 100,
        'DOT': 25, 'AVAX': 35, 'MATIC': 1.2, 'LINK': 15, 'UNI': 8,
        'LTC': 150, 'XRP': 0.6, 'DOGE': 0.08, 'SHIB': 0.00001, 'ATOM': 12
    }
    
    base_price = base_prices.get(symbol, 100)
    
    return {
        'symbol': symbol,
        'name': f'{symbol} Token',
        'price': base_price * random.uniform(0.9, 1.1),
        'change_24h': random.uniform(-10, 10),
        'volume_24h': random.uniform(1e9, 10e9),
        'market_cap': random.uniform(10e9, 100e9),
        'market_cap_rank': random.randint(1, 100),
        'circulating_supply': random.uniform(1e6, 1e9),
        'total_supply': random.uniform(1e6, 1e9),
        'max_supply': random.uniform(1e6, 1e9) if random.choice([True, False]) else None,
        'ath': base_price * random.uniform(1.2, 3.0),
        'ath_change': random.uniform(-80, -10),
        'atl': base_price * random.uniform(0.1, 0.8),
        'atl_change': random.uniform(50, 500),
        'rsi': random.uniform(20, 80),
        'volatility': random.uniform(0.02, 0.15),
        'sharpe_ratio': random.uniform(-1, 3),
        'beta': random.uniform(0.5, 2.0)
    }

def generate_historical_data(symbols, days=30):
    """生成历史价格数据"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    data = {'date': dates}
    
    for symbol in symbols:
        base_price = generate_currency_data(symbol)['price']
        prices = []
        current_price = base_price
        
        for _ in range(days):
            change = random.uniform(-0.05, 0.05)
            current_price *= (1 + change)
            prices.append(current_price)
        
        data[symbol] = prices
    
    return pd.DataFrame(data)

def create_comparison_chart(data, symbols):
    """创建比较图表"""
    fig = go.Figure()
    
    colors = ['#00d4aa', '#667eea', '#f093fb', '#ffa726', '#ff6b6b']
    
    for i, symbol in enumerate(symbols):
        if symbol in data.columns:
            # 标准化价格 (以第一天为基准)
            normalized_prices = (data[symbol] / data[symbol].iloc[0] - 1) * 100
            
            fig.add_trace(go.Scatter(
                x=data['date'],
                y=normalized_prices,
                mode='lines',
                name=symbol,
                line=dict(color=colors[i % len(colors)], width=3),
                hovertemplate=f'<b>{symbol}</b><br>%{{y:.2f}}%<br>%{{x}}<extra></extra>'
            ))
    
    fig.update_layout(
        title='价格表现比较 (标准化)',
        xaxis_title='日期',
        yaxis_title='价格变化 (%)',
        template='plotly_white',
        height=500,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def create_correlation_heatmap(data, symbols):
    """创建相关性热力图"""
    # 计算相关性矩阵
    price_data = data[symbols].pct_change().dropna()
    correlation_matrix = price_data.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix.round(2).values,
        texttemplate='%{text}',
        textfont={"size": 12},
        hovertemplate='<b>%{x} vs %{y}</b><br>相关性: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='货币相关性矩阵',
        template='plotly_white',
        height=400
    )
    
    return fig

def create_risk_return_scatter(currencies_data):
    """创建风险收益散点图"""
    symbols = [data['symbol'] for data in currencies_data]
    returns = [data['change_24h'] for data in currencies_data]
    volatilities = [data['volatility'] * 100 for data in currencies_data]
    market_caps = [data['market_cap'] for data in currencies_data]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=volatilities,
        y=returns,
        mode='markers+text',
        text=symbols,
        textposition='top center',
        marker=dict(
            size=[np.log(cap/1e9) * 5 for cap in market_caps],
            color=returns,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="24h收益率 (%)"),
            line=dict(width=2, color='white')
        ),
        hovertemplate='<b>%{text}</b><br>波动率: %{x:.2f}%<br>收益率: %{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title='风险收益分析',
        xaxis_title='波动率 (%)',
        yaxis_title='24小时收益率 (%)',
        template='plotly_white',
        height=500
    )
    
    return fig

def main():
    # 渲染导航栏
    render_navigation()
    
    # 渲染页面标题
    render_page_header(
        title="货币比较分析",
        description="对比不同货币表现，发现投资机会",
        icon="⚖️"
    )
    
    # 获取URL参数或session state中的货币列表
    query_params = st.query_params
    
    if 'symbols' in query_params:
        symbols_param = query_params['symbols']
        selected_currencies = symbols_param.split(',')
        st.session_state['comparison_list'] = selected_currencies
    elif 'comparison_list' in st.session_state:
        selected_currencies = st.session_state['comparison_list']
    else:
        selected_currencies = ['BTC', 'ETH', 'BNB']
    
    # 返回导航
    st.markdown("""
    <div class="back-nav">
        <h4>🔙 导航</h4>
    </div>
    """, unsafe_allow_html=True)
    
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
    
    with nav_col1:
        if st.button("← 返回概览", key="compare_back_overview"):
            st.switch_page("pages/1_🌍_货币概览.py")
    
    with nav_col2:
        if st.button("📈 详细分析", key="compare_nav_analysis"):
            st.switch_page("pages/2_📈_详细分析.py")
    
    with nav_col3:
        if st.button("🔍 高级筛选", key="compare_nav_filter"):
            st.switch_page("pages/4_🔍_高级筛选.py")
    
    with nav_col4:
        if st.button("📊 主仪表盘", key="compare_nav_main"):
            st.switch_page("src/app.py")
    
    # 主标题
    st.markdown("""
    <div class="compare-header">
        <h1>⚖️ 货币比较分析</h1>
        <p>多维度对比分析，发现投资机会</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 货币选择器
    all_currencies = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'AVAX', 'MATIC', 'LINK', 'UNI', 'LTC', 'XRP', 'DOGE', 'SHIB', 'ATOM']
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        new_currencies = st.multiselect(
            "选择要比较的货币 (最多5个)",
            all_currencies,
            default=selected_currencies[:5],
            max_selections=5,
            key="currency_multiselect"
        )
    
    with col2:
        if st.button("🔄 刷新数据", key="compare_refresh_data"):
            st.rerun()
    
    if new_currencies != selected_currencies:
        st.session_state['comparison_list'] = new_currencies
        # 更新URL参数
        st.query_params['symbols'] = ','.join(new_currencies)
        st.rerun()
    
    if not new_currencies:
        st.warning("请至少选择一个货币进行比较")
        return
    
    # 显示选中的货币
    st.markdown("**当前比较的货币:**")
    currency_tags = ""
    for currency in new_currencies:
        currency_tags += f'<span class="currency-tag">{currency}</span>'
    st.markdown(currency_tags, unsafe_allow_html=True)
    
    # 生成货币数据
    currencies_data = [generate_currency_data(symbol) for symbol in new_currencies]
    
    # 基本指标比较
    st.header("📊 基本指标比较")
    
    # 创建比较表格
    comparison_df = pd.DataFrame([
        {
            '货币': data['symbol'],
            '当前价格': f"${data['price']:,.2f}",
            '24h变化': f"{data['change_24h']:+.2f}%",
            '市值': f"${data['market_cap']/1e9:.1f}B",
            '交易量': f"${data['volume_24h']/1e9:.1f}B",
            '市值排名': f"#{data['market_cap_rank']}",
            'RSI': f"{data['rsi']:.1f}",
            '波动率': f"{data['volatility']*100:.1f}%"
        }
        for data in currencies_data
    ])
    
    st.dataframe(comparison_df, use_container_width=True)
    
    # 性能对比
    st.header("📈 性能对比")
    
    perf_col1, perf_col2 = st.columns(2)
    
    with perf_col1:
        st.subheader("24小时表现排名")
        
        # 按24h变化排序
        sorted_data = sorted(currencies_data, key=lambda x: x['change_24h'], reverse=True)
        
        for i, data in enumerate(sorted_data, 1):
            change = data['change_24h']
            if change > 0:
                style_class = "winner"
                icon = "🚀"
            elif change < -2:
                style_class = "loser"
                icon = "📉"
            else:
                style_class = "neutral"
                icon = "➡️"
            
            st.markdown(f"""
            <div class="{style_class}">
                {icon} #{i} {data['symbol']}: {change:+.2f}%
            </div>
            """, unsafe_allow_html=True)
            st.write("")
    
    with perf_col2:
        st.subheader("市值排名")
        
        # 按市值排序
        sorted_by_cap = sorted(currencies_data, key=lambda x: x['market_cap'], reverse=True)
        
        for i, data in enumerate(sorted_by_cap, 1):
            market_cap = data['market_cap']
            st.markdown(f"""
            <div class="comparison-card">
                <strong>#{i} {data['symbol']}</strong><br>
                市值: ${market_cap/1e9:.1f}B<br>
                排名: #{data['market_cap_rank']}
            </div>
            """, unsafe_allow_html=True)
    
    # 价格走势比较
    st.header("📈 价格走势比较")
    
    time_range = st.selectbox(
        "选择时间范围",
        ["7天", "30天", "90天", "1年"],
        index=1,
        key="price_time_range"
    )
    
    days_map = {"7天": 7, "30天": 30, "90天": 90, "1年": 365}
    days = days_map[time_range]
    
    historical_data = generate_historical_data(new_currencies, days)
    price_chart = create_comparison_chart(historical_data, new_currencies)
    st.plotly_chart(price_chart, use_container_width=True)
    
    # 相关性分析
    if len(new_currencies) > 1:
        st.header("🔗 相关性分析")
        correlation_chart = create_correlation_heatmap(historical_data, new_currencies)
        st.plotly_chart(correlation_chart, use_container_width=True)
        
        st.info("相关性接近1表示两个货币价格走势高度相关，接近-1表示负相关，接近0表示无相关性。")
    
    # 风险收益分析
    st.header("⚖️ 风险收益分析")
    
    risk_return_chart = create_risk_return_scatter(currencies_data)
    st.plotly_chart(risk_return_chart, use_container_width=True)
    
    st.info("气泡大小代表市值，颜色代表24小时收益率。理想投资标的应位于左上角（低风险高收益）。")
    
    # 技术指标对比
    st.header("🔧 技术指标对比")
    
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    
    with tech_col1:
        st.subheader("RSI 对比")
        for data in currencies_data:
            rsi = data['rsi']
            if rsi > 70:
                color = "🔴"
                status = "超买"
            elif rsi < 30:
                color = "🟢"
                status = "超卖"
            else:
                color = "🟡"
                status = "中性"
            
            st.write(f"{color} **{data['symbol']}**: {rsi:.1f} ({status})")
    
    with tech_col2:
        st.subheader("波动率对比")
        for data in currencies_data:
            volatility = data['volatility'] * 100
            if volatility > 10:
                risk_level = "高风险"
                color = "🔴"
            elif volatility > 5:
                risk_level = "中风险"
                color = "🟡"
            else:
                risk_level = "低风险"
                color = "🟢"
            
            st.write(f"{color} **{data['symbol']}**: {volatility:.1f}% ({risk_level})")
    
    with tech_col3:
        st.subheader("夏普比率对比")
        for data in currencies_data:
            sharpe = data['sharpe_ratio']
            if sharpe > 1:
                rating = "优秀"
                color = "🟢"
            elif sharpe > 0:
                rating = "良好"
                color = "🟡"
            else:
                rating = "较差"
                color = "🔴"
            
            st.write(f"{color} **{data['symbol']}**: {sharpe:.2f} ({rating})")
    
    # 侧边栏
    with st.sidebar:
        st.header("🛠️ 比较工具")
        
        st.subheader("📋 快速添加")
        quick_add = st.selectbox(
            "快速添加货币",
            [c for c in all_currencies if c not in new_currencies],
            key="quick_add_currency"
        )
        
        if st.button("添加到比较"):
            if len(new_currencies) < 5:
                new_list = new_currencies + [quick_add]
                st.session_state['comparison_list'] = new_list
                st.query_params['symbols'] = ','.join(new_list)
                st.rerun()
            else:
                st.warning("最多只能比较5个货币")
        
        st.subheader("🎯 预设组合")
        preset_groups = {
            "主流币": ["BTC", "ETH", "BNB"],
            "DeFi代币": ["UNI", "LINK", "AVAX"],
            "Layer1": ["ETH", "SOL", "ADA", "DOT"],
            "Meme币": ["DOGE", "SHIB"]
        }
        
        for group_name, symbols in preset_groups.items():
            if st.button(f"加载 {group_name}"):
                st.session_state['comparison_list'] = symbols
                st.query_params['symbols'] = ','.join(symbols)
                st.rerun()
        
        st.subheader("📊 导出数据")
        if st.button("导出比较数据"):
            csv_data = comparison_df.to_csv(index=False)
            st.download_button(
                label="下载CSV文件",
                data=csv_data,
                file_name=f"currency_comparison_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        st.subheader("🔗 分享比较")
        if st.button("生成分享链接"):
            share_url = f"http://localhost:8501/货币比较?symbols={','.join(new_currencies)}"
            st.code(share_url)
            st.info("复制上方链接即可分享此比较页面")
        
        st.subheader("⚠️ 风险提示")
        st.warning("""
        **投资风险提示:**
        - 加密货币投资存在高风险
        - 价格波动剧烈，可能造成重大损失
        - 请根据自身风险承受能力投资
        - 本分析仅供参考，不构成投资建议
        """)
    
    # 渲染页面底部
    render_footer()

if __name__ == "__main__":
    main()