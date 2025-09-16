"""
货币概览主页
独立网页 - 可通过 /货币概览 直接访问
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
    page_title="货币概览 - 加密货币分析平台",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义样式
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
    color: white;
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 2rem;
}

.metric-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    border-left: 4px solid #2a5298;
}

.currency-row {
    padding: 0.5rem;
    border-radius: 5px;
    margin: 0.2rem 0;
}

.currency-row:hover {
    background-color: #f0f2f6;
}

.positive {
    color: #00d4aa;
    font-weight: bold;
}

.negative {
    color: #ff6b6b;
    font-weight: bold;
}

.quick-nav {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}

.nav-button {
    background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 0.8rem 1.5rem;
    border: none;
    border-radius: 8px;
    text-decoration: none;
    display: inline-block;
    margin: 0.5rem;
    font-weight: bold;
    transition: transform 0.2s;
}

.nav-button:hover {
    transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)

def generate_market_data():
    """生成市场数据"""
    currencies = [
        'BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'AVAX', 'MATIC', 'LINK', 'UNI',
        'ATOM', 'ICP', 'FTM', 'ALGO', 'XTZ', 'EGLD', 'THETA', 'VET', 'FIL', 'TRX',
        'EOS', 'AAVE', 'MKR', 'COMP', 'SNX', 'YFI', 'UMA', 'BAL', 'CRV', 'SUSHI'
    ]
    
    data = []
    for i, currency in enumerate(currencies):
        price = random.uniform(0.1, 50000)
        change_24h = random.uniform(-15, 15)
        volume = random.uniform(1e6, 1e10)
        market_cap = random.uniform(1e8, 1e12)
        
        data.append({
            '排名': i + 1,
            '货币': currency,
            '价格': price,
            '价格显示': f"${price:.2f}",
            '24h涨跌': change_24h,
            '24h涨跌显示': f"{change_24h:+.2f}%",
            '24h交易量': f"${volume/1e9:.2f}B",
            '市值': f"${market_cap/1e9:.2f}B",
            '流通量': f"{random.uniform(1e6, 1e9):.0f}",
        })
    
    return pd.DataFrame(data)

def main():
    # 渲染导航栏
    render_navigation()
    
    # 渲染页面标题
    render_page_header(
        title="全球货币市场概览",
        description="实时监控全球货币市场动态，掌握投资先机",
        icon="🌍"
    )
    
    # 快速导航
    st.markdown("""
    <div class="quick-nav">
        <h3>🚀 快速导航</h3>
        <p>选择您需要的功能模块：</p>
    </div>
    """, unsafe_allow_html=True)
    
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
    
    with nav_col1:
        if st.button("📈 详细分析", key="overview_nav_analysis", help="深入分析单个货币"):
            st.switch_page("pages/2_📈_详细分析.py")
    
    with nav_col2:
        if st.button("⚖️ 货币比较", key="overview_nav_compare", help="对比多个货币"):
            st.switch_page("pages/3_⚖️_货币比较.py")
    
    with nav_col3:
        if st.button("🔍 高级筛选", key="overview_nav_filter", help="自定义筛选条件"):
            st.switch_page("pages/4_🔍_高级筛选.py")
    
    with nav_col4:
        if st.button("📊 实时仪表盘", key="overview_nav_dashboard", help="返回主仪表盘"):
            st.switch_page("src/app.py")
    
    # 市场统计
    st.header("📊 市场统计")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="总市值",
            value="$2.1T",
            delta="2.3%",
            help="全球加密货币总市值"
        )
    
    with col2:
        st.metric(
            label="24h交易量",
            value="$89.2B",
            delta="-1.2%",
            help="过去24小时总交易量"
        )
    
    with col3:
        st.metric(
            label="活跃货币",
            value="100",
            delta="0",
            help="当前追踪的货币数量"
        )
    
    with col4:
        st.metric(
            label="涨跌比",
            value="67:33",
            delta="5.2%",
            help="上涨vs下跌货币比例"
        )
    
    # 筛选器
    st.header("🔍 快速筛选")
    
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    
    with filter_col1:
        market_cap_filter = st.selectbox(
            "市值范围",
            ["全部", "大盘股 (>$10B)", "中盘股 ($1B-$10B)", "小盘股 (<$1B)"],
            help="按市值大小筛选"
        )
    
    with filter_col2:
        change_filter = st.selectbox(
            "24h涨跌",
            ["全部", "上涨 (>0%)", "下跌 (<0%)", "大涨 (>5%)", "大跌 (<-5%)"],
            help="按涨跌幅筛选"
        )
    
    with filter_col3:
        category_filter = st.selectbox(
            "货币类别",
            ["全部", "主流币", "DeFi", "Layer1", "Layer2", "Meme币"],
            help="按货币类别筛选"
        )
    
    with filter_col4:
        sort_by = st.selectbox(
            "排序方式",
            ["市值", "价格", "24h涨跌", "交易量"],
            help="选择排序依据"
        )
    
    # 主要货币列表
    st.header("💰 主要货币")
    
    # 生成数据
    df = generate_market_data()
    
    # 应用筛选（简化版）
    if change_filter == "上涨 (>0%)":
        df = df[df['24h涨跌'] > 0]
    elif change_filter == "下跌 (<0%)":
        df = df[df['24h涨跌'] < 0]
    elif change_filter == "大涨 (>5%)":
        df = df[df['24h涨跌'] > 5]
    elif change_filter == "大跌 (<-5%)":
        df = df[df['24h涨跌'] < -5]
    
    # 显示表格
    display_df = df[['排名', '货币', '价格显示', '24h涨跌显示', '24h交易量', '市值', '流通量']].copy()
    display_df.columns = ['排名', '货币', '价格', '24h涨跌', '24h交易量', '市值', '流通量']
    
    # 可选择的表格
    selected_rows = st.dataframe(
        display_df,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        hide_index=True
    )
    
    # 处理选择
    if selected_rows.selection.rows:
        selected_idx = selected_rows.selection.rows[0]
        selected_currency = df.iloc[selected_idx]['货币']
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.success(f"已选择: {selected_currency}")
        
        with col2:
            if st.button("📈 查看详细分析", key="view_detail"):
                # 保存选择的货币到session state
                st.session_state['selected_currency'] = selected_currency
                st.switch_page("pages/2_📈_详细分析.py")
        
        with col3:
            if st.button("⚖️ 添加到比较", key="add_compare"):
                # 添加到比较列表
                if 'comparison_list' not in st.session_state:
                    st.session_state['comparison_list'] = []
                
                if selected_currency not in st.session_state['comparison_list']:
                    st.session_state['comparison_list'].append(selected_currency)
                    st.success(f"已添加 {selected_currency} 到比较列表")
                else:
                    st.warning(f"{selected_currency} 已在比较列表中")
    
    # 市场热力图
    st.header("🔥 市场热力图")
    
    # 创建热力图数据
    heatmap_data = df.head(20).copy()
    heatmap_data['size'] = heatmap_data['价格'] / heatmap_data['价格'].max() * 100
    
    fig = px.treemap(
        heatmap_data,
        path=['货币'],
        values='size',
        color='24h涨跌',
        color_continuous_scale='RdYlGn',
        title="市场热力图 - 按市值大小和涨跌幅着色"
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # 侧边栏信息
    with st.sidebar:
        st.header("📋 页面信息")
        st.info("""
        **当前页面**: 货币概览
        
        **功能**:
        - 市场总体统计
        - 快速筛选和排序
        - 货币列表浏览
        - 市场热力图
        
        **操作提示**:
        - 点击表格行选择货币
        - 使用导航按钮跳转到其他功能
        - 筛选器可以快速过滤数据
        """)
        
        st.header("🔗 快速链接")
        st.markdown("""
        - [详细分析](/详细分析)
        - [货币比较](/货币比较)  
        - [高级筛选](/高级筛选)
        - [实时仪表盘](/)
        """)
        
        # 显示比较列表
        if 'comparison_list' in st.session_state and st.session_state['comparison_list']:
            st.header("⚖️ 比较列表")
            for currency in st.session_state['comparison_list']:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(currency)
                with col2:
                    if st.button("❌", key=f"remove_{currency}"):
                        st.session_state['comparison_list'].remove(currency)
                        st.rerun()
            
            if st.button("🔍 开始比较", key="start_compare"):
                st.switch_page("pages/3_⚖️_货币比较.py")
    
    # 渲染页面底部
    render_footer()

if __name__ == "__main__":
    main()