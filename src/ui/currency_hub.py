"""
货币比对中心 - 分层导航架构
基于专业交易软件的设计理念，提供清晰的页面层次结构
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random

class CurrencyHub:
    """货币比对中心主控制器"""
    
    def __init__(self):
        self.init_session_state()
    
    def init_session_state(self):
        """初始化会话状态"""
        if 'currency_page' not in st.session_state:
            st.session_state.currency_page = '概览'
        if 'selected_currency' not in st.session_state:
            st.session_state.selected_currency = 'BTC'
        if 'comparison_currencies' not in st.session_state:
            st.session_state.comparison_currencies = ['BTC', 'ETH']
    
    def render_navigation(self):
        """渲染导航栏"""
        st.markdown("""
        <style>
        .currency-nav {
            background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .nav-title {
            color: white;
            font-size: 1.5rem;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="currency-nav">', unsafe_allow_html=True)
            st.markdown('<div class="nav-title">🌍 货币比对中心</div>', unsafe_allow_html=True)
            
            # 页面导航
            pages = ['概览', '详细分析', '货币比较', '高级筛选']
            selected_page = st.selectbox(
                "选择页面",
                pages,
                index=pages.index(st.session_state.currency_page),
                key="page_selector"
            )
            
            if selected_page != st.session_state.currency_page:
                st.session_state.currency_page = selected_page
                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def render_main_interface(self):
        """渲染主界面"""
        self.render_navigation()
        
        # 根据选择的页面渲染对应内容
        if st.session_state.currency_page == '概览':
            self.render_overview_page()
        elif st.session_state.currency_page == '详细分析':
            self.render_analysis_page()
        elif st.session_state.currency_page == '货币比较':
            self.render_comparison_page()
        elif st.session_state.currency_page == '高级筛选':
            self.render_filter_page()
    
    def render_overview_page(self):
        """渲染概览页面"""
        st.header("📊 市场概览")
        
        # 快速统计
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总市值", "$2.1T", "2.3%")
        with col2:
            st.metric("24h交易量", "$89.2B", "-1.2%")
        with col3:
            st.metric("活跃货币", "100", "0")
        with col4:
            st.metric("涨跌比", "67:33", "5.2%")
        
        # 快速筛选
        st.subheader("🔍 快速筛选")
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            market_cap_filter = st.selectbox(
                "市值范围",
                ["全部", "大盘股 (>$10B)", "中盘股 ($1B-$10B)", "小盘股 (<$1B)"]
            )
        
        with filter_col2:
            change_filter = st.selectbox(
                "24h涨跌",
                ["全部", "上涨 (>0%)", "下跌 (<0%)", "大涨 (>5%)", "大跌 (<-5%)"]
            )
        
        with filter_col3:
            category_filter = st.selectbox(
                "货币类别",
                ["全部", "主流币", "DeFi", "Layer1", "Layer2", "Meme币"]
            )
        
        # 主要货币表格
        st.subheader("💰 主要货币")
        df = self.generate_currency_data(30)
        
        # 应用筛选
        filtered_df = self.apply_filters(df, market_cap_filter, change_filter, category_filter)
        
        # 可点击的表格
        selected_currency = st.dataframe(
            filtered_df,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row"
        )
        
        # 处理货币选择
        if selected_currency.selection.rows:
            selected_idx = selected_currency.selection.rows[0]
            currency_symbol = filtered_df.iloc[selected_idx]['货币']
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.info(f"已选择: {currency_symbol}")
            with col2:
                if st.button("查看详细分析", key="view_analysis"):
                    st.session_state.selected_currency = currency_symbol
                    st.session_state.currency_page = '详细分析'
                    st.rerun()
    
    def render_analysis_page(self):
        """渲染详细分析页面"""
        st.header(f"📈 {st.session_state.selected_currency} 详细分析")
        
        # 货币选择器
        currencies = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'AVAX', 'MATIC']
        selected = st.selectbox(
            "选择货币",
            currencies,
            index=currencies.index(st.session_state.selected_currency) if st.session_state.selected_currency in currencies else 0
        )
        
        if selected != st.session_state.selected_currency:
            st.session_state.selected_currency = selected
            st.rerun()
        
        # 基本信息
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("当前价格", "$45,230", "2.3%")
        with col2:
            st.metric("24h交易量", "$28.5B", "-1.2%")
        with col3:
            st.metric("市值", "$890B", "2.1%")
        with col4:
            st.metric("市值排名", "#1", "0")
        
        # 价格图表
        st.subheader("📊 价格走势")
        time_range = st.selectbox("时间范围", ["1天", "7天", "30天", "90天", "1年"], index=2)
        
        # 生成模拟价格数据
        chart_data = self.generate_price_chart_data(time_range)
        fig = self.create_price_chart(chart_data, selected)
        st.plotly_chart(fig, use_container_width=True)
        
        # 技术指标
        st.subheader("🔧 技术指标")
        indicator_col1, indicator_col2 = st.columns(2)
        
        with indicator_col1:
            st.write("**RSI (14)**: 65.2 (中性)")
            st.write("**MACD**: 买入信号")
            st.write("**布林带**: 上轨突破")
        
        with indicator_col2:
            st.write("**移动平均线**: 多头排列")
            st.write("**成交量**: 放量上涨")
            st.write("**支撑位**: $43,500")
        
        # 返回概览按钮
        if st.button("← 返回概览", key="back_to_overview"):
            st.session_state.currency_page = '概览'
            st.rerun()
    
    def render_comparison_page(self):
        """渲染货币比较页面"""
        st.header("⚖️ 货币比较分析")
        
        # 货币选择
        st.subheader("选择比较货币")
        currencies = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'AVAX', 'MATIC', 'LINK', 'UNI']
        
        selected_currencies = st.multiselect(
            "选择要比较的货币 (最多5个)",
            currencies,
            default=st.session_state.comparison_currencies[:5],
            max_selections=5
        )
        
        if selected_currencies:
            st.session_state.comparison_currencies = selected_currencies
            
            # 比较表格
            st.subheader("📊 对比数据")
            comparison_df = self.generate_comparison_data(selected_currencies)
            st.dataframe(comparison_df, use_container_width=True)
            
            # 相关性热力图
            st.subheader("🔥 相关性分析")
            correlation_data = self.generate_correlation_data(selected_currencies)
            fig_heatmap = px.imshow(
                correlation_data,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="货币价格相关性热力图"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # 性能对比图
            st.subheader("📈 性能对比")
            performance_data = self.generate_performance_data(selected_currencies)
            fig_performance = px.line(
                performance_data,
                x='日期',
                y='收益率',
                color='货币',
                title="30天收益率对比"
            )
            st.plotly_chart(fig_performance, use_container_width=True)
    
    def render_filter_page(self):
        """渲染高级筛选页面"""
        st.header("🔍 高级筛选器")
        
        # 筛选条件
        st.subheader("筛选条件")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**价格范围**")
            price_range = st.slider("价格 (USD)", 0, 100000, (100, 50000))
            
            st.write("**市值范围**")
            market_cap_range = st.slider("市值 (亿USD)", 0, 10000, (10, 5000))
            
            st.write("**交易量范围**")
            volume_range = st.slider("24h交易量 (亿USD)", 0, 500, (1, 100))
        
        with col2:
            st.write("**涨跌幅范围**")
            change_range = st.slider("24h涨跌幅 (%)", -50, 50, (-10, 10))
            
            st.write("**技术指标**")
            rsi_range = st.slider("RSI", 0, 100, (30, 70))
            
            st.write("**其他条件**")
            min_age = st.number_input("最小上市天数", min_value=0, value=365)
        
        # 应用筛选
        if st.button("应用筛选", type="primary"):
            filtered_data = self.apply_advanced_filters(
                price_range, market_cap_range, volume_range, 
                change_range, rsi_range, min_age
            )
            
            st.subheader("筛选结果")
            st.dataframe(filtered_data, use_container_width=True)
            
            st.success(f"找到 {len(filtered_data)} 个符合条件的货币")
    
    def generate_currency_data(self, count=30):
        """生成模拟货币数据"""
        currencies = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'AVAX', 'MATIC', 'LINK', 'UNI',
                     'ATOM', 'ICP', 'FTM', 'ALGO', 'XTZ', 'EGLD', 'THETA', 'VET', 'FIL', 'TRX',
                     'EOS', 'AAVE', 'MKR', 'COMP', 'SNX', 'YFI', 'UMA', 'BAL', 'CRV', 'SUSHI']
        
        data = []
        for i, currency in enumerate(currencies[:count]):
            price = random.uniform(0.1, 50000)
            change_24h = random.uniform(-15, 15)
            volume = random.uniform(1e6, 1e10)
            market_cap = random.uniform(1e8, 1e12)
            
            data.append({
                '排名': i + 1,
                '货币': currency,
                '价格': f"${price:.2f}",
                '24h涨跌': f"{change_24h:+.2f}%",
                '24h交易量': f"${volume/1e9:.2f}B",
                '市值': f"${market_cap/1e9:.2f}B",
                '流通量': f"{random.uniform(1e6, 1e9):.0f}",
                '涨跌色': 'green' if change_24h > 0 else 'red'
            })
        
        return pd.DataFrame(data)
    
    def apply_filters(self, df, market_cap_filter, change_filter, category_filter):
        """应用筛选条件"""
        # 这里可以根据实际筛选条件过滤数据
        # 目前返回原数据作为演示
        return df
    
    def generate_price_chart_data(self, time_range):
        """生成价格图表数据"""
        days_map = {"1天": 1, "7天": 7, "30天": 30, "90天": 90, "1年": 365}
        days = days_map.get(time_range, 30)
        
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        prices = []
        base_price = 45000
        
        for i in range(days):
            change = random.uniform(-0.05, 0.05)
            base_price *= (1 + change)
            prices.append(base_price)
        
        return pd.DataFrame({
            'Date': dates,
            'Price': prices
        })
    
    def create_price_chart(self, data, currency):
        """创建价格图表"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data['Date'],
            y=data['Price'],
            mode='lines',
            name=f'{currency} 价格',
            line=dict(color='#00d4aa', width=2)
        ))
        
        fig.update_layout(
            title=f'{currency} 价格走势',
            xaxis_title='日期',
            yaxis_title='价格 (USD)',
            template='plotly_dark',
            height=400
        )
        
        return fig
    
    def generate_comparison_data(self, currencies):
        """生成比较数据"""
        data = []
        for currency in currencies:
            data.append({
                '货币': currency,
                '价格': f"${random.uniform(0.1, 50000):.2f}",
                '24h涨跌': f"{random.uniform(-15, 15):+.2f}%",
                '7d涨跌': f"{random.uniform(-30, 30):+.2f}%",
                '30d涨跌': f"{random.uniform(-50, 50):+.2f}%",
                '市值': f"${random.uniform(1e8, 1e12)/1e9:.2f}B",
                'RSI': f"{random.uniform(20, 80):.1f}"
            })
        
        return pd.DataFrame(data)
    
    def generate_correlation_data(self, currencies):
        """生成相关性数据"""
        n = len(currencies)
        correlation_matrix = np.random.rand(n, n)
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        np.fill_diagonal(correlation_matrix, 1)
        
        return pd.DataFrame(
            correlation_matrix,
            index=currencies,
            columns=currencies
        )
    
    def generate_performance_data(self, currencies):
        """生成性能数据"""
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        data = []
        
        for currency in currencies:
            returns = np.cumsum(np.random.normal(0, 0.02, 30))
            for i, date in enumerate(dates):
                data.append({
                    '日期': date,
                    '货币': currency,
                    '收益率': returns[i] * 100
                })
        
        return pd.DataFrame(data)
    
    def apply_advanced_filters(self, price_range, market_cap_range, volume_range, 
                             change_range, rsi_range, min_age):
        """应用高级筛选"""
        # 生成符合筛选条件的模拟数据
        filtered_count = random.randint(5, 25)
        return self.generate_currency_data(filtered_count)

def apply_currency_hub_styles():
    """应用货币中心样式"""
    st.markdown("""
    <style>
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .currency-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .positive-change {
        color: #00d4aa;
        font-weight: bold;
    }
    
    .negative-change {
        color: #ff6b6b;
        font-weight: bold;
    }
    
    .page-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)