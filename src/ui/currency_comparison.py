"""
货币比对界面模块
支持100+种货币的多维度比对分析
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Tuple
import asyncio

class CurrencyComparison:
    """货币比对分析界面"""
    
    def __init__(self):
        self.view_modes = {
            "📊 表格视图": "table",
            "🎯 卡片视图": "cards", 
            "🔥 热力图": "heatmap",
            "📈 图表视图": "charts"
        }
        
        self.sort_options = {
            "价格": "price",
            "24h涨跌幅": "change_24h",
            "7d涨跌幅": "change_7d", 
            "成交量": "volume",
            "市值": "market_cap",
            "流通量": "circulating_supply"
        }
        
        self.filter_categories = {
            "全部": "all",
            "主流币": "major",
            "DeFi": "defi",
            "Layer1": "layer1",
            "Layer2": "layer2",
            "稳定币": "stablecoin",
            "Meme币": "meme"
        }
        
    def render_comparison_interface(self):
        """渲染货币比对主界面"""
        st.markdown("""
        <div class="currency-comparison-header">
            <h1>🌍 全球货币比对中心</h1>
            <p>实时监控100+种数字货币，多维度对比分析</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 控制面板
        self._render_control_panel()
        
        # 主要内容区域
        view_mode = st.session_state.get('comparison_view_mode', 'table')
        
        if view_mode == 'table':
            self._render_table_view()
        elif view_mode == 'cards':
            self._render_cards_view()
        elif view_mode == 'heatmap':
            self._render_heatmap_view()
        elif view_mode == 'charts':
            self._render_charts_view()
            
    def _render_control_panel(self):
        """渲染控制面板"""
        st.markdown("""
        <div class="control-panel">
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])
        
        with col1:
            view_mode = st.selectbox(
                "🎛️ 视图模式",
                options=list(self.view_modes.keys()),
                key="view_mode_selector"
            )
            st.session_state['comparison_view_mode'] = self.view_modes[view_mode]
            
        with col2:
            sort_by = st.selectbox(
                "📈 排序方式", 
                options=list(self.sort_options.keys()),
                key="sort_selector"
            )
            st.session_state['sort_by'] = self.sort_options[sort_by]
            
        with col3:
            category = st.selectbox(
                "🏷️ 分类筛选",
                options=list(self.filter_categories.keys()),
                key="category_selector"
            )
            st.session_state['filter_category'] = self.filter_categories[category]
            
        with col4:
            price_range = st.slider(
                "💰 价格范围 (USD)",
                min_value=0.0,
                max_value=100000.0,
                value=(0.0, 100000.0),
                step=100.0,
                key="price_range"
            )
            
        with col5:
            auto_refresh = st.checkbox("🔄 自动刷新", value=True, key="auto_refresh")
            if auto_refresh:
                refresh_interval = st.selectbox(
                    "刷新间隔",
                    options=["5秒", "10秒", "30秒", "1分钟"],
                    index=1,
                    key="refresh_interval"
                )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 快速筛选按钮
        st.markdown("### 🚀 快速筛选")
        quick_filter_cols = st.columns(6)
        
        filters = [
            ("🔥 热门", "trending"),
            ("📈 涨幅榜", "gainers"), 
            ("📉 跌幅榜", "losers"),
            ("💎 新币", "new_listings"),
            ("⚡ 高波动", "high_volatility"),
            ("🎯 关注列表", "watchlist")
        ]
        
        for i, (label, filter_type) in enumerate(filters):
            with quick_filter_cols[i]:
                if st.button(label, key=f"quick_filter_{filter_type}"):
                    st.session_state['quick_filter'] = filter_type
                    st.rerun()
    
    def _render_table_view(self):
        """渲染表格视图"""
        st.markdown("### 📊 详细数据表格")
        
        # 生成模拟数据
        data = self._generate_mock_data()
        
        # 应用筛选和排序
        filtered_data = self._apply_filters(data)
        
        # 自定义列选择
        col_selector_col, display_col = st.columns([1, 4])
        
        with col_selector_col:
            st.markdown("**显示列:**")
            show_columns = {
                "排名": st.checkbox("排名", value=True, key="show_rank"),
                "货币": st.checkbox("货币", value=True, key="show_currency"),
                "价格": st.checkbox("价格", value=True, key="show_price"),
                "24h变化": st.checkbox("24h变化", value=True, key="show_24h"),
                "7d变化": st.checkbox("7d变化", value=True, key="show_7d"),
                "成交量": st.checkbox("成交量", value=True, key="show_volume"),
                "市值": st.checkbox("市值", value=True, key="show_market_cap"),
                "图表": st.checkbox("迷你图表", value=True, key="show_chart")
            }
        
        with display_col:
            # 创建可配置的数据表格
            display_data = filtered_data.copy()
            
            # 格式化数据
            display_data = self._format_table_data(display_data, show_columns)
            
            # 使用streamlit的数据编辑器显示
            st.dataframe(
                display_data,
                use_container_width=True,
                height=600,
                column_config={
                    "排名": st.column_config.NumberColumn("排名", width="small"),
                    "货币": st.column_config.TextColumn("货币", width="medium"),
                    "价格": st.column_config.NumberColumn("价格 (USD)", format="$%.4f"),
                    "24h变化": st.column_config.NumberColumn("24h变化 (%)", format="%.2f%%"),
                    "7d变化": st.column_config.NumberColumn("7d变化 (%)", format="%.2f%%"),
                    "成交量": st.column_config.NumberColumn("成交量 (USD)", format="$%.0f"),
                    "市值": st.column_config.NumberColumn("市值 (USD)", format="$%.0f"),
                    "图表": st.column_config.LineChartColumn("价格走势", width="medium")
                }
            )
            
    def _render_cards_view(self):
        """渲染卡片视图"""
        st.markdown("### 🎯 货币卡片视图")
        
        data = self._generate_mock_data()
        filtered_data = self._apply_filters(data)
        
        # 分页设置
        items_per_page = 20
        total_items = len(filtered_data)
        total_pages = (total_items - 1) // items_per_page + 1
        
        page = st.selectbox(
            f"页面 (共 {total_pages} 页, {total_items} 个货币)",
            range(1, total_pages + 1),
            key="cards_page"
        )
        
        start_idx = (page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)
        page_data = filtered_data.iloc[start_idx:end_idx]
        
        # 渲染卡片网格
        cols = st.columns(4)
        
        for idx, (_, row) in enumerate(page_data.iterrows()):
            col_idx = idx % 4
            with cols[col_idx]:
                self._render_currency_card(row)
                
    def _render_currency_card(self, currency_data):
        """渲染单个货币卡片"""
        change_24h = currency_data['change_24h']
        change_color = "🟢" if change_24h >= 0 else "🔴"
        change_class = "positive" if change_24h >= 0 else "negative"
        
        st.markdown(f"""
        <div class="currency-card {change_class}">
            <div class="card-header">
                <h4>{currency_data['symbol']}</h4>
                <span class="rank">#{currency_data['rank']}</span>
            </div>
            <div class="card-price">
                <span class="price">${currency_data['price']:.4f}</span>
            </div>
            <div class="card-change">
                <span class="change-24h">{change_color} {change_24h:+.2f}%</span>
            </div>
            <div class="card-stats">
                <div class="stat">
                    <span class="label">成交量</span>
                    <span class="value">${currency_data['volume']:,.0f}</span>
                </div>
                <div class="stat">
                    <span class="label">市值</span>
                    <span class="value">${currency_data['market_cap']:,.0f}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    def _render_heatmap_view(self):
        """渲染热力图视图"""
        st.markdown("### 🔥 市场热力图")
        
        data = self._generate_mock_data()
        filtered_data = self._apply_filters(data)
        
        # 选择热力图指标
        heatmap_metric = st.selectbox(
            "选择热力图指标",
            options=["24h涨跌幅", "7d涨跌幅", "成交量", "市值"],
            key="heatmap_metric"
        )
        
        metric_mapping = {
            "24h涨跌幅": "change_24h",
            "7d涨跌幅": "change_7d", 
            "成交量": "volume",
            "市值": "market_cap"
        }
        
        metric_col = metric_mapping[heatmap_metric]
        
        # 创建热力图
        fig = self._create_heatmap(filtered_data, metric_col, heatmap_metric)
        st.plotly_chart(fig, use_container_width=True)
        
        # 热力图统计信息
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("最高值", f"{filtered_data[metric_col].max():.2f}")
        with col2:
            st.metric("最低值", f"{filtered_data[metric_col].min():.2f}")
        with col3:
            st.metric("平均值", f"{filtered_data[metric_col].mean():.2f}")
        with col4:
            st.metric("中位数", f"{filtered_data[metric_col].median():.2f}")
            
    def _render_charts_view(self):
        """渲染图表视图"""
        st.markdown("### 📈 多货币图表分析")
        
        data = self._generate_mock_data()
        filtered_data = self._apply_filters(data)
        
        # 图表类型选择
        chart_type = st.selectbox(
            "图表类型",
            options=["散点图分析", "相关性矩阵", "分布直方图", "时间序列对比"],
            key="chart_type"
        )
        
        if chart_type == "散点图分析":
            self._render_scatter_analysis(filtered_data)
        elif chart_type == "相关性矩阵":
            self._render_correlation_matrix(filtered_data)
        elif chart_type == "分布直方图":
            self._render_distribution_histogram(filtered_data)
        elif chart_type == "时间序列对比":
            self._render_time_series_comparison(filtered_data)
            
    def _render_scatter_analysis(self, data):
        """渲染散点图分析"""
        col1, col2 = st.columns(2)
        
        with col1:
            x_axis = st.selectbox("X轴", options=list(self.sort_options.keys()), key="scatter_x")
        with col2:
            y_axis = st.selectbox("Y轴", options=list(self.sort_options.keys()), index=1, key="scatter_y")
            
        x_col = self.sort_options[x_axis]
        y_col = self.sort_options[y_axis]
        
        fig = px.scatter(
            data, 
            x=x_col, 
            y=y_col,
            size='market_cap',
            color='change_24h',
            hover_name='symbol',
            title=f"{x_axis} vs {y_axis}",
            color_continuous_scale='RdYlGn'
        )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
    def _render_correlation_matrix(self, data):
        """渲染相关性矩阵"""
        numeric_cols = ['price', 'change_24h', 'change_7d', 'volume', 'market_cap']
        corr_matrix = data[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            title="货币指标相关性矩阵",
            color_continuous_scale='RdBu',
            aspect="auto"
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
    def _render_distribution_histogram(self, data):
        """渲染分布直方图"""
        metric = st.selectbox(
            "选择指标",
            options=list(self.sort_options.keys()),
            key="hist_metric"
        )
        
        metric_col = self.sort_options[metric]
        
        fig = px.histogram(
            data,
            x=metric_col,
            nbins=30,
            title=f"{metric} 分布直方图",
            marginal="box"
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
    def _render_time_series_comparison(self, data):
        """渲染时间序列对比"""
        # 选择要对比的货币
        selected_currencies = st.multiselect(
            "选择货币进行对比 (最多10个)",
            options=data['symbol'].tolist(),
            default=data['symbol'].head(5).tolist(),
            max_selections=10,
            key="time_series_currencies"
        )
        
        if selected_currencies:
            # 生成模拟时间序列数据
            time_series_data = self._generate_time_series_data(selected_currencies)
            
            fig = go.Figure()
            
            for currency in selected_currencies:
                fig.add_trace(go.Scatter(
                    x=time_series_data['date'],
                    y=time_series_data[currency],
                    mode='lines',
                    name=currency,
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                title="货币价格走势对比",
                xaxis_title="时间",
                yaxis_title="价格 (USD)",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
    def _generate_mock_data(self, num_currencies=100):
        """生成模拟货币数据"""
        np.random.seed(42)  # 确保数据一致性
        
        # 主流货币列表
        major_currencies = [
            "BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOGE", "DOT", "AVAX", "SHIB",
            "MATIC", "LTC", "UNI", "LINK", "ATOM", "XLM", "VET", "FIL", "TRX", "ETC"
        ]
        
        # 生成货币符号
        symbols = major_currencies + [f"TOKEN{i}" for i in range(len(major_currencies), num_currencies)]
        
        data = []
        for i, symbol in enumerate(symbols):
            # 生成随机但合理的数据
            base_price = np.random.lognormal(0, 2)  # 对数正态分布的价格
            
            currency_data = {
                'rank': i + 1,
                'symbol': symbol,
                'name': f"{symbol} Token",
                'price': base_price,
                'change_24h': np.random.normal(0, 5),  # 24小时变化
                'change_7d': np.random.normal(0, 15),   # 7天变化
                'volume': np.random.lognormal(15, 2),   # 成交量
                'market_cap': base_price * np.random.lognormal(15, 1),  # 市值
                'circulating_supply': np.random.lognormal(15, 1),
                'category': np.random.choice(['major', 'defi', 'layer1', 'layer2', 'stablecoin', 'meme'])
            }
            
            data.append(currency_data)
            
        return pd.DataFrame(data)
        
    def _apply_filters(self, data):
        """应用筛选条件"""
        filtered_data = data.copy()
        
        # 价格范围筛选
        if 'price_range' in st.session_state:
            price_min, price_max = st.session_state['price_range']
            filtered_data = filtered_data[
                (filtered_data['price'] >= price_min) & 
                (filtered_data['price'] <= price_max)
            ]
        
        # 分类筛选
        if 'filter_category' in st.session_state and st.session_state['filter_category'] != 'all':
            filtered_data = filtered_data[
                filtered_data['category'] == st.session_state['filter_category']
            ]
        
        # 排序
        if 'sort_by' in st.session_state:
            sort_col = st.session_state['sort_by']
            ascending = sort_col not in ['change_24h', 'change_7d', 'volume', 'market_cap']
            filtered_data = filtered_data.sort_values(sort_col, ascending=ascending)
            
        # 快速筛选
        if 'quick_filter' in st.session_state:
            quick_filter = st.session_state['quick_filter']
            if quick_filter == 'gainers':
                filtered_data = filtered_data.nlargest(20, 'change_24h')
            elif quick_filter == 'losers':
                filtered_data = filtered_data.nsmallest(20, 'change_24h')
            elif quick_filter == 'trending':
                filtered_data = filtered_data.nlargest(20, 'volume')
            elif quick_filter == 'high_volatility':
                filtered_data['volatility'] = abs(filtered_data['change_24h'])
                filtered_data = filtered_data.nlargest(20, 'volatility')
                
        return filtered_data.reset_index(drop=True)
        
    def _format_table_data(self, data, show_columns):
        """格式化表格数据"""
        formatted_data = pd.DataFrame()
        
        if show_columns.get("排名", False):
            formatted_data["排名"] = data['rank']
            
        if show_columns.get("货币", False):
            formatted_data["货币"] = data['symbol'] + " (" + data['name'] + ")"
            
        if show_columns.get("价格", False):
            formatted_data["价格"] = data['price']
            
        if show_columns.get("24h变化", False):
            formatted_data["24h变化"] = data['change_24h']
            
        if show_columns.get("7d变化", False):
            formatted_data["7d变化"] = data['change_7d']
            
        if show_columns.get("成交量", False):
            formatted_data["成交量"] = data['volume']
            
        if show_columns.get("市值", False):
            formatted_data["市值"] = data['market_cap']
            
        if show_columns.get("图表", False):
            # 生成迷你图表数据
            chart_data = []
            for _ in range(len(data)):
                # 生成7天的价格数据
                prices = np.random.normal(0, 0.02, 7).cumsum()
                chart_data.append(prices.tolist())
            formatted_data["图表"] = chart_data
            
        return formatted_data
        
    def _create_heatmap(self, data, metric_col, metric_name):
        """创建热力图"""
        # 计算网格大小
        n = len(data)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        
        # 创建网格坐标
        x_coords = []
        y_coords = []
        values = []
        symbols = []
        
        for i, (_, row) in enumerate(data.iterrows()):
            x = i % cols
            y = i // cols
            x_coords.append(x)
            y_coords.append(y)
            values.append(row[metric_col])
            symbols.append(row['symbol'])
            
        # 创建热力图
        fig = go.Figure(data=go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers+text',
            marker=dict(
                size=30,
                color=values,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title=metric_name)
            ),
            text=symbols,
            textposition="middle center",
            textfont=dict(size=8, color="white"),
            hovertemplate=f"<b>%{{text}}</b><br>{metric_name}: %{{marker.color}}<extra></extra>"
        ))
        
        fig.update_layout(
            title=f"货币 {metric_name} 热力图",
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            height=600,
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
        
    def _generate_time_series_data(self, currencies, days=30):
        """生成时间序列数据"""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            end=datetime.now(),
            freq='D'
        )
        
        data = {'date': dates}
        
        for currency in currencies:
            # 生成随机游走价格数据
            np.random.seed(hash(currency) % 1000)  # 基于货币名称的种子
            returns = np.random.normal(0, 0.03, len(dates))
            prices = 100 * np.exp(np.cumsum(returns))  # 从100开始的价格
            data[currency] = prices
            
        return pd.DataFrame(data)

# 样式定义
def apply_currency_comparison_styles():
    """应用货币比对界面样式"""
    st.markdown("""
    <style>
    .currency-comparison-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .currency-comparison-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .currency-comparison-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    .control-panel {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        border: 1px solid #e9ecef;
    }
    
    .currency-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #007bff;
        transition: transform 0.2s ease;
    }
    
    .currency-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    
    .currency-card.positive {
        border-left-color: #28a745;
    }
    
    .currency-card.negative {
        border-left-color: #dc3545;
    }
    
    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .card-header h4 {
        margin: 0;
        color: #333;
        font-weight: 600;
    }
    
    .rank {
        background: #6c757d;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .card-price {
        margin-bottom: 0.5rem;
    }
    
    .price {
        font-size: 1.5rem;
        font-weight: 700;
        color: #333;
    }
    
    .card-change {
        margin-bottom: 1rem;
    }
    
    .change-24h {
        font-size: 1rem;
        font-weight: 600;
    }
    
    .card-stats {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
    }
    
    .stat {
        text-align: center;
    }
    
    .stat .label {
        display: block;
        font-size: 0.8rem;
        color: #6c757d;
        margin-bottom: 0.25rem;
    }
    
    .stat .value {
        display: block;
        font-size: 0.9rem;
        font-weight: 600;
        color: #333;
    }
    
    /* 响应式设计 */
    @media (max-width: 768px) {
        .currency-comparison-header h1 {
            font-size: 2rem;
        }
        
        .control-panel {
            padding: 1rem;
        }
        
        .currency-card {
            padding: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# 创建全局实例
currency_comparison = CurrencyComparison()