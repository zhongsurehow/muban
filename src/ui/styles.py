"""
UI样式和主题配置
提供专业交易界面的样式定制
"""

import streamlit as st

def apply_trading_theme():
    """应用专业交易主题样式"""
    
    trading_css = """
    <style>
    /* 全局样式 */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    
    /* 专业交易界面样式 */
    .trading-widget {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .trading-widget h3 {
        color: white;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    /* 价格显示样式 */
    .price-positive {
        color: #00ff88 !important;
        font-weight: bold;
    }
    
    .price-negative {
        color: #ff4757 !important;
        font-weight: bold;
    }
    
    .price-neutral {
        color: #ffa502 !important;
        font-weight: bold;
    }
    
    /* 快捷操作按钮样式 */
    .quick-action-btn {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin: 0.2rem;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    
    .quick-action-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* 订单簿样式 */
    .orderbook-buy {
        background: rgba(0, 255, 136, 0.1);
        border-left: 3px solid #00ff88;
    }
    
    .orderbook-sell {
        background: rgba(255, 71, 87, 0.1);
        border-left: 3px solid #ff4757;
    }
    
    /* 市场概览卡片 */
    .market-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease;
    }
    
    .market-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    /* 预警样式 */
    .alert-high {
        background: linear-gradient(135deg, #ff4757, #ff3742);
        color: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        animation: pulse 2s infinite;
    }
    
    .alert-medium {
        background: linear-gradient(135deg, #ffa502, #ff6348);
        color: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .alert-low {
        background: linear-gradient(135deg, #7bed9f, #70a1ff);
        color: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    /* 布局选择器样式 */
    .layout-selector {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* 组件拖拽区域 */
    .widget-drag-area {
        border: 2px dashed rgba(255, 255, 255, 0.3);
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .widget-drag-area:hover {
        border-color: #667eea;
        background: rgba(102, 126, 234, 0.1);
    }
    
    /* 技术分析图表样式 */
    .chart-container {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* 投资组合样式 */
    .portfolio-item {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .portfolio-item:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
    }
    
    /* 响应式设计 */
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        
        .trading-widget {
            margin: 0.25rem 0;
            padding: 0.75rem;
        }
        
        .market-card {
            margin: 0.25rem;
            padding: 1rem;
        }
    }
    
    /* 滚动条样式 */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2, #667eea);
    }
    
    /* 数据表格样式 */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* 指标卡片样式 */
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
    }
    
    /* 状态指示器 */
    .status-online {
        color: #00ff88;
        font-weight: bold;
    }
    
    .status-offline {
        color: #ff4757;
        font-weight: bold;
    }
    
    .status-warning {
        color: #ffa502;
        font-weight: bold;
    }
    
    /* 动画效果 */
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .slide-in {
        animation: slideIn 0.3s ease-out;
    }
    
    @keyframes slideIn {
        from { transform: translateX(-100%); }
        to { transform: translateX(0); }
    }
    
    /* 加载动画 */
    .loading-spinner {
        border: 3px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        border-top: 3px solid #667eea;
        width: 30px;
        height: 30px;
        animation: spin 1s linear infinite;
        margin: 0 auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
    """
    
    st.markdown(trading_css, unsafe_allow_html=True)

def get_price_color(change):
    """根据价格变化获取颜色类名"""
    if change > 0:
        return "price-positive"
    elif change < 0:
        return "price-negative"
    else:
        return "price-neutral"

def get_alert_class(severity):
    """根据预警严重程度获取CSS类名"""
    severity_map = {
        'high': 'alert-high',
        'medium': 'alert-medium',
        'low': 'alert-low'
    }
    return severity_map.get(severity, 'alert-low')

def render_metric_card(title, value, delta=None, help_text=None):
    """渲染指标卡片"""
    delta_html = ""
    if delta is not None:
        delta_color = get_price_color(delta)
        delta_html = f'<div class="{delta_color}">{"+" if delta > 0 else ""}{delta}%</div>'
    
    help_html = ""
    if help_text:
        help_html = f'<small style="color: rgba(255,255,255,0.7);">{help_text}</small>'
    
    card_html = f"""
    <div class="metric-card fade-in">
        <h4 style="margin: 0; color: white;">{title}</h4>
        <h2 style="margin: 0.5rem 0; color: white;">{value}</h2>
        {delta_html}
        {help_html}
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)

def render_status_indicator(status, text):
    """渲染状态指示器"""
    status_class = f"status-{status}"
    status_html = f'<span class="{status_class}">● {text}</span>'
    st.markdown(status_html, unsafe_allow_html=True)

def render_loading_spinner():
    """渲染加载动画"""
    spinner_html = '<div class="loading-spinner"></div>'
    st.markdown(spinner_html, unsafe_allow_html=True)