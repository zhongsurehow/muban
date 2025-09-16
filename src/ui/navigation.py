"""
统一导航组件
提供所有页面间的导航功能
"""

import streamlit as st

def render_navigation():
    """渲染统一的导航栏"""
    st.markdown("""
    <style>
    .nav-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .nav-title {
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-align: center;
    }
    .nav-buttons {
        display: flex;
        justify-content: center;
        gap: 1rem;
        flex-wrap: wrap;
    }
    .nav-button {
        background: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.3);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        text-decoration: none;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    .nav-button:hover {
        background: rgba(255, 255, 255, 0.3);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .nav-button.active {
        background: rgba(255, 255, 255, 0.4);
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 获取当前页面
    current_page = st.session_state.get('current_page', 'main')
    
    st.markdown("""
    <div class="nav-container">
        <div class="nav-title">🌟 专业级货币分析平台</div>
        <div class="nav-buttons">
    """, unsafe_allow_html=True)
    
    # 导航按钮
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        if st.button("🏠 主页", key="nav_main", help="返回主页"):
            st.switch_page("src/app.py")
    
    with col2:
        if st.button("🌍 货币概览", key="nav_overview", help="查看货币市场概览"):
            st.switch_page("pages/1_🌍_货币概览.py")
    
    with col3:
        if st.button("📈 详细分析", key="nav_analysis", help="深入分析货币走势"):
            st.switch_page("pages/2_📈_详细分析.py")
    
    with col4:
        if st.button("⚖️ 货币比较", key="nav_compare", help="比较不同货币"):
            st.switch_page("pages/3_⚖️_货币比较.py")
    
    with col5:
        if st.button("🔍 高级筛选", key="nav_filter", help="使用高级筛选功能"):
            st.switch_page("pages/4_🔍_高级筛选.py")
    
    with col6:
        if st.button("📊 实时仪表盘", key="nav_dashboard", help="查看实时数据仪表盘"):
            st.switch_page("pages/5_📊_实时仪表盘.py")
    
    st.markdown("</div></div>", unsafe_allow_html=True)

def render_page_header(title: str, description: str = "", icon: str = ""):
    """渲染页面标题和描述"""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    ">
        <h1 style="
            color: #2c3e50;
            margin-bottom: 0.5rem;
            font-size: 2.5rem;
        ">{icon} {title}</h1>
        {f'<p style="color: #7f8c8d; font-size: 1.2rem; margin: 0;">{description}</p>' if description else ''}
    </div>
    """, unsafe_allow_html=True)

def render_quick_stats():
    """渲染快速统计信息"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📈 活跃货币对",
            value="156",
            delta="12"
        )
    
    with col2:
        st.metric(
            label="💰 总市值",
            value="$2.1T",
            delta="5.2%"
        )
    
    with col3:
        st.metric(
            label="🔄 24h交易量",
            value="$89.5B",
            delta="-2.1%"
        )
    
    with col4:
        st.metric(
            label="⚡ 套利机会",
            value="23",
            delta="7"
        )

def render_footer():
    """渲染页面底部信息"""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; padding: 1rem;">
        <p>🌟 专业级货币分析平台 | 实时数据 | 智能分析 | 风险管控</p>
        <p style="font-size: 0.9rem;">最后更新: {}</p>
    </div>
    """.format(st.session_state.get('last_update', '刚刚')), unsafe_allow_html=True)