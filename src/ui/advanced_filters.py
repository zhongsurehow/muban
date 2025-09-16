"""
高级筛选和性能优化模块
支持复杂的多维度筛选和大数据量的高效处理
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import asyncio
from datetime import datetime, timedelta
import time
from dataclasses import dataclass
from enum import Enum

class FilterOperator(Enum):
    """筛选操作符"""
    EQUALS = "等于"
    NOT_EQUALS = "不等于"
    GREATER_THAN = "大于"
    LESS_THAN = "小于"
    GREATER_EQUAL = "大于等于"
    LESS_EQUAL = "小于等于"
    BETWEEN = "介于"
    IN = "包含"
    NOT_IN = "不包含"
    CONTAINS = "文本包含"
    STARTS_WITH = "开头是"
    ENDS_WITH = "结尾是"

@dataclass
class FilterCondition:
    """筛选条件"""
    field: str
    operator: FilterOperator
    value: Any
    value2: Optional[Any] = None  # 用于BETWEEN操作

class AdvancedFilters:
    """高级筛选系统"""
    
    def __init__(self):
        self.numeric_fields = {
            "价格": "price",
            "24h涨跌幅": "change_24h",
            "7d涨跌幅": "change_7d",
            "成交量": "volume",
            "市值": "market_cap",
            "流通量": "circulating_supply",
            "排名": "rank"
        }
        
        self.text_fields = {
            "货币符号": "symbol",
            "货币名称": "name",
            "分类": "category"
        }
        
        self.all_fields = {**self.numeric_fields, **self.text_fields}
        
        # 预设筛选器
        self.preset_filters = {
            "🔥 热门币种": [
                FilterCondition("rank", FilterOperator.LESS_EQUAL, 50),
                FilterCondition("volume", FilterOperator.GREATER_THAN, 1000000)
            ],
            "💎 潜力币种": [
                FilterCondition("market_cap", FilterOperator.BETWEEN, [1000000, 100000000]),
                FilterCondition("change_24h", FilterOperator.GREATER_THAN, 5)
            ],
            "🚀 强势上涨": [
                FilterCondition("change_24h", FilterOperator.GREATER_THAN, 10),
                FilterCondition("change_7d", FilterOperator.GREATER_THAN, 20)
            ],
            "📉 超跌反弹": [
                FilterCondition("change_24h", FilterOperator.LESS_THAN, -10),
                FilterCondition("change_7d", FilterOperator.LESS_THAN, -20)
            ],
            "💰 大市值币种": [
                FilterCondition("market_cap", FilterOperator.GREATER_THAN, 1000000000),
                FilterCondition("rank", FilterOperator.LESS_EQUAL, 20)
            ],
            "⚡ 高波动币种": [
                FilterCondition("change_24h", FilterOperator.GREATER_THAN, 15),
            ]
        }
        
    def render_advanced_filter_panel(self):
        """渲染高级筛选面板"""
        st.markdown("### 🔍 高级筛选器")
        
        # 预设筛选器
        self._render_preset_filters()
        
        # 自定义筛选器
        self._render_custom_filters()
        
        # 筛选器管理
        self._render_filter_management()
        
    def _render_preset_filters(self):
        """渲染预设筛选器"""
        st.markdown("#### 📋 预设筛选器")
        
        preset_cols = st.columns(3)
        
        for i, (name, conditions) in enumerate(self.preset_filters.items()):
            col_idx = i % 3
            with preset_cols[col_idx]:
                if st.button(name, key=f"preset_{i}", use_container_width=True):
                    st.session_state['active_filters'] = conditions
                    st.session_state['filter_applied'] = True
                    st.rerun()
                    
    def _render_custom_filters(self):
        """渲染自定义筛选器"""
        st.markdown("#### ⚙️ 自定义筛选器")
        
        # 初始化筛选条件
        if 'custom_filters' not in st.session_state:
            st.session_state['custom_filters'] = []
            
        # 添加新筛选条件
        with st.expander("➕ 添加筛选条件", expanded=False):
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
            
            with col1:
                field = st.selectbox(
                    "字段",
                    options=list(self.all_fields.keys()),
                    key="new_filter_field"
                )
                
            with col2:
                field_type = "numeric" if field in self.numeric_fields else "text"
                operators = self._get_operators_for_field_type(field_type)
                operator = st.selectbox(
                    "操作符",
                    options=[op.value for op in operators],
                    key="new_filter_operator"
                )
                
            with col3:
                operator_enum = FilterOperator(operator)
                if operator_enum == FilterOperator.BETWEEN:
                    value1 = st.number_input("最小值", key="new_filter_value1")
                    value2 = st.number_input("最大值", key="new_filter_value2")
                    value = [value1, value2]
                elif operator_enum in [FilterOperator.IN, FilterOperator.NOT_IN]:
                    value = st.text_input("值 (逗号分隔)", key="new_filter_value").split(',')
                elif field_type == "numeric":
                    value = st.number_input("值", key="new_filter_value")
                else:
                    value = st.text_input("值", key="new_filter_value")
                    
            with col4:
                if st.button("添加", key="add_filter"):
                    condition = FilterCondition(
                        field=self.all_fields[field],
                        operator=FilterOperator(operator),
                        value=value
                    )
                    st.session_state['custom_filters'].append(condition)
                    st.rerun()
                    
        # 显示当前筛选条件
        if st.session_state['custom_filters']:
            st.markdown("#### 📝 当前筛选条件")
            
            for i, condition in enumerate(st.session_state['custom_filters']):
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    condition_text = self._format_condition_text(condition)
                    st.write(f"{i+1}. {condition_text}")
                    
                with col2:
                    if st.button("删除", key=f"delete_filter_{i}"):
                        st.session_state['custom_filters'].pop(i)
                        st.rerun()
                        
            # 应用筛选器按钮
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                if st.button("🔍 应用筛选", key="apply_custom_filters"):
                    st.session_state['active_filters'] = st.session_state['custom_filters']
                    st.session_state['filter_applied'] = True
                    st.rerun()
                    
            with col2:
                if st.button("🗑️ 清空筛选", key="clear_filters"):
                    st.session_state['custom_filters'] = []
                    st.session_state['active_filters'] = []
                    st.session_state['filter_applied'] = False
                    st.rerun()
                    
            with col3:
                if st.button("💾 保存筛选", key="save_filters"):
                    self._save_custom_filter()
                    
    def _render_filter_management(self):
        """渲染筛选器管理"""
        if 'saved_filters' in st.session_state and st.session_state['saved_filters']:
            st.markdown("#### 💾 已保存的筛选器")
            
            for name, conditions in st.session_state['saved_filters'].items():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"📁 {name}")
                    
                with col2:
                    if st.button("加载", key=f"load_{name}"):
                        st.session_state['custom_filters'] = conditions
                        st.rerun()
                        
                with col3:
                    if st.button("删除", key=f"delete_saved_{name}"):
                        del st.session_state['saved_filters'][name]
                        st.rerun()
                        
    def _get_operators_for_field_type(self, field_type: str) -> List[FilterOperator]:
        """根据字段类型获取可用操作符"""
        if field_type == "numeric":
            return [
                FilterOperator.EQUALS,
                FilterOperator.NOT_EQUALS,
                FilterOperator.GREATER_THAN,
                FilterOperator.LESS_THAN,
                FilterOperator.GREATER_EQUAL,
                FilterOperator.LESS_EQUAL,
                FilterOperator.BETWEEN
            ]
        else:
            return [
                FilterOperator.EQUALS,
                FilterOperator.NOT_EQUALS,
                FilterOperator.CONTAINS,
                FilterOperator.STARTS_WITH,
                FilterOperator.ENDS_WITH,
                FilterOperator.IN,
                FilterOperator.NOT_IN
            ]
            
    def _format_condition_text(self, condition: FilterCondition) -> str:
        """格式化筛选条件文本"""
        field_name = next(k for k, v in self.all_fields.items() if v == condition.field)
        
        if condition.operator == FilterOperator.BETWEEN:
            return f"{field_name} {condition.operator.value} {condition.value[0]} 和 {condition.value[1]}"
        elif condition.operator in [FilterOperator.IN, FilterOperator.NOT_IN]:
            return f"{field_name} {condition.operator.value} [{', '.join(map(str, condition.value))}]"
        else:
            return f"{field_name} {condition.operator.value} {condition.value}"
            
    def _save_custom_filter(self):
        """保存自定义筛选器"""
        if not st.session_state['custom_filters']:
            st.warning("没有筛选条件可保存")
            return
            
        filter_name = st.text_input("筛选器名称", key="save_filter_name")
        if filter_name:
            if 'saved_filters' not in st.session_state:
                st.session_state['saved_filters'] = {}
            st.session_state['saved_filters'][filter_name] = st.session_state['custom_filters'].copy()
            st.success(f"筛选器 '{filter_name}' 已保存")
            
    def apply_filters(self, data: pd.DataFrame, conditions: List[FilterCondition]) -> pd.DataFrame:
        """应用筛选条件"""
        if not conditions:
            return data
            
        filtered_data = data.copy()
        
        for condition in conditions:
            filtered_data = self._apply_single_condition(filtered_data, condition)
            
        return filtered_data
        
    def _apply_single_condition(self, data: pd.DataFrame, condition: FilterCondition) -> pd.DataFrame:
        """应用单个筛选条件"""
        field = condition.field
        operator = condition.operator
        value = condition.value
        
        if field not in data.columns:
            return data
            
        if operator == FilterOperator.EQUALS:
            return data[data[field] == value]
        elif operator == FilterOperator.NOT_EQUALS:
            return data[data[field] != value]
        elif operator == FilterOperator.GREATER_THAN:
            return data[data[field] > value]
        elif operator == FilterOperator.LESS_THAN:
            return data[data[field] < value]
        elif operator == FilterOperator.GREATER_EQUAL:
            return data[data[field] >= value]
        elif operator == FilterOperator.LESS_EQUAL:
            return data[data[field] <= value]
        elif operator == FilterOperator.BETWEEN:
            return data[(data[field] >= value[0]) & (data[field] <= value[1])]
        elif operator == FilterOperator.IN:
            return data[data[field].isin(value)]
        elif operator == FilterOperator.NOT_IN:
            return data[~data[field].isin(value)]
        elif operator == FilterOperator.CONTAINS:
            return data[data[field].str.contains(str(value), case=False, na=False)]
        elif operator == FilterOperator.STARTS_WITH:
            return data[data[field].str.startswith(str(value), na=False)]
        elif operator == FilterOperator.ENDS_WITH:
            return data[data[field].str.endswith(str(value), na=False)]
        else:
            return data

class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self):
        self.cache_duration = 300  # 5分钟缓存
        self.page_size = 50
        self.virtual_scroll_threshold = 1000
        
    def optimize_data_loading(self, data: pd.DataFrame, page: int = 1, 
                            page_size: Optional[int] = None) -> Tuple[pd.DataFrame, Dict]:
        """优化数据加载"""
        if page_size is None:
            page_size = self.page_size
            
        total_rows = len(data)
        total_pages = (total_rows - 1) // page_size + 1
        
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_rows)
        
        page_data = data.iloc[start_idx:end_idx]
        
        metadata = {
            'total_rows': total_rows,
            'total_pages': total_pages,
            'current_page': page,
            'page_size': page_size,
            'start_idx': start_idx,
            'end_idx': end_idx
        }
        
        return page_data, metadata
        
    def render_pagination_controls(self, metadata: Dict):
        """渲染分页控件"""
        col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
        
        current_page = metadata['current_page']
        total_pages = metadata['total_pages']
        
        with col1:
            if st.button("⏮️ 首页", disabled=current_page == 1, key="first_page"):
                st.session_state['current_page'] = 1
                st.rerun()
                
        with col2:
            if st.button("⏪ 上页", disabled=current_page == 1, key="prev_page"):
                st.session_state['current_page'] = current_page - 1
                st.rerun()
                
        with col3:
            page_input = st.number_input(
                f"页码 (共 {total_pages} 页)",
                min_value=1,
                max_value=total_pages,
                value=current_page,
                key="page_input"
            )
            if page_input != current_page:
                st.session_state['current_page'] = page_input
                st.rerun()
                
        with col4:
            if st.button("⏩ 下页", disabled=current_page == total_pages, key="next_page"):
                st.session_state['current_page'] = current_page + 1
                st.rerun()
                
        with col5:
            if st.button("⏭️ 末页", disabled=current_page == total_pages, key="last_page"):
                st.session_state['current_page'] = total_pages
                st.rerun()
                
        # 显示统计信息
        st.caption(f"显示第 {metadata['start_idx'] + 1}-{metadata['end_idx']} 条，共 {metadata['total_rows']} 条记录")
        
    def enable_virtual_scrolling(self, data: pd.DataFrame) -> bool:
        """判断是否启用虚拟滚动"""
        return len(data) > self.virtual_scroll_threshold
        
    @st.cache_data(ttl=300)  # 5分钟缓存
    def cache_expensive_computation(self, data_hash: str, computation_func, *args, **kwargs):
        """缓存昂贵的计算操作"""
        return computation_func(*args, **kwargs)
        
    def render_performance_settings(self):
        """渲染性能设置"""
        with st.expander("⚙️ 性能设置", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                page_size = st.selectbox(
                    "每页显示数量",
                    options=[25, 50, 100, 200],
                    index=1,
                    key="performance_page_size"
                )
                st.session_state['page_size'] = page_size
                
            with col2:
                auto_refresh = st.checkbox(
                    "自动刷新数据",
                    value=True,
                    key="performance_auto_refresh"
                )
                
                if auto_refresh:
                    refresh_interval = st.selectbox(
                        "刷新间隔",
                        options=[5, 10, 30, 60],
                        index=1,
                        key="performance_refresh_interval"
                    )
                    st.session_state['refresh_interval'] = refresh_interval
                    
            # 缓存控制
            col3, col4 = st.columns(2)
            
            with col3:
                if st.button("🗑️ 清除缓存", key="clear_cache"):
                    st.cache_data.clear()
                    st.success("缓存已清除")
                    
            with col4:
                cache_info = st.empty()
                # 这里可以显示缓存统计信息
                cache_info.info("缓存状态: 正常")

# 创建全局实例
advanced_filters = AdvancedFilters()
performance_optimizer = PerformanceOptimizer()