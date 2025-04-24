import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import base64
from io import BytesIO

# 设置matplotlib使用支持英文字符的字体
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# 全局设置 - 修改为云兼容方案
DATA_FILE = "data.xlsx"  # 使用相对路径
REQUIRED_COLS = {
    'p-aminophenol(g)': 'p-aminophenol (g)',
    'Acetic Anhydride(ml)': 'Acetic Anhydride (ml)',
    'PA:AA': 'PA:AA ratio',
    'Reaction time(min)': 'Reaction time (min)',
    'T(°C)': 'Temperature (°C)',
    'Crude weight(g)': 'Crude product weight (g)',
    'Purify weight(g)': 'Purified product weight (g)',
    'Yield(%)': 'Yield (%)'
}
st.set_page_config(page_title="Acetaminophen Synthesis Analysis System", layout="wide")

# ================ 辅助函数 ================
@st.cache_data(ttl=3600)  # 缓存1小时
def load_data():
    """加载数据，兼容云环境"""
    try:
        # 尝试从本地或云环境加载数据
        if os.path.exists(DATA_FILE):
            df = pd.read_excel(DATA_FILE)
        else:
            # 如果文件不存在，创建示例数据
            sample_data = {
                'p-aminophenol(g)': [5.0, 10.0],
                'Acetic Anhydride(ml)': [10.0, 20.0],
                'Reaction time(min)': [30, 45],
                'T(°C)': [80.0, 90.0],
                'Crude weight(g)': [6.2, 12.5],
                'Purify weight(g)': [5.8, 11.7]
            }
            df = pd.DataFrame(sample_data)
            # 计算衍生列
            df['PA:AA'] = df['p-aminophenol(g)'] / df['Acetic Anhydride(ml)']
            molar_ratio = 151.16 / 109.13
            df['Yield(%)'] = (df['Purify weight(g)'] / (df['p-aminophenol(g)'] * molar_ratio)) * 100
            df.to_excel(DATA_FILE, index=False)
        
        # 检查必要列
        missing_cols = [col for col in REQUIRED_COLS.keys() if col not in df.columns]
        if missing_cols:
            for col in missing_cols:
                df[col] = np.nan
        
        return df
    
    except Exception as e:
        st.error(f"加载数据失败: {str(e)}")
        return pd.DataFrame(columns=REQUIRED_COLS.keys())

def save_data(df):
    """保存数据到临时文件"""
    try:
        df.to_excel(DATA_FILE, index=False)
        return True
    except Exception as e:
        st.error(f"保存数据失败: {str(e)}")
        return False

def create_download_link(df):
    """生成数据下载链接"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data).decode()
    return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="acetaminophen_data.xlsx">下载Excel文件</a>'

# ================ 数据管理模块 ================
def data_management():
    st.title("对乙酰氨基酚合成数据管理")
    df = load_data()
    
    st.warning("""
    注意: 在Streamlit Cloud中，数据修改是临时的。
    请下载修改后的数据并手动上传到GitHub以持久化保存。
    """)
    
    # 数据编辑器
    st.subheader("实验数据表")
    edited_df = st.data_editor(
        df,
        column_config={
            "_selected_row": st.column_config.CheckboxColumn("选择")
        },
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        disabled=["PA:AA", "Yield(%)"]
    )
    
    # 操作按钮
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💾 保存修改", type="primary"):
            if save_data(edited_df):
                st.success("数据已保存(临时)!")
                st.rerun()
    with col2:
        if st.button("🔄 重置数据"):
            if os.path.exists(DATA_FILE):
                os.remove(DATA_FILE)
            st.rerun()
    with col3:
        st.markdown(create_download_link(edited_df), unsafe_allow_html=True)
    
    # 添加新数据表单 (保持原有逻辑)
    # ... [其余data_management()函数内容保持不变] ...

# ================ 其余模块保持不变 ================
# ... [show_analysis(), weight_prediction(), main()等函数保持不变] ...

if __name__ == "__main__":
    main()
