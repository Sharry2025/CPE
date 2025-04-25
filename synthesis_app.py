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

# 全局设置
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
@st.cache_data(ttl=3600)
def load_data():
    """加载或初始化数据文件"""
    try:
        if not os.path.exists(DATA_FILE):
            # 创建示例数据
            sample_data = {
                'p-aminophenol(g)': [5.0, 10.0],
                'Acetic Anhydride(ml)': [10.0, 20.0],
                'Reaction time(min)': [30, 45],
                'T(°C)': [80.0, 90.0],
                'Crude weight(g)': [6.2, 12.5],
                'Purify weight(g)': [5.8, 11.7]
            }
            df = pd.DataFrame(sample_data)
            df['PA:AA'] = df['p-aminophenol(g)'] / df['Acetic Anhydride(ml)']
            molar_ratio = 151.16 / 109.13
            df['Yield(%)'] = (df['Purify weight(g)'] / (df['p-aminophenol(g)'] * molar_ratio)) * 100
            df.to_excel(DATA_FILE, index=False)
            return df
        
        df = pd.read_excel(DATA_FILE)
        
        # 检查必要列
        missing_cols = [col for col in REQUIRED_COLS.keys() if col not in df.columns]
        if missing_cols:
            for col in missing_cols:
                df[col] = np.nan
        
        # 计算衍生列
        if 'PA:AA' not in df.columns and all(col in df.columns for col in ['p-aminophenol(g)', 'Acetic Anhydride(ml)']):
            df['PA:AA'] = df['p-aminophenol(g)'] / df['Acetic Anhydride(ml)']
        
        if 'Yield(%)' not in df.columns and all(col in df.columns for col in ['Purify weight(g)', 'p-aminophenol(g)']):
            molar_ratio = 151.16 / 109.13
            df['Yield(%)'] = (df['Purify weight(g)'] / (df['p-aminophenol(g)'] * molar_ratio)) * 100
        
        return df
    
    except Exception as e:
        st.error(f"加载数据失败: {str(e)}")
        return pd.DataFrame(columns=REQUIRED_COLS.keys())

def save_data(df):
    """保存数据到Excel"""
    try:
        df.to_excel(DATA_FILE, index=False)
        return True
    except Exception as e:
        st.error(f"保存失败: {str(e)}")
        return False

def create_download_link(df):
    """生成Excel下载链接"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data).decode()
    return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="acetaminophen_data.xlsx">下载Excel文件</a>'

# ================ 数据分析模块 ================
def show_analysis():
    # ... [保持原有show_analysis()函数内容不变] ...

# ================ 数据管理模块 ================
def data_management():
    """数据管理功能"""
    st.header("📊 数据管理")
    df = load_data()
    
    # 显示当前数据
    st.subheader("当前数据")
    st.dataframe(df)
    
    # 数据编辑功能
    st.subheader("编辑数据")
    edited_df = st.data_editor(df, num_rows="dynamic")
    if st.button("保存更改"):
        if save_data(edited_df):
            st.success("数据保存成功！")
        else:
            st.error("保存失败")
    
    # 数据下载
    st.subheader("数据导出")
    st.markdown(create_download_link(df), unsafe_allow_html=True)

# ================ 产量预测模块 ================
def weight_prediction():
    # ... [保持原有weight_prediction()函数内容不变] ...

# ================ 主应用 ================
def main():
    st.sidebar.title("导航")
    app_mode = st.sidebar.radio(
        "选择功能模块",
        ["数据分析", "数据管理", "产量预测"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **系统说明**:
    1. 在"数据管理"中录入实验数据
    2. 在"数据分析"中查看参数相关性
    3. 使用"产量预测"进行实验设计
    """)
    
    try:
        df = load_data()
        st.sidebar.markdown(f"当前数据量: {len(df)} 组实验")
    except:
        st.sidebar.warning("无法加载数据文件")
    
    if app_mode == "数据分析":
        show_analysis()
    elif app_mode == "数据管理":
        data_management()
    elif app_mode == "产量预测":
        weight_prediction()

# 确保main()函数被正确定义后调用
if __name__ == "__main__":
    main()
