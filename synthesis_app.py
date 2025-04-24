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

# Set matplotlib to use a font that supports English characters
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# Global settings - 修改为从GitHub读取数据
DATA_URL = "https://raw.githubusercontent.com/yourusername/CPE/main/data.xlsx"  # 替换为实际URL
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

# ================ Helper Functions ================
@st.cache_data(ttl=3600)  # 缓存1小时
def load_data():
    """Load data from GitHub with error handling"""
    try:
        # 尝试从GitHub加载数据
        df = pd.read_excel(DATA_URL)
        
        # 检查必要列
        missing_cols = [col for col in REQUIRED_COLS.keys() if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
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
        st.error(f"Failed to load data: {str(e)}")
        # 返回示例数据
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
        return df

def create_download_link(df):
    """Generate a download link for the DataFrame"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data).decode()
    return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="acetaminophen_data.xlsx">Download Excel File</a>'

# ================ Data Management Module ================
def data_management():
    st.title("Acetaminophen Synthesis Data Management")
    df = load_data()
    
    st.warning("""
    Note: In Streamlit Cloud, data editing is temporary. 
    Download the modified data and upload it to GitHub to persist changes.
    """)
    
    # Data editor
    st.subheader("Experimental Data Table")
    edited_df = st.data_editor(
        df,
        column_config={
            "_selected_row": st.column_config.CheckboxColumn("Select")
        },
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        disabled=["PA:AA", "Yield(%)"]
    )
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reset Data"):
            st.cache_data.clear()
            st.rerun()
    with col2:
        st.markdown(create_download_link(edited_df), unsafe_allow_html=True)
    
    # Add new data form (保持原有表单逻辑)
    # ... [其余 data_management() 函数内容保持不变] ...

# ================ 其余模块保持不变 ================
# ... [show_analysis(), weight_prediction(), main() 等函数保持不变] ...

if __name__ == "__main__":
    main()
