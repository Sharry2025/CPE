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

# Global settings
DATA_PATH = "data.xlsx"  # 改为相对路径
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
@st.cache_data(ttl=5)  # 修改1：缩短缓存时间为5秒
def load_data():
    """严格从 data.xlsx 加载数据，禁止自动生成示例数据"""
    try:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"数据文件 {DATA_PATH} 不存在！请上传有效的 data.xlsx")
        
        df = pd.read_excel(DATA_PATH)
        
        # 检查必要列是否完整
        missing_cols = [col for col in REQUIRED_COLS.keys() if col not in df.columns]
        if missing_cols:
            raise ValueError(f"数据文件缺少必要列: {', '.join(missing_cols)}")
        
        # 检查数据是否为空
        if df.empty:
            raise ValueError("数据文件为空！请提供有效数据")
            
        # 计算衍生列（如果原始数据完整）
        if 'PA:AA' not in df.columns and all(col in df.columns for col in ['p-aminophenol(g)', 'Acetic Anhydride(ml)']):
            df['PA:AA'] = df['p-aminophenol(g)'] / df['Acetic Anhydride(ml)']
        
        if 'Yield(%)' not in df.columns and all(col in df.columns for col in ['Purify weight(g)', 'p-aminophenol(g)']):
            molar_ratio = 151.16 / 109.13
            df['Yield(%)'] = (df['Purify weight(g)'] / (df['p-aminophenol(g)'] * molar_ratio)) * 100
        
        return df
    
    except Exception as e:
        st.error(f"数据加载失败: {str(e)}")
        st.stop()  # 终止应用

def save_data(df):
    """安全保存数据，禁止创建新文件"""
    try:
        if not os.path.exists(DATA_PATH):
            raise PermissionError("禁止自动创建文件，请确保 data.xlsx 已存在")
        df.to_excel(DATA_PATH, index=False)
        st.cache_data.clear()  # 修改2：保存后清除所有缓存
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

# ================ Data Analysis Module ================
def show_analysis():
    st.header("Experimental Data Analysis")
    df = load_data()  # 已强制检查数据有效性
    
    with st.expander("📊 Data Overview", expanded=True):
        st.dataframe(df, use_container_width=True)
        st.write(f"Total experiments: {len(df)}")
    
    st.subheader("Parameter Correlation Analysis")
    if len(df) < 2:
        st.warning("At least 2 data points required for correlation analysis")
        return
    
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            st.warning("Not enough numeric columns for correlation analysis")
            return
        
        corr_matrix = numeric_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                   center=0, linewidths=0.5, annot_kws={"size": 12})
        plt.title("Parameter Correlation Heatmap (Pearson)", pad=20)
        st.pyplot(plt)
        
        with st.expander("📌 Correlation Coefficient Guide"):
            st.markdown("""
            - **+1.0**: Perfect positive correlation
            - **+0.8 to +1.0**: Strong positive correlation
            - **+0.5 to +0.8**: Moderate positive correlation
            - **-0.5 to +0.5**: Weak or no correlation
            - **-0.8 to -0.5**: Moderate negative correlation
            - **-1.0 to -0.8**: Strong negative correlation
            - **-1.0**: Perfect negative correlation
            """)
    except Exception as e:
        st.error(f"Correlation analysis error: {str(e)}")

# ================ Data Management Module ================
def data_management():
    st.title("Acetaminophen Synthesis Data Management")
    
    # 强制重新加载数据，绕过缓存
    @st.cache_data(ttl=5, show_spinner=False)
    def load_data_no_cache():
        return pd.read_excel(DATA_PATH)
    
    try:
        df = load_data_no_cache()
    except Exception as e:
        st.error(f"加载数据失败: {str(e)}")
        st.stop()
    
    # 使用session_state保存删除状态
    if 'deleted_rows' not in st.session_state:
        st.session_state.deleted_rows = []
    
    with st.expander("ℹ️ Instructions", expanded=True):
        st.markdown("""
        ### User Guide:
        1. **Edit data**: Modify cells directly in the table
        2. **Add data**: Fill the form below and click "Add New Data"
        3. **Delete data**: Check rows and click "Delete Selected Rows"
        """)
    
    # Data editor - 添加唯一标识符列
    display_df = df.copy()
    display_df['_unique_id'] = range(len(display_df))
    
    edited_df = st.data_editor(
        display_df,
        column_config={
            "_selected_row": st.column_config.CheckboxColumn("Select"),
            "_unique_id": None  # 隐藏唯一ID列
        },
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        disabled=["PA:AA", "Yield(%)", "_unique_id"],
        key="data_editor"
    )
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💾 Save Changes", type="primary", key="save_changes"):
            # 移除临时列后再保存
            save_df = edited_df.drop(columns=['_selected_row', '_unique_id'], errors='ignore')
            if save_data(save_df):
                st.session_state.deleted_rows = []
                st.success("数据保存成功！")
                st.rerun()
    
    with col2:
        if st.button("🗑️ Delete Selected", key="delete_selected"):
            if "_selected_row" in edited_df.columns:
                # 获取要删除行的唯一ID
                deleted_ids = edited_df[edited_df["_selected_row"]]['_unique_id'].tolist()
                st.session_state.deleted_rows.extend(deleted_ids)
                
                # 立即从显示中移除
                remaining_df = edited_df[~edited_df["_unique_id"].isin(st.session_state.deleted_rows)]
                st.success(f"已标记删除 {len(deleted_ids)} 条记录，请点击保存确认")
                st.rerun()
    
    with col3:
        if st.button("🔄 Refresh Data", key="refresh_data"):
            st.cache_data.clear()
            if 'deleted_rows' in st.session_state:
                st.session_state.deleted_rows = []
            st.rerun()
    

    
    # Add new data form - 修改为三列布局
    st.subheader("Add New Experiment")
    with st.form("add_data_form"):
        cols = st.columns(3)  # 三列布局
        
        new_data = {}
        
        # 第一列: Reactant Parameters
        with cols[0]:
            st.markdown("**Reactant Parameters**")
            new_data['p-aminophenol(g)'] = st.number_input(
                REQUIRED_COLS['p-aminophenol(g)'], 
                min_value=0.0, step=0.01, format="%.2f",
                key="new_p_amino"
            )
            new_data['Acetic Anhydride(ml)'] = st.number_input(
                REQUIRED_COLS['Acetic Anhydride(ml)'], 
                min_value=0.0, step=0.01, format="%.2f",
                key="new_acetic"
            )
            # PA:AA 自动计算但允许手动修改
            pa_aa = (new_data['p-aminophenol(g)'] / new_data['Acetic Anhydride(ml)']) if new_data['Acetic Anhydride(ml)'] != 0 else 0.0
            new_data['PA:AA'] = st.number_input(
                REQUIRED_COLS['PA:AA'],
                min_value=0.0, max_value=2.0, value=float(pa_aa),
                step=0.01, format="%.4f",
                key="new_pa_aa"
            )
        
        # 第二列: Reaction Conditions
        with cols[1]:
            st.markdown("**Reaction Conditions**")
            new_data['Reaction time(min)'] = st.number_input(
                REQUIRED_COLS['Reaction time(min)'], 
                min_value=0, step=1,
                key="new_time"
            )
            new_data['T(°C)'] = st.number_input(
                REQUIRED_COLS['T(°C)'], 
                min_value=0.0, max_value=300.0, step=0.1, format="%.1f",
                key="new_temp"
            )
        
        # 第三列: Product Outcomes (纯手动输入)
        with cols[2]:
            st.markdown("**Product Outcomes**")
            new_data['Crude weight(g)'] = st.number_input(
                REQUIRED_COLS['Crude weight(g)'], 
                min_value=0.0, step=0.001, format="%.3f",
                key="new_crude"
            )
            new_data['Purify weight(g)'] = st.number_input(
                REQUIRED_COLS['Purify weight(g)'], 
                min_value=0.0, step=0.001, format="%.3f",
                key="new_purify"
            )
            new_data['Yield(%)'] = st.number_input(
                REQUIRED_COLS['Yield(%)'], 
                min_value=0.0, max_value=200.0, step=0.1, format="%.2f",
                key="new_yield"
            )
        
        if st.form_submit_button("✅ Add New Data", type="primary"):
            # 验证数据完整性
            if any(v is None for v in new_data.values()):
                st.error("所有字段都必须填写！")
            else:
                # 确保PA:AA使用手动输入的值
                new_data['PA:AA'] = new_data.get('PA:AA', 0.0)
                
                # 添加到数据框
                new_df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
                if save_data(new_df):
                    st.success("数据添加成功！")  # 修改3：添加成功提示
                    st.rerun()

# ================ Product Prediction Module ================
def weight_prediction():
    st.title("Acetaminophen Synthesis Yield Prediction")
    try:
        df = load_data()
    except:
        st.error("无法加载数据文件")
        return
    
    # 硬性数据要求
    MIN_DATA_ROWS = 5
    REQUIRED_PRED_COLS = ['p-aminophenol(g)', 'Acetic Anhydride(ml)', 'PA:AA',
                         'Reaction time(min)', 'T(°C)', 'Crude weight(g)', 'Purify weight(g)', 'Yield(%)']
    
    if len(df) < MIN_DATA_ROWS:
        st.error(f"至少需要 {MIN_DATA_ROWS} 条数据才能建模！当前数据量: {len(df)}")
        st.markdown("[➡️ 前往数据管理页面录入数据](#data-management)")
        return
    
    missing_cols = [col for col in REQUIRED_PRED_COLS if col not in df.columns]
    if missing_cols:
        st.error(f"缺失必要列: {', '.join(missing_cols)}")
        return
    
    # 建模逻辑
    try:
        X = df[['p-aminophenol(g)', 'Acetic Anhydride(ml)', 'PA:AA', 'Reaction time(min)', 'T(°C)']]
        y = df['Yield(%)']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # 显示模型性能
        with st.expander("📈 Model Performance"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}%")
            with col2:
                st.metric("R² Score", f"{r2_score(y_test, y_pred):.3f}")
        
        # 预测接口
        with st.form("prediction_form"):
            st.subheader("输入预测参数")
            input_data = {
                'p-aminophenol(g)': st.number_input(REQUIRED_COLS['p-aminophenol(g)'], min_value=0.1, max_value=50.0, value=5.0),
                'Acetic Anhydride(ml)': st.number_input(REQUIRED_COLS['Acetic Anhydride(ml)'], min_value=0.1, max_value=100.0, value=10.0),
                'Reaction time(min)': st.number_input(REQUIRED_COLS['Reaction time(min)'], min_value=1, max_value=300, value=30),
                'T(°C)': st.number_input(REQUIRED_COLS['T(°C)'], min_value=0.0, max_value=200.0, value=80.0)
            }
            
            if st.form_submit_button("🔮 Run Prediction"):
                X_input = pd.DataFrame([input_data])
                prediction = model.predict(X_input)[0]
                st.success(f"预测产量: {prediction:.1f}%")
                
    except Exception as e:
        st.error(f"建模失败: {str(e)}")

# ================ Main App ================
def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Select Module",
        ["Data Analysis", "Data Management", "Yield Prediction"],
        index=0
    )
    
    try:
        df = load_data()
        st.sidebar.markdown(f"Current data: {len(df)} experiments")
    except:
        st.sidebar.warning("Cannot load data file")
    
    if app_mode == "Data Analysis":
        show_analysis()
    elif app_mode == "Data Management":
        data_management()
    elif app_mode == "Yield Prediction":
        weight_prediction()

if __name__ == "__main__":
    main()
