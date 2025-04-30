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
DATA_PATH = "data.xlsx"
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
def load_fresh_data():
    """直接加载数据文件，完全绕过缓存系统"""
    try:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"数据文件 {DATA_PATH} 不存在！请上传有效的 data.xlsx")
        
        df = pd.read_excel(DATA_PATH)
        
        # 检查必要列是否完整
        missing_cols = [col for col in REQUIRED_COLS.keys() if col not in df.columns]
        if missing_cols:
            raise ValueError(f"数据文件缺少必要列: {', '.join(missing_cols)}")
        
        if df.empty:
            raise ValueError("数据文件为空！请提供有效数据")
            
        # 计算衍生列
        if 'PA:AA' not in df.columns and all(col in df.columns for col in ['p-aminophenol(g)', 'Acetic Anhydride(ml)']):
            df['PA:AA'] = df['p-aminophenol(g)'] / df['Acetic Anhydride(ml)']
        
        if 'Yield(%)' not in df.columns and all(col in df.columns for col in ['Purify weight(g)', 'p-aminophenol(g)']):
            molar_ratio = 151.16 / 109.13
            df['Yield(%)'] = (df['Purify weight(g)'] / (df['p-aminophenol(g)'] * molar_ratio)) * 100
        
        return df
    
    except Exception as e:
        st.error(f"数据加载失败: {str(e)}")
        st.stop()

def save_data(df):
    """保存数据并清除所有缓存状态"""
    try:
        df.to_excel(DATA_PATH, index=False)
        st.cache_data.clear()
        if 'data_modified' in st.session_state:
            st.session_state.data_modified = False
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
    df = load_fresh_data()
    
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
    df = load_data()
    
    # Instructions
    with st.expander("ℹ️ Instructions", expanded=True):
        st.markdown("""
        ### User Guide:
        1. **Edit data**: Modify cells directly in the table
        2. **Add data**: Fill the form below and click "Add New Data"
        3. **Delete data**: Check rows and click "Delete Selected Rows"
        
        ### Data Requirements:
        - All fields are required
        - Values must be positive numbers
        - Temperature: °C
        - Weights: g
        - Volume: ml
        - Time: min
        - PA:AA ratio is calculated automatically
        - Yield (%) is calculated automatically
        """)
    

    
    # Add new data form
    st.subheader("Add New Experiment")
    with st.form("add_data_form"):
        cols = st.columns(2)
        new_data = {}
        
        with cols[0]:
            st.markdown("#### Reactant Parameters")
            new_data['p-aminophenol(g)'] = st.number_input(
                REQUIRED_COLS['p-aminophenol(g)'], 
                min_value=0.0, step=0.01, format="%.2f"
            )
            new_data['Acetic Anhydride(ml)'] = st.number_input(
                REQUIRED_COLS['Acetic Anhydride(ml)'], 
                min_value=0.0, step=0.01, format="%.2f"
            )
            new_data['Reaction time(min)'] = st.number_input(
                REQUIRED_COLS['Reaction time(min)'], 
                min_value=0, step=1
            )
        
        with cols[1]:
            st.markdown("#### Reaction Conditions & Results")
            new_data['T(°C)'] = st.number_input(
                REQUIRED_COLS['T(°C)'], 
                min_value=0.0, max_value=300.0, step=0.1, format="%.1f"
            )
            new_data['Crude weight(g)'] = st.number_input(
                REQUIRED_COLS['Crude weight(g)'], 
                min_value=0.0, step=0.01, format="%.2f"
            )
            new_data['Purify weight(g)'] = st.number_input(
                REQUIRED_COLS['Purify weight(g)'], 
                min_value=0.0, step=0.01, format="%.2f"
            )
        
        if st.form_submit_button("✅ Add New Data", type="primary"):
            if any(pd.isna(val) for val in new_data.values()):
                st.error("All fields are required!")
            elif any(val < 0 for val in new_data.values()):
                st.error("Values cannot be negative!")
            else:
                # Calculate PA:AA and Yield(%)
                new_data['PA:AA'] = new_data['p-aminophenol(g)'] / new_data['Acetic Anhydride(ml)']
                molar_ratio = 151.16 / 109.13
                theo_yield = new_data['p-aminophenol(g)'] * molar_ratio
                new_data['Yield(%)'] = (new_data['Purify weight(g)'] / theo_yield) * 100
        
                new_df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)  # <-- Changed edited_df to df
                save_data(new_df)
                st.success("New data added!")
                st.rerun()

    # 数据删除功能
    st.subheader("Experimental Data Table")
    st.write("select rows to delete")
    
    # 创建带有复选框的DataFrame
    df_with_checkbox = df.copy()
    df_with_checkbox.insert(0, 'select', False)
    
    # 使用st.data_editor创建可编辑表格
    edited_df = st.data_editor(
        df_with_checkbox,
        column_config={
            "select": st.column_config.CheckboxColumn(
                "select",
                help="select rows to delete",
                default=False,
            )
        },
        hide_index=True,
        use_container_width=True
    )
    
    # 获取选中的行
    selected_rows = edited_df[edited_df['select']]
    
    if not selected_rows.empty:
        st.warning(f"delete {len(selected_rows)} row data")
        if st.button("confirm delete selected rows"):
            # 删除选中的行
            df = df.drop(selected_rows.index)
            save_data(df)
            st.success("success to delete selected rows!")
            st.rerun()  # 刷新页面显示更新后的数据
# ================ Product Prediction Module ================
def weight_prediction():
    st.title("Acetaminophen Synthesis Yield Prediction")
    df = load_fresh_data()
    
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
        # 定义特征列顺序（确保训练和预测时一致）
        feature_columns = ['p-aminophenol(g)', 'Acetic Anhydride(ml)', 'PA:AA', 'Reaction time(min)', 'T(°C)']
        
        X = df[feature_columns]
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
            p_amino = st.number_input(REQUIRED_COLS['p-aminophenol(g)'], min_value=0.1, max_value=50.0, value=5.0)
            acetic = st.number_input(REQUIRED_COLS['Acetic Anhydride(ml)'], min_value=0.1, max_value=100.0, value=10.0)
            time = st.number_input(REQUIRED_COLS['Reaction time(min)'], min_value=1, max_value=300, value=30)
            temp = st.number_input(REQUIRED_COLS['T(°C)'], min_value=0.0, max_value=200.0, value=80.0)
            
            # 计算PA:AA比值
            pa_aa_ratio = p_amino / acetic if acetic != 0 else 0.0
            
            if st.form_submit_button("🔮 Run Prediction"):
                # 按照训练时的特征顺序创建输入数据
                input_data = pd.DataFrame([[p_amino, acetic, pa_aa_ratio, time, temp]], 
                                        columns=feature_columns)
                
                prediction = model.predict(input_data)[0]
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
    
    df = load_fresh_data()
    st.sidebar.markdown(f"Current data: {len(df)} experiments")
    
    if app_mode == "Data Analysis":
        show_analysis()
    elif app_mode == "Data Management":
        data_management()
    elif app_mode == "Yield Prediction":
        weight_prediction()

if __name__ == "__main__":
    main()
