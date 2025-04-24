import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import tempfile

# Set matplotlib to use a font that supports English characters
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# Global settings - 修改为云兼容路径
DATA_PATH = os.path.join(tempfile.gettempdir(), "acetaminophen_data.xlsx")
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
def load_data():
    """Load or initialize data file with cloud compatibility"""
    try:
        # Create directory if needed
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        
        if not os.path.exists(DATA_PATH):
            # Initialize with sample data if file doesn't exist
            sample_data = {
                'p-aminophenol(g)': [5.0, 10.0],
                'Acetic Anhydride(ml)': [10.0, 20.0],
                'Reaction time(min)': [30, 45],
                'T(°C)': [80.0, 90.0],
                'Crude weight(g)': [6.2, 12.5],
                'Purify weight(g)': [5.8, 11.7]
            }
            df = pd.DataFrame(sample_data)
            # Calculate derived columns
            df['PA:AA'] = df['p-aminophenol(g)'] / df['Acetic Anhydride(ml)']
            molar_ratio = 151.16 / 109.13
            df['Yield(%)'] = (df['Purify weight(g)'] / (df['p-aminophenol(g)'] * molar_ratio)) * 100
            df.to_excel(DATA_PATH, index=False)
            return df
        
        # Load existing data
        df = pd.read_excel(DATA_PATH)
        
        # Check and repair missing columns
        missing_cols = [col for col in REQUIRED_COLS.keys() if col not in df.columns]
        if missing_cols:
            for col in missing_cols:
                df[col] = np.nan
        
        # Calculate PA:AA if missing
        if 'PA:AA' not in df.columns and all(col in df.columns for col in ['p-aminophenol(g)', 'Acetic Anhydride(ml)']):
            df['PA:AA'] = df['p-aminophenol(g)'] / df['Acetic Anhydride(ml)']
        
        # Calculate Yield(%) if missing
        if 'Yield(%)' not in df.columns and all(col in df.columns for col in ['Purify weight(g)', 'p-aminophenol(g)']):
            molar_ratio = 151.16 / 109.13
            df['Yield(%)'] = (df['Purify weight(g)'] / (df['p-aminophenol(g)'] * molar_ratio)) * 100
        
        return df
    
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return pd.DataFrame(columns=REQUIRED_COLS.keys())

def save_data(df):
    """Save data to Excel with error handling"""
    try:
        df.to_excel(DATA_PATH, index=False)
        return True
    except Exception as e:
        st.error(f"Failed to save data: {str(e)}")
        return False

# ================ Data Management Module ================
def data_management():
    st.title("Acetaminophen Synthesis Data Management")
    
    # Warning about cloud persistence
    st.warning("""
    ⚠️ Note: In Streamlit Cloud, data is not permanently saved between deployments. 
    For persistent storage, consider using a database or external storage.
    """)
    
    df = load_data()
    
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
        if st.button("💾 Save Changes", type="primary"):
            if save_data(edited_df):
                st.success("Data saved (temporary in cloud)!")
                st.rerun()
    with col2:
        if st.button("🔄 Reset to Sample Data"):
            # Clear the existing file to trigger sample data creation
            if os.path.exists(DATA_PATH):
                os.remove(DATA_PATH)
            st.rerun()
    
    # Add new data form (保持原有表单逻辑不变)
    # ... [其余 data_management() 函数内容保持不变] ...

# ================ 其余模块保持不变 ================
# ... [show_analysis(), weight_prediction(), main() 等函数保持不变] ...

if __name__ == "__main__":
    main()
