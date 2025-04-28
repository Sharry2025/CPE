import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import requests
from io import BytesIO
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import base64
import json

# Set matplotlib to use a font that supports English characters
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# Global settings - GitHub settings
try:
    GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
except KeyError:
    st.error("""
    ❌ GitHub token not found. Please configure:
    1. **Local development**: Create `.streamlit/secrets.toml` with `GITHUB_TOKEN="your_token"`  
    2. **Streamlit Cloud**: Go to App Settings → Secrets and add `GITHUB_TOKEN=your_token`
    """)
    st.stop()
except Exception as e:
    st.error(f"Failed to load GitHub token: {str(e)}")
    st.stop()

GITHUB_REPO_OWNER = "Sharry2025"
GITHUB_REPO_NAME = "CPE"
GITHUB_FILE_PATH = "data.xlsx"

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

# ================ GitHub API Functions ================
def get_github_file_sha():
    """Get the SHA hash of the current file on GitHub"""
    url = f"https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/contents/{GITHUB_FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()["sha"]
    else:
        st.error(f"Failed to get file SHA: {response.text}")
        return None

def update_github_file(new_content):
    """Update the file on GitHub with new content"""
    sha = get_github_file_sha()
    if not sha:
        return False
    
    url = f"https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/contents/{GITHUB_FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    
    excel_buffer = BytesIO()
    new_content.to_excel(excel_buffer, index=False)
    excel_buffer.seek(0)
    content_base64 = base64.b64encode(excel_buffer.read()).decode()
    
    data = {
        "message": "Update data.xlsx via Streamlit app",
        "content": content_base64,
        "sha": sha
    }
    
    response = requests.put(url, headers=headers, json=data)
    if response.status_code == 200:
        return True
    else:
        st.error(f"Failed to update file: {response.text}")
        return False

# ================ Helper Functions ================
def load_data():
    """Load data from GitHub"""
    try:
        url = f"https://raw.githubusercontent.com/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/main/{GITHUB_FILE_PATH}"
        response = requests.get(url)
        response.raise_for_status()
        df = pd.read_excel(BytesIO(response.content))
        
        missing_cols = [col for col in REQUIRED_COLS.keys() if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            st.stop()
        
        if 'PA:AA' not in df.columns:
            df['PA:AA'] = df['p-aminophenol(g)'] / df['Acetic Anhydride(ml)']
        
        if 'Yield(%)' not in df.columns:
            molar_ratio = 151.16 / 109.13
            df['Yield(%)'] = (df['Purify weight(g)'] / (df['p-aminophenol(g)'] * molar_ratio)) * 100
        
        return df
    
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        st.stop()

# ================ Data Management Module ================
def data_management():
    st.title("Acetaminophen Synthesis Data Management")
    df = load_data()
    
    # Display editable data
    st.subheader("Experimental Data Table")
    edited_df = st.data_editor(df, use_container_width=True, num_rows="dynamic")
    
    # Add new experiment form
    with st.expander("➕ Add New Experiment (Complete Reaction Data)", expanded=True):
        with st.form("add_experiment_form"):
            cols = st.columns([1, 1, 1])
            new_exp = {}
            
            with cols[0]:
                st.markdown("#### Reactant Parameters")
                new_exp['p-aminophenol(g)'] = st.number_input(
                    REQUIRED_COLS['p-aminophenol(g)'],
                    min_value=0.1, max_value=50.0, value=5.0, step=0.1,
                    format="%.1f"
                )
                new_exp['Acetic Anhydride(ml)'] = st.number_input(
                    REQUIRED_COLS['Acetic Anhydride(ml)'],
                    min_value=0.1, max_value=100.0, value=10.0, step=0.1,
                    format="%.1f"
                )
                new_exp['PA:AA'] = new_exp['p-aminophenol(g)'] / new_exp['Acetic Anhydride(ml)']
                st.metric("PA:AA Ratio", f"{new_exp['PA:AA']:.4f}")
            
            with cols[1]:
                st.markdown("#### Reaction Conditions")
                new_exp['Reaction time(min)'] = st.number_input(
                    REQUIRED_COLS['Reaction time(min)'],
                    min_value=1, max_value=300, value=60, step=1
                )
                new_exp['T(°C)'] = st.number_input(
                    REQUIRED_COLS['T(°C)'],
                    min_value=0.0, max_value=200.0, value=80.0, step=0.5,
                    format="%.1f"
                )
            
            with cols[2]:
                st.markdown("#### Product Outcomes")
                new_exp['Crude weight(g)'] = st.number_input(
                    REQUIRED_COLS['Crude weight(g)'],
                    min_value=0.0, max_value=100.0, value=0.0, step=0.001,
                    format="%.3f"
                )
                new_exp['Purify weight(g)'] = st.number_input(
                    REQUIRED_COLS['Purify weight(g)'],
                    min_value=0.0, max_value=100.0, value=0.0, step=0.001,
                    format="%.3f"
                )
                if new_exp['p-aminophenol(g)'] > 0:
                    molar_ratio = 151.16 / 109.13
                    theoretical_yield = new_exp['p-aminophenol(g)'] * molar_ratio
                    actual_yield = new_exp['Purify weight(g)']
                    new_exp['Yield(%)'] = (actual_yield / theoretical_yield) * 100
                    st.metric("Yield (%)", f"{new_exp['Yield(%)']:.2f}")
                else:
                    new_exp['Yield(%)'] = 0.0
            
            if st.form_submit_button("Add Complete Experiment", type="primary"):
                edited_df = pd.concat([edited_df, pd.DataFrame([new_exp])], ignore_index=True)
                st.success("New experiment data added!")
    
    # Save changes
    if st.button("💾 Save All Changes to GitHub", type="primary"):
        try:
            # Data validation
            if edited_df.isnull().values.any():
                st.error("Error: Missing values detected")
                return
                
            if update_github_file(edited_df):
                st.success("Data successfully updated on GitHub!")
                st.balloons()
                st.rerun()
            else:
                st.error("Failed to update GitHub")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ================ Other Modules (Keep Original) ================
def show_analysis():
    # ... (保持原有分析模块代码不变) ...

def weight_prediction():
    # ... (保持原有预测模块代码不变) ...

def main():
    # ... (保持原有主函数代码不变) ...

if __name__ == "__main__":
    main()