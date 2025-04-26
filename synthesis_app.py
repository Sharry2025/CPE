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

# Set matplotlib to use a font that supports English characters
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# Global settings - GitHub Raw URL for data.xlsx
DATA_URL = "https://raw.githubusercontent.com/Sharry2025/CPE/main/data.xlsx"  # Replace with your actual GitHub URL
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
    """Load data from GitHub or throw an error if file is missing/invalid"""
    try:
        # Download data from GitHub
        response = requests.get(DATA_URL)
        response.raise_for_status()  # Raise error for bad status codes
        df = pd.read_excel(BytesIO(response.content))
        
        # Validate required columns
        missing_cols = [col for col in REQUIRED_COLS.keys() if col not in df.columns]
        if missing_cols:
            st.error(f"Critical error: Missing required columns in data file: {', '.join(missing_cols)}")
            st.stop()  # Halt the app if columns are missing
        
        # Calculate derived columns
        if 'PA:AA' not in df.columns:
            df['PA:AA'] = df['p-aminophenol(g)'] / df['Acetic Anhydride(ml)']
        
        if 'Yield(%)' not in df.columns:
            molar_ratio = 151.16 / 109.13
            df['Yield(%)'] = (df['Purify weight(g)'] / (df['p-aminophenol(g)'] * molar_ratio)) * 100
        
        return df
    
    except Exception as e:
        st.error(f"Failed to load data from GitHub: {str(e)}")
        st.error("Please ensure:")
        st.error("1. data.xlsx exists in your GitHub repository")
        st.error("2. The file is in the correct format with all required columns")
        st.stop()  # Stop the app if data loading fails

# ================ Data Analysis Module ================
def show_analysis():
    st.header("Experimental Data Analysis")
    df = load_data()  # This will error out if data is invalid
    
    # Data overview
    with st.expander("📊 Data Overview", expanded=True):
        st.dataframe(df, use_container_width=True)
        st.write(f"Total experiments: {len(df)}")
    
    # Correlation analysis
    st.subheader("Parameter Correlation Analysis")
    if len(df) < 2:
        st.warning("At least 2 data points required for correlation analysis")
        return
    
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        # Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                   center=0, linewidths=0.5, annot_kws={"size": 12})
        plt.title("Parameter Correlation Heatmap (Pearson)", pad=20)
        st.pyplot(plt)
        
        # Correlation coefficient explanation
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
    st.warning("Data editing is disabled in cloud deployment. Edit data.xlsx directly in GitHub.")
    df = load_data()
    
    # Display read-only data
    st.subheader("Experimental Data Table (Read-only)")
    st.dataframe(df, use_container_width=True)
    
    # Add instructions for GitHub editing
    with st.expander("ℹ️ How to edit data", expanded=True):
        st.markdown("""
        ### To modify data:
        1. Go to your GitHub repository
        2. Edit the `data.xlsx` file directly
        3. The app will automatically reload the updated data
        
        ### Data Requirements:
        - All fields must be present in the Excel file
        - Values must be positive numbers
        - Column names must match exactly:
        ```
        p-aminophenol(g), Acetic Anhydride(ml), PA:AA, 
        Reaction time(min), T(°C), Crude weight(g), 
        Purify weight(g), Yield(%)
        ```
        """)

# ================ Product Prediction Module ================
def weight_prediction():
    st.title("Acetaminophen Synthesis Yield Prediction")
    df = load_data()
    
    # Data validation
    if len(df) < 5:
        st.error("⚠️ At least 5 experiments required for prediction model")
        st.info(f"Current data: {len(df)} experiments (minimum 5 required)")
        return
    
    # Model training
    try:
        X = df[['p-aminophenol(g)', 'Acetic Anhydride(ml)', 'PA:AA', 'Reaction time(min)', 'T(°C)']]
        y_crude = df['Crude weight(g)']
        y_purify = df['Purify weight(g)']
        y_yield = df['Yield(%)']
        
        # Random Forest models
        model_crude = RandomForestRegressor(n_estimators=100, random_state=42)
        model_purify = RandomForestRegressor(n_estimators=100, random_state=42)
        model_yield = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Train models
        X_train, X_test, yc_train, yc_test = train_test_split(X, y_crude, test_size=0.2)
        model_crude.fit(X_train, yc_train)
        
        X_train, X_test, yp_train, yp_test = train_test_split(X, y_purify, test_size=0.2)
        model_purify.fit(X_train, yp_train)
        
        X_train, X_test, yy_train, yy_test = train_test_split(X, y_yield, test_size=0.2)
        model_yield.fit(X_train, yy_train)
        
        # Model evaluation
        with st.expander("📈 Model Performance", expanded=False):
            yc_pred = model_crude.predict(X_test)
            yp_pred = model_purify.predict(X_test)
            yy_pred = model_yield.predict(X_test)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Crude Model MAE", f"{mean_absolute_error(yc_test, yc_pred):.3f} g")
                st.metric("Crude Model R²", f"{r2_score(yc_test, yc_pred):.3f}")
            with col2:
                st.metric("Purified Model MAE", f"{mean_absolute_error(yp_test, yp_pred):.3f} g")
                st.metric("Purified Model R²", f"{r2_score(yp_test, yp_pred):.3f}")
            with col3:
                st.metric("Yield Model MAE", f"{mean_absolute_error(yy_test, yy_pred):.3f}%")
                st.metric("Yield Model R²", f"{r2_score(yy_test, yy_pred):.3f}")
            
            st.info("MAE (Mean Absolute Error) - lower is better, R² (R-squared) - closer to 1 is better")
        
        # Prediction interface
        st.subheader("Yield Prediction")
        with st.form("prediction_form"):
            cols = st.columns(2)
            input_data = {}
            
            with cols[0]:
                st.markdown("#### Reactant Parameters")
                input_data['p-aminophenol(g)'] = st.number_input(
                    REQUIRED_COLS['p-aminophenol(g)'], 
                    min_value=0.1, max_value=50.0, value=5.0, step=0.1, 
                    format="%.1f", key="pred_p_amino"
                )
                input_data['Acetic Anhydride(ml)'] = st.number_input(
                    REQUIRED_COLS['Acetic Anhydride(ml)'], 
                    min_value=0.1, max_value=100.0, value=10.0, step=0.1,
                    format="%.1f", key="pred_acetic"
                )
                # Calculate and show PA:AA
                if input_data['Acetic Anhydride(ml)'] != 0:
                    pa_aa = input_data['p-aminophenol(g)'] / input_data['Acetic Anhydride(ml)']
                    st.metric("PA:AA Ratio", f"{pa_aa:.3f}")
                    input_data['PA:AA'] = pa_aa
            
            with cols[1]:
                st.markdown("#### Reaction Conditions")
                input_data['Reaction time(min)'] = st.number_input(
                    REQUIRED_COLS['Reaction time(min)'], 
                    min_value=1, max_value=300, value=30, step=1,
                    key="pred_time"
                )
                input_data['T(°C)'] = st.number_input(
                    REQUIRED_COLS['T(°C)'], 
                    min_value=0.0, max_value=200.0, value=80.0, step=0.5,
                    format="%.1f", key="pred_temp"
                )
            
            if st.form_submit_button("🔮 Run Prediction", type="primary"):
                try:
                    # Prepare input
                    X_input = pd.DataFrame([input_data])[['p-aminophenol(g)', 'Acetic Anhydride(ml)', 'PA:AA', 'Reaction time(min)', 'T(°C)']]
                    
                    # Predict
                    crude_pred = model_crude.predict(X_input)[0]
                    purify_pred = model_purify.predict(X_input)[0]
                    yield_pred = model_yield.predict(X_input)[0]
                    
                    # Display results
                    st.success("### Prediction Results")
                    res_cols = st.columns(3)
                    with res_cols[0]:
                        st.metric("Crude Product", f"{crude_pred:.2f} g")
                    with res_cols[1]:
                        st.metric("Purified Product", f"{purify_pred:.2f} g")
                    with res_cols[2]:
                        st.metric("Predicted Yield", f"{yield_pred:.1f}%")
                    
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
    
    except Exception as e:
        st.error(f"Model training error: {str(e)}")

# ================ Main App ================
def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Select Module",
        ["Data Analysis", "Data Management", "Yield Prediction"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **System Guide**:
    1. View correlations in "Data Analysis"
    2. Check data requirements in "Data Management"
    3. Make predictions in "Yield Prediction"
    """)
    
    # Show current data status
    try:
        df = load_data()
        st.sidebar.markdown(f"Current data: {len(df)} experiments")
    except:
        st.sidebar.error("Data loading failed")
    
    # Route to modules
    if app_mode == "Data Analysis":
        show_analysis()
    elif app_mode == "Data Management":
        data_management()
    elif app_mode == "Yield Prediction":
        weight_prediction()

if __name__ == "__main__":
    main()
