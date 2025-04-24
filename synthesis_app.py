import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Set matplotlib to use a font that supports English characters
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# Global settings
DATA_PATH = "D:/CPE/data.xlsx"
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
    """Load or initialize data file"""
    if not os.path.exists(DATA_PATH):
        pd.DataFrame(columns=REQUIRED_COLS.keys()).to_excel(DATA_PATH, index=False)
        return pd.DataFrame(columns=REQUIRED_COLS.keys())
    
    df = pd.read_excel(DATA_PATH)
    
    # Check for required columns
    missing_cols = [col for col in REQUIRED_COLS.keys() if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns in data file: {', '.join(missing_cols)}")
        st.info("Attempting to fix data file...")
        for col in missing_cols:
            if col not in df.columns:
                df[col] = np.nan
        df.to_excel(DATA_PATH, index=False)
    
    # Calculate PA:AA if missing
    if 'PA:AA' not in df.columns and all(col in df.columns for col in ['p-aminophenol(g)', 'Acetic Anhydride(ml)']):
        df['PA:AA'] = df['p-aminophenol(g)'] / df['Acetic Anhydride(ml)']
    
    # Calculate Yield(%) if missing
    if 'Yield(%)' not in df.columns and all(col in df.columns for col in ['Purify weight(g)', 'p-aminophenol(g)']):
        molar_ratio = 151.16 / 109.13  # Molecular weight ratio
        df['Yield(%)'] = (df['Purify weight(g)'] / (df['p-aminophenol(g)'] * molar_ratio)) * 100
    
    return df

def save_data(df):
    """Save data to Excel"""
    df.to_excel(DATA_PATH, index=False)

# ================ Data Analysis Module ================
def show_analysis():
    st.header("Experimental Data Analysis")
    df = load_data()
    
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
        
        if len(numeric_df.columns) < 2:
            st.warning("Not enough numeric columns for correlation analysis")
            return
        
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
        st.info("Please ensure the data contains numeric values")

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
        disabled=["PA:AA", "Yield(%)"]  # Auto-calculated columns
    )
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💾 Save Changes", help="Save all table modifications", type="primary"):
            save_data(edited_df)
            st.success("Data saved!")
            st.rerun()
    with col2:
        if st.button("🗑️ Delete Selected", help="Permanently delete selected rows"):
            if "_selected_row" in edited_df.columns:
                remaining_df = edited_df[~edited_df["_selected_row"]].drop("_selected_row", axis=1)
                save_data(remaining_df)
                st.success(f"Deleted {len(df) - len(remaining_df)} rows")
                st.rerun()
            else:
                st.warning("Please select rows to delete")
    with col3:
        if st.button("🔄 Refresh Data", help="Reload data file"):
            st.rerun()
    
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
                
                new_df = pd.concat([edited_df, pd.DataFrame([new_data])], ignore_index=True)
                save_data(new_df)
                st.success("New data added!")
                st.rerun()

# ================ Product Prediction Module ================
def weight_prediction():
    st.title("Acetaminophen Synthesis Yield Prediction")
    df = load_data()
    
    # Data validation
    if len(df) < 5:
        st.warning("⚠️ At least 5 experiments required for prediction model")
        st.info(f"Current data: {len(df)} experiments (minimum 5 recommended)")
        return
    
    required_pred_cols = ['p-aminophenol(g)', 'Acetic Anhydride(ml)', 'PA:AA',
                         'Reaction time(min)', 'T(°C)', 'Crude weight(g)', 'Purify weight(g)', 'Yield(%)']
    missing_cols = [col for col in required_pred_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns in data file: {', '.join(missing_cols)}")
        st.info("Please add complete data in Data Management")
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
        
    except Exception as e:
        st.error(f"Model training error: {str(e)}")
        st.info("Please check data format and missing values")
        return
    
    # Prediction interface
    st.subheader("Yield Prediction")
    with st.expander("📝 Prediction Guide", expanded=True):
        st.markdown("""
        ### Instructions:
        1. Enter reactant parameters and conditions
        2. Click "Run Prediction"
        3. View predicted results
        
        ### Parameter Ranges:
        - p-aminophenol: 0.1-50.0 g
        - Acetic Anhydride: 0.1-100.0 ml
        - Reaction time: 1-300 min
        - Temperature: 0-200 ℃
        """)
    
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
            else:
                st.warning("Acetic Anhydride volume cannot be 0")
        
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
                
                # Calculate theoretical yield
                molar_ratio = 151.16 / 109.13  # Molecular weight ratio
                theo_yield = input_data['p-aminophenol(g)'] * molar_ratio
                
                # Display results
                st.success("### Prediction Results")
                
                res_cols = st.columns(3)
                with res_cols[0]:
                    st.metric("Crude Product", f"{crude_pred:.2f} g")
                with res_cols[1]:
                    st.metric("Purified Product", f"{purify_pred:.2f} g")
                with res_cols[2]:
                    st.metric("Predicted Yield", f"{yield_pred:.1f}%")
                
                # Detailed info
                with st.expander("📊 Details", expanded=False):
                    st.markdown(f"""
                    **Theoretical Calculation**:
                    - Theoretical yield: {theo_yield:.2f} g (stoichiometric)
                    - Actual/Theoretical ratio: {purify_pred/theo_yield:.2%}
                    
                    **Model Info**:
                    - Random Forest Regression
                    - Trained on {len(df)} experiments
                    - Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
                    """)
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

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
    1. Enter data in "Data Management"
    2. View correlations in "Data Analysis"
    3. Make predictions in "Yield Prediction"
    """)
    
    # Show current data status
    try:
        df = load_data()
        st.sidebar.markdown(f"Current data: {len(df)} experiments")
    except:
        st.sidebar.warning("Cannot load data file")
    
    # Route to modules
    if app_mode == "Data Analysis":
        show_analysis()
    elif app_mode == "Data Management":
        data_management()
    elif app_mode == "Yield Prediction":
        weight_prediction()

if __name__ == "__main__":
    main()
