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

def get_data_ranges(df):
    """Get min and max values for each parameter from existing data"""
    ranges = {}
    for col in ['p-aminophenol(g)', 'Acetic Anhydride(ml)', 'PA:AA', 'Reaction time(min)', 'T(°C)']:
        if col in df.columns:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                ranges[col] = {
                    'min': float(col_data.min()),
                    'max': float(col_data.max())
                }
    return ranges

# ================ Model Training with Caching ================
@st.cache_resource(ttl=3600)  # Cache models for 1 hour
def train_models(df):
    """Train and cache the prediction models"""
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
        
        return {
            'crude_model': model_crude,
            'purify_model': model_purify,
            'yield_model': model_yield,
            'test_data': {
                'X_test': X_test,
                'yc_test': yc_test,
                'yp_test': yp_test,
                'yy_test': yy_test
            }
        }
    except Exception as e:
        st.error(f"Model training error: {str(e)}")
        return None


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
    
    # Get data ranges for validation
    data_ranges = get_data_ranges(df)
    if not data_ranges:
        st.error("Cannot determine data ranges from existing data")
        return
    
    # Check if all required ranges are available
    required_ranges = ['p-aminophenol(g)', 'Acetic Anhydride(ml)', 'PA:AA', 'Reaction time(min)', 'T(°C)']
    missing_ranges = [col for col in required_ranges if col not in data_ranges]
    if missing_ranges:
        st.error(f"Cannot determine ranges for: {', '.join(missing_ranges)}")
        st.info("Please add data with these parameters in Data Management")
        return
    
    # Train or load cached models
    models = train_models(df)
    if not models:
        st.error("Failed to train models")
        return
    
    # Model evaluation
    with st.expander("📈 Model Performance", expanded=False):
        yc_pred = models['crude_model'].predict(models['test_data']['X_test'])
        yp_pred = models['purify_model'].predict(models['test_data']['X_test'])
        yy_pred = models['yield_model'].predict(models['test_data']['X_test'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Crude Model MAE", f"{mean_absolute_error(models['test_data']['yc_test'], yc_pred):.3f} g")
            st.metric("Crude Model R²", f"{r2_score(models['test_data']['yc_test'], yc_pred):.3f}")
        with col2:
            st.metric("Purified Model MAE", f"{mean_absolute_error(models['test_data']['yp_test'], yp_pred):.3f} g")
            st.metric("Purified Model R²", f"{r2_score(models['test_data']['yp_test'], yp_pred):.3f}")
        with col3:
            st.metric("Yield Model MAE", f"{mean_absolute_error(models['test_data']['yy_test'], yy_pred):.3f}%")
            st.metric("Yield Model R²", f"{r2_score(models['test_data']['yy_test'], yy_pred):.3f}")
        
        st.info("MAE (Mean Absolute Error) - lower is better, R² (R-squared) - closer to 1 is better")
    
    # Prediction interface
    st.subheader("Yield Prediction")
    with st.expander("📝 Prediction Guide", expanded=True):
        st.markdown(f"""
        ### Instructions:
        1. Enter reactant parameters and conditions (must be within your experimental data ranges)
        2. Click "Run Prediction"
        3. View predicted results
        
        ### Parameter Ranges (from your data):
        - p-aminophenol: {data_ranges['p-aminophenol(g)']['min']:.1f}-{data_ranges['p-aminophenol(g)']['max']:.1f} g
        - Acetic Anhydride: {data_ranges['Acetic Anhydride(ml)']['min']:.1f}-{data_ranges['Acetic Anhydride(ml)']['max']:.1f} ml
        - PA:AA ratio: {data_ranges['PA:AA']['min']:.2f}-{data_ranges['PA:AA']['max']:.2f}
        - Reaction time: {data_ranges['Reaction time(min)']['min']:.0f}-{data_ranges['Reaction time(min)']['max']:.0f} min
        - Temperature: {data_ranges['T(°C)']['min']:.1f}-{data_ranges['T(°C)']['max']:.1f} ℃
        
        Predictions are only reliable within these ranges.
        """)
    
    with st.form("prediction_form"):
        cols = st.columns(2)
        input_data = {}
        validation_passed = True
        
        with cols[0]:
            st.markdown("#### Reactant Parameters")
            input_data['p-aminophenol(g)'] = st.number_input(
                REQUIRED_COLS['p-aminophenol(g)'], 
                min_value=float(data_ranges['p-aminophenol(g)']['min']),
                max_value=float(data_ranges['p-aminophenol(g)']['max']),
                value=float((data_ranges['p-aminophenol(g)']['min'] + data_ranges['p-aminophenol(g)']['max']) / 2),
                step=0.1, 
                format="%.1f", key="pred_p_amino"
            )
            input_data['Acetic Anhydride(ml)'] = st.number_input(
                REQUIRED_COLS['Acetic Anhydride(ml)'], 
                min_value=float(data_ranges['Acetic Anhydride(ml)']['min']),
                max_value=float(data_ranges['Acetic Anhydride(ml)']['max']),
                value=float((data_ranges['Acetic Anhydride(ml)']['min'] + data_ranges['Acetic Anhydride(ml)']['max']) / 2),
                step=0.1,
                format="%.1f", key="pred_acetic"
            )
            # Calculate and show PA:AA
            if input_data['Acetic Anhydride(ml)'] != 0:
                pa_aa = input_data['p-aminophenol(g)'] / input_data['Acetic Anhydride(ml)']
                st.metric("PA:AA Ratio", f"{pa_aa:.3f}")
                input_data['PA:AA'] = pa_aa
                # Validate PA:AA ratio
                if not (data_ranges['PA:AA']['min'] <= pa_aa <= data_ranges['PA:AA']['max']):
                    st.error(f"PA:AA ratio must be between {data_ranges['PA:AA']['min']:.2f} and {data_ranges['PA:AA']['max']:.2f}")
                    validation_passed = False
            else:
                st.warning("Acetic Anhydride volume cannot be 0")
                validation_passed = False
        
        with cols[1]:
            st.markdown("#### Reaction Conditions")
            input_data['Reaction time(min)'] = st.number_input(
                REQUIRED_COLS['Reaction time(min)'], 
                min_value=int(data_ranges['Reaction time(min)']['min']),
                max_value=int(data_ranges['Reaction time(min)']['max']),
                value=int((data_ranges['Reaction time(min)']['min'] + data_ranges['Reaction time(min)']['max']) / 2),
                step=1,
                key="pred_time"
            )
            input_data['T(°C)'] = st.number_input(
                REQUIRED_COLS['T(°C)'], 
                min_value=float(data_ranges['T(°C)']['min']),
                max_value=float(data_ranges['T(°C)']['max']),
                value=float((data_ranges['T(°C)']['min'] + data_ranges['T(°C)']['max']) / 2),
                step=0.5,
                format="%.1f", key="pred_temp"
            )
        
        if st.form_submit_button("🔮 Run Prediction", type="primary"):
            try:
                # Specifically check PA:AA, Reaction time, and Temperature
                critical_params = ['PA:AA', 'Reaction time(min)', 'T(°C)']
                for param in critical_params:
                    value = input_data[param]
                    if not (data_ranges[param]['min'] <= value <= data_ranges[param]['max']):
                        st.error(f"{REQUIRED_COLS[param]} must be between {data_ranges[param]['min']} and {data_ranges[param]['max']}")
                        validation_passed = False
                
                if not validation_passed:
                    st.error("超过预测范围，不能预测 (Exceeds prediction range, cannot predict)")
                    return
                
                # Prepare input
                X_input = pd.DataFrame([input_data])[['p-aminophenol(g)', 'Acetic Anhydride(ml)', 'PA:AA', 'Reaction time(min)', 'T(°C)']]
                
                # Predict using cached models
                crude_pred = models['crude_model'].predict(X_input)[0]
                purify_pred = models['purify_model'].predict(X_input)[0]
                yield_pred = models['yield_model'].predict(X_input)[0]
                
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
