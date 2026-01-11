# pages/1_Correlation.py
import streamlit as st
from Backend_API import add_data_point_to_bigquery, load_data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Wine Quality Correlation Analysis")
st.write("Explore correlations between wine features.")

# Load data with caching
@st.cache_data
def cached_load_data():
    df = load_data()
    # Pre-convert to numeric to avoid type issues downstream
    return df.astype(float)  

with st.spinner("Fetching data from BigQuery..."):
    df = cached_load_data()

# Limit rows for display (optional performance tweak)
if st.checkbox("Show raw data"):
    st.subheader("Raw Data")
    max_rows = st.slider("Limit rows to display (for speed)", 10, len(df), min(100, len(df)), key="row_limit")
    st.write(f"Displaying {max_rows} of {len(df)} rows.")
    st.dataframe(df.iloc[:max_rows])

# Filter columns
st.sidebar.header("Filter Columns")
selected_columns = st.sidebar.multiselect("Select columns for correlation", 
                                         df.columns, 
                                         default=df.columns.tolist())
filtered_df = df[selected_columns]

# Optimized correlation calculation
@st.cache_data
def get_correlation_matrix(df):
    # Use Pandas for Pearson (vectorized, much faster than pairwise pearsonr)
    pearson_corr = df.corr(method='pearson')
    
    # Use Spearman only for 'quality' pairs (vectorized where possible)
    if 'quality' in df.columns:
        spearman_corr = df.corr(method='spearman')
        # Replace only 'quality' rows/columns with Spearman values
        quality_mask = (pearson_corr.index == 'quality') | (pearson_corr.columns == 'quality')
        pearson_corr.loc[quality_mask, :] = spearman_corr.loc[quality_mask, :]
        pearson_corr.loc[:, quality_mask] = spearman_corr.loc[:, quality_mask]
    
    return pearson_corr

# Calculate correlation
correlation_matrix = get_correlation_matrix(filtered_df)

# Display correlation matrix
st.subheader("Correlation Matrix")
st.write("Pearson for numerical pairs, Spearman when 'quality' is involved.")
st.dataframe(correlation_matrix.style.format("{:.2f}"))

# Heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, 
            annot=True, 
            cmap="coolwarm", 
            vmin=-1, vmax=1, 
            center=0, 
            square=True, 
            ax=ax)
plt.title("Correlation Matrix of Wine Quality Data")
st.pyplot(fig)

# Form to add new data point
st.markdown("---")
st.subheader("Add New Wine Data Point")
st.write("Enter numeric values for the wine features to add to the BigQuery dataset.")

with st.form(key="add_data_form"):
    data_point = {}
    cols = st.columns(3)  # Organize inputs in 3 columns
    # Default ranges if data loading fails
    default_ranges = {
        'fixed acidity': (4.0, 16.0, 7.0),
        'volatile acidity': (0.1, 2.0, 0.5),
        'citric acid': (0.0, 1.0, 0.3),
        'residual sugar': (0.5, 20.0, 2.0),
        'chlorides': (0.01, 0.5, 0.08),
        'free sulfur dioxide': (1.0, 100.0, 15.0),
        'total sulfur dioxide': (1.0, 300.0, 50.0),
        'density': (0.98, 1.01, 0.995),
        'pH': (2.5, 4.5, 3.3),
        'sulphates': (0.2, 2.0, 0.6),
        'alcohol': (8.0, 15.0, 10.0),
        'quality': (0, 10, 5)
    }
    
    for i, col_name in enumerate(default_ranges.keys()):
        with cols[i % 3]:
            if df is not None and col_name in df.columns:
                min_val = float(df[col_name].min())
                max_val = float(df[col_name].max())
                default_val = float(df[col_name].mean())
            else:
                min_val, max_val, default_val = default_ranges[col_name]
            
            # Ensure valid range
            if min_val == max_val:
                max_val = min_val + 0.1
            default_val = max(min(default_val, max_val), min_val)
            
            if col_name == 'quality':
                data_point[col_name] = st.number_input(
                    col_name.replace('_', ' ').title(),
                    min_value=int(min_val),
                    max_value=int(max_val),
                    value=int(default_val),
                    step=1,
                    key=col_name
                )
            else:
                data_point[col_name] = st.number_input(
                    col_name.replace('_', ' ').title(),
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    step=0.01,
                    format="%.1f",
                    key=col_name
                )

    submit_button = st.form_submit_button("Add Data Point")

if submit_button:
    with st.spinner("Adding data point to BigQuery..."):
        success = add_data_point_to_bigquery(data_point)
        if success:
            st.button("Clear Cache to Refresh Data", on_click=st.cache_data.clear)
            st.info("Data point added! Click 'Clear Cache' to refresh analyses.")

# Interpretation
st.write("""
### Interpretation
- **Red**: Positive correlation (closer to 1 = stronger).
- **Blue**: Negative correlation (closer to -1 = stronger).
- **0**: No correlation.
- **Pearson**: Used for continuous numerical data (e.g., alcohol vs. pH).
- **Spearman**: Used for 'quality' (ordinal) vs. other features.
""")