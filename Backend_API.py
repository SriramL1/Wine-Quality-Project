# Backend_API.py
import os
from google.cloud import bigquery
import pandas as pd
import streamlit as st

# Set Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/srir5/OneDrive/Documents/Desktop/Wine_Quality_Project/wine-quality-analysis-2025.json"
# For deployment: client = bigquery.Client.from_service_account_info(st.secrets["gcp_service_account"])
client = bigquery.Client()

# BigQuery table details
PROJECT_ID = "wine-quality-analysis-2025"
DATASET_ID = "WineQT"
TABLE_ID = "WineQualityTable"
TABLE_REF = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

# Query to fetch existing data
QUERY = f"""
SELECT * EXCEPT (Id)
FROM `{TABLE_REF}`
"""

def load_data():
    """Fetches data from BigQuery and returns a DataFrame with numeric types enforced."""
    try:
        df = client.query(QUERY).to_dataframe()
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        if df.isna().any().any():
            st.warning("Some values in the dataset were coerced to NaN.")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        raise

def get_correlation_matrix(df):
    """Calculates the correlation matrix from a DataFrame."""
    return df.corr()

def add_data_point_to_bigquery(data_point, table_ref=TABLE_REF):
    """
    Adds a single data point to the BigQuery table, ensuring all values are numeric.
    
    Args:
        data_point: Dict with keys matching table columns (excluding Id).
        table_ref: BigQuery table reference (e.g., 'project.dataset.table').
    
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Expected columns (excluding Id)
        expected_columns = [
            'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol', 'quality'
        ]

        # Validate input
        if not all(key in expected_columns for key in data_point.keys()):
            st.error(f"Invalid columns provided. Expected: {expected_columns}")
            return False

        # Ensure all values are numeric
        df = pd.DataFrame([data_point], columns=expected_columns)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if df.isna().any().any():
            st.error("All values must be numeric. Please check your inputs.")
            return False

        # Fetch the table's schema dynamically
        table = client.get_table(table_ref)
        schema = table.schema

        # Filter schema to match expected columns (exclude Id if present)
        schema = [field for field in schema if field.name in expected_columns]

        # Validate schema compatibility
        for col in expected_columns:
            if col not in [field.name for field in schema]:
                st.error(f"Column {col} not found in BigQuery table schema.")
                return False

        # Convert quality to integer if required
        if 'quality' in df.columns:
            df['quality'] = df['quality'].astype(int)

        # Configure BigQuery job
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",  # Append to existing table
            schema=schema,
        )

        # Upload to BigQuery
        job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
        job.result()  # Wait for the job to complete

        st.success("Successfully added new data point to BigQuery!")
        return True

    except Exception as e:
        st.error(f"Error adding data point to BigQuery: {e}")
        return False

if __name__ == "__main__":
    # Example usage for testing locally
    df = load_data()
    print("Raw Data Head:")
    print(df.head())
    corr_matrix = get_correlation_matrix(df)
    print("\nCorrelation Matrix:")
    print(corr_matrix)