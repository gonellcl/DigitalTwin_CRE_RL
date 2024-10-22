import pandas as pd
import os


def preprocess_industry_data(file_path):
    """
    Preprocess the raw Department of Labor data for industry growth.

    Args:
    - file_path (str): Path to the raw CSV file.

    Returns:
    - industry_dfs (dict): Dictionary of DataFrames, each representing a unique industry.
    """
    # Load data
    data = pd.read_csv(file_path)

    # Make all column names lowercase
    data.columns = [col.lower() for col in data.columns]

    # Drop unnecessary columns
    data = data.drop(['bmrk yr', 'st', 'series', 'area', 'data type code', 'data type'], axis=1)

    # Copy data for transformation
    df_t = data.copy()

    # Convert 'year' column to datetime and set it as index
    df_t['date'] = pd.to_datetime(df_t['year'], format='%Y')
    df_t.set_index('date', inplace=True)
    df_t['growth_rate'] = df_t['average'].pct_change()  # Calculate growth rate

    return df_t


file_path = os.path.join(os.getcwd(), 'data', 'raw', 'DOL_data.csv')
industry_df = preprocess_industry_data(file_path)

print(industry_df.head())
