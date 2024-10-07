import pandas as pd
import numpy as np

# Load the raw dataset
def load_data(file_path):
    """
    Loads the raw data from a CSV file.
    Args:
    file_path (str): Path to the CSV file.
    Returns:
    pd.DataFrame: Loaded data as a DataFrame.
    """
    try:
        data = pd.read_csv(file_path, sep=';')
        return data
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

# Handle missing data and anomalies
def clean_data(df):
    """
    Cleans the dataset by handling missing values, transforming data types,
    and removing duplicates.
    Args:
    df (pd.DataFrame): Raw dataset.
    Returns:
    pd.DataFrame: Cleaned dataset.
    """
    # Drop duplicates
    df = df.drop_duplicates()

    # Handling missing or unknown values
    df.replace('unknown', np.nan, inplace=True)

    # Fill NaN values for categorical features with 'Unknown'
    categorical_columns = ['job', 'marital', 'education', 'contact', 'poutcome']
    df[categorical_columns] = df[categorical_columns].fillna('Unknown')

    # Fill NaN values for numerical features with median
    numerical_columns = ['balance', 'duration', 'campaign', 'pdays', 'previous']
    df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())

    # Convert categorical variables into category type
    for col in categorical_columns:
        df[col] = df[col].astype('category')

    # Convert target variable 'y' to binary (0 for 'no', 1 for 'yes')
    df['y'] = df['y'].map({'yes': 1, 'no': 0})

    return df

# Save cleaned data to CSV
def save_cleaned_data(df, output_path):
    """
    Saves the cleaned dataset to a CSV file.
    Args:
    df (pd.DataFrame): Cleaned dataset.
    output_path (str): File path to save the cleaned data.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

# Main function to run the data cleaning pipeline
if __name__ == "__main__":
    # Load raw data
    raw_data_path = "C:\\Users\\devan\\Desktop\\niyatiprojects24\\banking-sector\\data\\raw_data.csv"
    cleaned_data_path = "C:\\Users\\devan\\Desktop\\niyatiprojects24\\banking-sector\\data\\cleaned_data.csv"
    
    # Clean the data
    df_raw = load_data(raw_data_path)
    if df_raw is not None:
        df_cleaned = clean_data(df_raw)
        
        # Save the cleaned data
        save_cleaned_data(df_cleaned, cleaned_data_path)
