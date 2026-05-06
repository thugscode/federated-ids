"""
Data preprocessing module for CICIDS2017 dataset.

This module handles data cleaning, normalization, and feature engineering
for the CICIDS2017 intrusion detection dataset.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging
from typing import Tuple, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def task_1_load_and_merge_csvs(data_dir: str) -> pd.DataFrame:
    """
    Task 1: Load all 8 daily CSV files and merge into one dataframe.
    
    Handle encoding errors, strip column name whitespace.
    
    Expected files:
        - monday.csv, monday_plus.csv
        - tuesday.csv, tuesday_plus.csv
        - wednesday.csv, wednesday_plus.csv
        - thursday.csv, thursday_plus.csv
        - friday.csv, friday_plus.csv
    
    Args:
        data_dir: Path to directory containing raw CSV files
        
    Returns:
        Combined DataFrame from all CSV files
        
    Raises:
        FileNotFoundError: If no CSV files found in directory
    """
    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    dfs = []
    for csv_file in csv_files:
        logger.info(f"Loading {csv_file.name}")
        try:
            # Load with error handling for encoding
            df = pd.read_csv(csv_file, encoding='utf-8', on_bad_lines='skip')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_file, encoding='latin-1', on_bad_lines='skip')
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        dfs.append(df)
        logger.info(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Merge all dataframes
    df_combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined data shape: {df_combined.shape}")
    logger.info(f"Columns: {df_combined.columns.tolist()}")
    
    return df_combined


def task_2_remove_infinite_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Task 2: Remove infinite values in Flow_Bytes/s and Flow_Packets/s.
    
    Replace inf with NaN then drop those rows.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with infinite values removed
    """
    logger.info(f"Initial shape: {df.shape}")
    
    # Identify columns that might have infinite values
    inf_cols = ['Flow_Bytes/s', 'Flow_Packets/s'] if 'Flow_Bytes/s' in df.columns else []
    
    if inf_cols:
        # Replace infinite values with NaN
        for col in inf_cols:
            if col in df.columns:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                logger.info(f"Replaced infinite values in {col}")
        
        # Drop rows with NaN in these columns
        initial_len = len(df)
        df = df.dropna(subset=inf_cols)
        removed = initial_len - len(df)
        logger.info(f"Removed {removed} rows with infinite values. New shape: {df.shape}")
    
    return df


def task_3_remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Task 3: Remove duplicate rows.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with duplicates removed
    """
    initial_len = len(df)
    df = df.drop_duplicates()
    removed = initial_len - len(df)
    logger.info(f"Removed {removed} duplicate rows. New shape: {df.shape}")
    
    return df


def task_4_parse_and_unify_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Task 4: Parse and unify timestamps into a single datetime column.
    
    Creates a unified 'Timestamp' column from existing date/time columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with unified timestamp column
    """
    # Check for timestamp columns (varies by CICIDS2017 variant)
    timestamp_cols = ['Timestamp', 'Flow Start Time', 'Datetime']
    
    for col in timestamp_cols:
        if col in df.columns:
            logger.info(f"Found timestamp column: {col}")
            try:
                df['Timestamp'] = pd.to_datetime(df[col], format='%d/%m/%Y %H:%M:%S', errors='coerce')
                logger.info(f"Parsed {col} successfully")
                break
            except Exception as e:
                logger.warning(f"Failed to parse {col}: {e}")
    
    # If no timestamp column found, create a sequential one
    if 'Timestamp' not in df.columns:
        logger.warning("No timestamp column found, creating sequential timestamps")
        df['Timestamp'] = pd.date_range(start='2017-01-01', periods=len(df), freq='S')
    
    logger.info(f"Unified timestamp column created. Sample: {df['Timestamp'].head()}")
    
    return df


def task_5_encode_attack_type(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Task 5: Encode Attack_Type as integer labels, keep a label map dict.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (DataFrame with encoded labels, label mapping dictionary)
    """
    # Find the label column
    label_col = None
    possible_cols = ['Label', 'Attack_Type', 'Class']
    
    for col in possible_cols:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        raise ValueError(f"No label column found. Available columns: {df.columns.tolist()}")
    
    logger.info(f"Found label column: {label_col}")
    logger.info(f"Unique labels: {df[label_col].unique()}")
    
    # Encode labels
    le = LabelEncoder()
    df['Label'] = le.fit_transform(df[label_col])
    
    # Create label map
    label_map = dict(zip(le.classes_, le.transform(le.classes_)))
    logger.info(f"Label mapping: {label_map}")
    
    # Drop original label column if different
    if label_col != 'Label':
        df = df.drop(columns=[label_col])
    
    return df, label_map


def task_6_standard_scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Task 6: Standard scale all numeric features using StandardScaler.
    
    Fit on train split only, transform both train and test.
    
    Args:
        X_train: Training features
        X_test: Test features
        
    Returns:
        Tuple of (scaled X_train, scaled X_test, scaler object)
    """
    # Identify numeric columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    logger.info(f"Scaling {len(numeric_cols)} numeric features")
    
    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    logger.info("StandardScaler fitted on training data and applied to both sets")
    logger.info(f"Train - mean: {X_train_scaled[numeric_cols].mean().mean():.4f}, "
                f"std: {X_train_scaled[numeric_cols].std().mean():.4f}")
    
    return X_train_scaled, X_test_scaled, scaler


def task_7_save_processed_data(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                               y_train: pd.Series, y_test: pd.Series,
                               output_dir: str = "data/processed",
                               label_map: Dict = None):
    """
    Task 7: Save processed dataframe as parquet to data/processed/.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        output_dir: Output directory path
        label_map: Optional label mapping dictionary
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save data as parquet
    X_train.to_parquet(output_path / "X_train.parquet")
    X_test.to_parquet(output_path / "X_test.parquet")
    y_train.to_parquet(output_path / "y_train.parquet")
    y_test.to_parquet(output_path / "y_test.parquet")
    
    logger.info(f"Saved processed data to {output_path}")
    logger.info(f"  X_train: {X_train.shape} -> X_train.parquet")
    logger.info(f"  X_test: {X_test.shape} -> X_test.parquet")
    logger.info(f"  y_train: {y_train.shape} -> y_train.parquet")
    logger.info(f"  y_test: {y_test.shape} -> y_test.parquet")
    
    # Save label map if provided
    if label_map:
        import json
        with open(output_path / "label_map.json", "w") as f:
            json.dump(label_map, f, indent=2)
        logger.info(f"Saved label map to label_map.json")


def preprocess_pipeline(data_dir: str = "data/raw", 
                       output_dir: str = "data/processed",
                       test_size: float = 0.2, 
                       random_state: int = 42) -> Dict:
    """
    Complete preprocessing pipeline combining all tasks.
    
    Args:
        data_dir: Path to raw data directory
        output_dir: Path to save processed data
        test_size: Proportion of test set
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with processed data and metadata
    """
    logger.info("="*60)
    logger.info("Starting CICIDS2017 Preprocessing Pipeline")
    logger.info("="*60)
    
    # Task 1: Load and merge
    df = task_1_load_and_merge_csvs(data_dir)
    
    # Task 2: Remove infinite values
    df = task_2_remove_infinite_values(df)
    
    # Task 3: Remove duplicates
    df = task_3_remove_duplicates(df)
    
    # Task 4: Parse timestamps
    df = task_4_parse_and_unify_timestamps(df)
    
    # Task 5: Encode attack types
    df, label_map = task_5_encode_attack_type(df)
    
    # Separate features and labels
    X = df.drop(columns=['Label', 'Timestamp'])
    y = df['Label']
    
    # Remove any remaining non-numeric columns
    X = X.select_dtypes(include=[np.number])
    
    logger.info(f"Final features shape: {X.shape}")
    logger.info(f"Label distribution:\n{y.value_counts()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Train-test split: {X_train.shape} | {X_test.shape}")
    
    # Task 6: Scale features
    X_train_scaled, X_test_scaled, scaler = task_6_standard_scale_features(X_train, X_test)
    
    # Task 7: Save processed data
    task_7_save_processed_data(X_train_scaled, X_test_scaled, y_train, y_test, 
                              output_dir, label_map)
    
    result = {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "label_map": label_map,
        "feature_names": X.columns.tolist(),
    }
    
    logger.info("="*60)
    logger.info("Preprocessing pipeline completed successfully!")
    logger.info("="*60)
    
    return result


if __name__ == "__main__":
    import sys
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/processed"
    
    result = preprocess_pipeline(data_dir, output_dir)
