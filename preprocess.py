import pandas as pd
import numpy as np

def load_and_clean_data(train_path="train-test.csv", val_path="validation.csv"):
    # 1. Load datasets
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    
    # Track train vs val
    df_train['is_train'] = 1
    df_val['is_train'] = 0
    df_val['posted_rate'] = np.nan  # Target to predict
    
    # Combine for consistent feature processing
    df = pd.concat([df_train, df_val], ignore_index=True)
    
    # 2. Fix data quality issues
    # Fix negative weights via absolute value
    df['weight'] = df['weight'].abs()
    
    # Impute missing values
    df['weight'] = df['weight'].fillna(df['weight'].median())
    df['market_index'] = df['market_index'].fillna(df['market_index'].median())
    
    # 3. Parse dates & engineer temporal features
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # 4. Domain & Geospatial features
    # Estimated Rate Per Mile prior / distance features
    df['is_heavy_load'] = (df['weight'] > 40000).astype(int)
    
    # Split back into train and validation sets
    train_clean = df[df['is_train'] == 1].drop(columns=['is_train']).reset_index(drop=True)
    val_clean = df[df['is_train'] == 0].drop(columns=['is_train', 'posted_rate']).reset_index(drop=True)
    
    return train_clean, val_clean

if __name__ == "__main__":
    train_df, val_df = load_and_clean_data()
    print(f"Cleaned Train Shape: {train_df.shape}")
    print(f"Cleaned Validation Shape: {val_df.shape}")
    print("Sample missing counts in cleaned train:")
    print(train_df.isnull().sum()[train_df.isnull().sum() > 0])