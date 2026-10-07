import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb
from preprocess import load_and_clean_data

def train_and_evaluate():
    print("Loading datasets...")
    train_df, val_df = load_and_clean_data()
    dec_df = pd.read_csv("december-chart-inputs.csv")

    # 1. Coordinate mapping table from training data
    pickup_coords = train_df[['pickup', 'pickup_lat', 'pickup_lon']].drop_duplicates().set_index('pickup')
    delivery_coords = train_df[['delivery', 'delivery_lat', 'delivery_lon']].drop_duplicates().set_index('delivery')

    dec_df['pickup_lat'] = dec_df['pickup'].map(pickup_coords['pickup_lat'])
    dec_df['pickup_lon'] = dec_df['pickup'].map(pickup_coords['pickup_lon'])
    dec_df['delivery_lat'] = dec_df['delivery'].map(delivery_coords['delivery_lat'])
    dec_df['delivery_lon'] = dec_df['delivery'].map(delivery_coords['delivery_lon'])

    # 2. Date parsing & temporal feature engineering
    for df in [train_df, val_df, dec_df]:
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_year'] = df['date'].dt.dayofyear
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_heavy_load'] = (df['weight'] > 40000).astype(int)

    # 3. Dynamic market index assignment for December based on day-of-week seasonality
    dow_market_map = train_df.groupby('day_of_week')['market_index'].median()
    dow_quote_map = train_df.groupby('day_of_week')['quote_signal'].median()

    dec_df['market_index'] = dec_df['day_of_week'].map(dow_market_map)
    dec_df['quote_signal'] = dec_df['day_of_week'].map(dow_quote_map)

    # Categorical typing
    for df in [train_df, val_df, dec_df]:
        for col in ['pickup', 'delivery', 'equipment']:
            df[col] = df[col].astype('category')

    features = [
        'pickup', 'delivery', 'equipment', 
        'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon', 
        'distance', 'weight', 'month', 'day_of_week', 'day_of_year', 
        'is_weekend', 'market_index', 'quote_signal', 'is_heavy_load'
    ]

    X = train_df[features]
    y = train_df['posted_rate']
    X_val = val_df[features]
    X_dec = dec_df[features]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    val_predictions = np.zeros(len(val_df))
    dec_predictions = np.zeros(len(dec_df))

    print("\nTraining 5-Fold LightGBM Model...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            random_state=42,
            verbosity=-1
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        val_predictions += model.predict(X_val) / kf.n_splits
        dec_predictions += model.predict(X_dec) / kf.n_splits

    # Save outputs
    val_df['predicted_rate'] = val_predictions
    val_df[['load_id', 'predicted_rate']].to_csv("validation_predictions.csv", index=False)

    dec_df['predicted_rate'] = dec_predictions
    dec_cols = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    dec_df[dec_cols].to_csv("december_predictions.csv", index=False)
    print("Regenerated prediction CSVs with temporal seasonality.")

if __name__ == "__main__":
    train_and_evaluate()