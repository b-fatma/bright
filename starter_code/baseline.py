# baseline.py
"""
Random Forest Baseline for MovieLens Rating Prediction
"""

import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from load_data import load_competition_data, create_submission

def main():
    graph_data, train_df, test_df = load_competition_data('./data/public')
    
    x = graph_data['x'].numpy()
    num_users = graph_data['num_users']
    
    # Prepare training data
    user_idx = train_df['user_id'].values - 1
    movie_idx = train_df['movie_id'].values - 1 + num_users
    X_train = np.concatenate([x[user_idx], x[movie_idx]], axis=1)
    y_train = train_df['rating'].values
    
    # Train/val split
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Train model
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    
    # Validate
    val_pred = np.clip(rf.predict(X_val), 1.0, 5.0)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    
    # Predict test
    test_user_idx = test_df['user_id'].values - 1
    test_movie_idx = test_df['movie_id'].values - 1 + num_users
    X_test = np.concatenate([x[test_user_idx], x[test_movie_idx]], axis=1)
    predictions = np.clip(rf.predict(X_test), 1.0, 5.0)
    
    # Save
    os.makedirs('./submissions/inbox/baseline/run_001', exist_ok=True)
    submission_df = create_submission(test_df, predictions)
    submission_df.to_csv('./submissions/inbox/baseline/run_001/predictions.csv', index=False)
    
    print(f"Validation RMSE: {val_rmse:.4f}")
    print(f"Saved to submissions/inbox/baseline/run_001/predictions.csv")

if __name__ == "__main__":
    main()