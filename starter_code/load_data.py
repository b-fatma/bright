# load_data.py
import torch
import pandas as pd

def load_competition_data(data_dir='./data/public'):
    """
    Load MovieLens competition data
    
    Returns:
        graph_data: dict with x, edge_index, edge_attr, etc.
        train_df: DataFrame with user_id, movie_id, rating
        test_df: DataFrame with id, user_id, movie_id
    """
    # Load graph
    graph_data = torch.load(f'{data_dir}/graph_data.pt')
    
    # Load ratings
    train_df = pd.read_csv(f'{data_dir}/train_ratings.csv')
    test_df = pd.read_csv(f'{data_dir}/test_nodes.csv')
    
    return graph_data, train_df, test_df

def create_submission(test_df, predictions):
    """
    Create submission file
    
    Args:
        test_df: DataFrame from test_nodes.csv
        predictions: Array of predictions (same order as test_df)
    
    Returns:
        submission_df: DataFrame with id, y_pred columns
    """
    submission = pd.DataFrame({
        'id': test_df['id'],
        'y_pred': predictions
    })
    return submission