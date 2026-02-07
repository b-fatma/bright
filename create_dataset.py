"""
MovieLens 100K Dataset Preprocessing for GNN Competition
CORRECTED VERSION - Matches template requirements exactly
"""

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# ========================================
# CONFIGURATION 
# ========================================

# Input:  MovieLens 100K data 
INPUT_DATA_DIR = './ml-100k'

# Output: processed competition files
OUTPUT_DATA_DIR = './data'

# ========================================

GENRE_COLS = [f"genre_{i}" for i in range(19)]


def load_movielens_data():
    """Load MovieLens 100K data"""
    print("📚 Loading MovieLens 100K data...")
    print(f"   Input directory: {INPUT_DATA_DIR}")
    
    # Load ratings
    ratings = pd.read_csv(
        os.path.join(INPUT_DATA_DIR, 'u.data'),
        sep='\t',
        names=['user_id', 'movie_id', 'rating', 'timestamp']
    )

    # Load users
    users = pd.read_csv(
        os.path.join(INPUT_DATA_DIR, 'u.user'),
        sep='|',
        names=['user_id', 'age', 'gender', 'occupation', 'zip_code']
    )

    # Load movies
    movies_meta = pd.read_csv(
        os.path.join(INPUT_DATA_DIR, 'u.item'),
        sep='|',
        encoding='latin-1',
        usecols=range(5),
        names=['movie_id', 'title', 'release_date',
               'video_release_date', 'IMDb_URL']
    )

    # Load movie genres
    movie_genres = pd.read_csv(
        os.path.join(INPUT_DATA_DIR, 'u.item'),
        sep='|',
        encoding='latin-1',
        usecols=range(5, 24),
        header=None,
        names=GENRE_COLS
    )

    movies = pd.concat([movies_meta[['movie_id']], movie_genres], axis=1)

    print(f"  Ratings: {len(ratings)}")
    print(f"  Users: {len(users)}")
    print(f"  Movies: {len(movies)}")

    return ratings, users, movies


def create_user_features(users):
    """
    Create user feature matrix
    Features: age (normalized), gender (one-hot), occupation (one-hot)
    """
    print("\n🔧 Creating user features...")
    
    user_features = []
    
    # Encode gender
    gender_encoder = LabelEncoder()
    gender_encoded = gender_encoder.fit_transform(users['gender'])
    
    # Encode occupation
    occupation_encoder = LabelEncoder()
    occupation_encoded = occupation_encoder.fit_transform(users['occupation'])
    num_occupations = len(occupation_encoder.classes_)
    
    for idx, user in users.iterrows():
        feat = []
        
        # Age (normalized to 0-1)
        feat.append(user['age'] / 100.0)
        
        # Gender (binary)
        feat.append(float(gender_encoded[idx]))
        
        # Occupation (one-hot)
        occ_onehot = [0.0] * num_occupations
        occ_onehot[occupation_encoded[idx]] = 1.0
        feat.extend(occ_onehot)
        
        user_features.append(feat)
    
    user_features = torch.tensor(user_features, dtype=torch.float)
    print(f"  User feature shape: {user_features.shape}")
    
    return user_features, {
        'gender_encoder': gender_encoder,
        'occupation_encoder': occupation_encoder,
        'num_occupations': num_occupations
    }


def create_movie_features(movies):
    """
    Create movie feature matrix
    Features: 19 binary genre indicators
    """
    print("\n🎬 Creating movie features...")
    
    movie_features = torch.tensor(
        movies[GENRE_COLS].values,
        dtype=torch.float
    )

    print(f"  Movie feature shape: {movie_features.shape}")
    return movie_features


def create_bipartite_graph(ratings, num_users, num_movies, test_size=0.2, cold_start_threshold=5):
    """
    Create bipartite graph with cold start problem setup
    
    Cold start users/movies (with <= cold_start_threshold ratings) are placed in test set
    
    Node indexing:
    - Users: 0 to num_users-1
    - Movies: num_users to num_users+num_movies-1
    """
    print("\n🕸️ Creating bipartite graph with cold start problem...")
    
    # Convert to 0-indexed
    ratings = ratings.copy()
    ratings['user_idx'] = ratings['user_id'] - 1  # Users: 0 to 942
    ratings['movie_idx'] = ratings['movie_id'] - 1 + num_users  # Movies: 943 to 2624
    
    # Identify cold start users and movies
    user_counts = ratings['user_id'].value_counts()
    movie_counts = ratings['movie_id'].value_counts()
    
    cold_start_users = set(user_counts[user_counts <= cold_start_threshold].index)
    cold_start_movies = set(movie_counts[movie_counts <= cold_start_threshold].index)
    
    print(f"\n  ❄️  Cold Start Analysis (threshold: {cold_start_threshold} ratings):")
    print(f"     Cold start users: {len(cold_start_users)} / {num_users} ({len(cold_start_users)/num_users*100:.1f}%)")
    print(f"     Cold start movies: {len(cold_start_movies)} / {num_movies} ({len(cold_start_movies)/num_movies*100:.1f}%)")
    
    # Mark ratings as cold start if they involve cold start users or movies
    ratings['is_cold_start'] = ratings.apply(
        lambda row: (row['user_id'] in cold_start_users) or (row['movie_id'] in cold_start_movies),
        axis=1
    )
    
    cold_start_ratings = ratings[ratings['is_cold_start']]
    warm_start_ratings = ratings[~ratings['is_cold_start']]
    
    print(f"     Cold start ratings: {len(cold_start_ratings)} / {len(ratings)} ({len(cold_start_ratings)/len(ratings)*100:.1f}%)")
    print(f"     Warm start ratings: {len(warm_start_ratings)} / {len(ratings)} ({len(warm_start_ratings)/len(ratings)*100:.1f}%)")
    
    # Split warm start data for train/test (with stratification)
    num_warm_test = max(1, int(len(warm_start_ratings) * test_size))
    train_warm, test_warm = train_test_split(
        warm_start_ratings,
        test_size=num_warm_test,
        random_state=42,
        stratify=warm_start_ratings['rating'].apply(lambda x: int(x))
    )
    
    # Put all cold start ratings in test set
    test_ratings = pd.concat([test_warm, cold_start_ratings], ignore_index=True)
    train_ratings = train_warm
    
    print(f"\n  Train ratings: {len(train_ratings)}")
    print(f"  Test ratings: {len(test_ratings)}")
    print(f"  Test set composition:")
    print(f"    - Warm start: {len(test_warm)} ({len(test_warm)/len(test_ratings)*100:.1f}%)")
    print(f"    - Cold start: {len(cold_start_ratings)} ({len(cold_start_ratings)/len(test_ratings)*100:.1f}%)")
    
    # Create edge list for training (bidirectional for message passing)
    train_edges = []
    train_edge_attrs = []
    
    for _, row in train_ratings.iterrows():
        user_idx = int(row['user_idx'])
        movie_idx = int(row['movie_idx'])
        rating = row['rating']
        
        # User -> Movie
        train_edges.append([user_idx, movie_idx])
        train_edge_attrs.append(rating)
        
        # Movie -> User (bidirectional)
        train_edges.append([movie_idx, user_idx])
        train_edge_attrs.append(rating)
    
    train_edge_index = torch.tensor(train_edges, dtype=torch.long).t()
    train_edge_attr = torch.tensor(train_edge_attrs, dtype=torch.float).unsqueeze(1)
    
    print(f"  Train edge_index shape: {train_edge_index.shape}")
    print(f"  Train edge_attr shape: {train_edge_attr.shape}")
    
    return train_edge_index, train_edge_attr, train_ratings, test_ratings


def create_dataset():
    """
    Main function to create competition dataset
    """
    print("=" * 60)
    print("🎯 CREATING MOVIELENS GNN COMPETITION DATASET")
    print("=" * 60)
    
    # Load data
    ratings, users, movies = load_movielens_data()
    
    num_users = len(users)
    num_movies = len(movies)
    num_nodes = num_users + num_movies
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Users: {num_users}")
    print(f"  Movies: {num_movies}")
    print(f"  Total nodes: {num_nodes}")
    print(f"  Total ratings: {len(ratings)}")
    print(f"  Rating range: {ratings['rating'].min()}-{ratings['rating'].max()}")
    print(f"  Sparsity: {len(ratings)/(num_users*num_movies)*100:.2f}%")
    
    # Create features
    user_features, user_encoders = create_user_features(users)
    movie_features = create_movie_features(movies)
    
    # Pad features to same dimension
    max_dim = max(user_features.shape[1], movie_features.shape[1])
    
    if user_features.shape[1] < max_dim:
        padding = torch.zeros(user_features.shape[0], max_dim - user_features.shape[1])
        user_features = torch.cat([user_features, padding], dim=1)
    
    if movie_features.shape[1] < max_dim:
        padding = torch.zeros(movie_features.shape[0], max_dim - movie_features.shape[1])
        movie_features = torch.cat([movie_features, padding], dim=1)
    
    # Combine features
    x = torch.cat([user_features, movie_features], dim=0)
    print(f"\n✅ Combined feature matrix shape: {x.shape}")

    # Create bipartite graph
    train_edge_index, train_edge_attr, train_ratings, test_ratings = create_bipartite_graph(
        ratings, num_users, num_movies
    )
    
    # Create node type mask (0=user, 1=movie)
    node_type = torch.zeros(num_nodes, dtype=torch.long)
    node_type[num_users:] = 1
    
    # ========================================
    # SAVE PROCESSED DATA
    # ========================================
    print("\n💾 Saving dataset...")
    print(f"   Output directory: {OUTPUT_DATA_DIR}")
    
    # Create directory structure
    public_dir = os.path.join(OUTPUT_DATA_DIR, 'public')
    os.makedirs(public_dir, exist_ok=True)
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    
    # ===== PUBLIC DATA (participants can see) =====
    
    print("\n📂 Saving PUBLIC files (data/public/)...")
    
    # 1. Save graph structure
    torch.save({
        'x': x,
        'edge_index': train_edge_index,
        'edge_attr': train_edge_attr,
        'node_type': node_type,
        'num_users': num_users,
        'num_movies': num_movies,
        'num_nodes': num_nodes,
        'num_features': x.shape[1]
    }, os.path.join(public_dir, 'graph_data.pt'))
    print("  ✓ graph_data.pt")
    
    # 2. Save train ratings (PUBLIC)
    train_ratings[['user_id', 'movie_id', 'rating']].to_csv(
        os.path.join(public_dir, 'train_ratings.csv'), index=False
    )
    print("  ✓ train_ratings.csv")
    
    # 3. Save test nodes (PUBLIC - NO LABELS!)
    # CRITICAL: Add 'id' column as first column
    test_nodes = test_ratings[['user_id', 'movie_id']].copy()
    test_nodes.insert(0, 'id', range(len(test_nodes)))
    test_nodes.to_csv(os.path.join(public_dir, 'test_nodes.csv'), index=False)
    print("  ✓ test_nodes.csv (with id column)")
    
    # 4. Save ID mapping (PUBLIC - helps participants map IDs to graph indices)
    id_mapping = pd.DataFrame({
        'id': range(len(test_ratings)),
        'user_id': test_ratings['user_id'].values,
        'movie_id': test_ratings['movie_id'].values,
        'user_idx': test_ratings['user_idx'].values,
        'movie_idx': test_ratings['movie_idx'].values
    })
    id_mapping.to_csv(os.path.join(public_dir, 'test_id_mapping.csv'), index=False)
    print("  ✓ test_id_mapping.csv")
    
    # 5. Sample submission (PUBLIC)
    # CRITICAL: Format is 'id, y_pred' NOT 'user_id, movie_id, rating'
    sample_submission = pd.DataFrame({
        'id': range(len(test_nodes)),
        'y_pred': 3.0  # Naive baseline: predict average rating
    })
    sample_submission.to_csv(os.path.join(public_dir, 'sample_submission.csv'), index=False)
    print("  ✓ sample_submission.csv (id, y_pred format)")
    
    # ===== PRIVATE DATA (for scoring only - DO NOT COMMIT) =====
    
    print("\n🔒 Saving PRIVATE files (data/)...")
    
    # 6. Save test labels (PRIVATE - includes ratings and cold_start flag)
    # CRITICAL: Add 'id' column to match test_nodes.csv
    test_labels = test_ratings[['user_id', 'movie_id', 'rating', 'is_cold_start']].copy()
    test_labels.insert(0, 'id', range(len(test_labels)))
    test_labels.to_csv(os.path.join(OUTPUT_DATA_DIR, 'test_labels.csv'), index=False)
    print("  ✓ test_labels.csv (with id and is_cold_start columns)")
    
    print("\n⚠️  IMPORTANT: data/test_labels.csv is PRIVATE!")
    print("   This file should be stored in GitHub Secrets, NOT committed to repo!")
    
    # ===== METADATA =====
    
    # 7. Save metadata
    metadata = {
        'num_users': num_users,
        'num_movies': num_movies,
        'num_nodes': num_nodes,
        'num_features': x.shape[1],
        'num_train_ratings': len(train_ratings),
        'num_test_ratings': len(test_ratings),
        'rating_min': float(ratings['rating'].min()),
        'rating_max': float(ratings['rating'].max()),
        'user_feature_dim': user_features.shape[1],
        'movie_feature_dim': movie_features.shape[1],
        'genre_cols': GENRE_COLS,
        'user_encoders': user_encoders
    }
    
    with open(os.path.join(OUTPUT_DATA_DIR, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    print("  ✓ metadata.pkl")
    
    # ========================================
    # SUMMARY
    # ========================================
    
    print("\n" + "=" * 60)
    print("✅ DATASET CREATED SUCCESSFULLY!")
    print("=" * 60)
    
    print("\n📁 File Structure:")
    print("data/")
    print("├── public/                     [COMMIT TO REPO]")
    print("│   ├── graph_data.pt")
    print("│   ├── train_ratings.csv")
    print("│   ├── test_nodes.csv          ← Has 'id' column")
    print("│   ├── test_id_mapping.csv")
    print("│   └── sample_submission.csv   ← Format: id, y_pred")
    print("├── test_labels.csv             [🔒 GITHUB SECRETS ONLY]")
    print("└── metadata.pkl")
    
    print("\n📊 Statistics:")
    print(f"  Training ratings: {len(train_ratings):,}")
    print(f"  Test pairs: {len(test_ratings):,}")
    print(f"  Cold-start in test: {test_ratings['is_cold_start'].sum()} ({test_ratings['is_cold_start'].sum()/len(test_ratings)*100:.1f}%)")
    
    print("\n📈 Rating Distribution:")
    for rating in sorted(ratings['rating'].unique()):
        count = (ratings['rating'] == rating).sum()
        pct = count / len(ratings) * 100
        print(f"  {int(rating)}⭐: {count:5d} ({pct:5.2f}%)")
    
    print("\n" + "=" * 60)
    print("🎉 READY FOR COMPETITION!")
    print("=" * 60)
    
    # print("\n📋 Next Steps:")
    # print("1. Verify file formats:")
    # print(f"   head {os.path.join(OUTPUT_DATA_DIR, 'public', 'test_nodes.csv')}")
    # print("   → Should show: id,user_id,movie_id")
    # print(f"   head {os.path.join(OUTPUT_DATA_DIR, 'public', 'sample_submission.csv')}")
    # print("   → Should show: id,y_pred")
    # print("")
    # print("2. Upload to GitHub:")
    # print(f"   git add {os.path.join(OUTPUT_DATA_DIR, 'public')}/")
    # print("   git commit -m 'Add dataset'")
    # print("")
    # print("3. Store test labels in GitHub Secrets:")
    # print("   → Settings → Secrets → New secret")
    # print("   → Name: TEST_LABELS")
    # print(f"   → Value: Contents of {os.path.join(OUTPUT_DATA_DIR, 'test_labels.csv')}")
    
    return metadata


if __name__ == "__main__":
    # Verify input directory exists
    if not os.path.exists(INPUT_DATA_DIR):
        print(f"❌ ERROR: Input directory '{INPUT_DATA_DIR}' not found!")
        print(f"   Please download MovieLens 100K and extract to '{INPUT_DATA_DIR}'")
        print(f"   Download from: https://grouplens.org/datasets/movielens/100k/")
        exit(1)
    
    # Check for required files
    required_files = ['u.data', 'u.user', 'u.item']
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(INPUT_DATA_DIR, f))]
    
    if missing_files:
        print(f"❌ ERROR: Missing required files in '{INPUT_DATA_DIR}':")
        for f in missing_files:
            print(f"   - {f}")
        exit(1)
    
    print(f"✓ Found MovieLens data in '{INPUT_DATA_DIR}'")
    print(f"✓ Will save output to '{OUTPUT_DATA_DIR}'")
    print("")
    
    create_dataset()